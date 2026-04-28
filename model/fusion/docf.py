# model/fusion/docf.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 兼容你工程不同位置的 CARAFE 定义
try:
    from ..modules.ops.carafe import CARAFE
except Exception:
    try:
        from ..carafe import CARAFE
    except Exception:
        from ...carafe import CARAFE


# -----------------------------
# 基础组件
# -----------------------------
class SepConv(nn.Module):
    """Depthwise Separable Conv：轻量、收敛稳"""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch, eps=1e-3, momentum=1e-2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)


class WeightedAdd(nn.Module):
    """DOCF 双向传播中的可学习加权；ReLU确保非负，L1 归一避免数值不稳"""
    def __init__(self, n_inputs: int, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(n_inputs, dtype=torch.float32))

    def forward(self, xs):
        assert len(xs) == len(self.w), "WeightedAdd: inputs and weights size mismatch."
        w = torch.relu(self.w)
        w = w / (torch.sum(w) + self.eps)
        out = 0.0
        for i, x in enumerate(xs):
            out = out + w[i] * x
        return out


def _maybe_down(x: torch.Tensor, target_hw):
    """下采样到目标尺寸（用于 Bottom-Up 路）"""
    Ht, Wt = target_hw
    H, W = x.shape[-2:]
    if (H, W) == (Ht, Wt):
        return x
    scale_h = max(1, H // max(1, Ht))
    scale_w = max(1, W // max(1, Wt))
    k = max(2, min(scale_h, scale_w))
    return F.max_pool2d(x, kernel_size=k, stride=k)


def _maybe_up_carafe(x: torch.Tensor, target_hw, carafe2: 'CARAFE', carafe4: 'CARAFE'):
    """上采样到 (Ht, Wt)：优先用 2×/4× CARAFE，否则最近邻"""
    Ht, Wt = target_hw
    H, W = x.shape[-2:]
    if (H, W) == (Ht, Wt):
        return x
    sh = max(1, Ht // max(1, H))
    sw = max(1, Wt // max(1, W))
    if sh == 2 and sw == 2 and carafe2 is not None:
        return carafe2(x)
    if sh == 4 and sw == 4 and carafe4 is not None:
        return carafe4(x)
    return F.interpolate(x, size=(Ht, Wt), mode='nearest')


# -----------------------------
# DOCF: Task-differentiated Dynamic-Alpha Gate
# -----------------------------
class DOCFDynamicAlphaGate(nn.Module):
    """
    输入：2D feat + 3D feat（同尺度，同通道 inter_ch）
    输出：一个样本自适应的缩放因子 r(x) ∈ [r_min, r_max]
    设计：GAP+GMP（全局语义+局部峰值），MLP 输出
    """
    def __init__(self, inter_ch: int, hidden: int = 64, r_min: float = 0.5, r_max: float = 1.5):
        super().__init__()
        assert r_max > r_min > 0
        self.r_min = float(r_min)
        self.r_max = float(r_max)

        in_dim = 4 * inter_ch  # 2D(GAP+GMP)=2C, 3D(GAP+GMP)=2C
        hidden = int(hidden)

        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Linear(hidden, 1)

        # 让初始输出 r(x)=1.0：在 [r_min, r_max] 里对应 sigmoid=0.5 -> bias=0
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    @staticmethod
    def _gap_gmp(x: torch.Tensor) -> torch.Tensor:
        gap = F.adaptive_avg_pool2d(x, 1).flatten(1)
        gmp = F.adaptive_max_pool2d(x, 1).flatten(1)
        return torch.cat([gap, gmp], dim=1)

    def forward(self, f2d: torch.Tensor, f3d: torch.Tensor) -> torch.Tensor:
        v2 = self._gap_gmp(f2d)
        v3 = self._gap_gmp(f3d)
        v = torch.cat([v2, v3], dim=1)
        y = self.fc2(self.act(self.fc1(v)))
        s = torch.sigmoid(y)  # [B,1] in (0,1)
        r = self.r_min + (self.r_max - self.r_min) * s  # [B,1]
        return r.view(-1, 1, 1, 1)  # broadcast


# -----------------------------
# DOCF: Orthogonal Motion Residual Injection
# -----------------------------
def docf_orthogonal_motion_residual(f3d: torch.Tensor, f2d: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    计算 3D 相对 2D 的“正交残差”：
      proj = <f3d,f2d> / (||f2d||^2+eps) * f2d
      ortho = f3d - proj
    按像素做通道内投影，避免 3D 注入重复空间信息导致冲突。
    """
    num = (f3d * f2d).sum(dim=1, keepdim=True)
    den = (f2d * f2d).sum(dim=1, keepdim=True).clamp_min(eps)
    proj = (num / den) * f2d
    return f3d - proj


class SigmoidScalar(nn.Module):
    """raw -> sigmoid(raw) in (0,1)"""
    def __init__(self, p_init: float = 0.5):
        super().__init__()
        p = float(max(1e-6, min(1 - 1e-6, p_init)))
        raw = math.log(p / (1.0 - p))
        self.raw = nn.Parameter(torch.tensor(raw, dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        return torch.sigmoid(self.raw)


# -----------------------------
# 单层 DOCF（含 box/cls 两路）
# -----------------------------
class DOCFLayer(nn.Module):
    """
    一个 DOCF 层：正交残差分解 + 任务差异化注入 + 双向多尺度传播（box/cls 两路分离）
    DOCF:
      - DA-α：alpha 由 “可学习标量 * 动态门控 r(x)” 控制（按样本）
      - Ortho-Motion：把 3D 注入换成 “(1-γ)*3D + γ*(3D ⟂ 2D)” 以减少冲突
    """
    def __init__(self,
                 inter_ch: int,
                 c3d: int,
                 use_carafe: bool = True,
                 gating: bool = True,
                 # DOCF cfg
                 dynamic_alpha_enabled: bool = False,
                 dynamic_alpha_hidden: int = 64,
                 dynamic_alpha_range=(0.5, 1.5),
                 orthogonal_enabled: bool = False,
                 orthogonal_eps: float = 1e-4,
                 orthogonal_gamma_init_box: float = 0.35,
                 orthogonal_gamma_init_cls: float = 0.65,
                 alpha_init_box=(0.0, 0.12, 0.12),
                 alpha_init_cls=(0.0, 1.25, 1.25)):
        super().__init__()
        self.use_carafe, self.gating = use_carafe, gating

        # Phase A toggles
        self.dynamic_alpha_enabled = bool(dynamic_alpha_enabled)
        self.orthogonal_enabled = bool(orthogonal_enabled)
        self.orthogonal_eps = float(orthogonal_eps)

        # 3D -> 2D 投影 + BN
        self.proj3d_p3 = nn.Conv2d(c3d, inter_ch, 1, 1, 0, bias=False)
        self.proj3d_p4 = nn.Conv2d(c3d, inter_ch, 1, 1, 0, bias=False)
        self.proj3d_p5 = nn.Conv2d(c3d, inter_ch, 1, 1, 0, bias=False)
        self.bn3d_p3 = nn.BatchNorm2d(inter_ch, eps=1e-3, momentum=1e-2)
        self.bn3d_p4 = nn.BatchNorm2d(inter_ch, eps=1e-3, momentum=1e-2)
        self.bn3d_p5 = nn.BatchNorm2d(inter_ch, eps=1e-3, momentum=1e-2)

        # CARAFE 对齐器
        if use_carafe:
            self.carafe2 = CARAFE(inter_ch, up_factor=2, kernel_size=5)
            self.carafe4 = CARAFE(inter_ch, up_factor=4, kernel_size=5)
        else:
            self.carafe2 = self.carafe4 = None

        # -------- Top-Down（权重融合）--------
        self.td_w_p5_box = WeightedAdd(2)
        self.td_w_p4_box = WeightedAdd(3)
        self.td_w_p3_box = WeightedAdd(3)

        self.td_w_p5_cls = WeightedAdd(2)
        self.td_w_p4_cls = WeightedAdd(3)
        self.td_w_p3_cls = WeightedAdd(3)

        self.td_p5_box = SepConv(inter_ch, inter_ch)
        self.td_p4_box = SepConv(inter_ch, inter_ch)
        self.td_p3_box = SepConv(inter_ch, inter_ch)

        self.td_p5_cls = SepConv(inter_ch, inter_ch)
        self.td_p4_cls = SepConv(inter_ch, inter_ch)
        self.td_p3_cls = SepConv(inter_ch, inter_ch)

        # -------- Bottom-Up（权重融合）--------
        self.bu_w_p3_box = WeightedAdd(2)
        self.bu_w_p4_box = WeightedAdd(3)
        self.bu_w_p5_box = WeightedAdd(2)

        self.bu_w_p3_cls = WeightedAdd(2)
        self.bu_w_p4_cls = WeightedAdd(3)
        self.bu_w_p5_cls = WeightedAdd(2)

        self.bu_p3_box = SepConv(inter_ch, inter_ch)
        self.bu_p4_box = SepConv(inter_ch, inter_ch)
        self.bu_p5_box = SepConv(inter_ch, inter_ch)

        self.bu_p3_cls = SepConv(inter_ch, inter_ch)
        self.bu_p4_cls = SepConv(inter_ch, inter_ch)
        self.bu_p5_cls = SepConv(inter_ch, inter_ch)

        # 可选通道门控（缓解冲突）
        if gating:
            self.gate_box = nn.Sequential(nn.Conv2d(inter_ch, inter_ch, 1, 1, 0), nn.Sigmoid())
            self.gate_cls = nn.Sequential(nn.Conv2d(inter_ch, inter_ch, 1, 1, 0), nn.Sigmoid())
        else:
            self.gate_box = self.gate_cls = None

        # -------- 基础 α（与你现有实现一致，默认强助 cls / 弱助 box / P3 不注入）--------
        self.alpha_box_p3 = nn.Parameter(torch.tensor(float(alpha_init_box[0])))
        self.alpha_box_p4 = nn.Parameter(torch.tensor(float(alpha_init_box[1])))
        self.alpha_box_p5 = nn.Parameter(torch.tensor(float(alpha_init_box[2])))

        self.alpha_cls_p3 = nn.Parameter(torch.tensor(float(alpha_init_cls[0])))
        self.alpha_cls_p4 = nn.Parameter(torch.tensor(float(alpha_init_cls[1])))
        self.alpha_cls_p5 = nn.Parameter(torch.tensor(float(alpha_init_cls[2])))

        # -------- DOCF: 动态 α 门控（每尺度、每任务独立；更稳、更可控）--------
        if self.dynamic_alpha_enabled:
            r_min, r_max = dynamic_alpha_range
            self.da_box_p3 = DOCFDynamicAlphaGate(inter_ch, hidden=dynamic_alpha_hidden, r_min=r_min, r_max=r_max)
            self.da_box_p4 = DOCFDynamicAlphaGate(inter_ch, hidden=dynamic_alpha_hidden, r_min=r_min, r_max=r_max)
            self.da_box_p5 = DOCFDynamicAlphaGate(inter_ch, hidden=dynamic_alpha_hidden, r_min=r_min, r_max=r_max)
            self.da_cls_p3 = DOCFDynamicAlphaGate(inter_ch, hidden=dynamic_alpha_hidden, r_min=r_min, r_max=r_max)
            self.da_cls_p4 = DOCFDynamicAlphaGate(inter_ch, hidden=dynamic_alpha_hidden, r_min=r_min, r_max=r_max)
            self.da_cls_p5 = DOCFDynamicAlphaGate(inter_ch, hidden=dynamic_alpha_hidden, r_min=r_min, r_max=r_max)
        else:
            self.da_box_p3 = self.da_box_p4 = self.da_box_p5 = None
            self.da_cls_p3 = self.da_cls_p4 = self.da_cls_p5 = None

        # -------- DOCF: Ortho-Motion 混合系数 γ（box/cls 分离）--------
        if self.orthogonal_enabled:
            self.gamma_box = SigmoidScalar(p_init=orthogonal_gamma_init_box)
            self.gamma_cls = SigmoidScalar(p_init=orthogonal_gamma_init_cls)
        else:
            self.gamma_box = self.gamma_cls = None

    def _alpha(self, base: torch.Tensor, gate: torch.Tensor = None) -> torch.Tensor:
        """alpha = relu(base) * (gate if not None else 1)"""
        a = torch.relu(base)
        if gate is None:
            return a
        return a * gate

    def _mix_3d(self, f3d: torch.Tensor, f2d: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """
        f_inj = (1-γ)*f3d + γ*(f3d ⟂ f2d)
        """
        if gamma is None:
            return f3d
        ortho = docf_orthogonal_motion_residual(f3d, f2d, eps=self.orthogonal_eps)
        return (1.0 - gamma) * f3d + gamma * ortho

    def forward(self, ft_2d, ft_3d, hw_list):
        """
        ft_2d: [[box_p3, cls_p3], [box_p4, cls_p4], [box_p5, cls_p5]]  （各 [B,C,H,W]）
        ft_3d: [B, C3D, H5, W5]  或 [B, C3D, 1, H5, W5]
        hw_list: [(H3,W3), (H4,W4), (H5,W5)]
        """
        (H3, W3), (H4, W4), (H5, W5) = hw_list

        # 兼容含时间维的 3D 特征
        if ft_3d.dim() == 5:
            ft_3d = ft_3d.squeeze(2) if ft_3d.shape[2] == 1 else ft_3d.mean(dim=2)

        # 3D 投影 + BN 对齐
        proj_p5 = self.bn3d_p5(self.proj3d_p5(ft_3d))
        proj_p4 = self.bn3d_p4(self.proj3d_p4(ft_3d))
        proj_p3 = self.bn3d_p3(self.proj3d_p3(ft_3d))

        f3d_p5 = proj_p5
        f3d_p4 = _maybe_up_carafe(proj_p4, (H4, W4), self.carafe2, self.carafe4)
        f3d_p3 = _maybe_up_carafe(proj_p3, (H3, W3), self.carafe2, self.carafe4)

        # 2D 分支（已是 inter_ch）
        b3, c3 = ft_2d[0][0], ft_2d[0][1]
        b4, c4 = ft_2d[1][0], ft_2d[1][1]
        b5, c5 = ft_2d[2][0], ft_2d[2][1]

        # -------- DOCF: Ortho-Motion（减少 2D/3D 冗余冲突）--------
        if self.orthogonal_enabled and (self.gamma_box is not None) and (self.gamma_cls is not None):
            gb = self.gamma_box()  # scalar in (0,1)
            gc = self.gamma_cls()
            # 每尺度对齐到各自 2D 表征
            f3d_p3_box = self._mix_3d(f3d_p3, b3, gb)
            f3d_p4_box = self._mix_3d(f3d_p4, b4, gb)
            f3d_p5_box = self._mix_3d(f3d_p5, b5, gb)

            f3d_p3_cls = self._mix_3d(f3d_p3, c3, gc)
            f3d_p4_cls = self._mix_3d(f3d_p4, c4, gc)
            f3d_p5_cls = self._mix_3d(f3d_p5, c5, gc)
        else:
            f3d_p3_box = f3d_p4_box = f3d_p5_box = None
            f3d_p3_cls = f3d_p4_cls = f3d_p5_cls = None

        # -------- DOCF: Dynamic-Alpha（按样本调节注入强度）--------
        if self.dynamic_alpha_enabled:
            g_box_p3 = self.da_box_p3(b3, (f3d_p3_box if f3d_p3_box is not None else f3d_p3))
            g_box_p4 = self.da_box_p4(b4, (f3d_p4_box if f3d_p4_box is not None else f3d_p4))
            g_box_p5 = self.da_box_p5(b5, (f3d_p5_box if f3d_p5_box is not None else f3d_p5))

            g_cls_p3 = self.da_cls_p3(c3, (f3d_p3_cls if f3d_p3_cls is not None else f3d_p3))
            g_cls_p4 = self.da_cls_p4(c4, (f3d_p4_cls if f3d_p4_cls is not None else f3d_p4))
            g_cls_p5 = self.da_cls_p5(c5, (f3d_p5_cls if f3d_p5_cls is not None else f3d_p5))
        else:
            g_box_p3 = g_box_p4 = g_box_p5 = None
            g_cls_p3 = g_cls_p4 = g_cls_p5 = None

        # 选择最终注入特征（Ortho 打开就用混合后的，否则用原始）
        inj_b3 = f3d_p3_box if f3d_p3_box is not None else f3d_p3
        inj_b4 = f3d_p4_box if f3d_p4_box is not None else f3d_p4
        inj_b5 = f3d_p5_box if f3d_p5_box is not None else f3d_p5

        inj_c3 = f3d_p3_cls if f3d_p3_cls is not None else f3d_p3
        inj_c4 = f3d_p4_cls if f3d_p4_cls is not None else f3d_p4
        inj_c5 = f3d_p5_cls if f3d_p5_cls is not None else f3d_p5

        # 计算 alpha（非负化 + 动态门控）
        ab3 = self._alpha(self.alpha_box_p3, g_box_p3)
        ab4 = self._alpha(self.alpha_box_p4, g_box_p4)
        ab5 = self._alpha(self.alpha_box_p5, g_box_p5)

        ac3 = self._alpha(self.alpha_cls_p3, g_cls_p3)
        ac4 = self._alpha(self.alpha_cls_p4, g_cls_p4)
        ac5 = self._alpha(self.alpha_cls_p5, g_cls_p5)

        # ---------- Top-Down ----------
        td_b5 = self.td_p5_box(self.td_w_p5_box([b5, ab5 * inj_b5]))
        td_c5 = self.td_p5_cls(self.td_w_p5_cls([c5, ac5 * inj_c5]))

        up_b5 = _maybe_up_carafe(td_b5, (H4, W4), self.carafe2, self.carafe4)
        up_c5 = _maybe_up_carafe(td_c5, (H4, W4), self.carafe2, self.carafe4)

        td_b4 = self.td_p4_box(self.td_w_p4_box([b4, up_b5, ab4 * inj_b4]))
        td_c4 = self.td_p4_cls(self.td_w_p4_cls([c4, up_c5, ac4 * inj_c4]))

        up_b4 = _maybe_up_carafe(td_b4, (H3, W3), self.carafe2, self.carafe4)
        up_c4 = _maybe_up_carafe(td_c4, (H3, W3), self.carafe2, self.carafe4)

        td_b3 = self.td_p3_box(self.td_w_p3_box([b3, up_b4, ab3 * inj_b3]))
        td_c3 = self.td_p3_cls(self.td_w_p3_cls([c3, up_c4, ac3 * inj_c3]))

        # ---------- Bottom-Up ----------
        dn_b3 = self.bu_p3_box(self.bu_w_p3_box([td_b3, ab3 * inj_b3]))
        dn_c3 = self.bu_p3_cls(self.bu_w_p3_cls([td_c3, ac3 * inj_c3]))

        dn_b4 = self.bu_p4_box(self.bu_w_p4_box([td_b4, _maybe_down(dn_b3, (H4, W4)), ab4 * inj_b4]))
        dn_c4 = self.bu_p4_cls(self.bu_w_p4_cls([td_c4, _maybe_down(dn_c3, (H4, W4)), ac4 * inj_c4]))

        dn_b5 = self.bu_p5_box(self.bu_w_p5_box([td_b5, _maybe_down(dn_b4, (H5, W5))]))
        dn_c5 = self.bu_p5_cls(self.bu_w_p5_cls([td_c5, _maybe_down(dn_c4, (H5, W5))]))

        # 可选通道门控（缓解冲突）
        if self.gating and self.gate_box is not None:
            g3 = self.gate_box(dn_b3); dn_b3 = g3 * dn_b3 + (1.0 - g3) * td_b3
            g4 = self.gate_box(dn_b4); dn_b4 = g4 * dn_b4 + (1.0 - g4) * td_b4
            g5 = self.gate_box(dn_b5); dn_b5 = g5 * dn_b5 + (1.0 - g5) * td_b5

        if self.gating and self.gate_cls is not None:
            g3 = self.gate_cls(dn_c3); dn_c3 = g3 * dn_c3 + (1.0 - g3) * td_c3
            g4 = self.gate_cls(dn_c4); dn_c4 = g4 * dn_c4 + (1.0 - g4) * td_c4
            g5 = self.gate_cls(dn_c5); dn_c5 = g5 * dn_c5 + (1.0 - g5) * td_c5

        return [[dn_b3, dn_c3], [dn_b4, dn_c4], [dn_b5, dn_c5]]


# -----------------------------
# 外层 DOCFFusion（与 build_fusion.py 匹配）
# -----------------------------
class DOCFFusion(nn.Module):
    """
    - channels_2D: [[Cbox3,Ccls3],[Cbox4,Ccls4],[Cbox5,Ccls5]]
    - channels_3D: C3D (int)
    - interchannels: 融合输出通道（与 head filters 契约一致）
    - mode: 'decoupled'
    - num_layers/use_carafe/gating: from YAML
    - docf_cfg: dict（支持动态α/正交注入开关）
    - hw_info: [(H3,W3),(H4,W4),(H5,W5)]
    """
    def __init__(self,
                 channels_2D,
                 channels_3D: int,
                 interchannels: int,
                 mode: str = 'decoupled',
                 num_layers: int = 2,
                 use_carafe: bool = True,
                 gating: bool = True,
                 hw_info=None,
                 docf_cfg=None):
        super().__init__()
        assert mode == 'decoupled', "当前工程假设 decoupled 头"
        assert hw_info is not None and len(hw_info) == 3, "DOCFFusion 需要 [(H3,W3),(H4,W4),(H5,W5)]"
        self.hw_info = [(int(h), int(w)) for (h, w) in hw_info]

        # 2D -> interchannels 适配并归一（BN）
        self.adapt_p3_box = nn.Conv2d(channels_2D[0][0], interchannels, 1, 1, 0, bias=False)
        self.adapt_p4_box = nn.Conv2d(channels_2D[1][0], interchannels, 1, 1, 0, bias=False)
        self.adapt_p5_box = nn.Conv2d(channels_2D[2][0], interchannels, 1, 1, 0, bias=False)

        self.adapt_p3_cls = nn.Conv2d(channels_2D[0][1], interchannels, 1, 1, 0, bias=False)
        self.adapt_p4_cls = nn.Conv2d(channels_2D[1][1], interchannels, 1, 1, 0, bias=False)
        self.adapt_p5_cls = nn.Conv2d(channels_2D[2][1], interchannels, 1, 1, 0, bias=False)

        self.bn_p3_box = nn.BatchNorm2d(interchannels, eps=1e-3, momentum=1e-2)
        self.bn_p4_box = nn.BatchNorm2d(interchannels, eps=1e-3, momentum=1e-2)
        self.bn_p5_box = nn.BatchNorm2d(interchannels, eps=1e-3, momentum=1e-2)

        self.bn_p3_cls = nn.BatchNorm2d(interchannels, eps=1e-3, momentum=1e-2)
        self.bn_p4_cls = nn.BatchNorm2d(interchannels, eps=1e-3, momentum=1e-2)
        self.bn_p5_cls = nn.BatchNorm2d(interchannels, eps=1e-3, momentum=1e-2)

        # DOCF cfg defaults
        docf_cfg = docf_cfg or {}
        dynamic_alpha_enabled = bool(docf_cfg.get('dynamic_alpha_enabled', False))
        dynamic_alpha_hidden = int(docf_cfg.get('dynamic_alpha_hidden', 64))
        da_range = docf_cfg.get('dynamic_alpha_range', [0.5, 1.5])
        if isinstance(da_range, (list, tuple)) and len(da_range) == 2:
            dynamic_alpha_range = (float(da_range[0]), float(da_range[1]))
        else:
            dynamic_alpha_range = (0.5, 1.5)

        orth_enabled = bool(docf_cfg.get('orthogonal_enabled', False))
        orth_eps = float(docf_cfg.get('orthogonal_eps', 1e-4))
        orth_gamma_init_box = float(docf_cfg.get('orthogonal_gamma_init_box', 0.35))
        orth_gamma_init_cls = float(docf_cfg.get('orthogonal_gamma_init_cls', 0.65))

        # 可选：alpha init（不写则用默认）
        alpha_init = docf_cfg.get('alpha_init', {}) or {}
        alpha_box = alpha_init.get('box', [0.0, 0.12, 0.12])
        alpha_cls = alpha_init.get('cls', [0.0, 1.25, 1.25])
        if not (isinstance(alpha_box, (list, tuple)) and len(alpha_box) == 3):
            alpha_box = [0.0, 0.12, 0.12]
        if not (isinstance(alpha_cls, (list, tuple)) and len(alpha_cls) == 3):
            alpha_cls = [0.0, 1.25, 1.25]

        # 堆叠 DOCF 层
        self.layers = nn.ModuleList([
            DOCFLayer(
                interchannels, channels_3D,
                use_carafe=use_carafe,
                gating=gating,
                dynamic_alpha_enabled=dynamic_alpha_enabled,
                dynamic_alpha_hidden=dynamic_alpha_hidden,
                dynamic_alpha_range=dynamic_alpha_range,
                orthogonal_enabled=orth_enabled,
                orthogonal_eps=orth_eps,
                orthogonal_gamma_init_box=orth_gamma_init_box,
                orthogonal_gamma_init_cls=orth_gamma_init_cls,
                alpha_init_box=alpha_box,
                alpha_init_cls=alpha_cls,
            )
            for _ in range(int(num_layers))
        ])

    def forward(self, ft_2d, ft_3d):
        """
        ft_2d: [[box_p3, cls_p3], [box_p4, cls_p4], [box_p5, cls_p5]]
        ft_3d: [B, C3D, H5, W5] 或 [B, C3D, 1, H5, W5]
        """
        # 3D 维度兼容
        if ft_3d.dim() == 5:
            ft_3d = ft_3d.squeeze(2) if ft_3d.shape[2] == 1 else ft_3d.mean(dim=2)

        # 统一 2D 通道到 interchannels + BN
        b3 = self.bn_p3_box(self.adapt_p3_box(ft_2d[0][0])); c3 = self.bn_p3_cls(self.adapt_p3_cls(ft_2d[0][1]))
        b4 = self.bn_p4_box(self.adapt_p4_box(ft_2d[1][0])); c4 = self.bn_p4_cls(self.adapt_p4_cls(ft_2d[1][1]))
        b5 = self.bn_p5_box(self.adapt_p5_box(ft_2d[2][0])); c5 = self.bn_p5_cls(self.adapt_p5_cls(ft_2d[2][1]))
        feats = [[b3, c3], [b4, c4], [b5, c5]]

        # 堆叠 DOCF 层
        for ly in self.layers:
            feats = ly(feats, ft_3d, self.hw_info)

        return feats
