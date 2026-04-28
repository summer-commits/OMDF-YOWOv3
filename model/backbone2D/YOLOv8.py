# YOLOv8.py  - 可直接替换的增强版（STPE + sparse temporal compensation + content-adaptive reassembly）
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def pad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p


def fuse_conv(conv, norm):
    fused_conv = nn.Conv2d(conv.in_channels,
                           conv.out_channels,
                           kernel_size=conv.kernel_size,
                           stride=conv.stride,
                           padding=conv.padding,
                           groups=conv.groups,
                           bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)
    return fused_conv


# ------------------------------------------------------------
# Basic blocks
# ------------------------------------------------------------
class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None, d=1, g=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, pad(k, p, d), d, g, False)
        self.norm = nn.BatchNorm2d(out_ch, 0.001, 0.03)
        self.relu = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))


# ------------------------------------------------------------
# Involution + GRN （稳态替代：仅用于 P4/P5）
# ------------------------------------------------------------
class Involution2d(nn.Module):
    """
    Involution (CVPR2021): 位置自适应的卷积核（不扩大感受野、不引入强形变），更稳
    """
    def __init__(self, channels, kernel_size=5, stride=1, reduction=4):
        super().__init__()
        self.kernel_size, self.stride = kernel_size, stride
        hidden = max(channels // reduction, 16)
        self.reduce = nn.Conv2d(channels, hidden, 1)
        self.span   = nn.Conv2d(hidden, kernel_size * kernel_size, 1)
        self.unfold = nn.Unfold(kernel_size, padding=kernel_size // 2, stride=stride)
        self.proj   = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        weight = self.span(torch.relu_(self.reduce(x)))           # [B, k^2, H, W]
        B, C, H, W = x.shape
        K = self.kernel_size
        unfold = self.unfold(x).view(B, C, K * K, H // self.stride, W // self.stride)
        weight = weight.view(B, 1, K * K, H // self.stride, W // self.stride)
        out = (unfold * weight).sum(dim=2)                        # [B, C, H', W']
        return self.proj(out)


class GRN(nn.Module):
    """ Global Response Normalization (ConvNeXt-V2, 2023) """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.eps   = eps

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = x / (gx + self.eps)
        return self.gamma * (x * nx) + self.beta + x


class Residual(nn.Module):
    """
    残差块：默认 3x3 Conv + 3x3 Conv
    当 use_invo=True 时，替换中间 3x3 为 Involution，并在块尾加入 GRN 稳定训练
    """
    def __init__(self, ch, add=True, use_invo=False):
        super().__init__()
        self.add_m = add
        self.use_invo = use_invo
        if use_invo:
            self.res_m = nn.Sequential(
                nn.Conv2d(ch, ch, 1, bias=False), nn.BatchNorm2d(ch), nn.SiLU(True),
                Involution2d(ch, kernel_size=5, stride=1, reduction=4),
                GRN(ch),
                nn.Conv2d(ch, ch, 1, bias=False), nn.BatchNorm2d(ch)
            )
        else:
            self.res_m = nn.Sequential(Conv(ch, ch, 3), Conv(ch, ch, 3))

    def forward(self, x):
        y = self.res_m(x)
        return y + x if self.add_m else y


class CSP(nn.Module):
    def __init__(self, in_ch, out_ch, n=1, add=True, use_invo=False):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch // 2)
        self.conv2 = Conv(in_ch, out_ch // 2)
        self.conv3 = Conv((2 + n) * out_ch // 2, out_ch)
        self.res_m = nn.ModuleList(Residual(out_ch // 2, add, use_invo=use_invo) for _ in range(n))

    def forward(self, x):
        y = [self.conv1(x), self.conv2(x)]
        y.extend(m(y[-1]) for m in self.res_m)
        return self.conv3(torch.cat(y, dim=1))


class SPP(nn.Module):
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        self.conv1 = Conv(in_ch, in_ch // 2)
        self.conv2 = Conv(in_ch * 2, out_ch)
        self.res_m = nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.res_m(x)
        y2 = self.res_m(y1)
        return self.conv2(torch.cat([x, y1, y2, self.res_m(y2)], 1))


# ------------------------------------------------------------
# CARAFE++ (内容自适应上采样)
# ------------------------------------------------------------
class CARAFEpp(nn.Module):
    """
    轻量 CARAFE++：content-aware reassembly
    用于替代 F.interpolate/nn.Upsample，提升边界与小目标细节回流
    """
    def __init__(self, in_channels, scale=2, kernel_size=5, compress=4, group=1):
        super().__init__()
        self.scale, self.kernel_size, self.group = scale, kernel_size, group
        mid = max(in_channels // compress, 16)
        self.comp = nn.Conv2d(in_channels, mid, 1, bias=True)
        self.enc  = nn.Conv2d(mid, (kernel_size * kernel_size) * (scale * scale) * group, 3, padding=1, bias=True)

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, h, w = x.shape
        s, k, g = self.scale, self.kernel_size, self.group
        assert c % g == 0, f"CARAFEpp requires C divisible by group, got C={c}, group={g}"

        # 编码得到重组核权重，形状组织为 [B, g, s*s, k*k, H, W]
        feat = F.relu_(self.comp(x))
        w_k = self.enc(feat)  # [B, (k*k)*(s*s)*g, H, W]
        w_k = w_k.view(b, g, s * s, k * k, h, w)
        w_k = torch.softmax(w_k, dim=3)  # 在 k*k 维度上归一化

        # 展开输入邻域： [B, g, Cg, k*k, H, W]
        pad = (k - 1) // 2
        x_unf = F.unfold(x, kernel_size=k, padding=pad)  # [B, C*k*k, H*W]
        x_unf = x_unf.view(b, g, c // g, k * k, h, w)

        # 加权重组：对 k*k 维度求和，得到 [B, g, Cg, s*s, H, W]
        # 维度对齐：w_k -> [B,g,1,s*s,k*k,H,W]；x_unf -> [B,g,Cg,1,k*k,H,W]
        out = (w_k.unsqueeze(2) * x_unf.unsqueeze(3)).sum(dim=4)

        # 还原到 [B, C, H*s, W*s]
        out = out.view(b, c, s * s, h, w)  # [B, C, s*s, H, W]
        out = out.permute(0, 1, 3, 4, 2).contiguous()  # [B, C, H, W, s*s]
        out = out.view(b, c, h * s, w * s)
        return out


# ------------------------------------------------------------
# DarkNet Backbone（支持在 P4/P5 打开 Involution）
# ------------------------------------------------------------
class DarkNet(nn.Module):
    def __init__(self, width, depth, cfg2d=None):
        super().__init__()
        # 读取开关（来自 yaml: BACKBONE2D.YOLOv8.BLOCKS.*）
        blocks_cfg = {}
        if cfg2d is not None:
            blocks_cfg = cfg2d.get('BLOCKS', {})
        use_invo_p4 = bool(blocks_cfg.get('involution_p4_enabled', False))
        use_invo_p5 = bool(blocks_cfg.get('involution_p5_enabled', False))

        p1 = [Conv(width[0], width[1], 3, 2)]
        p2 = [Conv(width[1], width[2], 3, 2),
              CSP(width[2], width[2], depth[0], use_invo=False)]
        p3 = [Conv(width[2], width[3], 3, 2),
              CSP(width[3], width[3], depth[1], use_invo=False)]
        p4 = [Conv(width[3], width[4], 3, 2),
              CSP(width[4], width[4], depth[2], use_invo=use_invo_p4)]
        p5 = [Conv(width[4], width[5], 3, 2),
              CSP(width[5], width[5], depth[0], use_invo=use_invo_p5),
              SPP(width[5], width[5])]

        self.p1 = nn.Sequential(*p1)
        self.p2 = nn.Sequential(*p2)
        self.p3 = nn.Sequential(*p3)
        self.p4 = nn.Sequential(*p4)
        self.p5 = nn.Sequential(*p5)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p3, p4, p5  # P3(1/8), P4(1/16), P5(1/32)


# ------------------------------------------------------------
# STPE（Spatially-Aware Texture and Pattern Enhancement：语义筛选 + 短时补偿 + 结构重组）
# ------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, q_in, k_in, v_in):
        B, Nq, D = q_in.shape
        Nk = k_in.shape[1]
        q = self.q(q_in).view(B, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(k_in).view(B, Nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(v_in).view(B, Nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, Nq, D)
        out = self.proj(out)
        return out


class STPEBlock(nn.Module):
    def __init__(self, in_channels_list, embed_dim=128, k_coarse=8, k_fine=24, num_heads=4, dropout=0.0):
        super().__init__()
        assert len(in_channels_list) == 3, "expect three scales (p3, p4, p5)"
        self.in_ch = in_channels_list
        self.embed_dim = embed_dim
        self.k_coarse = k_coarse
        self.k_fine = k_fine
        self.num_heads = num_heads

        self.proj_in = nn.ModuleList([nn.Conv2d(c, embed_dim, kernel_size=1, bias=False) for c in in_channels_list])
        self.proj_out = nn.ModuleList([nn.Conv2d(embed_dim, c, kernel_size=1, bias=False) for c in in_channels_list])

        self.token_score = nn.Sequential(
            nn.Linear(embed_dim, max(embed_dim // 4, 1)),
            nn.SiLU(inplace=True),
            nn.Linear(max(embed_dim // 4, 1), 1)
        )

        self.coarse_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.cross_attn  = MultiHeadAttention(embed_dim, num_heads, dropout)

        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(inplace=True),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _to_tokens(self, x, proj):
        B, C, H, W = x.shape
        t = proj(x).flatten(2).permute(0, 2, 1).contiguous()
        return t, H, W

    def _gather_by_index(self, tokens, idx):
        B = tokens.shape[0]
        batch_idx = torch.arange(B, device=tokens.device).unsqueeze(1)
        return tokens[batch_idx, idx]

    def forward(self, feats):
        assert len(feats) == 3
        B = feats[0].shape[0]
        tokens_list, HWs, Ss = [], [], []
        for i, x in enumerate(feats):
            t, H, W = self._to_tokens(x, self.proj_in[i])
            tokens_list.append(t)
            HWs.append((H, W))
            Ss.append(t.shape[1])

        device = tokens_list[0].device
        D = self.embed_dim

        coarse_idx, fine_idx = [], []
        for t, S in zip(tokens_list, Ss):
            scores = self.token_score(self.norm(t)).squeeze(-1)
            kc = min(self.k_coarse, S); kf = min(self.k_fine, S)
            ic = torch.topk(scores, kc, dim=1)[1] if kc > 0 else torch.zeros((B, 0), dtype=torch.long, device=device)
            if kf <= 0:
                fi = torch.zeros((B, 0), dtype=torch.long, device=device)
            elif kf <= kc:
                fi = ic[:, :kf]
            else:
                fi = torch.topk(scores, kf, dim=1)[1]
            coarse_idx.append(ic); fine_idx.append(fi)

        coarse_toks = [self._gather_by_index(t, idx) for t, idx in zip(tokens_list, coarse_idx) if idx.shape[1] > 0]
        if len(coarse_toks) == 0:
            return feats
        coarse_cat = torch.cat(coarse_toks, dim=1)

        coarse_upd = self.coarse_attn(coarse_cat, coarse_cat, coarse_cat)
        coarse_upd = self.dropout(self.ffn(self.norm(coarse_upd)) + coarse_upd)

        out_feats = []
        for i, (orig_t, idx_f, S, (H, W)) in enumerate(zip(tokens_list, fine_idx, Ss, HWs)):
            if idx_f.shape[1] == 0:
                out_t = orig_t
            else:
                fine_sel = self._gather_by_index(orig_t, idx_f)
                updated = self.cross_attn(fine_sel, coarse_upd, coarse_upd)
                updated = self.dropout(self.ffn(self.norm(updated)) + updated)
                out_t = orig_t.clone()
                batch_idx = torch.arange(B, device=device).unsqueeze(1)
                out_t[batch_idx, idx_f] = out_t[batch_idx, idx_f] + updated

            out_map = out_t.permute(0, 2, 1).contiguous().view(B, D, H, W)
            out_map = self.proj_out[i](out_map)
            out_feats.append(out_map)

        return out_feats

# ------------------------------------------------------------
# Temporal Shift（与现有一致，放在残差支路思路）
# ------------------------------------------------------------
class TemporalShift(nn.Module):
    def __init__(self, n_segment=8, fold_div=16):
        super().__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x):
        orig_dim = x.dim()
        if orig_dim == 4:
            nt, c, h, w = x.shape
            if self.n_segment is None or self.n_segment <= 0:
                n = 1; t = nt
            else:
                if nt % self.n_segment == 0:
                    n = nt // self.n_segment; t = self.n_segment
                else:
                    n = 1; t = nt
            if t == 1:
                return x
            x = x.view(n, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
        elif orig_dim == 5:
            n, c, t, h, w = x.shape
            if t == 1:
                return x
        else:
            raise ValueError("TemporalShift expects 4D or 5D tensor")

        n, c, t, h, w = x.size()
        fold = max(1, c // self.fold_div) if self.fold_div > 0 else 0
        if fold == 0:
            out = x
        else:
            out = x.clone()
            out[:, :fold, :-1] = x[:, :fold, 1:]   # 前移
            out[:, fold:2*fold, 1:] = x[:, fold:2*fold, :-1]  # 后移
        if orig_dim == 4:
            out = out.permute(0, 2, 1, 3, 4).contiguous().view(n * t, c, h, w)
        return out


# ------------------------------------------------------------
# PAdapter（可选，带 learnable alpha）
# ------------------------------------------------------------
class PAdapter(nn.Module):
    def __init__(self, in_channels=None, bottleneck=32, n_segment=8, alpha_init=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.bottleneck = bottleneck
        self.n_segment = n_segment
        self.initialized = False
        self.conv1 = None; self.dw = None; self.conv2 = None
        self.act = nn.ReLU(inplace=True)

        # raw param -> sigmoid(raw)=alpha_init
        raw = math.log(alpha_init / (1.0 - alpha_init + 1e-12) + 1e-12)
        self.alpha_raw = nn.Parameter(torch.tensor(raw, dtype=torch.float32))

    def _lazy_init(self, in_c, device):
        mid = self.bottleneck if self.bottleneck is not None else max(in_c // 4, 32)
        self.conv1 = nn.Conv3d(in_c, mid, kernel_size=1, bias=True)
        self.dw    = nn.Conv3d(mid, mid, kernel_size=(3, 3, 3), padding=1, groups=mid, bias=True)
        self.conv2 = nn.Conv3d(mid, in_c, kernel_size=1, bias=True)
        self.initialized = True
        self.to(device)

    def forward(self, x):
        orig_4d = False
        if x.dim() == 4:
            nt, c, h, w = x.shape
            if self.n_segment is None or self.n_segment <= 0:
                n = 1; t = nt
            else:
                if nt % self.n_segment == 0:
                    n = nt // self.n_segment; t = self.n_segment
                else:
                    n = 1; t = nt
            if t == 1:
                return x
            x = x.view(n, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
            orig_4d = True
        elif x.dim() == 5:
            n, c, t, h, w = x.shape
            if t == 1:
                return x
        else:
            raise ValueError("PAdapter expects 4D or 5D tensor")

        if not self.initialized:
            self._lazy_init(x.size(1), x.device)

        res = self.conv1(x)
        res = self.act(self.dw(res))
        res = self.conv2(res)

        alpha = torch.sigmoid(self.alpha_raw)
        out = x + alpha * res

        if orig_4d:
            n, c, t, h, w = out.size()
            out = out.permute(0, 2, 1, 3, 4).contiguous().view(n * t, c, h, w)
        return out


# ------------------------------------------------------------
# DarkFPN（加 STPE、可选 TSM / PAdapter、可选 CARAFE++ 上采样）
# ------------------------------------------------------------
class DarkFPN(nn.Module):
    def __init__(self, width, depth, cfg2d=None):
        super().__init__()
        # neck blocks
        self.h1 = CSP(width[4] + width[5], width[4], depth[0], False)
        self.h2 = CSP(width[3] + width[4], width[3], depth[0], False)
        self.h3 = Conv(width[3], width[3], 3, 2)
        self.h4 = CSP(width[3] + width[4], width[4], depth[0], False)
        self.h5 = Conv(width[4], width[4], 3, 2)
        self.h6 = CSP(width[4] + width[5], width[5], depth[0], False)

        # 默认开关
        self.stpe_enabled = False
        self.temp_shift_enabled = False
        self.padapter_enabled = False
        self.carafepp_enabled = False

        # 缺省参数
        self.n_segment = 1
        self.temp_shift_fold_div = 16
        self.padapter_bottleneck = 32
        self.padapter_alpha_init = 0.1

        # 每尺度开关
        self.temp_shift_p3 = False
        self.temp_shift_p4 = True
        self.temp_shift_p5 = True
        self.padapter_p3 = False
        self.padapter_p4 = False
        self.padapter_p5 = False

        # 从 cfg2d 解析（对应 yaml: BACKBONE2D.YOLOv8.*）
        stpe_cfg = {}
        neck_cfg = {}
        if cfg2d is not None:
            stpe_cfg = cfg2d.get('STPE', {}) or {}
            neck_cfg = cfg2d.get('NECK', {}) or {}

            # STPE
            if stpe_cfg.get('enabled', False):
                in_channels_list = [width[3], width[4], width[5]]
                self.stpe = STPEBlock(
                    in_channels_list=in_channels_list,
                    embed_dim=stpe_cfg.get('embed_dim', 128),
                    k_coarse=stpe_cfg.get('k_coarse', 8),
                    k_fine=stpe_cfg.get('k_fine', 24),
                    num_heads=stpe_cfg.get('num_heads', 4),
                    dropout=stpe_cfg.get('dropout', 0.0)
                )
                self.stpe_enabled = True

            # Temporal Shift
            self.temp_shift_enabled = bool(stpe_cfg.get('temp_shift_enabled', stpe_cfg.get('TEMP_SHIFT_ENABLED', False)))
            self.temp_shift_fold_div = int(stpe_cfg.get('temp_shift_fold_div', stpe_cfg.get('TEMP_SHIFT_FOLD_DIV', 16)))
            self.temp_shift_p3 = bool(stpe_cfg.get('temp_shift_p3_enabled', stpe_cfg.get('TEMP_SHIFT_P3_ENABLED', False)))
            self.temp_shift_p4 = bool(stpe_cfg.get('temp_shift_p4_enabled', stpe_cfg.get('TEMP_SHIFT_P4_ENABLED', True)))
            self.temp_shift_p5 = bool(stpe_cfg.get('temp_shift_p5_enabled', stpe_cfg.get('TEMP_SHIFT_P5_ENABLED', True)))

            # PAdapter
            self.padapter_enabled = bool(stpe_cfg.get('padapter_enabled', stpe_cfg.get('PADAPTER_ENABLED', False)))
            self.padapter_bottleneck = int(stpe_cfg.get('padapter_bottleneck', stpe_cfg.get('PADAPTER_BOTTLENECK', 32)))
            self.padapter_alpha_init = float(stpe_cfg.get('padapter_alpha_init', stpe_cfg.get('PADAPTER_ALPHA_INIT', 0.1)))
            self.n_segment = int(stpe_cfg.get('n_segment', stpe_cfg.get('N_SEGMENT', 1)))
            self.padapter_p3 = bool(stpe_cfg.get('padapter_p3_enabled', stpe_cfg.get('PADAPTER_P3_ENABLED', False)))
            self.padapter_p4 = bool(stpe_cfg.get('padapter_p4_enabled', stpe_cfg.get('PADAPTER_P4_ENABLED', False)))
            self.padapter_p5 = bool(stpe_cfg.get('padapter_p5_enabled', stpe_cfg.get('PADAPTER_P5_ENABLED', True)))

            # CARAFE++ 开关
            self.carafepp_enabled = bool(neck_cfg.get('carafepp_enabled', False))

        # 实例化 TSM
        if self.temp_shift_enabled:
            self._temp_p3 = TemporalShift(n_segment=self.n_segment, fold_div=self.temp_shift_fold_div) if self.temp_shift_p3 else None
            self._temp_p4 = TemporalShift(n_segment=self.n_segment, fold_div=self.temp_shift_fold_div) if self.temp_shift_p4 else None
            self._temp_p5 = TemporalShift(n_segment=self.n_segment, fold_div=self.temp_shift_fold_div) if self.temp_shift_p5 else None
        else:
            self._temp_p3 = self._temp_p4 = self._temp_p5 = None

        # PAdapter lazy placeholders
        self._padapter_p3 = None
        self._padapter_p4 = None
        self._padapter_p5 = None

        # ---------- B-1：持久化 CARAFE++ 上采样算子（两处 2× 常用路径） ----------
        # 说明：
        # - up2_p5：用于 P5 -> P4 的 2× 上采样，输入通道通常为 width[5]
        # - up2_h1：用于 h1 -> P3 的 2× 上采样，输入通道通常为 width[4]（h1 的输出通道）
        if self.carafepp_enabled:
            try:
                self.up2_p5 = CARAFEpp(width[5], scale=2)
                self.up2_h1 = CARAFEpp(width[4], scale=2)
                # 记录通道，用于 _upsample 前向匹配
                self._up2_p5_c = width[5]
                self._up2_h1_c = width[4]
            except Exception:
                # 任意异常则关闭 carafepp，回退最近邻
                self.carafepp_enabled = False

    def _lazy_init_padapter(self, attr_name, sample_feat, enabled_flag):
        if not enabled_flag:
            return
        if getattr(self, attr_name) is None:
            in_c = sample_feat.shape[1]
            pad = PAdapter(in_channels=in_c, bottleneck=self.padapter_bottleneck,
                           n_segment=self.n_segment, alpha_init=self.padapter_alpha_init)
            pad.to(sample_feat.device)
            setattr(self, attr_name, pad)

    # ---------- B-2：重写 _upsample（优先使用持久化 CARAFE++，否则回退最近邻） ----------
    def _upsample(self, x, scale=2):
        if scale == 1:
            return x
        if self.carafepp_enabled and scale == 2:
            c = x.shape[1]
            # 通道匹配才使用已持久化的 CARAFE++，避免运行期动态 new
            if hasattr(self, 'up2_p5') and c == getattr(self, '_up2_p5_c', -1):
                return self.up2_p5(x)
            if hasattr(self, 'up2_h1') and c == getattr(self, '_up2_h1_c', -1):
                return self.up2_h1(x)
            # 其他通道则回退最近邻（也可根据需要再行持久化更多分支）
            return F.interpolate(x, scale_factor=scale, mode='nearest')
        else:
            # 统一回退最近邻，避免在前向中构造新算子
            return F.interpolate(x, scale_factor=scale, mode='nearest')

    def forward(self, x):
        p3, p4, p5 = x

        # Temporal Shift
        if self._temp_p3 is not None:
            try: p3 = self._temp_p3(p3)
            except Exception: pass
        if self._temp_p4 is not None:
            try: p4 = self._temp_p4(p4)
            except Exception: pass
        if self._temp_p5 is not None:
            try: p5 = self._temp_p5(p5)
            except Exception: pass

        # STPE
        if self.stpe_enabled:
            try:
                p3, p4, p5 = self.stpe([p3, p4, p5])
            except Exception:
                pass

        # PAdapter
        if self.padapter_enabled:
            try:
                self._lazy_init_padapter('_padapter_p3', p3, self.padapter_p3)
                self._lazy_init_padapter('_padapter_p4', p4, self.padapter_p4)
                self._lazy_init_padapter('_padapter_p5', p5, self.padapter_p5)
            except Exception:
                pass
            if self.padapter_p3 and self._padapter_p3 is not None:
                try: p3 = self._padapter_p3(p3)
                except Exception: pass
            if self.padapter_p4 and self._padapter_p4 is not None:
                try: p4 = self._padapter_p4(p4)
                except Exception: pass
            if self.padapter_p5 and self._padapter_p5 is not None:
                try: p5 = self._padapter_p5(p5)
                except Exception: pass

        # Neck 路径（两处上采样采用 CARAFE++ 可选）
        up_p5 = self._upsample(p5, scale=2)             # P5 -> P4 对齐（优先 up2_p5）
        h1 = self.h1(torch.cat([up_p5, p4], 1))

        up_h1 = self._upsample(h1, scale=2)             # h1 -> P3 对齐（优先 up2_h1）
        h2 = self.h2(torch.cat([up_h1, p3], 1))

        h4 = self.h4(torch.cat([self.h3(h2), h1], 1))
        h6 = self.h6(torch.cat([self.h5(h4), p5], 1))
        return h2, h4, h6


# ------------------------------------------------------------
# Wrapper
# ------------------------------------------------------------
class YOLO(nn.Module):
    def __init__(self, width, depth, cfg2d=None, pretrain_path=None):
        super().__init__()
        self.net = DarkNet(width, depth, cfg2d=cfg2d)
        self.fpn = DarkFPN(width, depth, cfg2d=cfg2d)
        self.pretrain_path = pretrain_path

    def forward(self, x):
        x = self.net(x)
        return self.fpn(x)

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self

    # ---------- B-3：更鲁棒的预训练权重加载 ----------
    def load_pretrain(self):
        if not getattr(self, 'pretrain_path', None):
            return
        try:
            ckpt = torch.load(self.pretrain_path, map_location='cpu')
        except Exception as e:
            print(f"[YOLOv8] warn: failed to load pretrain from {self.pretrain_path}: {e}", flush=True)
            return

        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            pre = ckpt['state_dict']
        elif isinstance(ckpt, dict):
            pre = ckpt
        else:
            print("[YOLOv8] warn: unexpected checkpoint format; skip.", flush=True)
            return

        new_sd = self.state_dict()
        matched = 0
        for k, v in pre.items():
            k2 = k.replace('module.', '')
            if k2 in new_sd and new_sd[k2].shape == v.shape:
                new_sd[k2] = v
                matched += 1
        self.load_state_dict(new_sd, strict=False)
        print(f"backbone2D : YOLOv8 pretrained loaded! (matched {matched}/{len(new_sd)})", flush=True)


def build_yolov8(config):
    """
    读取：
    - config['BACKBONE2D']['YOLOv8']['ver']            -> n/s/m/l/x
    - config['BACKBONE2D']['YOLOv8']['PRETRAIN'][ver]  -> 预训练权重
    - config['BACKBONE2D']['YOLOv8']['STPE']            -> STPE 参数
    - config['BACKBONE2D']['YOLOv8']['NECK']           -> carafepp_enabled
    - config['BACKBONE2D']['YOLOv8']['BLOCKS']         -> involution_p4_enabled / involution_p5_enabled
    """
    cfg2d = config['BACKBONE2D']['YOLOv8']
    ver = cfg2d['ver']
    assert ver in ['n', 's', 'm', 'l', 'x'], "wrong version of YOLOv8!"
    pretrain_path = cfg2d['PRETRAIN'][ver]

    if ver == 'n':
        depth = [1, 2, 2]; width = [3, 16, 32, 64, 128, 256]
    elif ver == 's':
        depth = [1, 2, 2]; width = [3, 32, 64, 128, 256, 512]
    elif ver == 'm':
        depth = [2, 4, 4]; width = [3, 48, 96, 192, 384, 576]
    elif ver == 'l':
        depth = [3, 6, 6]; width = [3, 64, 128, 256, 512, 512]
    elif ver == 'x':
        depth = [3, 6, 6]; width = [3, 80, 160, 320, 640, 640]

    return YOLO(width, depth, cfg2d=cfg2d, pretrain_path=pretrain_path)
