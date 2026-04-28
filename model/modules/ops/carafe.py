# model/ops/carafe.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CARAFE(nn.Module):
    """
    Minimal CARAFE upsampler (content-aware reassembly of features).
    - up_factor: 2 or 4 (常用在 FPN 的 2x/4x 上采样)
    - kernel_size: 重组核大小，5/3 常见
    参考实现要点：先用轻量编码器预测重组权重（s^2 * k^2），
    再对输入做 unfold 得到 k^2 邻域，按权重重组并用 pixel-shuffle 样式重排到高分辨率。
    """
    def __init__(self, in_channels, up_factor=2, kernel_size=5, compress_ratio=4):
        super().__init__()
        assert up_factor in [2, 4]
        self.up = up_factor
        self.k = kernel_size
        hidden = max(1, in_channels // compress_ratio)

        # 轻量编码器：C -> hidden -> (s^2 * k^2)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, (self.up * self.up) * (self.k * self.k), 1, 1, 0, bias=True)
        )

    def forward(self, x):
        """
        x: [B, C, H, W] -> out: [B, C, H*s, W*s]
        """
        B, C, H, W = x.shape
        s, k = self.up, self.k

        # 预测重组权重并 softmax 到 k^2
        kernel = self.encoder(x)                           # [B, s^2*k^2, H, W]
        kernel = kernel.view(B, s*s, k*k, H, W)
        kernel = torch.softmax(kernel, dim=2)              # over k^2

        # 邻域展开 & 重组
        patches = F.unfold(x, kernel_size=k, padding=k//2) # [B, C*k^2, H*W]
        patches = patches.view(B, C, k*k, H, W)            # [B, C, k^2, H, W]

        # 逐位置加权求和 -> [B, s^2, C, H, W]
        out = torch.einsum('bsqhw,bcqhw->bschw', kernel, patches)

        # 像素重排到高分辨率
        out = out.view(B, s, s, C, H, W).permute(0, 3, 4, 1, 5, 2).contiguous()
        out = out.view(B, C, H * s, W * s)
        return out
