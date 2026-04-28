import torch
import torch.nn as nn
import torch.nn.functional as F

class PAdapter(nn.Module):
    """
    Lightweight 3D adapter for short-range spatio-temporal fusion.
    Accepts (N, C, T, H, W) or (N*T, C, H, W).
    Uses lazy init if in_channels is None (but we will pass in_channels for clarity).
    """
    def __init__(self, in_channels=None, bottleneck=64, n_segment=8):
        super().__init__()
        self.in_channels = in_channels
        self.bottleneck = bottleneck
        self.n_segment = n_segment
        self.inited = False

        # placeholders
        self.conv1 = None
        self.dw = None
        self.conv2 = None
        self.act = nn.ReLU(inplace=True)

    def _lazy_init(self, in_channels):
        mid = self.bottleneck if self.bottleneck is not None else max(in_channels // 4, 32)
        self.conv1 = nn.Conv3d(in_channels, mid, kernel_size=1, bias=True)
        self.dw = nn.Conv3d(mid, mid, kernel_size=(3,3,3), padding=1, groups=mid, bias=True)
        self.conv2 = nn.Conv3d(mid, in_channels, kernel_size=1, bias=True)
        self.inited = True
        self.in_channels = in_channels

    def forward(self, x):
        orig_4d = False
        if x.dim() == 4:
            # (N*T, C, H, W)
            nt, c, h, w = x.shape
            if self.n_segment is None:
                raise ValueError("PAdapter requires n_segment for 4D input")
            if nt % self.n_segment != 0:
                raise ValueError(f"PAdapter: (N*T)={nt} not divisible by n_segment={self.n_segment}")
            n = nt // self.n_segment
            x = x.view(n, self.n_segment, c, h, w).permute(0,2,1,3,4).contiguous()  # -> (N, C, T, H, W)
            orig_4d = True

        # now x: (N, C, T, H, W)
        if not self.inited:
            self._lazy_init(x.size(1))
            # move parameters to same device as input
            self.to(x.device)

        out = self.conv1(x)
        out = self.act(self.dw(out))
        out = self.conv2(out)
        out = out + x  # residual

        if orig_4d:
            n, c, t, h, w = out.size()
            out = out.permute(0,2,1,3,4).contiguous().view(n * t, c, h, w)
        return out
