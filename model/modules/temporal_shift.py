import torch
import torch.nn as nn

class TemporalShift(nn.Module):
    """
    Temporal Shift module, supports inputs:
      - (N, C, T, H, W)
      - (N*T, C, H, W)  (frames flattened, common in many detection pipelines)
    Usage:
      ts = TemporalShift(n_segment=8, fold_div=8)
      y = ts(x)
    """
    def __init__(self, n_segment=8, fold_div=8):
        super().__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x):
        # Accept 4D or 5D tensors
        if x.dim() == 4:
            # (N*T, C, H, W) -> (N, C, T, H, W)
            nt, c, h, w = x.size()
            if self.n_segment is None or self.n_segment <= 0:
                raise ValueError("TemporalShift: n_segment must be set for 4D input")
            if nt % self.n_segment != 0:
                # try to infer n as integer
                n = nt // self.n_segment
                if n * self.n_segment != nt:
                    raise ValueError(f"TemporalShift: cannot reshape (N*T)={nt} with n_segment={self.n_segment}")
            else:
                n = nt // self.n_segment
            x = x.view(n, self.n_segment, c, h, w).permute(0,2,1,3,4).contiguous()  # (N, C, T, H, W)
        elif x.dim() == 5:
            n, c, t, h, w = x.size()
            if self.n_segment is None:
                self.n_segment = t
        else:
            raise ValueError("TemporalShift expects 4D or 5D tensor")

        n, c, t, h, w = x.size()
        fold = c // self.fold_div
        if fold == 0:
            out = x
        else:
            out = torch.zeros_like(x)
            # shift left: channels [0:fold] take from future (t+1)
            out[:, :fold, :-1] = x[:, :fold, 1:]
            # shift right: channels [fold:2*fold] take from past (t-1)
            out[:, fold:2*fold, 1:] = x[:, fold:2*fold, :-1]
            # keep the rest
            out[:, 2*fold:] = x[:, 2*fold:]

        # restore original shape
        if out.dim() == 5 and x is not None:
            if out.shape[0] * out.shape[2] == n * t:
                # if original was 4D, return to (N*T, C, H, W)
                if x is not None and hasattr(self, 'fold_div'):
                    # detect original input dim by seeing if we originally reshaped (best-effort)
                    pass
        # If original input was 4D we need to return to shape (N*T, C, H, W)
        # Heuristic: if input was 4D, 'n' and 't' are defined above and product equals original nt
        # But we already changed x, so to be safe, detect by checking calling stack shape isn't available.
        # Simpler: we branch earlier based on original input dimension.
        # For clarity we repeat reshape based on current dims:
        if x.dim() == 5:
            # but we don't know original dim here; earlier branch handled reshape
            pass

        # To correctly restore, check if incoming 'x' originally was 4D:
        # We'll assume if code reached here, the input was 5D (simple case) -> return 5D
        # However we did create out from possibly reshaped x: safe restore:
        if hasattr(self, '_input_was_4d') and self._input_was_4d:
            out = out.permute(0,2,1,3,4).contiguous().view(n * t, c, h, w)
            # reset flag
            self._input_was_4d = False
        else:
            # keep as (N,C,T,H,W)
            pass

        return out

    # small override: wrap to properly detect 4D origin
    def __call__(self, x):
        # set flag and call forward for correct reshape restore
        if x.dim() == 4:
            self._input_was_4d = True
        else:
            self._input_was_4d = False
        return super().__call__(x)
