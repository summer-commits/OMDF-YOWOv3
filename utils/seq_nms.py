# utils/seq_nms.py
# -*- coding: utf-8 -*-
from typing import List, Tuple
import numpy as np

__all__ = ["seq_nms", "seq_nms_torch"]

def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    x11 = np.maximum(a[:, None, 0], b[None, :, 0])
    y11 = np.maximum(a[:, None, 1], b[None, :, 1])
    x22 = np.minimum(a[:, None, 2], b[None, :, 2])
    y22 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0.0, x22 - x11) * np.maximum(0.0, y22 - y11)
    area_a = np.maximum(0.0, (a[:, 2] - a[:, 0])) * np.maximum(0.0, (a[:, 3] - a[:, 1]))
    area_b = np.maximum(0.0, (b[:, 2] - b[:, 0])) * np.maximum(0.0, (b[:, 3] - b[:, 1]))
    union = area_a[:, None] + area_b[None, :] - inter + 1e-6
    return inter / union

def _center_dist_norm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    归一化中心距离：用当前框对角线作为尺度，避免大框与小框阈值不一致。
    return: (Na,Nb)
    """
    ax = (a[:, 0] + a[:, 2]) * 0.5
    ay = (a[:, 1] + a[:, 3]) * 0.5
    bx = (b[:, 0] + b[:, 2]) * 0.5
    by = (b[:, 1] + b[:, 3]) * 0.5
    # 当前框对角线
    a_diag = np.maximum(1e-6, np.sqrt((a[:, 2]-a[:, 0])**2 + (a[:, 3]-a[:, 1])**2))[:, None]
    dist = np.sqrt((ax[:, None]-bx[None, :])**2 + (ay[:, None]-by[None, :])**2)
    return dist / a_diag  # Na,Nb

def _mutual_best(iou: np.ndarray) -> np.ndarray:
    """
    互为最佳匹配：A->B 与 B->A 都是对方的最大 IoU。
    iou: (Na,Nb)
    return mask: (Na,Nb) bool
    """
    if iou.size == 0:
        return np.zeros_like(iou, dtype=bool)
    a2b = np.zeros_like(iou, dtype=bool)
    b2a = np.zeros_like(iou, dtype=bool)
    # A->B 最大
    a2b[np.arange(iou.shape[0]), np.argmax(iou, axis=1)] = True
    # B->A 最大
    b2a[np.argmax(iou, axis=0), np.arange(iou.shape[1])] = True
    return a2b & b2a

def seq_nms(
    frames_dets: List[np.ndarray],
    window: int = 5,           # 更保守：窗口小些
    iou_thr: float = 0.35,     # IoU 阈值略高，减少误连
    decay: float = 0.8,        # 时间衰减
    score_col: int = 4,
    cls_col: int = 5,
    center_dist_thr: float = 0.4,   # 归一化中心距离阈值（相对当前框对角线）
    k_support: int = 2,             # 至少有多少邻帧支持才托分
    lambda_boost: float = 0.30,     # 托分强度（凸组合权重）
    cap: float = 0.15,              # 单帧最大加分上限
) -> List[np.ndarray]:
    """
    保守版 Seq-NMS：多证据、互为最佳、凸组合+上限。
    frames_dets: list[np.ndarray], 每帧 (N,6)=[x1,y1,x2,y2,score,cls]
    """
    T = len(frames_dets)
    if T == 0:
        return frames_dets

    out = []
    for f in frames_dets:
        if f is None or len(f) == 0:
            out.append(np.zeros((0, 6), dtype=np.float32))
        else:
            f = np.asarray(f, dtype=np.float32)
            if f.ndim != 2 or f.shape[1] < 6:
                raise ValueError(f"Each frame must be (N,6), got {f.shape}")
            out.append(f.copy())

    # 类别集合
    classes = set()
    for f in out:
        if f.size > 0:
            classes.update(f[:, cls_col].astype(np.int32).tolist())

    for cls in classes:
        boxes, scores, idxs = [], [], []
        for t in range(T):
            f = out[t]
            if f.size == 0:
                boxes.append(np.zeros((0, 4), dtype=np.float32))
                scores.append(np.zeros((0,), dtype=np.float32))
                idxs.append(np.zeros((0,), dtype=np.int64))
                continue
            m = (f[:, cls_col].astype(np.int32) == cls)
            idt = np.where(m)[0]
            idxs.append(idt)
            if idt.size > 0:
                boxes.append(f[idt, :4])
                scores.append(f[idt, score_col])
            else:
                boxes.append(np.zeros((0, 4), dtype=np.float32))
                scores.append(np.zeros((0,), dtype=np.float32))

        for t in range(T):
            Na = boxes[t].shape[0]
            if Na == 0:
                continue
            A = boxes[t]
            sA = scores[t].copy()

            # 收集支持证据
            support_counts = np.zeros((Na,), dtype=np.int32)
            best_neighbor = np.zeros((Na,), dtype=np.float32)  # 衰减后的最佳邻居分数

            for dt in range(1, window + 1):
                w = (decay ** dt)

                for tt in (t - dt, t + dt):
                    if tt < 0 or tt >= T or boxes[tt].shape[0] == 0:
                        continue
                    B = boxes[tt]; sB = scores[tt]
                    iou = _iou_xyxy(A, B)
                    dist = _center_dist_norm(A, B)
                    # 过滤：IoU 与 中心距离
                    mask = (iou >= iou_thr) & (dist <= center_dist_thr)
                    if not np.any(mask):
                        continue
                    # 互为最佳匹配
                    mutual = _mutual_best(iou) & mask
                    if not np.any(mutual):
                        continue
                    # 对每个 A_i，取满足 mutual 的 B 中最大分数
                    cand = (sB[None, :] * mutual.astype(np.float32))
                    # 若该行全 0，max 仍为 0，不影响
                    max_b = cand.max(axis=1) * w
                    # 更新支持计数（该 dt 下有至少一个匹配视作 +1）
                    support_counts += (max_b > 0).astype(np.int32)
                    # 更新最佳邻居分数
                    best_neighbor = np.maximum(best_neighbor, max_b)

            # 应用保守托分：需满足支持数阈值
            need_boost = support_counts >= k_support
            if np.any(need_boost):
                s_new = (1.0 - lambda_boost) * sA + lambda_boost * best_neighbor
                # 限制不降分
                s_new = np.maximum(s_new, sA)
                # 限制最大增量
                s_new = np.minimum(s_new, sA + cap)
                sA[need_boost] = s_new[need_boost]

            # 回写
            if idxs[t].size > 0:
                out[t][idxs[t], score_col] = sA

    return out


def seq_nms_torch(
    frames_dets: List["torch.Tensor"],
    window: int = 5,
    iou_thr: float = 0.35,
    decay: float = 0.8,
    score_col: int = 4,
    cls_col: int = 5,
    center_dist_thr: float = 0.4,
    k_support: int = 2,
    lambda_boost: float = 0.30,
    cap: float = 0.15,
) -> List["torch.Tensor"]:
    import torch
    np_in = [d.detach().cpu().numpy() if isinstance(d, torch.Tensor) else d for d in frames_dets]
    np_out = seq_nms(np_in, window, iou_thr, decay, score_col, cls_col,
                     center_dist_thr, k_support, lambda_boost, cap)
    out = []
    for d, ref in zip(np_out, frames_dets):
        if isinstance(ref, torch.Tensor):
            out.append(torch.from_numpy(d).to(ref.device))
        else:
            out.append(d)
    return out
