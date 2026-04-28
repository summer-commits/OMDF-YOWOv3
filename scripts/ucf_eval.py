# scripts/ucf_eval.py
import torch
import torch.utils.data as data
import numpy as np
import tqdm

# 保持你原本的 import 结构（删除未用 import，避免 IDE 报未使用）
from cus_datasets.build_dataset import build_dataset
from cus_datasets.collate_fn import collate_fn
from model.TSN.YOWOv3 import build_yowov3
from utils.box import non_max_suppression, box_iou
from evaluator.eval import compute_ap
from utils.flops import get_info


def _get_cfg(config: dict, path: list, default=None):
    """从多层字典里安全取值，例如 path=['TEST','score_thr']。"""
    cur = config if isinstance(config, dict) else {}
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _to_numpy_frames(outputs, device):
    """List[Tensor(N,6)] -> List[np.ndarray(N,6)]，空帧转 (0,6)。"""
    np_frames = []
    for out in outputs:
        if out is None or len(out) == 0:
            np_frames.append(np.zeros((0, 6), dtype=np.float32))
        else:
            np_frames.append(out.detach().to("cpu").numpy().astype(np.float32))
    return np_frames


def _numpy_frames_to_torch(np_frames, device):
    """List[np.ndarray(N,6)] -> List[Tensor(N,6)]，放回原设备。"""
    frames = []
    for arr in np_frames:
        if arr is None:
            frames.append(torch.zeros((0, 6), device=device))
        else:
            t = torch.from_numpy(arr)
            frames.append(t.to(device))
    return frames


@torch.no_grad()
def eval(config):
    """
    评估入口：
    - 读取 TEST 中的 score_thr / nms_iou_thr / max_per_img（若缺省则给默认）
    - 在逐帧 NMS 之后可选执行 Seq-NMS（POSTPROC.SEQ_NMS.enabled）
    - 评测逻辑与原脚本保持一致
    """

    # -------------------------
    # 构建数据与模型
    # -------------------------
    dataset = build_dataset(config, phase='test')
    dataloader = data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=False
    )

    model = build_yowov3(config)
    get_info(config, model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    # -------------------------
    # IoU 指标设置（保持原来只算 0.5）
    # -------------------------
    iou_v = torch.tensor([0.5], device=device)
    n_iou = iou_v.numel()

    m_pre = 0.0
    m_rec = 0.0
    map50 = 0.0
    mean_ap = 0.0
    metrics = []

    # -------------------------
    # 读取 TEST / POSTPROC 参数（带默认）
    # -------------------------
    score_thr = float(_get_cfg(config, ['TEST', 'score_thr'], 0.01))     # 默认放宽，提高召回
    nms_iou_thr = float(_get_cfg(config, ['TEST', 'nms_iou_thr'], 0.60)) # 默认稍放宽
    max_per_img = int(_get_cfg(config, ['TEST', 'max_per_img'], 500))

    seq_enabled = bool(_get_cfg(config, ['POSTPROC', 'SEQ_NMS', 'enabled'], False))
    # 保守版 Seq-NMS 推荐默认（更利于稳住 mAP）
    seq_window = int(_get_cfg(config, ['POSTPROC', 'SEQ_NMS', 'window'], 5))
    seq_iou_thr = float(_get_cfg(config, ['POSTPROC', 'SEQ_NMS', 'iou_thr'], 0.35))
    seq_decay = float(_get_cfg(config, ['POSTPROC', 'SEQ_NMS', 'decay'], 0.8))
    seq_center_thr = float(_get_cfg(config, ['POSTPROC', 'SEQ_NMS', 'center_dist_thr'], 0.4))
    seq_k_support = int(_get_cfg(config, ['POSTPROC', 'SEQ_NMS', 'k_support'], 2))
    seq_lambda = float(_get_cfg(config, ['POSTPROC', 'SEQ_NMS', 'lambda_boost'], 0.30))
    seq_cap = float(_get_cfg(config, ['POSTPROC', 'SEQ_NMS', 'cap'], 0.15))

    # -------------------------
    # 评估循环
    # -------------------------
    p_bar = tqdm.tqdm(dataloader, desc=('%10s' * 3) % ('precision', 'recall', 'mAP'))
    with torch.no_grad():
        for batch_clip, batch_bboxes, batch_labels in p_bar:
            batch_clip = batch_clip.to(device)

            # 组装 targets（与原脚本一致）
            targets = []
            for i, (bboxes, labels) in enumerate(zip(batch_bboxes, batch_labels)):
                target = torch.zeros((bboxes.shape[0], 6), dtype=torch.float32)
                target[:, 0] = i
                target[:, 1] = labels
                target[:, 2:] = bboxes
                targets.append(target)
            targets = torch.cat(targets, dim=0).to(device)

            height = config['img_size']
            width = config['img_size']

            # 前向
            outputs = model(batch_clip)

            # 坐标转像素（与原脚本一致）
            targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)

            # -------------------------
            # 逐帧 NMS（读取 TEST 阈值）
            # -------------------------
            try:
                # 尽量使用含 max_det 的新签名
                outputs = non_max_suppression(outputs, score_thr, nms_iou_thr, max_det=max_per_img)
            except TypeError:
                # 回退到老签名（没有 max_det）
                outputs = non_max_suppression(outputs, score_thr, nms_iou_thr)

            # -------------------------
            # Seq-NMS（可选，训练零侵入）
            # -------------------------
            if seq_enabled:
                # 惰性引入，避免未安装时的 import 错误
                try:
                    from utils.seq_nms import seq_nms as seq_nms_numpy
                except Exception:
                    # 如果你的实现是 torch 版，也可以换成：from utils.seq_nms import seq_nms_torch as seq_nms_numpy
                    from utils.seq_nms import seq_nms as seq_nms_numpy

                np_frames = _to_numpy_frames(outputs, device)
                np_frames = seq_nms_numpy(
                    np_frames,
                    window=seq_window,
                    iou_thr=seq_iou_thr,
                    decay=seq_decay,
                    center_dist_thr=seq_center_thr,
                    k_support=seq_k_support,
                    lambda_boost=seq_lambda,
                    cap=seq_cap
                )
                outputs = _numpy_frames_to_torch(np_frames, device)

            # -------------------------
            # 逐样本评测（保持原逻辑）
            # -------------------------
            for i, output in enumerate(outputs):
                labels = targets[targets[:, 0] == i, 1:]
                correct = torch.zeros(output.shape[0], n_iou, dtype=torch.bool, device=device)

                if output.shape[0] == 0:
                    if labels.shape[0]:
                        metrics.append((correct, *torch.zeros((3, 0), device=device)))
                    continue

                detections = output.clone()

                if labels.shape[0]:
                    tbox = labels[:, 1:5].clone()  # target boxes
                    correct_np = np.zeros((detections.shape[0], iou_v.shape[0]), dtype=bool)

                    t_tensor = torch.cat((labels[:, 0:1], tbox), 1)
                    iou = box_iou(t_tensor[:, 1:], detections[:, :4])
                    correct_class = t_tensor[:, 0:1] == detections[:, 5]
                    for j in range(len(iou_v)):
                        x = torch.where((iou >= iou_v[j]) & correct_class)
                        if x[0].shape[0]:
                            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
                            matches = matches.detach().cpu().numpy()
                            if x[0].shape[0] > 1:
                                matches = matches[matches[:, 2].argsort()[::-1]]
                                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                            correct_np[matches[:, 1].astype(int), j] = True
                    correct = torch.tensor(correct_np, dtype=torch.bool, device=iou_v.device)
                metrics.append((correct, output[:, 4], output[:, 5], labels[:, 0]))

        # -------------------------
        # 汇总与打印（保持原逻辑与输出）
        # -------------------------
        metrics = [torch.cat(x, 0).cpu().numpy() for x in zip(*metrics)]
        if len(metrics) and metrics[0].any():
            tp, fp, m_pre, m_rec, map50, mean_ap = compute_ap(*metrics)

        print('%10.3g' * 3 % (m_pre, m_rec, mean_ap), flush=True)

        model.float()
        print(map50, flush=True)
        print(flush=True)
        print("=================================================================", flush=True)
        print(flush=True)
        print(mean_ap, flush=True)
