# scripts/ava_eval.py
# -*- coding: utf-8 -*-
import csv
import tqdm
import torch
import numpy as np
import torch.utils.data as data

from cus_datasets.build_dataset import build_dataset
from model.TSN.YOWOv3 import build_yowov3
from utils.flops import get_info
from utils.box import non_max_suppression
from evaluator.Evaluation import get_ava_performance

# ✅ 如你已把保守版 Seq-NMS 放在 utils/seq_nms.py，并命名为 seq_nms
#    若你的函数名不同，请在此改成对应的名字
from utils.seq_nms import seq_nms as seq_nms_numpy


def ava_eval_collate_fn(batch):
    """
    你的 AVA DataLoader 的专用 collate：
    返回：
      clips: (B, C, T, H, W)
      vid_n: list[str]  每个样本所属视频名
      secs : list[int]  每个样本的关键帧秒标（或时间戳）
    """
    clips, vid_n, secs = [], [], []
    for b in batch:
        clips.append(b[0])
        vid_n.append(b[3])
        secs.append(b[4])
    clips = torch.stack(clips, dim=0)
    return clips, vid_n, secs


def _get_cfg(config: dict, path: list, default=None):
    """从字典读取多级配置，缺省返回 default。"""
    cur = config if isinstance(config, dict) else {}
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


@torch.no_grad()
def eval(config):
    """
    评估流程：
      1) 逐帧前向，逐帧硬筛 NMS（读 TEST.* 阈值）
      2) 将同一视频的各帧候选收集起来（保持秒序）
      3) 若开启 POSTPROC.SEQ_NMS：对该视频整段做保守 Seq-NMS
      4) 最终归一化坐标、黑名单过滤、写 CSV
      5) 调用 AVA 官方评测
    """

    # --------- 构建数据与模型 ----------
    dataset = build_dataset(config, phase='test')
    bs = int(_get_cfg(config, ['EVAL', 'batch_size'], 32))
    num_workers = int(_get_cfg(config, ['EVAL', 'num_workers'], 6))

    dataloader = data.DataLoader(
        dataset,
        batch_size=bs,
        shuffle=False,
        collate_fn=ava_eval_collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )

    model = build_yowov3(config)
    get_info(config, model)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device).eval()

    # --------- NMS & Seq-NMS 配置 ----------
    # 逐帧硬筛（先宽松一些，后面再由 Seq-NMS 纠偏/抬分）
    score_thr   = float(_get_cfg(config, ['TEST', 'score_thr'], 0.10))  # 你的原默认 0.1
    nms_iou_thr = float(_get_cfg(config, ['TEST', 'nms_iou_thr'], 0.50))
    max_per_img = int(_get_cfg(config,   ['TEST', 'max_per_img'], 500))

    # Seq-NMS（保守 v2）
    seq_cfg     = _get_cfg(config, ['POSTPROC', 'SEQ_NMS'], {})
    seq_enabled = bool(seq_cfg.get('enabled', False))
    seq_window  = int(seq_cfg.get('window', 5))
    seq_iou     = float(seq_cfg.get('iou_thr', 0.40))          # 比较保守
    seq_decay   = float(seq_cfg.get('decay', 0.8))
    seq_center  = float(seq_cfg.get('center_dist_thr', 0.4))
    seq_ksupp   = int(seq_cfg.get('k_support', 2))
    seq_lambda  = float(seq_cfg.get('lambda_boost', 0.20))
    seq_cap     = float(seq_cfg.get('cap', 0.10))

    # --------- 其他通用配置 ----------
    H = W = int(config.get('img_size', 224))  # 你的管线里 eval/训练通常同一尺寸
    ava_result_file = config['detections']

    # AVA 官方黑名单（不计分的动作类，与你原脚本一致）
    black_list = set([2, 16, 18, 19, 21, 23, 25, 31, 32, 33, 35, 39, 40, 42, 44, 50, 53, 55, 71, 75])

    # --------- 第一阶段：逐帧前向 + 逐帧 NMS，按视频聚合 ----------
    # 存储结构： frames_by_video[vid] = [(sec, dets_tensor_Nx6_cpu), ...]
    from collections import defaultdict
    frames_by_video = defaultdict(list)

    pbar = tqdm.tqdm(dataloader, desc='[AVA-Eval] forward+frame-NMS')
    for clips, video_names, secs in pbar:
        clips = clips.to(device, non_blocking=True)
        outputs_batch = model(clips)   # 形状依实现而定，这里与原脚本一致逐样本取出
        outputs_batch = outputs_batch.detach().cpu()

        for outputs, vid, sec in zip(outputs_batch, video_names, secs):
            # non_max_suppression 期望 batch 维：这里对单样本加一维
            det = non_max_suppression(outputs.unsqueeze(0), score_thr, nms_iou_thr, max_det=max_per_img)[0]
            # 统一存为 CPU Tensor[N,6]： [x1,y1,x2,y2,score,cls]（像素坐标，未归一）
            if det is None:
                det = torch.zeros((0, 6), dtype=torch.float32)
            frames_by_video[vid].append((int(sec), det))

    # --------- 第二阶段：对每个视频执行（可选）Seq-NMS ----------
    results = []  # 最终写 CSV 的列表
    pbar2 = tqdm.tqdm(frames_by_video.items(), desc='[AVA-Eval] seq-NMS + write')

    for vid, items in pbar2:
        # 按时间排序，确保与窗口一致
        items.sort(key=lambda x: x[0])  # (sec, dets)
        secs_sorted = [s for s, _ in items]
        dets_sorted = [d for _, d in items]

        if seq_enabled:
            # 转 numpy list（float32），逐帧阵列形状 [Ni,6]
            np_frames = []
            for det in dets_sorted:
                if det is None or det.numel() == 0:
                    np_frames.append(np.zeros((0, 6), dtype=np.float32))
                else:
                    np_frames.append(det.numpy().astype(np.float32))

            # 调用你的保守版 Seq-NMS（互为最佳 + 多证据 + 凸组合 + cap）
            np_frames = seq_nms_numpy(
                np_frames,
                window=seq_window,
                iou_thr=seq_iou,
                decay=seq_decay,
                center_dist_thr=seq_center,
                k_support=seq_ksupp,
                lambda_boost=seq_lambda,
                cap=seq_cap
            )

            # 转回 Torch，方便后续同一流程（仍然是像素坐标）
            dets_sorted = [torch.from_numpy(arr) if isinstance(arr, np.ndarray) else arr
                           for arr in np_frames]

        # --------- 第三阶段：写 CSV（归一化 + 黑名单过滤） ----------
        for sec, det in zip(secs_sorted, dets_sorted):
            if det is None or det.numel() == 0:
                continue
            # 归一化到 [0,1]
            det_norm = det.clone().float()
            det_norm[:, 0] /= W
            det_norm[:, 1] /= H
            det_norm[:, 2] /= W
            det_norm[:, 3] /= H

            # 过滤黑名单，并写入（注意 AVA 类号从 1 开始；你的 cls 是 0-based）
            for row in det_norm:
                x1, y1, x2, y2, score, cls = row.tolist()
                cls_ava = int(cls) + 1
                if cls_ava in black_list:
                    continue
                results.append([
                    vid,
                    sec,
                    round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3),
                    cls_ava,
                    round(float(score), 3)
                ])

    # --------- 写文件并跑官方评测 ----------
    with open(ava_result_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(results)

    # 你的原调用方式（保持一致）
    get_ava_performance.eval(config)
