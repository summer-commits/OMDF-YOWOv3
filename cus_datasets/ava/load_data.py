import torch
import torch.utils.data as data
import os
import csv
import cv2
import numpy as np
from PIL import Image
from cus_datasets.ucf.transforms import Augmentation, UCF_transform


class AVA_dataset(data.Dataset):
    """
    Custom AVA-like dataset loader for custom frame-indexed data (one timestamp per frame).
    Assumes CSV rows: video_name, frame_idx, x1, y1, x2, y2, action_id, instance_id
    """

    def __init__(self,
                 root_path,
                 split_path,
                 data_path,
                 clip_length,
                 sampling_rate,
                 img_size,
                 transform=None,
                 phase='train'):
        # 基础属性
        self.root_path = root_path
        self.split_path = os.path.join(root_path, 'annotations', split_path)
        self.data_path = os.path.join(root_path, data_path)
        self.clip_length = clip_length
        self.sampling_rate = sampling_rate
        self.transform = transform if transform is not None else Augmentation()
        self.num_classes = 8
        self.phase = phase
        self.img_size = img_size

        # 日志：记录本轮加载帧路径
        self.loaded_frame_paths = []

        # 缓存视频帧数
        self._frame_counts = {}

        # 加载并排序标注
        self.read_ann_csv()

    def read_ann_csv(self):
        ann_dict = {}
        with open(self.split_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                video, frame_idx, x1, y1, x2, y2, cls, _ = row
                key = f"{video}/{frame_idx}"
                box = "/".join([x1, y1, x2, y2])
                ann_dict.setdefault(key, {}).setdefault(box, []).append(int(cls))
        # 保留合法索引并排序
        filtered = []
        for key in ann_dict.keys():
            _, fidx_str = key.split('/')
            try:
                fidx = int(fidx_str)
            except ValueError:
                continue
            if fidx >= 0:
                filtered.append(key)
        self.data_dict = ann_dict
        self.data_list = sorted(filtered, key=lambda k: int(k.split('/')[1]))
        self.data_len = len(self.data_list)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index, get_origin_image=False):
        # 解析 video_name 与 frame 索引
        video_name, fidx_str = self.data_list[index].split('/')
        try:
            fidx = int(fidx_str)
        except ValueError:
            fidx = 0
        key_frame_idx = fidx + 1
        video_path = os.path.join(self.data_path, video_name)

        # 计算该视频最大帧数
        if video_name not in self._frame_counts:
            try:
                files = [f for f in os.listdir(video_path) if f.endswith('.jpg')]
                idxs = [int(f.split('_')[-1].split('.')[0]) for f in files]
                self._frame_counts[video_name] = max(idxs) if idxs else 1
            except FileNotFoundError:
                self._frame_counts[video_name] = 1
        max_idx = self._frame_counts[video_name]
        print(video_name, fidx_str, key_frame_idx, max_idx)

        # 采样 clip
        clip = []
        for i in reversed(range(self.clip_length)):
            cur_idx = key_frame_idx - i * self.sampling_rate
            cur_idx = max(1, min(cur_idx, max_idx))
            frame_name = f"{video_name}_{cur_idx:06d}.jpg"
            frame_path = os.path.join(video_path, frame_name)

            # 记录路径
            self.loaded_frame_paths.append(frame_path)
            img = Image.open(frame_path).convert('RGB')
            clip.append(img)

        # 可选原图读取
        if get_origin_image:
            orig_name = f"{video_name}_{key_frame_idx:06d}.jpg"
            original_image = cv2.imread(os.path.join(video_path, orig_name))

        # 解析标注
        W, H = clip[-1].size
        boxes, labels = [], []
        for box_key, cls_list in self.data_dict[self.data_list[index]].items():
            x1, y1, x2, y2 = map(float, box_key.split('/'))
            boxes.append([x1 * W, y1 * H, x2 * W, y2 * H])
            onehot = np.zeros(self.num_classes, dtype=np.float32)
            for cls in cls_list:
                onehot[cls - 1] = 1.0
            labels.append(onehot)
        boxes = np.array(boxes, np.float32)
        labels = np.array(labels, np.float32)
        targets = np.concatenate([boxes, labels], axis=1)

        # 变换
        clip, targets = self.transform(clip, targets)
        boxes = targets[:, :4]
        labels = targets[:, 4:]

        if get_origin_image:
            return original_image, clip, boxes, labels
        if self.phase == 'test':
            return clip, boxes, labels, video_name, fidx_str
        return clip, boxes, labels

    def reset_loaded_log(self):
        """清空本轮加载帧日志"""
        self.loaded_frame_paths.clear()

    def get_loaded_log(self):
        """返回本轮加载帧路径列表"""
        return list(self.loaded_frame_paths)


def build_ava_dataset(config, phase):
    root = config['data_root']
    clip_len = config['clip_length']
    sr = config['sampling_rate']
    img_size = config['img_size']

    if phase == 'train':
        split = 'ava_v2.2/ava_train_v2.2.csv'
        transform = Augmentation(img_size=img_size)
    else:
        split = 'ava_v2.2/ava_val_v2.2.csv'
        transform = UCF_transform(img_size=img_size)

    return AVA_dataset(
        root_path=root,
        split_path=split,
        data_path='frames',
        clip_length=clip_len,
        sampling_rate=sr,
        img_size=img_size,
        transform=transform,
        phase=phase
    )








