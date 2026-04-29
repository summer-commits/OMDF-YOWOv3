import math
import os
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class MaxPool3dSamePadding(nn.MaxPool3d):

    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        (batch, channel, t, h, w) = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)


class Unit3D(nn.Module):

    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 activation_fn=F.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):
        super(Unit3D, self).__init__()

        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding

        # padding=0 here, we do manual pad in forward for SAME padding
        self.conv3d = nn.Conv3d(in_channels=in_channels,
                                out_channels=self._output_channels,
                                kernel_size=self._kernel_shape,
                                stride=self._stride,
                                padding=0,
                                bias=self._use_bias)

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        (batch, channel, t, h, w) = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        x = F.pad(x, pad)

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()

        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels[0], kernel_shape=[1, 1, 1], padding=0,
                         name=name + '/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels[1], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=out_channels[2], kernel_shape=[3, 3, 3],
                          name=name + '/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels[3], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=out_channels[4], kernel_shape=[3, 3, 3],
                          name=name + '/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3],
                                        stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels[5], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_3/Conv3d_0b_1x1')
        self.name = name

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


# =========================
# LTCC: Long-term Temporal Compactness Condensation
# =========================
class GateParam(nn.Module):
    """Scalar gate with sigmoid, initialized to a small probability (e.g., 0.1) for safe residual injection."""
    def __init__(self, p_init=0.1):
        super().__init__()
        raw = math.log(p_init / (1.0 - p_init + 1e-12) + 1e-12)
        self.raw = nn.Parameter(torch.tensor(raw, dtype=torch.float32))

    def forward(self):
        return torch.sigmoid(self.raw)


class LTCCMultiGranularityContextAggregation(nn.Module):
    """
    Multi-dilation depthwise temporal conv aggregation (k=3, dilations=[1,2,3]),
    residual with a learnable gate.
    """
    def __init__(self, channels, dilations=(1, 2, 3), p_init=0.1):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv3d(channels, channels, kernel_size=(3, 1, 1),
                      padding=(d, 0, 0), dilation=(d, 1, 1),
                      groups=channels, bias=False)
            for d in dilations
        ])
        self.fuse = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.bn   = nn.BatchNorm3d(channels, eps=1e-3, momentum=1e-2)
        self.gate = GateParam(p_init=p_init)

    def forward(self, x):
        y = 0
        for b in self.branches:
            y = y + b(x)
        y = self.fuse(y)
        y = self.bn(y)
        return x + self.gate() * y


class LTCCStateChangeEnhancement(nn.Module):
    """
    Lightweight Semantic Difference 3D:
    depthwise 3x3x3 + pointwise 1x1x1, residual with a learnable gate.
    """
    def __init__(self, channels, p_init=0.1):
        super().__init__()
        self.dw = nn.Conv3d(channels, channels, kernel_size=3, padding=1,
                            groups=channels, bias=False)
        self.pw = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm3d(channels, eps=1e-3, momentum=1e-2)
        self.gate = GateParam(p_init=p_init)
        self._init_diff_kernel()

    def _init_diff_kernel(self):
        k = torch.ones((1, 1, 3, 3, 3), dtype=torch.float32)
        k[:, :, 1, 1, 1] = -26.0
        with torch.no_grad():
            C = self.dw.in_channels
            self.dw.weight.copy_(k.repeat(C, 1, 1, 1, 1) / 26.0)
            nn.init.kaiming_normal_(self.pw.weight, nonlinearity='relu')

    def forward(self, x):
        y = self.dw(x)
        y = self.pw(y)
        y = self.bn(y)
        return x + self.gate() * y


class LTCCTemporalAttentionCompression(nn.Module):
    """
    Temporal attention pooling along T:
    attn = softmax(Conv1x1x1(x) along T) ; output = sum_t attn_t * x_t
    Keeps spatial dims; outputs T=1.
    """
    def __init__(self, channels):
        super().__init__()
        self.attn = nn.Conv3d(channels, 1, kernel_size=1, bias=True)

    def forward(self, x):
        # x: [B, C, T, H, W]
        w = self.attn(x)                   # [B, 1, T, H, W]
        w = torch.softmax(w, dim=2)        # softmax over time
        y = (x * w).sum(dim=2, keepdim=True)  # [B, C, 1, H, W]
        return y

# =========================
# Inception I3D Backbone
# =========================
class InceptionI3d(nn.Module):
    """Inception-v1 I3D backbone (features only, no classification head)."""

    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

    def __init__(self, pretrain_path, spatial_squeeze=True,
                 final_endpoint='Logits', name='inception_i3d',
                 in_channels=3, dropout_keep_prob=0.5,
                 # LTCC switches (all default False => behavior = original)
                 ltcc_context_enabled=False, ltcc_state_enabled=False, ltcc_attention_enabled=False):
        super(InceptionI3d, self).__init__()

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None
        self.pretrain_path = pretrain_path

        self.ltcc_context_enabled = bool(ltcc_context_enabled)
        self.ltcc_state_enabled = bool(ltcc_state_enabled)
        self.ltcc_attention_enabled = bool(ltcc_attention_enabled)

        self.end_points = {}

        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=[7, 7, 7],
                                            stride=(2, 2, 2), padding=(3, 3, 3), name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0,
                                            name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1,
                                            name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(192, [64, 96, 128, 16, 32, 32], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(256, [128, 128, 192, 32, 96, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2), padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(160 + 224 + 64 + 64, [128, 128, 256, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(128 + 256 + 64 + 64, [112, 144, 288, 32, 64, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(112 + 288 + 64 + 64, [256, 160, 320, 32, 128, 128], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2), padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(256 + 320 + 128 + 128, [256, 160, 320, 32, 128, 128], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(256 + 320 + 128 + 128, [384, 192, 384, 48, 128, 128], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Logits'
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)

        # register endpoints
        self.build()

        # LTCC after Mixed_5c (out channels = 1024): context aggregation -> state-change enhancement -> temporal compression
        out_c = 384 + 384 + 128 + 128
        if self.ltcc_context_enabled:
            self.ltcc_context = LTCCMultiGranularityContextAggregation(out_c, dilations=(1, 2, 3), p_init=0.1)
        if self.ltcc_state_enabled:
            self.ltcc_state_change = LTCCStateChangeEnhancement(out_c, p_init=0.1)
        if self.ltcc_attention_enabled:
            self.ltcc_temporal_attention = LTCCTemporalAttentionCompression(out_c)

        # 末端“温和白化”层：压强度，稳注入
        self.post_conv = nn.Conv3d(out_c, out_c, kernel_size=1, bias=False)
        self.post_bn = nn.BatchNorm3d(out_c, eps=1e-3, momentum=1e-2)
        with torch.no_grad():
            self.post_bn.weight.fill_(0.5)
            self.post_bn.bias.zero_()

    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.logits = Unit3D(in_channels=384 + 384 + 128 + 128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')

    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])

    def forward(self, x):
        # standard Inception flow up to Mixed_5c
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)  # use _modules to work with DataParallel

        # --- LTCC: multi-granularity context aggregation + state-change enhancement ---
        if getattr(self, 'ltcc_context_enabled', False):
            x = self.ltcc_context(x)
        if getattr(self, 'ltcc_state_enabled', False):
            x = self.ltcc_state_change(x)

        # --- LTCC: temporal attention compression to T=1 (keep H,W) ---
        if getattr(self, 'ltcc_attention_enabled', False):
            x = self.ltcc_temporal_attention(x)  # [B,C,1,H,W]
        else:
            x = F.avg_pool3d(x, kernel_size=(x.shape[2], 1, 1), stride=1)  # [B,C,1,H,W]

        # ★ 温和白化（控制注入强度与统计）
        x = self.post_conv(x)
        x = self.post_bn(x)     # [B,C,1,H,W]

        return x

    def extract_features(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        return self.avg_pool(x)

    def load_pretrain(self):
        if self.pretrain_path is None:
            return
        state_dict = self.state_dict()

        # robust loader
        try:
            pretrain_state_dict = torch.load(self.pretrain_path, map_location='cpu')
        except TypeError:
            pretrain_state_dict = torch.load(self.pretrain_path, map_location='cpu')

        if isinstance(pretrain_state_dict, dict) and 'state_dict' in pretrain_state_dict:
            pretrain_state_dict = pretrain_state_dict['state_dict']

        new_sd = {}
        for k, v in pretrain_state_dict.items():
            k2 = k[7:] if k.startswith('module.') else k
            if k2 in state_dict and state_dict[k2].shape == v.shape:
                new_sd[k2] = v

        state_dict.update(new_sd)
        self.load_state_dict(state_dict, strict=False)
        print("backbone3D : i3d pretrained loaded!", flush=True)


# =========================
# Builder (keeps API stable)
# =========================
def build_i3d(config):
    """
    Reads YAML like:
    BACKBONE3D:
      I3D:
        PRETRAIN:
          default: weights/rgb_imagenet.pth
        LTCC:
          enabled: true
          multi_granularity_temporal_context: true
          state_change_enhancement: true
          temporal_attention_compression: true

    """
    cfg_i3d = config['BACKBONE3D']['I3D']
    pretrain_path = cfg_i3d['PRETRAIN']['default']

    ltcc_cfg = cfg_i3d.get('LTCC', {}) if isinstance(cfg_i3d, dict) else {}
    ltcc_enabled = bool(ltcc_cfg.get('enabled', True))
    ltcc_context_enabled = ltcc_enabled and bool(
        ltcc_cfg.get('multi_granularity_temporal_context', False)
    )
    ltcc_state_enabled = ltcc_enabled and bool(
        ltcc_cfg.get('state_change_enhancement', False)
    )
    ltcc_attention_enabled = ltcc_enabled and bool(
        ltcc_cfg.get('temporal_attention_compression', False)
    )

    return InceptionI3d(in_channels=3,
                        pretrain_path=pretrain_path,
                        ltcc_context_enabled=ltcc_context_enabled,
                        ltcc_state_enabled=ltcc_state_enabled,
                        ltcc_attention_enabled=ltcc_attention_enabled)
