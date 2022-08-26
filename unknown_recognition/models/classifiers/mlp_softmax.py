import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.utils.weight_init import (constant_init, kaiming_init,
                                        trunc_normal_)
from mmcv.runner import BaseModule, force_fp32
from torch.nn.modules.batchnorm import _BatchNorm
from torch import Tensor

from ..builder import CLASSIFIER, build_loss
from ..losses import accuracy


@CLASSIFIER.register_module()
class MLPSoftmax(BaseModule):
    def __init__(
        self, 
        in_channels=19,
        hidden_channels=256,
        num_classes=2,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0),
        ignore_index=255,
        # sampler=None,
        init_cfg=None
    ):
        super(MLPSoftmax, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        if isinstance(loss_decode, dict):
            self.loss_decode = build_loss(loss_decode)
        elif isinstance(loss_decode, (list, tuple)):
            self.loss_decode = nn.ModuleList()
            for loss in loss_decode:
                self.loss_decode.append(build_loss(loss))
        else:
            raise TypeError(f'loss_decode must be a dict or sequence of dict,\
                but got {type(loss_decode)}')

        self.fc1 = nn.Conv2d(self.in_channels, hidden_channels, 1)
        self.fc2 = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.fc3 = nn.Conv2d(hidden_channels, self.num_classes, 1)

    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    if 'ffn' in n:
                        nn.init.normal_(m.bias, mean=0., std=1e-6)
                    else:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                kaiming_init(m, mode='fan_in', bias=0.)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
                constant_init(m, val=1.0, bias=0.)

    def forward(self, inputs: Tensor):
        assert inputs.shape[1] == self.in_channels, \
            f"input channels should be {self.in_channels}, but get {inputs.shape[1]}"

        output = F.relu(self.fc1(inputs))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output

    def forward_train(self, softmax: Tensor, img_metas, gt_semantic_seg):
        seg_logits = self.forward(softmax)
        loss = self.losses(seg_logits, gt_semantic_seg)
        return loss

    def forward_test(self, softmax: Tensor, img_metas):
        return self.forward(softmax)

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        loss = dict()

        seg_weight = None

        seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)

        loss['acc_seg'] = accuracy(
            seg_logit, seg_label, ignore_index=self.ignore_index)
        return loss