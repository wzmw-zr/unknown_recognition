# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import builder
from ..builder import UNKNOWN_DETECTOR
from .base_unknown_detector import BaseUnknownDetector


@UNKNOWN_DETECTOR.register_module()
class UnknownDetectorSoftmax(BaseUnknownDetector):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 classifier, 
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(UnknownDetectorSoftmax, self).__init__(init_cfg)
        if pretrained is not None:
            assert classifier.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            classifier.pretrained = pretrained
        self.classifier = builder.build_classifier(classifier)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _classifier_forward_train(self, softmax, img_metas, gt_semantic_seg):
        return self.classifier.forward_train(softmax, img_metas, gt_semantic_seg)

    def forward_train(self, softmax, img_metas, gt_semantic_seg):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = self._classifier_forward_train(softmax, img_metas, gt_semantic_seg)
        return losses

    def _classifier_forward_test(self, softmax, img_meta):
        return self.classifier.forward_test(softmax, img_meta)

    def inference(self, softmax, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['whole']

        seg_logit = self._classifier_forward_test(softmax, img_meta)

        output = F.softmax(seg_logit, dim=1)

        return output

    def forward_test(self, softmax, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(softmax, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def forward(self, softmax, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(softmax, img_metas, **kwargs)
        else:
            return self.forward_test(softmax, img_metas, **kwargs)

    def simple_test(self, softmax, img_meta, **kwargs):
        seg_logit = self.inference(softmax, img_meta, rescale=True)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, **kwargs):
        """Placeholder for augmentation test."""
        raise NotImplementedError