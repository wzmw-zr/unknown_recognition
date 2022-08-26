# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from unittest import result

import mmcv
import numpy as np
import torch

from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadAnnotations(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        img_bytes = self.file_client.get(filename)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        # modify if custom classes
        if results.get('label_map', None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class LoadLogit(object):
    """
    Load logit data from `.npy` file.

    - Required keys are "logit_prefix" and "logit_info" (a dict that must contain the
    key "logit_filename"). 

    - Added or updated keys are "logit_filename", "logit".

    Args:
        to_float32 (bool): Whether to convert the loaded logit to a float32
            numpy array. 
    """

    def __init__(self,
                 to_float32=True):
        self.to_float32 = to_float32

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if results.get('logit_prefix') is not None:
            logit_filename = osp.join(results['logit_prefix'],
                                results['logit_info']['logit_filename'])
        else:
            logit_filename = results['logit_info']['logit_filename']
        logit = np.load(logit_filename)
        if self.to_float32:
            logit = logit.astype(np.float32)

        results['logit_filename'] = logit_filename
        results['logit'] = logit
        results['seg_fields'].append("logit")
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        return repr_str


@PIPELINES.register_module()
class LoadSoftmax(object):
    """
    Load softmax data from `.npy` file.

    - Required keys are "softmax_prefix" and "softmax_info" (a dict that must contain the
    key "softmax_filename"). 

    - Added or updated keys are "softmax_filename", "softmax".

    Args:
        to_float32 (bool): Whether to convert the loaded softmax to a float32
            numpy array. 
    """

    def __init__(self,
                 to_float32=True):
        self.to_float32 = to_float32

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if results.get('softmax_prefix') is not None:
            softmax_filename = osp.join(results['softmax_prefix'],
                                results['softmax_info']['softmax_filename'])
        else:
            softmax_filename = results['softmax_info']['softmax_filename']
        softmax = np.load(softmax_filename)
        if self.to_float32:
            softmax = softmax.astype(np.float32)

        results['softmax_filename'] = softmax_filename
        results['softmax'] = softmax
        results['seg_fields'].append("softmax")
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        return repr_str


@PIPELINES.register_module()
class LoadSoftmaxFromLogit(object):
    """ Load Softmax from Logit value (to save disk space).
    """
    def __init__(self) -> None:
        pass

    def __call__(self, results):
        logit = results.get("logit", None)
        assert logit is not None, \
            f"logit should not be None"
        softmax = torch.nn.functional.softmax(torch.as_tensor(logit), dim=0)
        results["softmax"] = softmax
        results["seg_fields"].append("softmax")
        return results