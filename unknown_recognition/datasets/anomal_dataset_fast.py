# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
import os.path as osp

import mmcv
import cv2
import numpy as np
from mmcv.utils import print_log

from unknown_recognition.datasets.custom import CustomDataset
from unknown_recognition.utils import get_root_logger
from .builder import DATASETS
from .pipelines import Compose, LoadAnnotations


@DATASETS.register_module()
class AnomalDatasetFast(CustomDataset):
    """Custom dataset for semantic segmentation. An example of file structure
    is as followed.

    .. code-block:: none

        ├── data
        │   ├── my_dataset
        │   │   ├── logit_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{logit_suffix}
        │   │   │   │   ├── xxx{logit_suffix}
        │   │   │   │   ├── xxx{logit_suffix}
        │   │   │   ├── val
        │   │   ├── softmax_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{softmax_suffix}
        │   │   │   │   ├── xxx{softmax_suffix}
        │   │   │   │   ├── xxx{softmax_suffix}
        │   │   │   ├── val
        │   │   ├── ann_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{seg_map_suffix}
        │   │   │   │   ├── yyy{seg_map_suffix}
        │   │   │   │   ├── zzz{seg_map_suffix}
        │   │   │   ├── val

    The img/gt_semantic_seg pair of CustomDataset should be of the same
    except suffix. A valid img/gt_semantic_seg filename pair should be like
    ``xxx{img_suffix}`` and ``xxx{seg_map_suffix}`` (extension is also included
    in the suffix). If split is given, then ``xxx`` is specified in txt file.
    Otherwise, all files in ``img_dir/``and ``ann_dir`` will be loaded.
    Please refer to ``docs/en/tutorials/new_dataset.md`` for more details.


    Args:
        pipeline (list[dict]): Processing pipeline
        img_dir (str): Path to image directory
        img_suffix (str): Suffix of images. Default: '.jpg'
        ann_dir (str, optional): Path to annotation directory. Default: None
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        split (str, optional): Split txt file. If split is specified, only
            file with suffix in the splits will be loaded. Otherwise, all
            images in img_dir/ann_dir will be loaded. Default: None
        data_root (str, optional): Data root for img_dir/ann_dir. Default:
            None.
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
        ignore_index (int): The label index to be ignored. Default: 255
        reduce_zero_label (bool): Whether to mark label zero as ignored.
            Default: False
        classes (str | Sequence[str], optional): Specify classes to load.
            If is None, ``cls.CLASSES`` will be used. Default: None.
        palette (Sequence[Sequence[int]]] | np.ndarray | None):
            The palette of segmentation map. If None is given, and
            self.PALETTE is None, random palette will be generated.
            Default: None
        gt_seg_map_loader_cfg (dict, optional): build LoadAnnotations to
            load gt for evaluation, load from disk by default. Default: None.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    CLASSES = ('unknown', 'known')

    PALETTE = [[255, 255, 255], [0, 0, 0]]


    def __init__(self,
                 pipeline,
                 logit_dir,
                 logit_suffix='.npy',
                 ann_dir=None,
                 seg_map_suffix='_TrainIds.png',
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None,
                 gt_seg_map_loader_cfg=None,
                 file_client_args=dict(backend='disk')):
        self.pipeline = Compose(pipeline)

        self.logit_dir = logit_dir
        self.logit_suffix = logit_suffix

        self.ann_dir = ann_dir
        self.seg_map_suffix = seg_map_suffix

        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes, palette)
        self.gt_seg_map_loader = LoadAnnotations(
        ) if gt_seg_map_loader_cfg is None else LoadAnnotations(
            **gt_seg_map_loader_cfg)

        self.file_client_args = file_client_args
        self.file_client = mmcv.FileClient.infer_client(self.file_client_args)

        if test_mode:
            assert self.CLASSES is not None, \
                '`cls.CLASSES` or `classes` should be specified when testing'

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.logit_dir):
                self.logit_dir = osp.join(self.data_root, self.logit_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)

        # load data and annotations (meta information)
        self.data_infos = self.load_annotations(self.logit_dir, self.logit_suffix,
                                               self.ann_dir, self.seg_map_suffix)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)

    def load_annotations(self, logit_dir, logit_suffix, ann_dir, seg_map_suffix):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        data_infos = []
        for logit_filename in self.file_client.list_dir_or_file(
                dir_path=logit_dir,
                list_dir=False,
                suffix=logit_suffix,
                recursive=True):
            logit_path = osp.join(logit_dir, logit_filename)
            # Load logit into RAM, save IO cost.
            logit = np.load(logit_path).astype(np.float32)
            data_info = dict(
                logit_info=dict(
                    logit_filename=logit_filename,
                    logit=logit
                )
            )
            # Load gtFine into RAM, save IO cost.
            if ann_dir is not None:
                seg_map = logit_filename.replace(logit_suffix, seg_map_suffix)
                seg_map_path = osp.join(ann_dir, seg_map)
                gt_semantic_seg = cv2.imread(seg_map_path, cv2.IMREAD_UNCHANGED).squeeze().astype(np.uint8)
                data_info['ann'] = dict(seg_map=seg_map, gt_semantic_seg=gt_semantic_seg)
            data_infos.append(data_info)
        data_infos = sorted(data_infos, key=lambda x: x['logit_info']['logit_filename'])

        print_log(f'Loaded {len(data_infos)} logit files', logger=get_root_logger())
        return data_infos

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results["logit_prefix"] = self.logit_dir
        results['seg_prefix'] = self.ann_dir

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        logit = deepcopy(self.data_infos[idx]["logit_info"]["logit"])
        gt_semantic_seg = deepcopy(self.data_infos[idx]["ann"]["gt_semantic_seg"])
        # ann_info = self.data_infos[idx]["ann"]
        results = dict(logit=logit, gt_semantic_seg=gt_semantic_seg)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by
                pipeline.
        """

        logit = deepcopy(self.data_infos[idx]["logit_info"]["logit"])
        results = dict(logit=logit)
        self.pre_pipeline(results)
        return self.pipeline(results)
