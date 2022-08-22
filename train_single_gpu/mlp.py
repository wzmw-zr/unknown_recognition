import mmcv
import os.path as osp
from mmcv import Config
from unknown_recognition.datasets import build_dataset
from unknown_recognition.models import build_unknown_detector
from unknown_recognition.apis import train_unknown_detector
from unknown_recognition.apis import set_random_seed
import torch

import argparse
import os.path as osp

import mmcv

def parse_args():
    parser = argparse.ArgumentParser(
        description='ignore edges for semantic segmentation')
    parser.add_argument('train_gt_directory', help='train_gt_directory')
    parser.add_argument('output_path', help="output_path")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    cfg = Config.fromfile("configs/mlp/mlp_256x1024_40k_anomal_dataset.py")

    # print(f'Config:\n{cfg.pretty_text}')

    CLASSES = ('unknown', 'known')

    PALETTE = [[255, 255, 255], [0, 0, 0]]

    cfg.checkpoint_config.meta = dict(
        CLASSES=CLASSES,
        PALETTE=PALETTE)

    # cfg.norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
    # cfg.model.decode_head.norm_cfg = cfg.norm_cfg
    # cfg.model.decode_head.loss_decode.avg_non_ignore = True
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.optimizer.lr = 0.00006

    """
    simulate multi-gpu with single gpu
    """
    # cfg.log_config = dict(
    #     interval=400, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
    # cfg.optimizer_config = dict(type="GradientCumulativeSimulateDDPOptimizerHook", cumulative_iters=8)
    # cfg.runner = dict(type='IterBasedRunner', max_iters=1280000)
    # cfg.checkpoint_config.interval = 128000
    # cfg.evaluation = dict(interval=128000, metric='mIoU', pre_eval=True)

    # cfg.gpu_ids = range(1)
    # cfg.work_dir = './work_dirs/cityscapes/segformer_mit-b0'
    # cfg.data.samples_per_gpu = 1
    # cfg.data.workers_per_gpu = 1

    print(f'Config:\n{cfg.pretty_text}')

    datasets = build_dataset(cfg.data.train)

    model = build_unknown_detector(cfg.model, train_cfg=cfg.get(
        "train_cfg"), test_cfg=cfg.get("test_cfg"))
    model.CLASSES = CLASSES
    # Create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    train_unknown_detector(model, datasets, cfg, distributed=False,
                    validate=True, meta=dict())