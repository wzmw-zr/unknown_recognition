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
    parser.add_argument("lr", help="learning_rate", type=float)
    parser.add_argument("epoch", help="epoch", type=int)
    parser.add_argument("policy", help="policy", type=str)
    parser.add_argument("unknown_ratio", help="unknown_ratio", type=float)
    parser.add_argument("known_ratio", help="known_ratio", type=float)
    parser.add_argument("norm_type", help="norm_type", type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    lr = args.lr
    epoch = args.epoch
    policy = args.policy
    unknown_ratio = args.unknown_ratio
    known_ratio = args.known_ratio
    norm_type = args.norm_type
    norm_type = norm_type if norm_type != "None" else None

    cfg = Config.fromfile("configs/mlp/mlp_softmax_anomal_dataset_logit_adjust_loss_300_epoch.py")

    # print(f'Config:\n{cfg.pretty_text}')

    CLASSES = ('unknown', 'known')

    PALETTE = [[255, 255, 255], [0, 0, 0]]

    cfg.model.classifier.loss_decode.class_ratio = [unknown_ratio, known_ratio]
    cfg.model.classifier.norm = norm_type

    cfg.checkpoint_config.meta = dict(
        CLASSES=CLASSES,
        PALETTE=PALETTE)
    cfg.checkpoint_config.interval = epoch // 10

    # cfg.norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
    # cfg.model.decode_head.norm_cfg = cfg.norm_cfg
    # cfg.model.decode_head.loss_decode.avg_non_ignore = True
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.optimizer.lr = lr

    """
    simulate multi-gpu with single gpu
    """
    # cfg.log_config = dict(
    #     interval=400, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
    # cfg.optimizer_config = dict(type="GradientCumulativeSimulateDDPOptimizerHook", cumulative_iters=8)
    cfg.log_config.interval = 50
    cfg.runner = dict(type='EpochBasedRunner', max_epochs=epoch)
    cfg.lr_config.policy = policy
    # cfg.checkpoint_config.interval = 128000
    cfg.evaluation = dict(interval=10, metric=["mIoU", "mFscore", "mDice"])

    cfg.device = "cuda"
    cfg.gpu_ids = range(1)
    cfg.data.samples_per_gpu = 16
    cfg.data.workers_per_gpu = 2
    cfg.work_dir = f'./work_dirs/anomal_datasets/mlp_softmax_lr_{lr}_epoch_{epoch}_policy_{policy}_unknown_{unknown_ratio}_known_{known_ratio}'

    print(f'Config:\n{cfg.pretty_text}')

    datasets = build_dataset(cfg.data.train)

    model = build_unknown_detector(cfg.model, train_cfg=cfg.get(
        "train_cfg"), test_cfg=cfg.get("test_cfg"))
    model.CLASSES = CLASSES
    # Create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    train_unknown_detector(model, datasets, cfg, distributed=False,
                    validate=True, meta=dict())