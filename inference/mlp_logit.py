from unknown_recognition.apis import init_unknown_detector, inference_unknown_detector, show_result_pyplot
import os
import argparse
import mmcv
import numpy as np
import torch
import cv2
from mmcv import Config

"""
Usage Example:

python inference/mlp_logit.py configs/mlp/mlp_logit_tversky_loss_100_epoch.py \
LN data/anomal_campusE1/ test \
work_dirs/anomal_datasets/mlp_logit_lr_0.001_epoch_20_policy_fixed_logit_norm_local_layer_norm_LN_0.3_0.7/latest.pth \
mlp_logit_predictions
"""

def parse_args():
    parser = argparse.ArgumentParser(
        description='ignore edges for semantic segmentation')
    parser.add_argument("config_file", help="config_file", type=str)
    parser.add_argument("norm_type", help="norm_type", type=str)
    parser.add_argument("dataset_path", help="dataset_path", type=str)
    parser.add_argument("split", help="split", type=str)
    parser.add_argument("checkpoint_file", help="checkpoint_file", type=str)
    parser.add_argument("output_path", help="output_path", type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    config_file = args.config_file
    norm_type = args.norm_type
    dataset_path = args.dataset_path
    split = args.split
    checkpoint_file = args.checkpoint_file
    output_path = args.output_path

    logit_dir = os.path.join(dataset_path, "logit", split)
    gtFine_dir = os.path.join(dataset_path, "gtFine", split)
    output_dir = os.path.join(output_path, split)
    pred_output_dir = os.path.join(output_path, split, "prediction")
    gt_output_dir = os.path.join(output_path, split, "gtFine")

    logit_suffix = ".npy"
    gtFine_suffix = "_TrainIds.png"
    pred_output_suffix = ".png"
    gt_output_suffix = ".png"

    cfg = Config.fromfile(config_file)
    cfg.model.classifier.norm = norm_type

    model = init_unknown_detector(cfg, checkpoint_file, device='cuda')

    mmcv.mkdir_or_exist(output_path)

    for logit_file in mmcv.scandir(logit_dir, logit_suffix, recursive=True):
        # input file path and output file path
        logit_file_path = os.path.join(logit_dir, logit_file)
        gtFine_file_path = os.path.join(gtFine_dir, logit_file.replace(logit_suffix, gtFine_suffix))
        pred_output_file_path = os.path.join(pred_output_dir, logit_file.replace(logit_suffix, pred_output_suffix))
        gt_output_file_path = os.path.join(gt_output_dir, logit_file.replace(logit_suffix, gt_output_suffix))
        print(f"inference {logit_file_path}")
        # load input files
        logit = np.load(logit_file_path)
        logit = torch.as_tensor(logit, device="cuda", dtype=torch.float32)
        logit = (logit - torch.min(logit, dim=0)[0]) / (torch.max(logit, dim=0)[0] - torch.min(logit, dim=0)[0])
        logit = logit.unsqueeze(dim=0)
        img_meta = None
        pred = model.simple_test(logit, img_meta)[0]
        gt = cv2.imread(gtFine_file_path, cv2.IMREAD_UNCHANGED).astype(np.uint8)

        # generate color images of prediction and gt
        H, W = pred.shape
        pred_gray = np.zeros((H, W, 3), dtype=np.uint8)
        gt_gray = np.zeros((H, W, 3), dtype=np.uint8)
        unknown_mask = pred == 0
        known_mask = pred == 1
        ignore_mask = gt == 255
        pred_gray[unknown_mask] = [122, 122, 122]
        pred_gray[known_mask] = [255, 255, 255]
        pred_gray[ignore_mask] = [0, 0, 0]

        unknown_mask = gt == 0
        known_mask = gt == 1
        gt_gray[unknown_mask] = [122, 122, 122]
        gt_gray[known_mask] = [255, 255, 255]
        gt_gray[ignore_mask] = [0, 0, 0]

        mmcv.mkdir_or_exist(os.path.dirname(pred_output_file_path))
        mmcv.mkdir_or_exist(os.path.dirname(gt_output_file_path))
        cv2.imwrite(pred_output_file_path, pred_gray)
        cv2.imwrite(gt_output_file_path, gt_gray)