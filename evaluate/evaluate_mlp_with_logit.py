from unknown_recognition.apis import init_unknown_detector
import os
import argparse
import mmcv
import numpy as np
import torch
import cv2
from unknown_recognition.core.evaluation import eval_unknown_metrics, eval_metrics
from mmcv import Config

def parse_args():
    parser = argparse.ArgumentParser(
        description='ignore edges for semantic segmentation')
    parser.add_argument("config_file", help="config_file", type=str)
    parser.add_argument("checkpoint_file", help="checkpoint_file", type=str)
    parser.add_argument("dataset_path", help="dataset_path", type=str)
    parser.add_argument("output_path", help="output_path", type=str)
    parser.add_argument("norm", help="norm", type=str)
    parser.add_argument("split", help="split", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    config_file = args.config_file
    checkpoint_file = args.checkpoint_file
    dataset_path = args.dataset_path
    output_path = args.output_path
    norm = args.norm
    norm = norm if norm != "None" else None
    split = args.split

    logit_dir = os.path.join(dataset_path, "logit", split)
    gt_dir = os.path.join(dataset_path, "gtFine", split)
    seg_pred_dir = os.path.join(dataset_path, "seg_pred", split)
    seg_gt_dir = os.path.join(dataset_path, "seg_gt", split)
    output_dir = os.path.join(output_path, "color", split)

    logit_suffix = ".npy"
    gt_suffix = "_TrainIds.png"
    # seg_pred_suffix = "_seg_pred_labelTrainIds.png"
    # seg_gt_suffix = "_seg_gt_labelTrainIds.png"
    color_suffix = ".png"

    cfg = Config.fromfile(config_file)
    cfg.model.classifier.norm = norm

    model = init_unknown_detector(cfg, checkpoint_file, device='cuda:0')

    mmcv.mkdir_or_exist(output_path)

    all_seg_preds = list()
    all_seg_gts = list()
    all_gts = list()
    all_results = list()

    for logit_file in mmcv.scandir(logit_dir, logit_suffix, recursive=True):
        logit_file_path = os.path.join(logit_dir, logit_file)
        # seg_pred_file_path = os.path.join(seg_pred_dir, logit_file.replace(logit_suffix, seg_pred_suffix))
        # seg_gt_file_path = os.path.join(seg_gt_dir, logit_file.replace(logit_suffix, seg_gt_suffix))
        gt_file_path = os.path.join(gt_dir, logit_file.replace(logit_suffix, gt_suffix))
        color_file_path = os.path.join(output_dir, logit_file.replace(logit_suffix, color_suffix))
        print(f"inference {logit_file_path}")

        # seg_pred = cv2.imread(seg_pred_file_path, cv2.IMREAD_UNCHANGED).astype(np.uint8)
        # seg_gt = cv2.imread(seg_gt_file_path, cv2.IMREAD_UNCHANGED).astype(np.uint8)
        gt = cv2.imread(gt_file_path, cv2.IMREAD_UNCHANGED).astype(np.uint8)

        logit = np.load(logit_file_path)
        logit = (logit - np.min(logit)) / (np.max(logit) - np.min(logit))
        # logit = (logit - np.min(logit, axis=0)) / (np.max(logit, axis=0) - np.min(logit, axis=0))
        logit = torch.as_tensor(logit, device="cuda")
        logit = logit.unsqueeze(dim=0)
        img_meta = None
        result = model.simple_test(logit, img_meta)[0]

        # all_seg_preds.append(seg_pred)
        # all_seg_gts.append(seg_gt)
        all_gts.append(gt)
        all_results.append(result)

        H, W = result.shape
        color = np.zeros((H, W, 3), dtype=np.uint8)
        color[result == 0] = [88, 88, 88]
        color[result == 1] = [255, 255, 255]
        color[gt == 255] = [0, 0, 0]

        color = cv2.resize(color, (W * 4, H * 4))
        mmcv.mkdir_or_exist(os.path.dirname(color_file_path))
        cv2.imwrite(color_file_path, color)
    
    metrics = eval_metrics(all_results, all_gts, num_classes=2, ignore_index=255, metrics=["mIoU", "mFscore", "mDice"])
    print(metrics)
    # unknown_metrics = eval_unknown_metrics(all_seg_preds, all_seg_gts, 19, all_results, 255)
    # for key in unknown_metrics.keys():
    #     print(key)
    #     print(unknown_metrics[key])
