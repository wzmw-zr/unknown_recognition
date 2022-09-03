from unknown_recognition.apis import init_unknown_detector, inference_unknown_detector, show_result_pyplot
import os
import argparse
import mmcv
import numpy as np
import torch
import cv2

def parse_args():
    parser = argparse.ArgumentParser(
        description='ignore edges for semantic segmentation')
    parser.add_argument("config_file", help="config_file", type=str)
    parser.add_argument("checkpoint_file", help="checkpoint_file", type=str)
    parser.add_argument("dataset_path", help="dataset_path", type=str)
    parser.add_argument("output_path", help="output_path", type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    config_file = args.config_file
    checkpoint_file = args.checkpoint_file
    dataset_path = args.dataset_path
    output_path = args.output_path

    logit_dir = os.path.join(dataset_path, "logit")
    softmax_dir = os.path.join(dataset_path, "softmax")
    logit_suffix = "_logit.npy"
    softmax_suffix = "_softmax.npy"
    output_suffix = ".png"

    model = init_unknown_detector(config_file, checkpoint_file, device='cuda:0')

    mmcv.mkdir_or_exist(output_path)

    for logit_file in mmcv.scandir(logit_dir, logit_suffix, recursive=True):
        logit_file_path = os.path.join(logit_dir, logit_file)
        softmax_file_path = os.path.join(softmax_dir, logit_file.replace(logit_suffix, softmax_suffix))
        print(f"inference {logit_file_path}")
        logit = np.load(logit_file_path)
        logit = (logit - np.min(logit)) / (np.max(logit) - np.min(logit))
        logit = torch.as_tensor(logit, device="cuda")
        logit = logit.unsqueeze(dim=0)
        # softmax = np.load(softmax_file_path)
        img_meta = None
        result = model.simple_test(logit, img_meta)[0]
        H, W = result.shape
        gray = np.zeros((H, W, 3), dtype=np.uint8)
        unknown_mask = result == 0
        known_mask = result == 1
        gray[unknown_mask] = [255, 255, 255]
        gray[known_mask] = [0, 0, 0]
        output_file_path = os.path.join(output_path, logit_file.replace(logit_suffix, output_suffix))
        mmcv.mkdir_or_exist(os.path.dirname(output_file_path))
        cv2.imwrite(output_file_path, gray)