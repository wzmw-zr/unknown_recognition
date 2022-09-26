import mmcv
import os
import argparse 
import cv2
import numpy as np
import joblib
import torch
from sklearn import linear_model
from tqdm import tqdm
from unknown_recognition.core import eval_metrics



def parse_args():
    parser = argparse.ArgumentParser(
        description='Decision Tree')
    parser.add_argument("dataset_dir", help="dataset_dir", type=str)
    parser.add_argument("input_type", help="input_type", type=str)
    args = parser.parse_args()
    return args

logit_suffix = ".npy"
gtFine_suffix = "_TrainIds.png"


def load_dataset(dataset_dir: str, split: str, input_type: str):
    logit_dir = os.path.join(dataset_dir, "logit", split)
    gtFine_dir = os.path.join(dataset_dir, "gtFine", split)
    logit_files = list()
    gtFine_files = list()
    for logit_file in mmcv.scandir(logit_dir, logit_suffix, True):
        logit_file_path = os.path.join(logit_dir, logit_file)
        gtFine_file_path = os.path.join(gtFine_dir, logit_file.replace(logit_suffix, gtFine_suffix))
        logit_files.append(logit_file_path)
        gtFine_files.append(gtFine_file_path)
    C, H, W = np.load(logit_files[0]).shape
    N = len(logit_files)
    if "concat" in input_type:
        samples = np.zeros((N, 2, H, W), dtype=np.float32)
    else:
        samples = np.zeros((N, H, W), dtype=np.float32)
    gtFines = np.zeros((N, H, W), dtype=np.uint8)
    for i, (logit_file, gtFine_file) in tqdm(enumerate(zip(logit_files, gtFine_files))):
        logit = np.load(logit_file)
        gtFine = cv2.imread(gtFine_file, cv2.IMREAD_UNCHANGED).astype(np.uint8)
        if input_type == "max_logit":
            logit = (logit - np.min(logit)) / (np.max(logit) - np.min(logit))
            max_logit = np.max(logit, axis=0)
            samples[i] = max_logit
        elif input_type == "L1_softmax_distance":
            logit = torch.as_tensor(logit)
            prob = torch.nn.functional.softmax(logit, dim=0)
            softmax_distance = torch.sum(prob, dim=0)
            softmax_distance = softmax_distance.cpu().numpy()
            samples[i] = softmax_distance
        elif input_type == "L2_softmax_distance":
            logit = torch.as_tensor(logit)
            prob = torch.nn.functional.softmax(logit, dim=0)
            softmax_distance = torch.sqrt(torch.sum(prob ** 2, dim=0))
            softmax_distance = softmax_distance.cpu().numpy()
            samples[i] = softmax_distance
        elif input_type == "max_logit_add_L1_softmax_distance":
            prob = torch.nn.functional.softmax(torch.as_tensor(logit), dim=0)
            softmax_distance = torch.sum(prob, dim=0)
            softmax_distance = softmax_distance.cpu().numpy()
            logit = (logit - np.min(logit)) / (np.max(logit) - np.min(logit))
            max_logit = np.max(logit, axis=0)
            samples[i] = max_logit + softmax_distance
        elif input_type == "max_logit_add_L2_softmax_distance":
            prob = torch.nn.functional.softmax(torch.as_tensor(logit), dim=0)
            softmax_distance = torch.sqrt(torch.sum(prob ** 2, dim=0))
            softmax_distance = softmax_distance.cpu().numpy()
            logit = (logit - np.min(logit)) / (np.max(logit) - np.min(logit))
            max_logit = np.max(logit, axis=0)
            samples[i] = max_logit + softmax_distance
        elif input_type == "max_logit_concat_L1_softmax_distance":
            prob = torch.nn.functional.softmax(torch.as_tensor(logit), dim=0)
            softmax_distance = torch.sum(prob, dim=0)
            softmax_distance = softmax_distance.cpu().numpy()
            logit = (logit - np.min(logit)) / (np.max(logit) - np.min(logit))
            max_logit = np.max(logit, axis=0)
            sample = np.stack([max_logit, softmax_distance])
            samples[i] = sample
        elif input_type == "max_logit_concat_L2_softmax_distance":
            prob = torch.nn.functional.softmax(torch.as_tensor(logit), dim=0)
            softmax_distance = torch.sqrt(torch.sum(prob ** 2, dim=0))
            softmax_distance = softmax_distance.cpu().numpy()
            logit = (logit - np.min(logit)) / (np.max(logit) - np.min(logit))
            max_logit = np.max(logit, axis=0)
            sample = np.stack([max_logit, softmax_distance])
            samples[i] = sample
        else:
            raise NotImplementedError
        gtFines[i] = gtFine
    print(f"load {N} logit files")
    if "concat" in input_type:
        samples = samples.transpose(0, 2, 3, 1)
        samples = samples.reshape(N * H * W, -1)
    else:
        samples = samples.reshape(N * H * W)
    gtFines = gtFines.reshape(N * H * W)
    mask = gtFines != 255
    samples = samples[mask]
    gtFines = gtFines[mask]
    samples = samples.reshape(len(samples), -1)
    return samples, gtFines

"""
Need large RAM, cannot train now.
"""
if __name__ == "__main__":
    args = parse_args()
    num_class = 2
    dataset_dir = args.dataset_dir
    input_type = args.input_type

    train_samples, train_labels = load_dataset(dataset_dir, "train", input_type)
    test_samples, test_labels = load_dataset(dataset_dir, "test", input_type)
    classifier = linear_model.LogisticRegression()
    classifier.fit(train_samples, train_labels)
    joblib.dump(classifier, "logistic_regression.pkl")


    print("=================================")
    print("========Confusing matrix=========")
    print("=================================")
    test_pred = classifier.predict(test_samples)

    hist = np.bincount(
        num_class *  test_labels.astype(np.int32) + test_pred.astype(np.int32), minlength=num_class ** 2
    ).reshape(num_class, num_class)

    print(hist)
    all_acc = np.diag(hist).sum() / hist.sum()
    acc = np.diag(hist) / np.sum(hist, axis=1)
    intersection = np.diag(hist)
    union = np.sum(hist, axis=1) + np.sum(hist, axis=0) - intersection
    precision = np.diag(hist) / np.sum(hist, axis=0)
    recall = np.diag(hist) / np.sum(hist, axis=1)
    print(f"all_acc = {all_acc}")
    print(f"acc = {acc}")
    print(f"precision = {precision}")
    print(f"recall = {recall}")
    print(f"iou = {intersection / union}")


    if input_type == "max_logit_add_L2_softmax_distance":
        print("=================================")
        print("=====accumulate each image=======")
        print("=================================")
        logit_path = os.path.join(dataset_dir, "logit", "test")
        gtFine_path = os.path.join(dataset_dir, "gtFine", "test")
        prediction_path = os.path.join("logistic_regression_prediction", "test")

        prediction_list = []
        gtFine_list = []
        for logit_file in mmcv.scandir(logit_path, logit_suffix, True):
            logit_file_path = os.path.join(logit_path, logit_file)
            gt_file_path = os.path.join(gtFine_path, logit_file.replace(logit_suffix, gtFine_suffix))
            prediction_file_path = os.path.join(prediction_path, logit_file.replace(logit_suffix, ".png"))

            logit = np.load(logit_file_path)
            gt = cv2.imread(gt_file_path, cv2.IMREAD_UNCHANGED).astype(np.uint8)
            H, W = logit.shape[-2:]
            prob = torch.nn.functional.softmax(torch.as_tensor(logit), dim=0)
            softmax_distance = torch.sqrt(torch.sum(prob ** 2, dim=0)).cpu().numpy()
            logit = (logit - np.min(logit)) / (np.max(logit) - np.min(logit))
            max_logit = np.max(logit, axis=0)
            input_feature = max_logit + softmax_distance
            input_feature = input_feature.reshape(-1, 1)
            prediction = classifier.predict(input_feature)
            prediction = prediction.reshape(H, W)
            prediction_list.append(prediction)
            gtFine_list.append(gt)

            color = np.zeros((H, W, 3), dtype=np.uint8)
            color[prediction == 0] = [122, 122, 122]
            color[prediction == 1] = [255, 255, 255]
            color[gt == 255] = [0, 0, 0]
            mmcv.mkdir_or_exist(os.path.dirname(prediction_file_path))
            cv2.imwrite(prediction_file_path, color)

        metrics = eval_metrics(prediction_list, gtFine_list, 2, 255, ["mIoU", "mFscore"])
        print(f"all_acc = {metrics['aAcc']}")
        print(f"acc = {metrics['Acc']}")
        print(f"precision = {metrics['Precision']}")
        print(f"recall = {metrics['Recall']}")
        print(f"iou = {metrics['IoU']}")
        print(f"fscore = {metrics['Fscore']}")