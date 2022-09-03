from collections import OrderedDict
from typing import List

import numpy as np
import torch


def f_score(precision, recall, beta=1):
    """calculate the f-score value.

    Args:
        precision (float | torch.Tensor): The precision value.
        recall (float | torch.Tensor): The recall value.
        beta (int): Determines the weight of recall in the combined score.
            Default: False.

    Returns:
        [torch.tensor]: The f-score value.
    """
    score = (1 + beta**2) * (precision * recall) / (
        (beta**2 * precision) + recall)
    return score


def intersect_and_union(pred_label,
                        label,
                        num_classes,
                        ignore_index):
    """Calculate intersection and Union.

    Args:
        pred_label (ndarray): Prediction result map.
        label (ndarray): Ground truth map.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.

     Returns:
         torch.Tensor: The intersection of prediction and ground truth
            histogram on all classes.
         torch.Tensor: The union of prediction and ground truth histogram on
            all classes.
         torch.Tensor: The prediction histogram on all classes.
         torch.Tensor: The ground truth histogram on all classes.
    """

    assert isinstance(pred_label, np.ndarray), \
        f"pred_label should be np.ndarray, but get {type(pred_label)}"
    assert isinstance(label, np.ndarray), \
        f"label should be np.ndarray, but get {type(label)}"

    pred_label = torch.from_numpy((pred_label))
    label = torch.from_numpy(label)

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect = torch.histc(
        intersect.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_pred_label = torch.histc(
        pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_label = torch.histc(
        label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label


def total_intersect_and_union(results,
                              gt_maps,
                              num_classes,
                              ignore_index):
    """Calculate Total Intersection and Union.

    Args:
        results (list[ndarray]): List of prediction maps.
        gt_maps (list[ndarray]): list of ground truth maps.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes.
         ndarray: The union of prediction and ground truth histogram on all
             classes.
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """
    total_area_intersect = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_union = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_pred_label = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_label = torch.zeros((num_classes, ), dtype=torch.float64)
    for result, gt_seg_map in zip(results, gt_maps):
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(
                result, gt_seg_map, num_classes, ignore_index)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    return total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label


def total_area_to_metrics(total_area_intersect,
                          total_area_union,
                          total_area_pred_label,
                          total_area_label,
                          beta=1):
    """Calculate evaluation metrics
    Args:
        total_area_intersect (Tensor): The intersection of prediction and
            ground truth histogram on all classes.
        total_area_union (Tensor): The union of prediction and ground truth
            histogram on all classes.
        total_area_pred_label (Tensor): The prediction histogram on all
            classes.
        total_area_label (Tensor): The ground truth histogram on all classes.
        beta (float): parameter used in f_score.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """

    all_acc = total_area_intersect.sum() / total_area_label.sum()
    ret_metrics = OrderedDict({'all_Acc': all_acc})

    acc = total_area_intersect / total_area_label
    iou = total_area_intersect / total_area_union
    precision = total_area_intersect / total_area_pred_label
    recall = total_area_intersect / total_area_label
    f_value = torch.tensor(
        [f_score(x[0], x[1], beta) for x in zip(precision, recall)])

    ret_metrics["Acc"] = acc
    ret_metrics["IoU"] = iou
    ret_metrics["Precision"] = precision
    ret_metrics["Recall"] = recall
    ret_metrics["Fscore"] = f_value

    ret_metrics = {
        metric: value.numpy()
        for metric, value in ret_metrics.items()
    }

    return ret_metrics


def eval_metrics(results,
                 gt_maps,
                 num_classes,
                 ignore_index,
                 beta=1):
    """Calculate evaluation metrics
    Args:
        results (list[ndarray]): List of prediction maps.
        gt_maps (list[ndarray]): list of ground truth maps.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        beta (float): parameter used in f_score.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """

    total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label = total_intersect_and_union(
            results, gt_maps, num_classes, ignore_index)
    ret_metrics = total_area_to_metrics(total_area_intersect, total_area_union,
                                        total_area_pred_label, total_area_label,
                                        beta)
    return ret_metrics


def eval_unknown_metrics(
    seg_preds: List[np.ndarray],
    seg_gts: List[np.ndarray],
    num_seg_classes: int,
    unknown_preds: List[np.ndarray],
    ignore_index: int = 255
):
    seg_preds = np.asarray(seg_preds)
    seg_gts = np.asarray(seg_gts)
    unknown_preds = np.asarray(unknown_preds)
    metrics_per_class = dict()
    # Calculcate each class's metric
    for label in range(num_seg_classes):
        # Generate unknown task ground truth for each semantic segmentation class.
        # Let prediction area 'P', ground truth area 'G', intersection area 'I',
        # ignored area 'IGN', total area 'T'.
        # In unknown task ground truth:
        #   1. Let points in intersection area (I) be 1.
        #   2. Let points in area (P - I) be 0.
        #   3. Let points in (IGN + (T - P)) be 255.
        pred_areas_mask = seg_preds == label
        gt_areas_mask = seg_gts == label
        intersect_mask = pred_areas_mask & gt_areas_mask
        ignore_areas_mask = (seg_gts == ignore_index) | (~pred_areas_mask)
        unknown_gts = np.zeros_like(seg_preds)
        unknown_gts[intersect_mask] = 1
        unknown_gts[ignore_areas_mask] = ignore_index
        metrics = eval_metrics(list(unknown_preds), list(
            unknown_gts), num_classes=2, ignore_index=ignore_index)
        metrics_per_class[label] = metrics
    return metrics_per_class


"""
Usage example:

    ignore_index = 255
    num_seg_classes = 19
    all_pred_seg = [ ... ] # list of np.ndarray
    all_gt_seg = [ ... ] # list of np.ndarray
    all_unknown_task_pred = [ ... ] # list of np.ndarray

    metrics = eval_unknown_metrics(all_pred_seg all_gt_seg, call_unkown_task_pred, num_seg_classes, ignore_index)
"""
