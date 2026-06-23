"""
utils/metrics.py
计算目标检测评估指标：AP, mAP, 精度, 召回率, F1, 混淆矩阵, PR曲线数据等
"""
import numpy as np
import torch
from collections import defaultdict
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve
import warnings

warnings.filterwarnings('ignore')


def compute_iou(box1, box2):
    """计算两个边界框的IoU (xyxy格式)"""
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = inter_area / (area1 + area2 - inter_area + 1e-7)
    return iou


def compute_ap(recall, precision):
    """计算AP (P-R曲线下面积)"""
    # 添加边界点
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))
    # 使precision单调递减
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    # 计算面积
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
    return ap


def evaluate_predictions(predictions, groundtruths, iou_threshold=0.5, num_classes=4):
    """
    评估预测结果
    参数:
        predictions: list of dicts, 每个dict包含 'boxes', 'scores', 'labels'
        groundtruths: list of dicts, 每个dict包含 'boxes', 'labels'
        iou_threshold: IoU阈值
        num_classes: 类别数
    返回:
        metrics: dict 包含各类AP, mAP, 精度, 召回率, F1, 混淆矩阵, PR曲线数据等
    """
    assert len(predictions) == len(groundtruths)
    # 存储所有检测结果和真实目标
    all_detections = [[] for _ in range(num_classes)]
    all_annotations = [[] for _ in range(num_classes)]

    for pred, gt in zip(predictions, groundtruths):
        # 真实目标
        for box, label in zip(gt['boxes'], gt['labels']):
            all_annotations[int(label)].append({'box': box, 'used': False})
        # 预测目标
        for box, score, label in zip(pred['boxes'], pred['scores'], pred['labels']):
            all_detections[int(label)].append({'box': box, 'score': score})

    # 计算每个类别的AP和PR曲线数据
    ap_per_class = []
    pr_curves = {}  # {class_id: {'precision': [...], 'recall': [...]}}
    class_metrics = {}

    for c in range(num_classes):
        detections = all_detections[c]
        annotations = all_annotations[c]
        if len(annotations) == 0:
            ap_per_class.append(0.0)
            continue

        # 按置信度降序排序
        detections = sorted(detections, key=lambda x: x['score'], reverse=True)
        tp = np.zeros(len(detections))
        fp = np.zeros(len(detections))

        # 匹配
        for d_idx, det in enumerate(detections):
            best_iou = 0
            best_anno_idx = -1
            for a_idx, ann in enumerate(annotations):
                if ann['used']:
                    continue
                iou = compute_iou(det['box'], ann['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_anno_idx = a_idx
            if best_iou >= iou_threshold:
                tp[d_idx] = 1
                annotations[best_anno_idx]['used'] = True
            else:
                fp[d_idx] = 1

        # 累积TP和FP
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        precision = tp_cum / (tp_cum + fp_cum + 1e-7)
        recall = tp_cum / len(annotations)
        ap = compute_ap(recall, precision)
        ap_per_class.append(ap)
        pr_curves[c] = {'precision': precision.tolist(), 'recall': recall.tolist()}

        # 计算该类别的精度和召回率（在IoU阈值下的最终值）
        final_precision = tp_cum[-1] / (tp_cum[-1] + fp_cum[-1] + 1e-7)
        final_recall = tp_cum[-1] / len(annotations)
        f1 = 2 * final_precision * final_recall / (final_precision + final_recall + 1e-7)
        class_metrics[c] = {'precision': final_precision, 'recall': final_recall, 'f1': f1, 'ap': ap}

    mAP = np.mean(ap_per_class)

    # 总体精度和召回率（宏观平均）
    macro_precision = np.mean([class_metrics[c]['precision'] for c in range(num_classes) if c in class_metrics])
    macro_recall = np.mean([class_metrics[c]['recall'] for c in range(num_classes) if c in class_metrics])
    macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall + 1e-7)

    # 混淆矩阵
    # 构建所有预测和真实标签列表
    all_pred_labels = []
    all_true_labels = []
    for pred, gt in zip(predictions, groundtruths):
        # 真实标签（每个框对应一个）
        for box, label in zip(gt['boxes'], gt['labels']):
            all_true_labels.append(int(label))
        # 预测标签（仅保留匹配的？直接取所有预测，匹配逻辑较复杂）
        # 简化：为每个预测找一个最佳匹配真实标签，没有匹配则视为假正类（额外类）
        # 这里提供标准方法：使用匈牙利匹配，为每个预测分配真实标签
        gt_boxes = gt['boxes']
        gt_labels = gt['labels']
        pred_boxes = pred['boxes']
        pred_labels = pred['labels']
        if len(pred_boxes) == 0:
            continue
        # 构建IoU矩阵
        iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
        for i, pbox in enumerate(pred_boxes):
            for j, gbox in enumerate(gt_boxes):
                iou_matrix[i, j] = compute_iou(pbox, gbox)
        # 贪心匹配
        matched = np.zeros(len(gt_boxes), dtype=bool)
        for i in range(len(pred_boxes)):
            best_j = -1
            best_iou = iou_threshold
            for j in range(len(gt_boxes)):
                if not matched[j] and iou_matrix[i, j] > best_iou:
                    best_iou = iou_matrix[i, j]
                    best_j = j
            if best_j != -1:
                all_pred_labels.append(pred_labels[i])
                all_true_labels.append(gt_labels[best_j])
                matched[best_j] = True
            else:
                # 假正类，真实类别为背景（用-1表示）
                all_pred_labels.append(pred_labels[i])
                all_true_labels.append(-1)
        # 未匹配的真实目标为假负类，这里不直接体现在混淆矩阵中，可以通过计算MR得到
    # 使用sklearn混淆矩阵（限制类别范围0~num_classes-1, 背景忽略或单独处理）
    # 这里构建完整混淆矩阵（包含背景）
    labels = list(range(num_classes)) + [-1]
    cm = confusion_matrix(all_true_labels, all_pred_labels, labels=labels)
    # 移除背景行/列（如果需要）

    metrics = {
        'mAP': mAP,
        'ap_per_class': ap_per_class,
        'class_metrics': class_metrics,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'pr_curves': pr_curves,
        'confusion_matrix': cm.tolist(),
        'num_annotations': sum(len(ann) for ann in all_annotations),
        'num_predictions': sum(len(det) for det in all_detections)
    }
    return metrics


def compute_mr_fdr(predictions, groundtruths, iou_threshold=0.5):
    """计算漏检率(MR)和误检率(FDR)"""
    total_gt = 0
    total_fp = 0
    total_tp = 0
    for pred, gt in zip(predictions, groundtruths):
        gt_boxes = gt['boxes']
        total_gt += len(gt_boxes)
        pred_boxes = pred['boxes']
        # 匹配
        matched_gt = set()
        for pbox in pred_boxes:
            best_iou = 0
            best_idx = -1
            for j, gbox in enumerate(gt_boxes):
                iou = compute_iou(pbox, gbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j
            if best_iou >= iou_threshold and best_idx not in matched_gt:
                matched_gt.add(best_idx)
                total_tp += 1
            else:
                total_fp += 1
    fn = total_gt - total_tp
    mr = fn / (total_gt + 1e-7)
    fdr = total_fp / (total_tp + total_fp + 1e-7)
    return mr, fdr