import os
import json
import torch
from ultralytics import YOLO
from pathlib import Path
from .metrics import evaluate_predictions, compute_mr_fdr


def load_groundtruths(data_yaml, split='test'):
    import yaml
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    # 数据集根目录
    data_root = Path(data['path'])
    img_dir = data_root / data[split]
    label_dir = data_root / data[split].replace('images', 'labels')
    # 获取所有图像文件
    img_files = list(img_dir.glob('*.*'))
    groundtruths = []
    for img_path in img_files:
        label_path = label_dir / (img_path.stem + '.txt')
        boxes = []
        labels = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, xc, yc, w, h = map(float, parts)
                        boxes.append([xc, yc, w, h])
                        labels.append(int(cls))
        groundtruths.append({'boxes': boxes, 'labels': labels, 'img_path': str(img_path)})
    return groundtruths, img_dir


def get_image_size(img_path):
    from PIL import Image
    with Image.open(img_path) as img:
        return img.size


def normalize_to_xyxy(box_norm, img_w, img_h):
    xc, yc, w, h = box_norm
    x1 = (xc - w / 2) * img_w
    y1 = (yc - h / 2) * img_h
    x2 = (xc + w / 2) * img_w
    y2 = (yc + h / 2) * img_h
    return [max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)]


def evaluate_model(model_path, data_yaml, iou_threshold=0.5, conf_threshold=0.001, device='cuda'):
    model = YOLO(model_path)
    groundtruths_raw, img_dir = load_groundtruths(data_yaml, split='test')
    # 获取图像尺寸
    img_sizes = [get_image_size(gt['img_path']) for gt in groundtruths_raw]
    groundtruths = []
    for i, gt in enumerate(groundtruths_raw):
        img_w, img_h = img_sizes[i]
        boxes_abs = []
        for box in gt['boxes']:
            x1, y1, x2, y2 = normalize_to_xyxy(box, img_w, img_h)
            boxes_abs.append([x1, y1, x2, y2])
        groundtruths.append({'boxes': boxes_abs, 'labels': gt['labels']})

    results = model.predict(source=str(img_dir), imgsz=640, conf=conf_threshold, device=device, verbose=False)
    predictions = []
    for res in results:
        if res.boxes is not None:
            boxes = res.boxes.xyxy.cpu().numpy().tolist()
            scores = res.boxes.conf.cpu().numpy().tolist()
            labels = res.boxes.cls.cpu().numpy().astype(int).tolist()
        else:
            boxes, scores, labels = [], [], []
        predictions.append({'boxes': boxes, 'scores': scores, 'labels': labels})

    metrics = evaluate_predictions(predictions, groundtruths, iou_threshold, num_classes=4)
    mr, fdr = compute_mr_fdr(predictions, groundtruths, iou_threshold)
    metrics['MR'] = mr
    metrics['FDR'] = fdr
    metrics['predictions'] = predictions
    metrics['groundtruths'] = groundtruths
    return metrics


def save_predictions_json(predictions, groundtruths, save_path):
    data = []
    for pred, gt in zip(predictions, groundtruths):
        data.append({
            'pred_boxes': pred['boxes'],
            'pred_scores': pred['scores'],
            'pred_labels': pred['labels'],
            'gt_boxes': gt['boxes'],
            'gt_labels': gt['labels']
        })
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)


def evaluate_all_models(model_paths, data_yaml, output_dir='evaluation_results'):
    os.makedirs(output_dir, exist_ok=True)
    all_metrics = {}
    for name, path in model_paths.items():
        print(f"Evaluating {name}...")
        metrics = evaluate_model(path, data_yaml)
        pred_path = os.path.join(output_dir, f'{name}_predictions.json')
        save_predictions_json(metrics['predictions'], metrics['groundtruths'], pred_path)
        metrics.pop('predictions', None)
        metrics.pop('groundtruths', None)
        all_metrics[name] = metrics
        with open(os.path.join(output_dir, f'{name}_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
    return all_metrics