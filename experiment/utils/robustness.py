"""
utils/robustness.py
提供图像干扰函数和鲁棒性评估工具
"""
import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
from utils.evaluate import load_groundtruths, normalize_to_xyxy, get_image_size
from utils.metrics import evaluate_predictions, compute_mr_fdr
import json
from tqdm import tqdm


# ---------- 干扰函数 ----------
def apply_brightness_adjust(image, factor):
    """
    调整亮度
    factor: 亮度因子，<1变暗，>1变亮
    """
    return np.clip(image * factor, 0, 255).astype(np.uint8)


def apply_gaussian_noise(image, sigma=25):
    """
    添加高斯噪声
    sigma: 噪声标准差
    """
    noise = np.random.normal(0, sigma, image.shape)
    noisy = image + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_poisson_noise(image):
    """
    添加泊松噪声
    """
    # 泊松噪声与像素值有关，先归一化到0-1再缩放
    img_norm = image.astype(np.float64) / 255.0
    noisy = np.random.poisson(img_norm * 255) / 255.0 * 255
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_motion_blur(image, kernel_size=15, angle=45):
    """
    添加运动模糊
    kernel_size: 核大小
    angle: 运动方向角度
    """
    # 创建运动模糊核
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    rad = np.deg2rad(angle)
    x = int(center + (center - 1) * np.cos(rad))
    y = int(center + (center - 1) * np.sin(rad))
    cv2.line(kernel, (center, center), (x, y), 1, 1)
    kernel = kernel / kernel.sum()
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred


# ---------- 评估函数 ----------
def evaluate_with_corruption(model_path, data_yaml, corruption_func, corruption_name,
                             iou_threshold=0.5, conf_threshold=0.001, device='cuda'):
    """
    对测试集应用干扰后评估模型
    """
    model = YOLO(model_path)
    groundtruths_raw, img_dir = load_groundtruths(data_yaml, split='test')
    img_files = list(Path(img_dir).glob('*.*'))

    # 获取所有图像尺寸
    img_sizes = [get_image_size(str(p)) for p in img_files]

    # 转换真实标签为绝对坐标
    groundtruths = []
    for i, gt in enumerate(groundtruths_raw):
        img_w, img_h = img_sizes[i]
        boxes_abs = [normalize_to_xyxy(box, img_w, img_h) for box in gt['boxes']]
        groundtruths.append({'boxes': boxes_abs, 'labels': gt['labels']})

    # 对每张图像应用干扰并进行预测
    predictions = []
    for i, img_path in enumerate(tqdm(img_files, desc=corruption_name)):
        # 读取图像
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 应用干扰
        corrupted_img = corruption_func(img)
        # 保存为临时文件或直接传递？YOLO predict 支持 numpy array
        results = model.predict(corrupted_img, imgsz=640, conf=conf_threshold, device=device, verbose=False)
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
            scores = results[0].boxes.conf.cpu().numpy().tolist()
            labels = results[0].boxes.cls.cpu().numpy().astype(int).tolist()
        else:
            boxes, scores, labels = [], [], []
        predictions.append({'boxes': boxes, 'scores': scores, 'labels': labels})

    # 评估
    metrics = evaluate_predictions(predictions, groundtruths, iou_threshold, num_classes=4)
    mr, fdr = compute_mr_fdr(predictions, groundtruths, iou_threshold)
    metrics['MR'] = mr
    metrics['FDR'] = fdr
    metrics['corruption_name'] = corruption_name
    return metrics, predictions


def run_all_robustness_tests(model_path, data_yaml, output_dir, device='cuda'):
    """
    运行所有鲁棒性测试
    """
    os.makedirs(output_dir, exist_ok=True)

    # 定义干扰函数和名称
    tests = [
        ('original', lambda img: img),  # 无干扰，作为基线
        ('brightness_-30', lambda img: apply_brightness_adjust(img, 0.7)),
        ('brightness_+30', lambda img: apply_brightness_adjust(img, 1.3)),
        ('gaussian_noise', lambda img: apply_gaussian_noise(img, sigma=25)),
        ('poisson_noise', apply_poisson_noise),
        ('motion_blur', lambda img: apply_motion_blur(img, kernel_size=15, angle=45))
    ]

    all_results = {}
    for name, func in tests:
        print(f"Testing {name}...")
        metrics, preds = evaluate_with_corruption(model_path, data_yaml, func, name, device=device)
        all_results[name] = metrics

        # 保存预测结果JSON
        pred_path = os.path.join(output_dir, f'{name}_predictions.json')
        with open(pred_path, 'w') as f:
            json.dump(preds, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        # 保存指标
        metrics_path = os.path.join(output_dir, f'{name}_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

    # 汇总结果到表格
    summary = {}
    for name, metrics in all_results.items():
        summary[name] = {
            'mAP@0.5': metrics['mAP'],
            'Precision': metrics['macro_precision'],
            'Recall': metrics['macro_recall'],
            'MR': metrics['MR'],
            'FDR': metrics['FDR']
        }
    summary_path = os.path.join(output_dir, 'robustness_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # 打印表格
    print("\n=== Robustness Test Results ===")
    print(f"{'Condition':<20} {'mAP@0.5':>10} {'Precision':>10} {'Recall':>10} {'MR':>10} {'FDR':>10}")
    for name, vals in summary.items():
        print(
            f"{name:<20} {vals['mAP@0.5']:>10.1f} {vals['Precision']:>10.1f} {vals['Recall']:>10.1f} {vals['MR']:>10.2f} {vals['FDR']:>10.2f}")

    return all_results