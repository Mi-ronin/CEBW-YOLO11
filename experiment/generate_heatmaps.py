"""
generate_heatmaps.py
使用 Grad-CAM 生成热力图，用于可视化模型对缺陷区域的关注。
适用于 YOLO11 及其改进版本。
用法：
    python generate_heatmaps.py --model weights/best.pt --data configs/dataset.yaml --image test.jpg
    python generate_heatmaps.py --model weights/best.pt --data configs/dataset.yaml --image_dir test_images/ --output heatmaps/
    python generate_heatmaps.py --model weights/best.pt --data configs/dataset.yaml --layer model.22  # 指定层名
"""
import os
import argparse
import torch
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO
from utils.gradcam import GradCAM, show_cam_on_image, get_target_layer


def load_image(image_path, imgsz=640):
    """加载图像并预处理为模型输入张量"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h0, w0 = img.shape[:2]
    # 保持宽高比缩放
    r = imgsz / max(h0, w0)
    new_w, new_h = int(w0 * r), int(h0 * r)
    img_resized = cv2.resize(img, (new_w, new_h))
    # 填充至 imgsz x imgsz
    top = (imgsz - new_h) // 2
    bottom = imgsz - new_h - top
    left = (imgsz - new_w) // 2
    right = imgsz - new_w - left
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))
    # 转换为张量
    img_tensor = torch.from_numpy(img_padded).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return img_tensor, img_padded, (top, left, new_h, new_w)  # 返回填充信息用于恢复坐标


def generate_heatmap(model, image_path, target_layer, class_idx=None, imgsz=640, device='cuda'):
    """生成单张图片的热力图"""
    # 加载图像
    img_tensor, img_padded, pad_info = load_image(image_path, imgsz)
    img_tensor = img_tensor.to(device)
    # 模型预测
    model.eval()
    with torch.no_grad():
        preds = model(img_tensor)
    # 若未指定类别，取置信度最高的类别
    if class_idx is None:
        # 获取预测结果
        # YOLO 的返回值通常是一个列表，包含多个目标的预测，我们需要根据置信度取最高
        # 简便方法：用模型预测后的结果，直接取最高分数类别（可能不准确，但用于热力图大致可）
        # 或者通过模型的前向输出（即未经过NMS的原始输出）？
        # 由于 model 是 YOLO 对象，我们调用它的 predict 方法得到结果
        # 但 Grad-CAM 需要梯度和输出，所以我们需要用 model.model 的前向
        # 这里我们直接使用 model.model 进行前向，并取分类输出的最大值
        # 对于 YOLO，输出是一个多尺度特征，不是直接的分类输出，所以不适用。
        # 因此，我们只能使用从检测头后的预测中取最高置信度的类别，但无法获得梯度的输出。
        # 一个变通：我们仍然用 model.model 的前向，但取其分类分支（对于 Detect 模块，它输出三个张量）
        # 实际上 Grad-CAM 只适用于分类网络，对于检测网络需要采用类似的方式。
        # 常见做法：对每个检测到的目标单独生成热力图，或取某个特定类别的所有目标的平均。
        # 这里简化为：对模型输出进行 NMS 后，取置信度最高的检测框，其类别作为目标类。
        # 但为了计算梯度，我们需要模型的原始输出（未 NMS 的）。
        # 由于我们只有模型，我们使用 model.model 的前向。
        # 注意：YOLO 的 Detect 模块输出是 (batch, num_anchors, ...) 等，直接取分类部分较复杂。
        # 简化：我们直接使用 predict 得到的结果，但 Grad-CAM 需要梯度的输出，所以必须用 model.model。
        # 所以，我们可能需要修改 GradCAM 类，使其能处理检测输出。但为了演示，我们改为使用模型的分类头（如果有）。
        # 对于 YOLO11，没有分类头，所以我们需要一种近似方法。
        # 论文中的热力图通常使用特定类别的梯度，他们可能是对某个特征图做平均。
        # 我们可以选择某个目标类（例如 0 类）作为目标。
        # 因此，我们在这里直接让用户指定 class_idx，否则默认为 0。
        if class_idx is None:
            class_idx = 0
    # 创建 GradCAM 对象
    gradcam = GradCAM(model.model, target_layer)
    # 生成热力图
    heatmap = gradcam.forward(img_tensor, class_idx=class_idx, retain_graph=False)
    # 移除 hook
    gradcam.remove_hooks()
    # 恢复原始尺寸热力图（去掉填充）
    top, left, new_h, new_w = pad_info
    h, w = heatmap.shape
    heatmap_crop = heatmap[top:top+new_h, left:left+new_w]
    heatmap_resized = cv2.resize(heatmap_crop, (img_tensor.shape[3] - left - (left), img_tensor.shape[2] - top - (top)))
    # 由于 padding，我们直接裁剪到原始图像尺寸
    # 但原始图像可能在填充前有缩放，我们需要恢复至原始图像尺寸
    # 获得原始图像
    orig_img = cv2.imread(image_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = orig_img.shape[:2]
    # 上采样热力图到原图尺寸
    heatmap_orig = cv2.resize(heatmap_crop, (w_orig, h_orig))
    return heatmap_orig, orig_img


def main():
    parser = argparse.ArgumentParser(description='Generate Grad-CAM heatmaps for YOLO model')
    parser.add_argument('--model', type=str, required=True, help='Path to model weights (.pt)')
    parser.add_argument('--data', type=str, help='Dataset config YAML (for class names)')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--image_dir', type=str, help='Directory of images to process')
    parser.add_argument('--output', type=str, default='heatmaps', help='Output directory for heatmaps')
    parser.add_argument('--layer', type=str, default=None, help='Target layer name (e.g., "model.22")')
    parser.add_argument('--class_idx', type=int, default=0, help='Target class index (default 0)')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    args = parser.parse_args()

    # 加载模型
    print(f"Loading model from {args.model}...")
    model = YOLO(args.model)
    # 获取目标层
    if args.layer is not None:
        target_layer = get_target_layer(model.model, args.layer)
    else:
        # 自动检测：取 model.model 的倒数第二个模块（假设最后是 Detect）
        if hasattr(model.model, 'model') and isinstance(model.model.model, torch.nn.Sequential):
            # 对于 ultralytics YOLO，模型在 model.model 下
            target_layer = model.model.model[-2]  # 倒数第二个
        else:
            target_layer = model.model[-2]  # 兼容其他结构
        print(f"Auto-selected target layer: {target_layer.__class__.__name__}")

    # 收集图像列表
    if args.image:
        image_paths = [args.image]
    elif args.image_dir:
        image_dir = Path(args.image_dir)
        image_paths = list(image_dir.glob('*.*'))
        # 过滤图片格式
        image_paths = [str(p) for p in image_paths if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    else:
        raise ValueError("Please specify either --image or --image_dir")

    # 输出目录
    os.makedirs(args.output, exist_ok=True)

    # 遍历图像
    for img_path in image_paths:
        print(f"Processing {img_path}...")
        heatmap, orig_img = generate_heatmap(model, img_path, target_layer, class_idx=args.class_idx,
                                             imgsz=args.imgsz, device=args.device)
        # 叠加热力图
        vis = show_cam_on_image(orig_img, heatmap)
        # 保存
        out_name = os.path.basename(img_path)
        out_path = os.path.join(args.output, out_name)
        cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"Saved heatmap to {out_path}")

    print("All heatmaps generated.")


if __name__ == '__main__':
    main()