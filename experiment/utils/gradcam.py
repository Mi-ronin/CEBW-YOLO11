"""
utils/gradcam.py
Grad-CAM 实现，支持任意 PyTorch 模型的指定层。
使用方法：
    from utils.gradcam import GradCAM, show_cam_on_image
    model = ...  # 加载模型
    target_layer = model.model[-2]  # 示例：Neck 最后一层
    gradcam = GradCAM(model, target_layer)
    heatmap = gradcam(input_tensor, class_idx)
    vis = show_cam_on_image(original_image, heatmap)
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from collections import OrderedDict


class GradCAM:
    """
    针对任意目标层的 Grad-CAM
    """
    def __init__(self, model, target_layer):
        """
        Args:
            model: PyTorch 模型
            target_layer: 需要可视化的层（nn.Module 实例）
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handles = []

        # 注册前向和反向钩子
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            # 保存激活值
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # 保存梯度
            self.gradients = grad_output[0].detach()

        handle_f = self.target_layer.register_forward_hook(forward_hook)
        handle_b = self.target_layer.register_backward_hook(backward_hook)
        self.handles = [handle_f, handle_b]

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()

    def forward(self, x, class_idx=None, retain_graph=False):
        """
        执行前向传播并计算梯度（如果需要）
        Args:
            x: 输入张量 [B,C,H,W]
            class_idx: 目标类别索引，若为None则取预测置信度最高的类别
            retain_graph: 是否保留计算图
        Returns:
            heatmap: 归一化的热力图 [H,W] (0~1)
        """
        # 前向
        output = self.model(x)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # 清空梯度
        self.model.zero_grad()
        # 对目标类别计算梯度
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1.0
        output.backward(gradient=one_hot, retain_graph=retain_graph)

        # 获取梯度和激活
        gradients = self.gradients  # [B, C, H', W']
        activations = self.activations  # [B, C, H', W']

        # 计算权重：全局平均池化梯度 (GAP)
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [B, C, 1, 1]

        # 加权组合
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # [B, 1, H', W']
        # 应用 ReLU
        cam = F.relu(cam)
        # 归一化到 [0,1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        # 上采样到输入尺寸
        cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
        # 移除批次和通道维度
        heatmap = cam[0, 0].cpu().numpy()
        return heatmap


def show_cam_on_image(img, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    将热力图叠加到原始图像上
    Args:
        img: 原始图像 (H,W,3) 0~255 uint8 或 0~1 float
        heatmap: 热力图 (H,W) 0~1 float
        alpha: 透明度
        colormap: OpenCV 颜色映射
    Returns:
        vis: 叠加后的图像 (H,W,3) uint8
    """
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    # 生成彩色热力图
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, colormap)
    # 叠加
    vis = cv2.addWeighted(img, 1-alpha, heatmap_color, alpha, 0)
    return vis


def get_target_layer(model, layer_name=None):
    """
    根据层名称或自动查找获取目标层。
    如果未指定，尝试找到最接近 Detect 模块的前一层（即 Neck 最后一层）。
    """
    if layer_name is not None:
        # 按名称查找
        for name, module in model.named_modules():
            if name == layer_name:
                return module
        raise ValueError(f"Layer '{layer_name}' not found in model.")
    else:
        # 自动查找：遍历所有模块，找到 Detect 模块，取前一层的模块
        # Detect 模块是最后一个模块？通常模型结构为 Sequential，Detect 在最后
        modules = list(model.modules())
        # 逆序查找 Detect 类
        detect_cls = None
        try:
            from ultralytics.nn.modules import Detect
            detect_cls = Detect
        except ImportError:
            # 如果 ultralytics 未安装，尝试从模型结构判断
            pass
        # 简单方法：取 model.model 的倒数第二个模块（假设最后是 Detect）
        if hasattr(model, 'model') and isinstance(model.model, torch.nn.Sequential):
            # 如果模型是 YOLO 类型的，它的 model.model 是 Sequential
            if len(model.model) >= 2:
                # 通常最后一个模块是 Detect，我们取倒数第二个
                return model.model[-2]
            else:
                raise ValueError("Cannot automatically find target layer.")
        else:
            # 递归查找
            last_detect = None
            for module in modules:
                if detect_cls is not None and isinstance(module, detect_cls):
                    last_detect = module
            if last_detect is not None:
                # 需要找到该模块的父模块列表中的前一个
                # 用 named_children 获取，但比较复杂，建议用户显式指定
                raise ValueError("Auto detection of target layer not reliable. Please specify layer name.")
            else:
                raise ValueError("Detect module not found. Please specify target layer name.")