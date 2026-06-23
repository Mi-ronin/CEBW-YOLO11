import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class WiseInnerIoULoss(nn.Module):
    """
    Wise-Inner-IoU Loss Function
    结合 WIoU v3 的动态非单调聚焦机制 与 Inner-IoU 的内部区域对齐机制。
    公式: L_WiseInnerIoU = exp(d^2/(Wg^2+Hg^2)) * f(β) * (1 - IoU^inner)
    其中:
        - d: 预测框与真实框中心点距离
        - Wg, Hg: 真实框与预测框的最小包围框的宽和高
        - β: 离群度 (outlier degree)，用于动态梯度增益
        - IoU^inner: 使用辅助边界框计算的 Inner-IoU
    参数:
        ratio: Inner-IoU 辅助框尺度因子 (论文表3: 0.8)
        momentum: WIoU 动量更新系数 (默认0.9)
        eps: 防止除零的小常数
    """
    def __init__(self, ratio=0.8, momentum=0.9, eps=1e-7):
        super(WiseInnerIoULoss, self).__init__()
        self.ratio = ratio          # Inner-IoU 辅助框尺度因子
        self.momentum = momentum    # WIoU 动量更新系数
        self.eps = eps

        # 注册动量缓冲区: IoU损失的历史均值 (用于计算 β)
        self.register_buffer('iou_mean', torch.tensor(1.0))

    def forward(self, pred_boxes, target_boxes, anchor_points=None):
        """
        计算 Wise-Inner-IoU 损失
        参数:
            pred_boxes: 预测边界框，形状 [B, N, 4] 或 [N, 4]，格式 (x1, y1, x2, y2)
            target_boxes: 真实边界框，形状 [B, N, 4] 或 [N, 4]
            anchor_points: 可选，锚点坐标，用于计算中心点距离 (格式 (cx, cy))
        返回:
            loss: 标量损失值
        """
        # 确保格式一致并计算 IoU
        # 将坐标转换为 (x1, y1, x2, y2)
        if pred_boxes.size(-1) == 4:
            pred_x1, pred_y1, pred_x2, pred_y2 = pred_boxes.unbind(dim=-1)
            target_x1, target_y1, target_x2, target_y2 = target_boxes.unbind(dim=-1)
        else:
            raise ValueError("Box format must be (x1,y1,x2,y2)")

        # 计算预测框和真实框的宽度、高度
        pw = pred_x2 - pred_x1
        ph = pred_y2 - pred_y1
        gw = target_x2 - target_x1
        gh = target_y2 - target_y1

        # 计算中心点坐标
        pcx = (pred_x1 + pred_x2) / 2
        pcy = (pred_y1 + pred_y2) / 2
        gcx = (target_x1 + target_x2) / 2
        gcy = (target_y1 + target_y2) / 2

        # ========== 1. Inner-IoU 计算 (公式 25-31) ==========
        # 辅助边界框尺度因子 ratio
        inner_ratio = self.ratio

        # 真实框的辅助边界框 (内部或外部)
        inner_bx1 = gcx - (gw * inner_ratio) / 2
        inner_by1 = gcy - (gh * inner_ratio) / 2
        inner_bx2 = gcx + (gw * inner_ratio) / 2
        inner_by2 = gcy + (gh * inner_ratio) / 2

        # 预测框的辅助边界框 (相同尺度因子)
        inner_px1 = pcx - (pw * inner_ratio) / 2
        inner_py1 = pcy - (ph * inner_ratio) / 2
        inner_px2 = pcx + (pw * inner_ratio) / 2
        inner_py2 = pcy + (ph * inner_ratio) / 2

        # 计算辅助框的交集区域
        inter_x1 = torch.max(inner_bx1, inner_px1)
        inter_y1 = torch.max(inner_by1, inner_py1)
        inter_x2 = torch.min(inner_bx2, inner_px2)
        inter_y2 = torch.min(inner_by2, inner_py2)
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

        # 辅助框的并集面积
        area_gt = gw * gh * (inner_ratio ** 2)
        area_pred = pw * ph * (inner_ratio ** 2)
        union_area = area_gt + area_pred - inter_area + self.eps

        inner_iou = inter_area / union_area  # IoU^inner (公式31)
        inner_iou = torch.clamp(inner_iou, min=0, max=1)

        # ========== 2. WIoU v3 动态聚焦因子 ==========
        # 计算中心点距离平方 d^2 (公式33)
        d2 = (pcx - gcx) ** 2 + (pcy - gcy) ** 2

        # 计算最小包围框的宽和高
        min_x = torch.min(pred_x1, target_x1)
        min_y = torch.min(pred_y1, target_y1)
        max_x = torch.max(pred_x2, target_x2)
        max_y = torch.max(pred_y2, target_y2)
        Wg = max_x - min_x
        Hg = max_y - min_y

        # 距离代价因子 R_WIoU (公式23)
        # 注意: 论文中分母为 (Wg^2 + Hg^2)，为避免除零，加上 eps
        denominator = Wg ** 2 + Hg ** 2 + self.eps
        R_wIoU = torch.exp(d2 / denominator)

        # 计算离群度 β (公式22, 35 使用 Inner-IoU 的损失分量)
        iou_loss = 1 - inner_iou
        # 动量更新 IoU 损失的历史均值
        with torch.no_grad():
            self.iou_mean = self.momentum * self.iou_mean + (1 - self.momentum) * iou_loss.detach().mean()
        beta = iou_loss / (self.iou_mean + self.eps)

        # 动态梯度增益函数 f(β) (非单调聚焦机制)
        # 根据 WIoU v3 论文，f(β) = β / (β^2 + 0.5) 或使用简化形式: f(β) = β / (δ * β^2 + 1)
        # 这里采用论文中的公式: f(β) = β / (β^2 + 0.5)
        f_beta = beta / (beta ** 2 + 0.5 + self.eps)
        f_beta = torch.clamp(f_beta, max=2.0)  # 限制最大增益

        # ========== 3. 最终 Wise-Inner-IoU 损失 (公式32) ==========
        loss = R_wIoU * f_beta * iou_loss

        # 返回平均损失
        return loss.mean()


class WiseInnerIoULossV2(nn.Module):
    """
    为方便集成到 YOLO 训练框架中，提供更完整的版本，
    包含 anchor 信息处理和自适应 ratio 衰减。
    该版本额外支持:
        - 根据训练 epoch 动态调整 ratio (可选)
        - 直接接收模型输出的边界框偏移量
    """
    def __init__(self, ratio=0.8, momentum=0.9, eps=1e-7, dynamic_ratio=False, total_epochs=300):
        super().__init__()
        self.base_ratio = ratio
        self.momentum = momentum
        self.eps = eps
        self.dynamic_ratio = dynamic_ratio
        self.total_epochs = total_epochs
        self.register_buffer('iou_mean', torch.tensor(1.0))

    def forward(self, pred_boxes, target_boxes, epoch=None):
        """
        pred_boxes, target_boxes: 形状 [B, N, 4] (x1, y1, x2, y2)
        epoch: 当前训练轮次，用于动态调整 ratio
        """
        # 动态调整 Inner-IoU 的 ratio: 从 0.8 逐渐增加到 1.2 或保持不变
        if self.dynamic_ratio and epoch is not None:
            # 线性变化: 前期使用较小 ratio 加速收敛，后期增大 ratio 扩大回归范围
            ratio = self.base_ratio + (epoch / self.total_epochs) * 0.4  # 0.8 -> 1.2
            ratio = min(ratio, 1.2)
        else:
            ratio = self.base_ratio

        # 解包坐标
        px1, py1, px2, py2 = pred_boxes.unbind(dim=-1)
        tx1, ty1, tx2, ty2 = target_boxes.unbind(dim=-1)

        pw = px2 - px1
        ph = py2 - py1
        gw = tx2 - tx1
        gh = ty2 - ty1

        pcx = (px1 + px2) / 2
        pcy = (py1 + py2) / 2
        gcx = (tx1 + tx2) / 2
        gcy = (ty1 + ty2) / 2

        # Inner-IoU 辅助框
        inner_tx1 = gcx - gw * ratio / 2
        inner_ty1 = gcy - gh * ratio / 2
        inner_tx2 = gcx + gw * ratio / 2
        inner_ty2 = gcy + gh * ratio / 2

        inner_px1 = pcx - pw * ratio / 2
        inner_py1 = pcy - ph * ratio / 2
        inner_px2 = pcx + pw * ratio / 2
        inner_py2 = pcy + ph * ratio / 2

        # 交集
        inter_x1 = torch.max(inner_tx1, inner_px1)
        inter_y1 = torch.max(inner_ty1, inner_py1)
        inter_x2 = torch.min(inner_tx2, inner_px2)
        inter_y2 = torch.min(inner_ty2, inner_py2)
        inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

        # 并集
        area_gt = gw * gh * (ratio ** 2)
        area_pred = pw * ph * (ratio ** 2)
        union = area_gt + area_pred - inter + self.eps

        inner_iou = inter / union
        inner_iou = inner_iou.clamp(0, 1)

        # WIoU 相关量
        d2 = (pcx - gcx) ** 2 + (pcy - gcy) ** 2
        # 最小外接框
        min_x = torch.min(px1, tx1)
        min_y = torch.min(py1, ty1)
        max_x = torch.max(px2, tx2)
        max_y = torch.max(py2, ty2)
        Wg = max_x - min_x
        Hg = max_y - min_y
        denom = Wg ** 2 + Hg ** 2 + self.eps
        R_wiou = torch.exp(d2 / denom)

        iou_loss = 1 - inner_iou
        with torch.no_grad():
            self.iou_mean = self.momentum * self.iou_mean + (1 - self.momentum) * iou_loss.detach().mean()

        beta = iou_loss / (self.iou_mean + self.eps)
        f_beta = beta / (beta ** 2 + 0.5 + self.eps)
        f_beta = torch.clamp(f_beta, max=2.0)

        loss = R_wiou * f_beta * iou_loss
        return loss.mean()

# 测试代码
if __name__ == "Improved Wise-Inner-IoU Loss Function__main__":
    # 模拟预测框和真实框 (batch_size=2, 每个图像3个目标)
    pred = torch.tensor([[[10, 20, 50, 60], [30, 40, 80, 90], [5, 5, 15, 15]]],
                         dtype=torch.float32) * 1.0
    target = torch.tensor([[[12, 22, 52, 62], [28, 38, 78, 88], [6, 6, 14, 14]]],
                          dtype=torch.float32)

    loss_fn = WiseInnerIoULoss(ratio=0.8)
    loss = loss_fn(pred, target)
    print(f"Wise-Inner-IoU Loss: {loss.item():.4f}")

    # 测试动态 ratio 版本
    loss_fn_v2 = WiseInnerIoULossV2(ratio=0.8, dynamic_ratio=True, total_epochs=300)
    loss_v2 = loss_fn_v2(pred, target, epoch=150)
    print(f"Dynamic ratio loss: {loss_v2.item():.4f}")