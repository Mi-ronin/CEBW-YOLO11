import torch
import torch.nn as nn

# -------------------- 基础卷积模块 (CBS) --------------------
class CBS(nn.Module):
    """Conv + BatchNorm + SiLU (CBS模块，常见于YOLO系列)"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1, dilation=1):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding=padding, groups=groups, bias=False, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)  # 使用SiLU激活函数

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# -------------------- DWR 模块 --------------------
class DWR(nn.Module):
    """
    Dilation-wise Residual (DWR) 模块
    包含两个阶段:
        1) Region-wise Residual (RR): 3x3 卷积 + BN + SiLU，局部重构特征
        2) Semantic-wise Residual (SR): 多扩张率深度可分离卷积 + 逐点卷积融合 + 残差连接
    最终与原始输入相加，增强多尺度特征表达。
    """
    def __init__(self, channels, dilations=[1, 3, 5]):
        """
        Args:
            channels: 输入/输出通道数
            dilations: 扩张率列表，每个分支使用不同的扩张率
        """
        super().__init__()
        # ---------- RR阶段 ----------
        self.rr = CBS(channels, channels, kernel_size=3, stride=1, padding=1)

        # ---------- SR阶段 ----------
        # 多个并行的深度可分离卷积分支（仅Depthwise Conv）
        self.depthwise_convs = nn.ModuleList()
        for d in dilations:
            # padding = dilation 以保证特征图尺寸不变
            padding = d
            dw_conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1,
                                padding=padding, groups=channels, bias=False, dilation=d)
            self.depthwise_convs.append(dw_conv)

        # 融合所有分支的逐点卷积 (Pointwise Conv)
        num_branches = len(dilations)
        self.pw_conv = CBS(channels * num_branches, channels, kernel_size=1, stride=1, padding=0)

        # BN 和激活（可选，逐点卷积中CBS已包含，此处不再重复）
        # 注意：每个深度可分离卷积后未加BN+激活，统一在拼接后通过pw_conv处理

    def forward(self, x):
        identity = x  # 原始输入用于最终残差连接

        # RR阶段
        rr_out = self.rr(x)  # [B, C, H, W]

        # SR阶段：多分支深度可分离卷积
        sr_branches = []
        for dw_conv in self.depthwise_convs:
            branch_out = dw_conv(rr_out)  # 深度卷积，保持通道数不变
            sr_branches.append(branch_out)

        # 沿通道维度拼接
        sr_concat = torch.cat(sr_branches, dim=1)  # [B, C*num_branches, H, W]

        # 逐点卷积融合多尺度信息
        sr_out = self.pw_conv(sr_concat)  # [B, C, H, W]

        # 残差连接
        out = identity + sr_out
        return out


# -------------------- C3K-DWR 模块 (基础残差块) --------------------
class C3K_DWR(nn.Module):
    """
    基于 DWR 构建的 C3K 风格模块。
    类似于 YOLO 中的 Bottleneck，包含两个 CBS 和一个 DWR 核心，
    并带有 shortcut 选项。
    """
    def __init__(self, in_channels, out_channels, shortcut=True, dilations=[1,3,5]):
        super().__init__()
        self.shortcut = shortcut and in_channels == out_channels
        # 第一个卷积降低通道数（可选，此处保持通道不变，若需扩展可调整）
        self.cv1 = CBS(in_channels, out_channels, kernel_size=1, stride=1)  # 通道变换
        self.dwr = DWR(out_channels, dilations=dilations)  # DWR 核心模块
        self.cv2 = CBS(out_channels, out_channels, kernel_size=3, stride=1)  # 可选后处理

    def forward(self, x):
        y = self.cv1(x)
        y = self.dwr(y)
        y = self.cv2(y)
        if self.shortcut:
            return x + y
        else:
            return y


# -------------------- C3K2-DWR 模块 (类似 C2f，用于多级特征提取) --------------------
class C3K2_DWR(nn.Module):
    """
    替代 YOLO 中 C3k2 的模块，将内部 Bottleneck 替换为 DWR 模块。
    结构：输入 → CBS1 → split → (n个C3K_DWR) → concat → CBS2
    实现多尺度特征的有效提取与融合。
    """
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, dilations=[1,3,5]):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            n: C3K_DWR 模块重复次数
            shortcut: 是否使用残差连接
            dilations: DWR 中使用的扩张率列表
        """
        super().__init__()
        # 中间隐藏通道数，通常设为 out_channels 的一半
        hidden_channels = out_channels // 2

        # 第一个 CBS，将输入通道映射到隐藏通道的两倍（便于 split）
        self.cv1 = CBS(in_channels, 2 * hidden_channels, kernel_size=1, stride=1)

        # 一系列 C3K_DWR 模块
        self.m = nn.Sequential(*[
            C3K_DWR(hidden_channels, hidden_channels, shortcut=shortcut, dilations=dilations)
            for _ in range(n)
        ])

        # 输出前的 CBS
        self.cv2 = CBS(hidden_channels * 2, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        # 经过第一个卷积，通道数变为 2*hidden_channels
        y = self.cv1(x)  # [B, 2*hidden, H, W]

        # 沿通道维度分割成两个分支
        y1, y2 = y.chunk(2, dim=1)  # 每个分支 [B, hidden, H, W]

        # 分支2经过 n 个 C3K_DWR 模块
        y2 = self.m(y2)

        # 将两个分支拼接
        out = torch.cat([y1, y2], dim=1)  # [B, 2*hidden, H, W]

        # 最终卷积调整通道数到 out_channels
        return self.cv2(out)


# -------------------- 测试代码 --------------------
if __name__ == "C3K2_DWR__main__":
    # 测试 DWR 模块
    dwr = DWR(channels=64, dilations=[1, 3, 5])
    x = torch.randn(2, 64, 32, 32)
    out_dwr = dwr(x)
    print(f"DWR output shape: {out_dwr.shape}")

    # 测试 C3K_DWR 模块
    c3k_dwr = C3K_DWR(in_channels=64, out_channels=64, shortcut=True)
    out_c3k = c3k_dwr(x)
    print(f"C3K_DWR output shape: {out_c3k.shape}")

    # 测试 C3K2_DWR 模块 (n=2 表示两个 DWR 块)
    c3k2_dwr = C3K2_DWR(in_channels=64, out_channels=128, n=2, shortcut=True)
    out_c3k2 = c3k2_dwr(x)
    print(f"C3K2_DWR output shape: {out_c3k2.shape}")

    # 统计参数量
    total_params = sum(p.numel() for p in c3k2_dwr.parameters())
    print(f"C3K2_DWR (n=2) parameters: {total_params:,}")