import torch
import torch.nn as nn
import torch.nn.functional as F


class EAC(nn.Module):
    """
    Efficient Attention Compression (EAC) 模块


    设计思路：
        1. 压缩 (Compression)   ：使用 2×2、4×4 平均池化降低空间分辨率，并结合 1×1 卷积降通道（压缩率=4）
        2. 建模 (Modeling)      ：拼接多尺度特征后，通过矩阵乘法（Mat MUI）计算注意力权重
        3. 重建 (Reconstruction)：利用生成的注意力权重加权原始输入，并通过残差连接保持信息流

    结构对应图13：
        Input -> 3x3 Conv -> (2x2 AvgPool, 4x4 AvgPool) -> Concat -> Mat MUI -> 8x8 AvgPool -> Attention weights
        最终将注意力权重上采样并应用于原始特征。
    """

    def __init__(self, channels, reduction=4):
        """
        Args:
            channels: 输入特征图的通道数 C
            reduction: 压缩率 (默认4，对应 EAC compression rate = 4)
        """
        super().__init__()
        # 压缩后的通道数
        compressed_channels = max(1, channels // reduction)

        # 1. 初始卷积，稳定特征分布（图中第一个 3x3 Conv）
        self.init_conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

        # 2. 多尺度平均池化分支（对应图中 2x2, 4x4 AvgPool）
        #    每个分支先池化，再通过 1x1 卷积将通道压缩至 compressed_channels
        self.pool_branch2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(channels, compressed_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(compressed_channels),
            nn.SiLU(inplace=True)
        )
        self.pool_branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=4, stride=4),
            nn.Conv2d(channels, compressed_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(compressed_channels),
            nn.SiLU(inplace=True)
        )
        # 图中另有 8x8 AvgPool，此处将其用于最终的注意力权重聚合（见 forward 中的第八行）
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 相当于 8x8 池化后取全局平均（实际尺寸适应任意输入）

        # 3. 矩阵乘法部分（Mat MUI）：基于拼接后的多尺度特征生成注意力权重
        #    先将各分支特征上采样到同一空间尺寸，然后拼接，再通过全连接层得到通道权重
        #    这里设计为轻量的两层 MLP（模拟矩阵乘法的非线性映射）
        self.fc = nn.Sequential(
            nn.Linear(compressed_channels * 2, compressed_channels, bias=False),  # *2 因为有 2×2 和 4×4 两个分支
            nn.SiLU(inplace=True),
            nn.Linear(compressed_channels, channels, bias=False),
            nn.Sigmoid()  # 输出 0~1 的注意力权重
        )

    def forward(self, x):
        """
        Args:
            x: 输入特征图 [B, C, H, W]
        Returns:
            经过注意力加权的特征图，尺寸与 x 相同
        """
        identity = x

        # ----- 压缩阶段 -----
        # 初始卷积
        out = self.init_conv(x)
        out = self.bn(out)
        out = self.act(out)

        # 多尺度池化压缩
        feat2 = self.pool_branch2(out)  # [B, C/r, H/2, W/2]
        feat4 = self.pool_branch4(out)  # [B, C/r, H/4, W/4]

        # ----- 建模阶段（Mat MUI）-----
        # 将两个尺度的特征上采样至相同空间大小（以较大者 H/2,W/2 为准）
        feat4_up = F.interpolate(feat4, size=feat2.shape[2:], mode='bilinear', align_corners=False)
        # 沿通道拼接
        concat_feat = torch.cat([feat2, feat4_up], dim=1)  # [B, 2*(C/r), H/2, W/2]

        # 全局平均池化 + 8x8 AvgPool 效果（图中最后的 8x8 池化）
        # 此处用 AdaptiveAvgPool2d(1) 替代，以获得全局描述向量
        global_desc = concat_feat.mean(dim=[2, 3])  # [B, 2*(C/r)]

        # 通过全连接层生成通道注意力权重
        attention_weights = self.fc(global_desc).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]

        # ----- 重建阶段 -----
        # 注意力加权 + 残差连接
        out = identity * attention_weights
        out = out + identity  # 残差连接增强稳定性（可根据需要移除）
        return out

# 测试
if __name__ == "EAC__main__":
    # 创建模块，压缩率 = 4
    eac = EAC(channels=64, reduction=4)
    dummy = torch.randn(2, 64, 80, 80)
    out = eac(dummy)
    print(f"Input shape: {dummy.shape}, Output shape: {out.shape}")  # 应保持 [2,64,80,80]

    # 统计参数量
    total_params = sum(p.numel() for p in eac.parameters())
    print(f"EAC 参数量: {total_params:,}")