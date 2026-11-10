# -*- coding: utf-8 -*-
"""Plain 网络（无捷径连接）—— ResNet 论文中的"对照组"。

论文：Deep Residual Learning for Image Recognition (arXiv:1512.03385) 图 4 左侧 / 表 6。
作用：与 ResNet 同深度、同结构，唯一区别是【没有捷径连接】。
预期现象（我们要复现的"退化问题"）：
  - plain-32 的训练误差比 plain-20 更高（更深反而更差，且不是过拟合）；
  - 同深度的 ResNet 则训练误差随深度继续下降。

CIFAR 版结构（论文 4.2 节）：
  1 个 3x3 卷积(3->16) + 3 个 stage（16/32/64 通道），
  每个 stage 有 n=(depth-2)/6 个 [3x3卷积 + BN + ReLU]，
  stage 之间用 stride=2 的卷积下采样，最后全局平均池化 + 全连接。
"""


import torch.nn as nn


def conv_bn_relu(in_ch: int, out_ch: int, stride: int = 1):
    """一个 '3x3卷积 + BN + ReLU' 基本单元（Plain 网的积木）。"""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,
                  padding=1, bias=False),   # bias=False：BN 里有可学习的偏移，卷积不再需要 bias
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class PlainNet(nn.Module):
    """CIFAR 版 Plain 网络。depth 必须满足 (depth-2) 能被 6 整除，如 20/32/44。"""

    def __init__(self, depth: int = 20, num_classes: int = 10):
        super().__init__()
        assert (depth - 2) % 6 == 0, "depth 需满足 (depth-2) % 6 == 0，例如 20/32/44"
        n = (depth - 2) // 6  # 每个 stage 的残差块数（= 卷积层数的一半）

        layers = [conv_bn_relu(3, 16, stride=1)]                # 入口卷积
        # 每个 stage 的卷积层数 = 2n（与 ResNet 的 n 个残差块 × 2 层卷积对齐，
        # 保证与同名 ResNet 深度、通道完全一致，只差捷径连接）
        layers += [conv_bn_relu(16, 16) for _ in range(2 * n)]  # stage1：16 通道
        layers += [conv_bn_relu(16, 32, stride=2)]              # stage2 第一层：下采样
        layers += [conv_bn_relu(32, 32) for _ in range(2 * n - 1)]
        layers += [conv_bn_relu(32, 64, stride=2)]              # stage3 第一层：下采样
        layers += [conv_bn_relu(64, 64) for _ in range(2 * n - 1)]

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化 -> [B, 64, 1, 1]
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = self.pool(out).flatten(1)
        return self.fc(out)
