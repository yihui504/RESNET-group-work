# -*- coding: utf-8 -*-
"""ResNet-v1（原始版）—— CVPR 2016（arXiv 2015.12.10）。

论文：Deep Residual Learning for Image Recognition (arXiv:1512.03385)。
核心思想：让堆叠层拟合残差 F(x) = H(x) - x，输出为 y = F(x) + x；
恒等捷径不引入任何参数/算力，却让深层网络可以正常优化。

本文件实现 CIFAR 版（论文 4.2 节，3 个 stage，BasicBlock）：
  ResNet-20/32/44/56/110，depth 满足 (depth-2) % 6 == 0。

BasicBlock（v1，post-activation）：
  branch = BN-ReLU 结构在卷积【后】：conv3x3 -> BN -> ReLU -> conv3x3 -> BN
  out    = ReLU( branch(x) + shortcut(x) )
  通道数变化（stage 过渡）时，shortcut 用 1x1 卷积 + BN 对齐形状。
"""


import torch.nn as nn


def conv3x3(in_ch: int, out_ch: int, stride: int = 1):
    """3x3 卷积。bias=False 是惯例：BN 自带可学习偏移。"""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """v1 基本残差块：两个 3x3 卷积 + 一条捷径。"""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride)  # stage 第一块时 stride=2（下采样）
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)

        # 捷径：形状不变时用恒等映射（零参数！）；变化时用 1x1 卷积对齐
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        branch = self.relu(self.bn1(self.conv1(x)))   # 主分支：两层卷积
        branch = self.bn2(self.conv2(branch))
        out = self.relu(branch + self.shortcut(x))    # 关键一步：相加后再激活
        return out


class ResNetV1(nn.Module):
    """CIFAR 版 ResNet-v1。"""

    def __init__(self, depth: int = 32, num_classes: int = 10):
        super().__init__()
        assert (depth - 2) % 6 == 0, "depth 需满足 (depth-2) % 6 == 0，例如 20/32/110"
        n = (depth - 2) // 6  # 每个 stage 的残差块数

        self.conv1 = conv3x3(3, 16)   # 入口：单层 3x3 卷积（CIFAR 图像小，不需要 7x7+池化）
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(16, 16, n, stride=1)
        self.layer2 = self._make_layer(16, 32, n, stride=2)  # 32x32 -> 16x16
        self.layer3 = self._make_layer(32, 64, n, stride=2)  # 16x16 -> 8x8

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    @staticmethod
    def _make_layer(in_ch, out_ch, blocks, stride):
        """堆 blocks 个 BasicBlock；只有第一块负责下采样/升维。"""
        layers = [BasicBlock(in_ch, out_ch, stride)]
        layers += [BasicBlock(out_ch, out_ch) for _ in range(blocks - 1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)
