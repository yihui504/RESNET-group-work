# -*- coding: utf-8 -*-
"""SE-ResNet（消融实验用）—— CVPR 2018（arXiv:1709.01507）。

论文：Squeeze-and-Excitation Networks（ILSVRC 2017 冠军）。
SE 模块：通道注意力的"即插即用"配件。
  1) Squeeze：全局平均池化，把每个通道压成一个数 -> [B, C]
  2) Excitation：两层小 FC（降维 1/r -> ReLU -> 升维回 C -> Sigmoid），
     得到每个通道的"重要性权重"（0~1）
  3) Scale：用权重逐通道乘回特征图

插入位置：残差分支最后一层 BN 之后、与捷径相加之前。
消融目的：验证"不动主干结构、只加注意力"能带来提升（论文中普遍 +1pt 左右）。
"""


import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力模块。"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        # 通道很少时（如 16）防止降维到 0
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)          # [B,C,H,W] -> [B,C,1,1]
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),     # 降维（瓶颈）
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),     # 升维回 C
            nn.Sigmoid(),                                 # 输出 0~1 的通道权重
        )

    def forward(self, x):
        b, c = x.shape[0], x.shape[1]
        w = self.avg_pool(x).flatten(1)   # squeeze: [B, C]
        w = self.fc(w).view(b, c, 1, 1)   # excitation: [B, C, 1, 1]
        return x * w                       # scale: 逐通道加权


class SEBasicBlock(nn.Module):
    """v1 BasicBlock + SE：分支末端先过 SE，再与捷径相加。"""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, reduction: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch, reduction)             # <-- 与 v1 的唯一区别

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y = self.se(y)            # 先做通道重标定
        return self.relu(y + self.shortcut(x))


class SEResNetV1(nn.Module):
    """SE-ResNet-v1（CIFAR 版）。结构与 resnet_v1.ResNetV1 完全一致，只是块里加了 SE。"""

    def __init__(self, depth: int = 32, num_classes: int = 10):
        super().__init__()
        assert (depth - 2) % 6 == 0, "depth 需满足 (depth-2) % 6 == 0，例如 20/32"
        n = (depth - 2) // 6

        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(16, 16, n, stride=1)
        self.layer2 = self._make_layer(16, 32, n, stride=2)
        self.layer3 = self._make_layer(32, 64, n, stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    @staticmethod
    def _make_layer(in_ch, out_ch, blocks, stride):
        layers = [SEBasicBlock(in_ch, out_ch, stride)]
        layers += [SEBasicBlock(out_ch, out_ch) for _ in range(blocks - 1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)
