# -*- coding: utf-8 -*-
"""ResNeXt —— CVPR 2017（arXiv:1611.05431）。

论文：Aggregated Residual Transformations for Deep Neural Networks。
核心观点：在"深度"和"宽度"之外引入第三个维度——基数（cardinality）：
把一条卷积分支拆成 C 条并行的小分支（等价于分组卷积），
在相同算力下通常比"更深/更宽"更有效。

本文件实现 CIFAR 版 ResNeXt-29（16x4d），与论文附录一致：
  - 入口 3x3 卷积到 64 通道；
  - 3 个 stage，每个 stage 3 个 bottleneck 块（(29-2)/9=3）；
  - 每块中间通道 D = cardinality * base_width = 16 * 4 = 64；
  - 3x3 卷积用 groups=cardinality（分组卷积），输出通道 256/512/1024。

bottleneck（v1 风格，加法后接 ReLU）：
  1x1(降维到 D) -> 3x3分组卷积(D->D) -> 1x1(升维到 out) -> 与捷径相加 -> ReLU
"""


import torch.nn as nn


class ResNeXtBottleneck(nn.Module):
    """ResNeXt bottleneck 块：1x1 -> 分组3x3 -> 1x1。"""

    def __init__(self, in_ch: int, out_ch: int, cardinality: int = 16,
                 base_width: int = 4, stride: int = 1):
        super().__init__()
        D = cardinality * base_width   # 中间总通道数（每条小分支 base_width 个通道）

        # 1x1 降维：out 通道特征 -> D 通道
        self.conv1 = nn.Conv2d(in_ch, D, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(D)
        # 3x3 分组卷积：分成 cardinality 组，每组 base_width 个通道
        self.conv2 = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1,
                               groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(D)
        # 1x1 升维：D -> out_ch
        self.conv3 = nn.Conv2d(D, out_ch, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.relu(self.bn2(self.conv2(y)))   # 这一步就是"16 条并行小分支"
        y = self.bn3(self.conv3(y))
        return self.relu(y + self.shortcut(x))


class ResNeXt(nn.Module):
    """CIFAR 版 ResNeXt。depth 满足 (depth-2) % 9 == 0（如 29）。"""

    def __init__(self, cardinality: int = 16, base_width: int = 4,
                 depth: int = 29, num_classes: int = 10):
        super().__init__()
        assert (depth - 2) % 9 == 0, "depth 需满足 (depth-2) % 9 == 0，例如 29"
        n = (depth - 2) // 9   # 每个 stage 的 bottleneck 数

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, 256, n, cardinality, base_width, stride=1)
        self.layer2 = self._make_layer(256, 512, n, cardinality, base_width, stride=2)
        self.layer3 = self._make_layer(512, 1024, n, cardinality, base_width, stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

    @staticmethod
    def _make_layer(in_ch, out_ch, blocks, cardinality, base_width, stride):
        layers = [ResNeXtBottleneck(in_ch, out_ch, cardinality, base_width, stride)]
        layers += [ResNeXtBottleneck(out_ch, out_ch, cardinality, base_width)
                   for _ in range(blocks - 1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)
