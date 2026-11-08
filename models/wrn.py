# -*- coding: utf-8 -*-
"""Wide ResNet（WRN）—— arXiv:1605.07146。

论文：Wide Residual Networks。
核心观点：与其把网络做得"极深"（ResNet-1001），不如做得"宽"——
薄而极深的网络存在特征复用衰减问题，加宽训练更快、并行度更高。
WRN-28-10：28 层（4 个块/stage）、加宽因子 10（通道数 16/32/64 -> 160/320/640），
块内加 dropout=0.3，CIFAR-10 错误率 4.00%（约 96% top-1），参数 36.5M。

块结构：pre-activation 风格的 b(3,3) 块 + dropout：
  y = ReLU(BN1(x));  y = conv3x3(y);  y = dropout(y);
  y = ReLU(BN2(y));  y = conv3x3(y);  out = y + shortcut(x)
"""


import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_ch: int, out_ch: int, stride: int = 1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class WRNBasicBlock(nn.Module):
    """WRN 基本块：pre-act + dropout。"""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.3):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = conv3x3(in_ch, out_ch, stride)
        self.dropout = nn.Dropout(p=dropout)   # WRN 的正则化：块内 dropout
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = conv3x3(out_ch, out_ch)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1,
                                      stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        y = F.relu(self.bn1(x))          # pre-activation
        shortcut = self.shortcut(y) if not isinstance(self.shortcut, nn.Identity) else x
        y = self.conv1(y)
        y = self.dropout(y)
        y = self.conv2(F.relu(self.bn2(y)))
        return y + shortcut


class WideResNet(nn.Module):
    """WRN-{depth}-{widen}，depth 满足 (depth-4) % 6 == 0（如 16/22/28）。"""

    def __init__(self, depth: int = 28, widen_factor: int = 10,
                 dropout: float = 0.3, num_classes: int = 10):
        super().__init__()
        assert (depth - 4) % 6 == 0, "depth 需满足 (depth-4) % 6 == 0，例如 28"
        n = (depth - 4) // 6  # 每个 stage 的块数（WRN-28 -> 4）
        k = widen_factor

        widths = [16 * k, 32 * k, 64 * k]      # 160/320/640（k=10 时）

        self.conv1 = conv3x3(3, 16)            # 入口仍是 16 通道，之后再加宽
        self.layer1 = self._make_layer(16, widths[0], n, stride=1, dropout=dropout)
        self.layer2 = self._make_layer(widths[0], widths[1], n, stride=2, dropout=dropout)
        self.layer3 = self._make_layer(widths[1], widths[2], n, stride=2, dropout=dropout)

        self.bn_final = nn.BatchNorm2d(widths[2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(widths[2], num_classes)

        # WRN 论文的初始化方式：对较深网络更稳
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    @staticmethod
    def _make_layer(in_ch, out_ch, blocks, stride, dropout):
        layers = [WRNBasicBlock(in_ch, out_ch, stride, dropout)]
        layers += [WRNBasicBlock(out_ch, out_ch, dropout=dropout)
                   for _ in range(blocks - 1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.relu(self.bn_final(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)
