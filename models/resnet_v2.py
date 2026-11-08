# -*- coding: utf-8 -*-
"""Pre-act ResNet-v2 —— ECCV 2016（arXiv:1603.05027）。

论文：Identity Mappings in Deep Residual Networks。
动机：分析"捷径上不加任何操作、加法后不接 ReLU"（恒等映射）时，
信号可以在任意两个块之间无障碍直传（公式推导见论文图 1-4）。
做法：把 BN+ReLU 从卷积【后】移到卷积【前】（pre-activation）。

与 v1 的直观区别（初学者重点看 forward）：
  v1: out = ReLU( conv-BN-ReLU-conv-BN (x) + x )
  v2: out = conv-BN-ReLU-conv ( BN-ReLU(x) ) + shortcut(BN-ReLU(x))
      —— 激活挪到前面，加法之后【没有】 ReLU/BN，保证加法结果是"干净"的求和。

CIFAR 版还有一个细节：v2 没有 v1 那个"入口 conv+BN"，
输入图像直接进第一个 pre-act 块（论文 4.2 节 Figure 6），最后 BN-ReLU 再池化。
"""


import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_ch: int, out_ch: int, stride: int = 1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class PreActBlock(nn.Module):
    """v2 pre-activation 基本块。"""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        # 注意：BN 建在"进入本块的通道数"上（bn1 作用于输入 x）
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = conv3x3(in_ch, out_ch, stride)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = conv3x3(out_ch, out_ch)

        # 捷径挂在"激活之后"的信号上；形状变化时用 1x1 卷积（不需要 BN，
        # 因为 bn1 已经做过了）
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1,
                                      stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        # 先激活（BN+ReLU），再做卷积 —— 这就是 "pre-activation"
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)          # 捷径也取激活后的值
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + shortcut                   # 加法后【不】接激活


class PreActResNet(nn.Module):
    """CIFAR 版 Pre-act ResNet（v2）。depth 满足 (depth-2) % 6 == 0。"""

    def __init__(self, depth: int = 32, num_classes: int = 10):
        super().__init__()
        assert (depth - 2) % 6 == 0, "depth 需满足 (depth-2) % 6 == 0，例如 20/32"
        n = (depth - 2) // 6

        # v2：没有入口 conv，图像(3 通道)直接进第一个块
        self.layer1 = self._make_layer(3, 16, n, stride=1)
        self.layer2 = self._make_layer(16, 32, n, stride=2)
        self.layer3 = self._make_layer(32, 64, n, stride=2)

        self.bn_final = nn.BatchNorm2d(64)      # 收尾的 BN+ReLU
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    @staticmethod
    def _make_layer(in_ch, out_ch, blocks, stride):
        layers = [PreActBlock(in_ch, out_ch, stride)]
        layers += [PreActBlock(out_ch, out_ch) for _ in range(blocks - 1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.relu(self.bn_final(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)
