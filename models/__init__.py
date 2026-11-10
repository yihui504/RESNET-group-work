# -*- coding: utf-8 -*-
"""模型注册表：把命令行里的模型名映射到具体的构造函数。

用法（在 train.py 中）：
    from models import build_model
    net = build_model("resnet32")
"""

from .plain import PlainNet
from .resnet_v1 import ResNetV1
from .resnet_v2 import PreActResNet
from .wrn import WideResNet
from .resnext import ResNeXt
from .se_resnet import SEResNetV1

# 名字 -> 构造函数。lambda 里写清楚默认超参数，方便查阅。
MODEL_REGISTRY = {
    # ---- 对照组（不是 ResNet 版本，只用来复现"退化问题"）----
    "plain20": lambda: PlainNet(depth=20),
    "plain32": lambda: PlainNet(depth=32),
    "plain56": lambda: PlainNet(depth=56),   # 深 plain：复现"加深反而变差"
    "plain110": lambda: PlainNet(depth=110), # 更深的 plain：退化现象最明显
    # ---- 历史版本 1：ResNet-v1（CVPR 2016）----
    "resnet20": lambda: ResNetV1(depth=20),
    "resnet32": lambda: ResNetV1(depth=32),
    "resnet44": lambda: ResNetV1(depth=44),
    "resnet56": lambda: ResNetV1(depth=56),
    "resnet110": lambda: ResNetV1(depth=110),  # 可选：给 ResNeXt 做对比
    # ---- 历史版本 2：Pre-act ResNet-v2（ECCV 2016）----
    "preact20": lambda: PreActResNet(depth=20),
    "preact32": lambda: PreActResNet(depth=32),
    # "preact110": lambda: PreActResNet(depth=110),  # 暂缓：补实验只保留 plain110 + seresnet110
    # 需要恢复时取消注释即可（实现本身完好，冒烟测试曾通过）
    # ---- 历史版本 3：WRN-28-10（2016）----
    "wrn28_2": lambda: WideResNet(depth=28, widen_factor=2),   # 小模型：冒烟测试用
    "wrn28_10": lambda: WideResNet(depth=28, widen_factor=10),
    # ---- 历史版本 4：ResNeXt-29 16x4d（CVPR 2017）----
    "resnext29_16x4d": lambda: ResNeXt(cardinality=16, base_width=4, depth=29),
    # ---- 消融：SE-ResNet（CVPR 2018 的 SE 模块插到 v1 上）----
    "seresnet20": lambda: SEResNetV1(depth=20),
    "seresnet32": lambda: SEResNetV1(depth=32),
    "seresnet110": lambda: SEResNetV1(depth=110),  # SE 论文同款设置（ResNet-110）
}


def build_model(name: str):
    """根据名字构建模型；名字写错时列出所有可用名字，方便排查。"""
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"未知模型名 '{name}'。可用模型：{sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name]()


def list_models():
    """返回全部可用模型名。"""
    return sorted(MODEL_REGISTRY)
