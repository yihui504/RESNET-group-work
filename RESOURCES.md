# Resources

教学素材按可信度分层。所有课程数字优先取自本项目实测（最高优先级），其次为论文原文，最后为二手讲解。

## 项目自有资产（一手，最高优先级）

| 资源 | 路径 | 用途 |
|---|---|---|
| 实验分析报告（13 模型全部数字） | docs/实验分析报告.md | 所有课程的实验数据来源 |
| 十年演进调研报告（53 篇文献） | docs/ResNet发展脉络调研报告.md | L5 演进叙事与文献数字 |
| 参考文献清单（分组+分工） | docs/references.md | 每课的 primary source 索引 |
| 实测训练历史 | results/*.json | 任何被质疑的数字可回溯 |
| 图表 | figures/*.png | PPT 直接使用 |

## 一手论文（课程引用来源）

| 论文 | 链接 | 用于课程 |
|---|---|---|
| ResNet 原论文 §1–§4.2 | arxiv.org/abs/1512.03385 | L1 退化、L2 残差块 |
| Identity Mappings §1–§2 | arxiv.org/abs/1603.05027 | L2 pre-act |
| WRN | arxiv.org/abs/1605.07146 | L3 加宽 |
| ResNeXt | arxiv.org/abs/1611.05431 | L3 基数 |
| SENet | arxiv.org/abs/1709.01507 | L4 SE |
| Veit 集成 / Balduzzi 梯度破碎 / Li 损失面 | arXiv:1605.06431 / 1702.08591 / 1712.09913 | L4 理论三件套 |
| ConvNeXt | arxiv.org/abs/2201.03545 | L5 |
| ResNet Strikes Back | arxiv.org/abs/2110.00476 | L6 配方类提问 |

## 二手讲解（可选补充，用于直觉建立）

- 3Blue1Brown 神经网络系列（youtube.com/@3blue1brown）——零基础直觉
- CS231n 卷积网络讲义（cs231n.github.io）——CNN 标准教材级讲义

## 社区（wisdom 阶段）

- 知乎/CSDN ResNet 复现帖（中文，用于对照常见坑）
- GitHub Discussions: pytorch/vision（torchvision ResNet 实现讨论）
