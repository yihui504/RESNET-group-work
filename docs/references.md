# 参考文献清单（按用途分组）

> 供三人分工查阅与报告引用。标注 ★ 的为核心必读（直接对应我们实现的模型/报告核心章节）；
> 「用于」说明该文献在本课程设计中的落点。完整脉络见 [ResNet发展脉络调研报告.md](./ResNet发展脉络调研报告.md)。

## 一、核心必读（8 篇，直接对应实现与实验）

| # | 文献 | 链接 | 用于 |
|---|---|---|---|
| ★1 | He et al. *Deep Residual Learning for Image Recognition*, CVPR 2016 | [arXiv:1512.03385](https://arxiv.org/abs/1512.03385) | 残差思想来源；表 6 的 plain/resnet 20–110 CIFAR 数字是退化对照的文献基准（`models/plain.py`、`models/resnet_v1.py`、报告 §4.1） |
| ★2 | He et al. *Identity Mappings in Deep Residual Networks*, ECCV 2016 | [arXiv:1603.05027](https://arxiv.org/abs/1603.05027) | Pre-act v2 块内顺序（`models/resnet_v2.py`；preact32 负结果的解释依据，报告 §4.5） |
| ★3 | Zagoruyko & Komodakis. *Wide Residual Networks*, 2016 | [arXiv:1605.07146](https://arxiv.org/abs/1605.07146) | WRN-28-10 结构与 dropout=0.3 设置；96.0% 是我们唯一精确对齐的基准（报告 §4.2、表 3） |
| ★4 | Xie et al. *Aggregated Residual Transformations for Deep Neural Networks*, CVPR 2017 | [arXiv:1611.05431](https://arxiv.org/abs/1611.05431) | ResNeXt-29 16×4d 基数设计与分组卷积（`models/resnext.py`；报告 §4.3） |
| ★5 | Hu et al. *Squeeze-and-Excitation Networks*, CVPR 2018 | [arXiv:1709.01507](https://arxiv.org/abs/1709.01507) | SE 模块；其 ResNet-110 增益 +1.2pt 是我们 SE 消融设计的依据（`models/se_resnet.py`；报告 §4.4） |
| ★6 | Wightman et al. *ResNet Strikes Back*, 2021 | [arXiv:2110.00476](https://arxiv.org/abs/2110.00476) | 训练配方重要性的论据；支撑我们统一 200ep+cosine 的设置合理性（报告 §2、§7） |
| ★7 | Liu et al. *A ConvNet for the 2020s*, CVPR 2022 | [arXiv:2201.03545](https://arxiv.org/abs/2201.03545) | "残差骨架被一切现代架构继承"叙事的主证据（PPT 结论页、调研报告 §3.3） |
| ★8 | Gomez et al. *Reversible Residual Networks*, NeurIPS 2017（可逆残差，延伸） | [arXiv:1707.04585](https://arxiv.org/abs/1707.04585) | 残差思想的分支延伸（PPT 演进时间线） |

## 二、理论解释三件套（报告 §6"为什么捷径有效"）

| # | 文献 | 链接 | 视角 |
|---|---|---|---|
| ★9 | Veit et al. *Residual Networks Behave Like Ensembles of Relatively Shallow Networks*, NeurIPS 2016 | [arXiv:1605.06431](https://arxiv.org/abs/1605.06431) | 集成视角：梯度主要来自短路径 |
| ★10 | Balduzzi et al. *The Shattered Gradients Problem*, ICML 2017 | [arXiv:1702.08591](https://arxiv.org/abs/1702.08591) | 优化视角：梯度相关性指数衰减 → 亚线性 |
| ★11 | Li et al. *Visualizing the Loss Landscape of Neural Nets*, NeurIPS 2018 | [arXiv:1712.09913](https://arxiv.org/abs/1712.09913) | 几何视角：跳连平滑损失面 |

## 三、奠基期拓展（2015–2018，PPT 时间线用）

| # | 文献 | 链接 | 一句话 |
|---|---|---|---|
| 12 | Huang et al. *Deep Networks with Stochastic Depth*, ECCV 2016 | [arXiv:1603.09382](https://arxiv.org/abs/1603.09382) | 训练时随机丢块，1200+ 层可训 |
| 13 | Huang et al. *Densely Connected Convolutional Networks*, CVPR 2017（最佳论文） | [arXiv:1608.06993](https://arxiv.org/abs/1608.06993) | 残差"相加"推广为稠密"拼接" |
| 14 | torchvision ResNet 官方文档 | [docs.pytorch.org](https://docs.pytorch.org/vision/0.15/models/resnet.html) | v1.5 工程变体说明（下采样位置） |
| 15 | mmclassification ResNet 复现基准 | [mmclassification](https://mmclassification.readthedocs.io/en/stable/papers/resnet.html) | CIFAR-10 各深度官方复现精度 |

## 四、成熟期（2019–2022）

| # | 文献 | 链接 | 一句话 |
|---|---|---|---|
| 16 | Gao et al. *Res2Net*, TPAMI 2021 | [arXiv:1904.01169](https://arxiv.org/abs/1904.01169) | 块内多尺度分组残差连接 |
| 17 | Zhang et al. *ResNeSt*, 2020 | [arXiv:2004.08955](https://arxiv.org/abs/2004.08955) | Split-Attention，ResNeXt 注意力化 |
| 18 | Ridnik et al. *TResNet*, 2020 | [arXiv:2003.13630](https://arxiv.org/abs/2003.13630) | GPU 真实吞吐导向重构 |
| 19 | Bello et al. *Revisiting ResNets (ResNet-RS)*, 2021 | [arXiv:2103.07579](https://arxiv.org/abs/2103.07579) | 解耦结构/训练/缩放，RS-420 达 84.4% |
| 20 | Radosavovic et al. *RegNet*, CVPR 2020 | [arXiv:2003.13678](https://arxiv.org/abs/2003.13678) | 设计空间参数化 |
| 21 | Yu et al. *MetaFormer/PoolFormer*, CVPR 2022 | [arXiv:2111.11418](https://arxiv.org/abs/2111.11418) | 残差宏观架构是 Transformer 成功的骨架 |
| 22 | Ding et al. *RepVGG*, CVPR 2021 | [arXiv:2101.03697](https://arxiv.org/abs/2101.03697) | 推理期去残差的重参数化 |
| 23 | Vasu et al. *MobileOne*, CVPR 2023 | [arXiv:2206.04040](https://arxiv.org/abs/2206.04040) | 移动端重参数化 |
| 24 | Brock et al. *NFNet*, 2021 | [arXiv:2102.06171](https://arxiv.org/abs/2102.06171) | AGC 支撑无归一化层训练 |

## 五、最新进展（2023–2025，"残差成为通用骨架"证据）

| # | 文献 | 链接 | 一句话 |
|---|---|---|---|
| 25 | Woo et al. *ConvNeXt V2*, CVPR 2023 | [arXiv:2301.00808](https://arxiv.org/abs/2301.00808) | FCMAE+GRN，卷积自监督 86.3% |
| 26 | Chen et al. *VanillaNet*, NeurIPS 2023 | [arXiv:2305.12972](https://arxiv.org/abs/2305.12972) | 训练用残差、推理去残差 |
| 27 | Chen et al. *FasterNet*, CVPR 2023 | [arXiv:2303.03667](https://arxiv.org/abs/2303.03667) | PConv 部分卷积+少激活 |
| 28 | Wang et al. *RepViT*, CVPR 2024 | [arXiv:2307.09283](https://arxiv.org/abs/2307.09283) | 端侧 80%@1ms 首破 |
| 29 | Cai et al. *EfficientViT*, CVPR 2023 | [arXiv:2305.07027](https://arxiv.org/abs/2305.07027) | 级联组注意力 |
| 30 | Wang et al. *InternImage*, CVPR 2023 | [arXiv:2211.05778](https://arxiv.org/abs/2211.05778) | DCNv3，G 版 90.1% 开源最高 |
| 31 | Xiong et al. *DCNv4/FlashInternImage*, 2024 | [arXiv:2401.06197](https://arxiv.org/abs/2401.06197) | 可变形卷积提速 50–80% |
| 32 | Liu et al. *SLaK*, ICML 2022 | [arXiv:2207.03620](https://arxiv.org/abs/2207.03620) | 51×51 大核复兴标志 |
| 33 | Dai et al. *TransNeXt*, CVPR 2024 | [arXiv:2311.17132](https://arxiv.org/abs/2311.17132) | 指出残差深度退化，块内仍用残差 |
| 34 | Fan et al. *RMT*, CVPR 2024 | [arXiv:2309.11523](https://arxiv.org/abs/2309.11523) | 2D 距离衰减先验 |
| 35 | Liu et al. *VMamba*, NeurIPS 2024 | [arXiv:2401.10166](https://arxiv.org/abs/2401.10166) | 论文明确 VSS 块"本质是带跳连的残差网络" |
| 36 | Zhu et al. *Vision Mamba (Vim)*, ICML 2024 | [arXiv:2401.09417](https://arxiv.org/abs/2401.09417) | 线性复杂度 SSM 骨干 |
| 37 | Hatamizadeh et al. *MambaVision*, CVPR 2025 | [CVPR 2025 开放获取](https://openaccess.thecvf.com/content/CVPR2025/papers/Hatamizadeh_MambaVision_A_Hybrid_Mamba-Transformer_Vision_Backbone_CVPR_2025_paper.pdf) | 混合 Mamba-Transformer，B 版 84.2% |
| 38 | Peebles & Xie. *DiT*, ICCV 2023 | [arXiv:2212.09748](https://arxiv.org/abs/2212.09748) | adaLN-Zero 零初始化残差，扩散模型 SOTA |
| 39 | Vasu et al. *MobileCLIP*, CVPR 2024 | [CVPR 2024 开放获取](https://openaccess.thecvf.com/content/CVPR2024/papers/Vasu_MobileCLIP_Fast_Image-Text_Models_through_Multi-Modal_Reinforced_Training_CVPR_2024_paper.pdf) | 卷积混合编码器落地端侧 CLIP |
| 40 | *Orthogonal Residual Update*, 2025 | [arXiv:2505.11881](https://arxiv.org/abs/2505.11881) | 残差只沿正交方向更新 |
| 41 | *ResNets Are Deeper Than You Think*, 2025 | [arXiv:2506.14386](https://arxiv.org/abs/2506.14386) | 残差网居于不同函数空间 |
| 42 | *DeepCrossAttention*, 2025 | [arXiv:2502.06785](https://arxiv.org/abs/2502.06785) | 可学习的输入依赖残差权重 |
| 43 | *OverLoCK*, 2025 | [arXiv:2502.20087](https://arxiv.org/abs/2502.20087) | 动态大核骨干，T 版 84.2% |

## 六、官方代码仓库（复现参考）

| # | 仓库 | 链接 | 用于 |
|---|---|---|---|
| 44 | WRN 官方（szagoruyko） | [github.com/szagoruyko/wide-residual-networks](https://github.com/szagoruyko/wide-residual-networks) | 对照我们的 WRN 实现 |
| 45 | ResNeXt 官方（facebookresearch） | [github.com/facebookresearch/ResNeXt](https://github.com/facebookresearch/ResNeXt) | CIFAR 29 层设置来源 |
| 46 | SENet 官方（hujie-frank） | [github.com/hujie-frank/SENet](https://github.com/hujie-frank/SENet) | SE 模块细节 |
| 47 | ResNet Strikes Back（timm） | [github.com/rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models) | A1 配方细节 |
| 48 | ConvNeXt 官方 | [github.com/facebookresearch/ConvNeXt](https://github.com/facebookresearch/ConvNeXt) | 现代化路线对照 |
| 49 | InternImage 官方 | [github.com/OpenGVLab/InternImage](https://github.com/OpenGVLab/InternImage) | 全系精度表 |
| 50 | VMamba 官方 | [github.com/MzeroMiko/VMamba](https://github.com/MzeroMiko/VMamba) | VSS 块结构 |
| 51 | 本项目仓库 | [github.com/yihui504/RESNET-group-work](https://github.com/yihui504/RESNET-group-work) | 全部实现与实验记录 |

> 说明：★ 共 11 篇（1–11）建议三人分工精读：成员 A 负责 #1/6，成员 B 负责 #2/4/5，成员 C 负责 #3/7 与理论三件套 #9–11。其余文献按 PPT 板块需要选读。
