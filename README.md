# ResNet 课程设计 · 复现代码库

在 CIFAR-10 上复现 ResNet 十年演进主线的性能对比：Plain 对照 + ResNet-v1 + Pre-act v2 + WRN + ResNeXt + SE 消融。

配套文档：[复现实施规划.md](./docs/复现实施规划.md)（里程碑/分工/风险）、[ResNet发展脉络调研报告.md](./docs/ResNet发展脉络调研报告.md)（文献脉络）。

## 1. 环境安装

```bash
pip install -r requirements.txt
# GPU 版 PyTorch 若未安装，去 https://pytorch.org 按显卡选择命令，例如：
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

验证环境：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 2. 快速开始（三步）

```bash
# 第一步：小模型 5 分钟冒烟测试（确认环境没问题）
python train.py --model wrn28_2 --epochs 3

# 第二步：跑主线实验（每个模型一条命令；首次运行会自动下载 CIFAR-10）
python train.py --model plain20
python train.py --model plain32
python train.py --model resnet20
python train.py --model resnet32
python train.py --model preact32
python train.py --model resnext29_16x4d
python train.py --model wrn28_10
python train.py --model seresnet32      # SE 消融（与 resnet32 对比）

# 第三步：一键汇总出图 + 生成对比表
python plot.py
```

图表输出在 `figures/`，汇总表在 `figures/summary.md`（可直接贴进报告）。

## 3. 全部可用模型

| 命令行名 | 说明 | 参数量参考 |
|---|---|---|
| `plain20` / `plain32` / `plain56` / `plain110` | 无捷径对照网（复现退化问题，与同名 ResNet 同深度同参数量） | 0.27M / 0.46M / 0.85M / 1.73M |
| `resnet20` / `resnet32` / `resnet44` / `resnet56` / `resnet110` | ResNet-v1（CVPR 2016） | 0.27M~1.7M |
| `preact20` / `preact32` | Pre-act ResNet-v2（ECCV 2016） | 同深度 v1 略少 |
| `wrn28_2` / `wrn28_10` | Wide ResNet（`wrn28_2` 是调试用小模型） | 1.5M / 36.5M |
| `resnext29_16x4d` | ResNeXt-29 16×4d（CVPR 2017） | 1.35M |
| `seresnet20` / `seresnet32` | SE-ResNet（CVPR 2018 消融） | 比 v1 略多 |

## 4. 统一训练配置（公平对比的关键，别改）

SGD lr=0.1 / momentum=0.9 / weight_decay=5e-4 · cosine 退火 · 200 epochs · batch 128 · seed 42 · 标准增强（RandomCrop+Flip）。每 epoch 记录 train/test 指标，实时写入 `results/`（中断不丢数据）。

## 5. 常见问题

| 问题 | 处置 |
|---|---|
| 显存不足（OOM） | `--batch-size 64 --lr 0.05`（学习率随批大小减半） |
| 跑得太慢想先看趋势 | 加 `--epochs 100`；WRN 先用 `wrn28_2` 验证 |
| FLOPs 显示未统计 | `pip install thop` 后重跑该模型 |
| 想保存模型权重 | 加 `--save-ckpt`（存到 `results/*_best.pth`） |
| 退化现象不明显 | 确认没改训练配置；换 `--seed 123` 重跑；检查 plain 模型确实带 BN |
| 曲线图中文乱码 | 本脚本已配置微软雅黑；如仍乱码，把 plot.py 图表标题改成英文即可 |

## 6. 实验代码结构

```
models/plain.py      对照组（无捷径）
models/resnet_v1.py  历史版本1：原始 ResNet（BasicBlock）
models/resnet_v2.py  历史版本2：Pre-act ResNet（块内顺序改造）
models/wrn.py        历史版本3：加宽残差网络（宽度轴 + dropout）
models/resnext.py    历史版本4：分组卷积/基数轴
models/se_resnet.py  消融：通道注意力即插即用
train.py             统一训练循环（所有模型共用，保证公平）
plot.py              汇总出图 + summary.md 对比表
```
