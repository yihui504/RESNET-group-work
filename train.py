# -*- coding: utf-8 -*-
"""统一训练脚本：一个命令训练本项目的任何模型。

示例：
    python train.py --model resnet32              # 训练 ResNet-32（200 epochs）
    python train.py --model wrn28_2 --epochs 5    # 小模型快速冒烟测试
    python train.py --model plain20 --epochs 100  # 少轮数也能看出退化趋势

每次训练会把【每个 epoch 的完整历史】写入 results/{模型名}_e{轮数}.json，
之后用 plot.py 一键汇总出图。
"""


import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from models import build_model, list_models


# ---------------------------------------------------------------- 工具函数

def set_seed(seed: int):
    """固定随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dataloaders(data_root: str, batch_size: int, num_workers: int):
    """CIFAR-10 数据加载 + 标准数据增强（所有模型统一，保证公平）。"""
    mean = (0.4914, 0.4822, 0.4465)   # CIFAR-10 训练集的通道均值
    std = (0.2470, 0.2435, 0.2616)    # 通道标准差

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),       # 随机裁剪（平移）
        transforms.RandomHorizontalFlip(),          # 随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=test_tf)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    """训练一个 epoch，返回 (平均loss, top-1准确率)。"""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """在测试集上评估，返回 (平均loss, top-1准确率)。"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100.0 * correct / total


def count_flops(model):
    """用 thop 估算 FLOPs；没安装则返回 None（不影响其他指标）。"""
    try:
        from thop import profile
        dummy = torch.randn(1, 3, 32, 32)
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
        return flops / 1e9   # 单位：GFLOPs
    except Exception:
        return None


# ---------------------------------------------------------------- 主流程

def main():
    parser = argparse.ArgumentParser(description="ResNet 课程设计统一训练脚本")
    parser.add_argument("--model", type=str, required=True,
                        help=f"模型名，可选：{list_models()}")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1,
                        help="初始学习率（batch 减半时建议也减半）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--out-dir", type=str, default="./results")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-ckpt", action="store_true",
                        help="保存最佳模型权重（可选，默认只存训练历史）")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] {device}")
    if device.type == "cpu":
        print("[警告] 未检测到 GPU，训练会非常慢（可用小模型+少轮数先跑通流程）")

    # 1. 数据
    train_loader, test_loader = get_dataloaders(
        args.data_root, args.batch_size, args.num_workers)

    # 2. 模型
    model = build_model(args.model).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6   # 单位：M
    flops = count_flops(model)
    print(f"[模型] {args.model} | 参数量 {n_params:.2f}M"
          + (f" | FLOPs {flops:.2f}G" if flops else " | FLOPs 未统计(可 pip install thop)"))

    # 3. 统一训练配置（所有模型相同，保证公平对比）
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)   # 学习率 0.1 -> 0 余弦退火

    # 4. 训练循环
    history = []
    best_acc = 0.0
    t_start = time.time()
    result_path = os.path.join(
        args.out_dir, f"{args.model}_e{args.epochs}.json")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        epoch_time = time.time() - t0

        if test_acc > best_acc:
            best_acc = test_acc
            if args.save_ckpt:
                torch.save(model.state_dict(),
                           os.path.join(args.out_dir, f"{args.model}_best.pth"))

        history.append({
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "test_loss": round(test_loss, 4),
            "test_acc": round(test_acc, 2),
            "epoch_time_sec": round(epoch_time, 1),
        })

        # 每个 epoch 都保存一次 JSON（即使中途断电也不丢历史）
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump({
                "model": args.model,
                "args": vars(args),
                "params_M": round(n_params, 3),
                "flops_G": round(flops, 3) if flops else None,
                "best_acc": best_acc,
                "total_time_hour": round((time.time() - t_start) / 3600, 3),
                "history": history,
            }, f, ensure_ascii=False, indent=2)

        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"[epoch {epoch:>3}/{args.epochs}] "
                  f"train_loss {train_loss:.3f} | train_acc {train_acc:5.2f}% | "
                  f"test_acc {test_acc:5.2f}% | best {best_acc:5.2f}% | "
                  f"{epoch_time:.0f}s")

    print(f"\n[完成] {args.model} 最佳测试准确率: {best_acc:.2f}%")
    print(f"[完成] 训练历史已保存: {result_path}")


if __name__ == "__main__":
    main()
