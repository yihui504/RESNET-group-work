# -*- coding: utf-8 -*-
"""冒烟测试：不训练，只验证每个模型能构建、能前向传播、输出形状正确。

用法：
    python tests/smoke_test.py              # 自动选 GPU（没有则 CPU）
    python tests/smoke_test.py --device cpu

全部 OK 说明代码库结构正确，可以放心开训。
"""


import argparse
import os
import sys
import time

# 把项目根目录加进 sys.path，使 tests/ 下也能 import models
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models import MODEL_REGISTRY


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None,
                        help="cuda 或 cpu，默认自动选择")
    parser.add_argument("--batch", type=int, default=4, help="测试批大小")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[冒烟测试] 设备: {device}，共 {len(MODEL_REGISTRY)} 个模型\n")

    x = torch.randn(args.batch, 3, 32, 32, device=device)  # 模拟一个 CIFAR-10 batch
    print(f"{'模型':<20}{'参数量(M)':>10}  {'输出形状':<14}耗时    状态")
    print("-" * 62)

    n_ok, n_fail = 0, 0
    for name, ctor in sorted(MODEL_REGISTRY.items()):
        try:
            t0 = time.time()
            net = ctor().to(device)
            net.eval()                      # 关闭 dropout/BN 的训练行为
            with torch.no_grad():
                y = net(x)
            assert y.shape == (args.batch, 10), f"输出形状 {tuple(y.shape)} 不对"
            params = sum(p.numel() for p in net.parameters()) / 1e6
            print(f"{name:<20}{params:>9.3f}  {str(tuple(y.shape)):<14}"
                  f"{time.time() - t0:5.1f}s  OK")
            n_ok += 1
        except Exception as e:
            print(f"{name:<20}{'-':>10}  {'-':<14}{'-':>7}  失败: {e}")
            n_fail += 1

    print("-" * 62)
    print(f"结果: {n_ok} 个通过, {n_fail} 个失败")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
