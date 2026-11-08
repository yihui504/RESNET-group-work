# -*- coding: utf-8 -*-
"""汇总绘图脚本：读取 results/ 下所有训练 JSON，一键生成报告全部图表。

用法：
    python plot.py                 # 读 ./results，图存到 ./figures
    python plot.py --results-dir results --out-dir figures

输出：
    figures/fig_accuracy.png    所有模型测试准确率曲线
    figures/fig_loss.png        所有模型训练损失曲线
    figures/fig_degradation.png 退化问题对照图（plain vs resnet 训练误差）
    figures/fig_best_acc.png    最佳准确率柱状图（标注参数量）
    figures/summary.md          统一对比表（可直接贴进报告）
"""


import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")   # 无显示器环境也能保存图片
import matplotlib.pyplot as plt

# 中文字体（Windows 自带微软雅黑/黑体；其他系统自动回退）
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------- 读取数据

def load_all(results_dir):
    """读取目录下全部 *_e*.json，返回 {模型名: 数据} 。"""
    runs = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*_e*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        name = data.get("model", os.path.basename(path))
        runs[name] = data   # 同名模型后跑的覆盖先跑的
    return runs


def get_series(data, key):
    """从 history 里抽出某一列。"""
    return [h[key] for h in data["history"]]


# ---------------------------------------------------------------- 四张图

def plot_accuracy(runs, out_dir):
    plt.figure(figsize=(9, 6))
    for name, data in runs.items():
        plt.plot(get_series(data, "epoch"),
                 get_series(data, "test_acc"),
                 label=f"{name} ({data['best_acc']:.2f}%)")
    plt.xlabel("Epoch")
    plt.ylabel("测试集 Top-1 准确率 (%)")
    plt.title("各模型测试准确率曲线（CIFAR-10）")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig_accuracy.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print("已生成:", path)


def plot_loss(runs, out_dir):
    plt.figure(figsize=(9, 6))
    for name, data in runs.items():
        plt.plot(get_series(data, "epoch"),
                 get_series(data, "train_loss"),
                 label=name)
    plt.xlabel("Epoch")
    plt.ylabel("训练损失 (CrossEntropy)")
    plt.title("各模型训练损失曲线（CIFAR-10）")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig_loss.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print("已生成:", path)


def plot_degradation(runs, out_dir):
    """论文图 4 的复现：plain 加深后训练误差反而更高；ResNet 则正常下降。"""
    keys = [k for k in ["plain20", "plain32", "resnet20", "resnet32"]
            if k in runs]
    if not keys:
        print("跳过退化图：还没有 plain/resnet 的结果")
        return
    plt.figure(figsize=(9, 6))
    for name in keys:
        data = runs[name]
        train_err = [100 - a for a in get_series(data, "train_acc")]
        plt.plot(get_series(data, "epoch"), train_err, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("训练误差 (%) = 100 - 训练准确率")
    plt.title("退化问题复现：加深 Plain 网误差反而升高，捷径连接解决之")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig_degradation.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print("已生成:", path)


def plot_best_acc(runs, out_dir):
    if not runs:
        return
    items = sorted(runs.items(), key=lambda kv: kv[1]["best_acc"])
    names = [k for k, _ in items]
    accs = [v["best_acc"] for _, v in items]
    params = [v.get("params_M") or 0 for _, v in items]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(names, accs, color="#4C72B0")
    for bar, acc, p in zip(bars, accs, params):
        plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                 f"{acc:.2f}%  ({p:.2f}M参数)", va="center", fontsize=9)
    plt.xlabel("最佳测试 Top-1 准确率 (%)")
    plt.title("各模型最佳准确率对比（括号内为参数量）")
    plt.xlim(0, 108)
    plt.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    path = os.path.join(out_dir, "fig_best_acc.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print("已生成:", path)


# ---------------------------------------------------------------- 汇总表

def write_summary(runs, out_dir):
    rows = sorted(runs.items(), key=lambda kv: kv[1]["best_acc"],
                  reverse=True)
    lines = [
        "# 实验结果汇总表（自动生成，可直接贴入报告）",
        "",
        "| 模型 | 最佳 Top-1 (%) | 参数量 (M) | FLOPs (G) | Epochs | 总训练时长 (h) |",
        "|---|---|---|---|---|---|",
    ]
    for name, d in rows:
        flops = d.get("flops_G")
        lines.append(
            f"| {name} | {d['best_acc']:.2f} | {d.get('params_M', '-')} | "
            f"{flops if flops else '-'} | "
            f"{d.get('args', {}).get('epochs', '-')} | "
            f"{d.get('total_time_hour', '-')} |")
    lines.append("")
    lines.append("> 趋势核对清单：① plain32 训练误差应高于 plain20；"
                 "② resnet 优于同深度 plain；③ preact 收敛更快/精度持平更好；"
                 "④ wrn28_10 明显最高；⑤ resnext 优于深窄 resnet；⑥ seresnet 略优于 resnet。")
    path = os.path.join(out_dir, "summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("已生成:", path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument("--out-dir", type=str, default="./figures")
    args = parser.parse_args()

    runs = load_all(args.results_dir)
    if not runs:
        print(f"目录 {args.results_dir} 下没有实验结果，请先运行 train.py")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"共读取 {len(runs)} 个实验：{list(runs)}\n")
    plot_accuracy(runs, args.out_dir)
    plot_loss(runs, args.out_dir)
    plot_degradation(runs, args.out_dir)
    plot_best_acc(runs, args.out_dir)
    write_summary(runs, args.out_dir)


if __name__ == "__main__":
    main()
