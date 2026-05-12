"""
多模型对比分析脚本

用法:
    python scripts/compare_models.py
    python scripts/compare_models.py --log_dir logs --output_dir outputs/comparison

读取所有模型的 train_log.csv 和评估报告，自动生成：
1. 对比汇总表 (Markdown)
2. 对比柱状图 (Accuracy / F1 / 参数量 / 训练时间)
3. 合并 Loss / Accuracy 曲线图 (4条线在同一图上)
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="多模型对比分析")
    parser.add_argument("--log_dir", type=str, default="logs", help="训练日志目录")
    parser.add_argument("--report_dir", type=str, default="outputs/reports", help="评估报告目录")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="权重目录")
    parser.add_argument("--output_dir", type=str, default="outputs/comparison", help="对比结果输出目录")
    return parser.parse_args()


# ---------- 数据读取 ----------

def read_train_log(log_path: Path) -> list:
    """读取 train_log.csv"""
    rows = []
    if not log_path.exists():
        return rows
    with open(log_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "epoch": int(row.get("epoch", 0)),
                "train_loss": float(row.get("train_loss", 0)),
                "train_accuracy": float(row.get("train_accuracy", 0)),
                "val_loss": float(row.get("val_loss", 0)),
                "val_accuracy": float(row.get("val_accuracy", 0)),
                "lr": float(row.get("lr", 0)),
                "time": float(row.get("time", 0)),
                "model": row.get("model", "unknown"),
                "total_params": int(row.get("total_params", 0)) if row.get("total_params") else 0,
                "trainable_params": int(row.get("trainable_params", 0)) if row.get("trainable_params") else 0,
            })
    return rows


def read_eval_report(report_path: Path) -> dict:
    """从评估报告 Markdown 中提取关键指标"""
    metrics = {
        "accuracy": None,
        "macro_precision": None,
        "macro_recall": None,
        "macro_f1": None,
        "per_class": {},
    }
    if not report_path.exists():
        return metrics

    text = report_path.read_text(encoding="utf-8")
    # 总体指标表
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("| Overall Accuracy"):
            metrics["accuracy"] = float(line.split("|")[2].strip())
        elif line.startswith("| Macro Precision"):
            metrics["macro_precision"] = float(line.split("|")[2].strip())
        elif line.startswith("| Macro Recall"):
            metrics["macro_recall"] = float(line.split("|")[2].strip())
        elif line.startswith("| Macro F1"):
            metrics["macro_f1"] = float(line.split("|")[2].strip())

    return metrics


def read_config_snapshot(snapshot_path: Path) -> dict:
    """读取训练配置快照"""
    try:
        import yaml
        if snapshot_path.exists():
            with open(snapshot_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except ImportError:
        pass
    return {}


def discover_models(log_dir: Path, report_dir: Path, ckpt_dir: Path) -> list:
    """自动发现已训练的模型"""
    model_names = set()

    # 优先从单模型日志 train_log_{name}.csv 发现
    for p in log_dir.glob("train_log_*.csv"):
        name = p.stem.replace("train_log_", "")
        model_names.add(name)

    # 也从合并日志 CSV 中的 model 列获取
    csv_path = log_dir / "train_log.csv"
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                m = row.get("model", "").strip()
                if m:
                    model_names.add(m)

    # 从评估报告文件名发现
    if report_dir.exists():
        for p in report_dir.glob("evaluation_report_*.md"):
            name = p.stem.replace("evaluation_report_", "")
            model_names.add(name)

    # 从 checkpoint 文件名发现
    if ckpt_dir.exists():
        for p in ckpt_dir.glob("best_*.pth"):
            name = p.stem.replace("best_", "")
            model_names.add(name)

    return sorted(model_names)


# ---------- 但实际上 autodl 脚本是按模型分别保存日志的 ----------
# 每个 log 文件以模型名区分（config_snapshot_xxx.yaml, train_log.csv 合在一起）
# 需要更灵活的处理

def gather_model_data(log_dir: Path, report_dir: Path, ckpt_dir: Path) -> dict:
    """
    收集所有模型的数据
    Returns: {model_name: {log: [...], report: {...}, config: {...}, checkpoint: str}}
    """
    model_names = discover_models(log_dir, report_dir, ckpt_dir)
    data = {}

    for model_name in model_names:
        entry = {
            "log": [],
            "report": {},
            "config": {},
            "checkpoint": None,
        }

        # 优先读单模型日志 train_log_{name}.csv
        per_model_csv = log_dir / f"train_log_{model_name}.csv"
        if per_model_csv.exists():
            entry["log"] = read_train_log(per_model_csv)
        else:
            # 回退：从合并日志中过滤
            csv_path = log_dir / "train_log.csv"
            all_logs = read_train_log(csv_path)
            entry["log"] = [r for r in all_logs if r.get("model") == model_name]
            # 如果合并日志也没有 model 列（旧格式），且只有一个模型，全量使用
            if not entry["log"] and all_logs and len(model_names) == 1:
                entry["log"] = all_logs

        # 评估报告
        report_path = report_dir / f"evaluation_report_{model_name}.md"
        entry["report"] = read_eval_report(report_path)

        # 配置快照
        snapshot_path = log_dir / f"config_snapshot_{model_name}.yaml"
        entry["config"] = read_config_snapshot(snapshot_path)

        # checkpoint
        ckpt_path = ckpt_dir / f"best_{model_name}.pth"
        if ckpt_path.exists():
            entry["checkpoint"] = str(ckpt_path)
            try:
                import torch
                ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
                if "state_dict" in ckpt:
                    total = sum(v.numel() for v in ckpt["state_dict"].values())
                    entry["total_params"] = total
                entry["best_epoch"] = ckpt.get("epoch", "?")
                entry["val_accuracy"] = ckpt.get("val_accuracy", "?")
            except Exception:
                pass

        data[model_name] = entry

    return data


# ---------- 生成图表 ----------

def save_combined_curves(model_data: dict, output_dir: Path):
    """生成 4 模型 Loss + Accuracy 合并曲线图"""
    if not model_data:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    markers = ["o", "s", "^", "D"]

    for idx, (model_name, data) in enumerate(sorted(model_data.items())):
        log = data.get("log", [])
        if not log:
            continue

        epochs = [r["epoch"] for r in log]
        val_loss = [r["val_loss"] for r in log]
        val_acc = [r["val_accuracy"] for r in log]

        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]

        ax1.plot(epochs, val_loss, color=color, marker=marker, markersize=2,
                 label=model_name)
        ax2.plot(epochs, val_acc, color=color, marker=marker, markersize=2,
                 label=model_name)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation Loss")
    ax1.set_title("Validation Loss 对比")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Accuracy")
    ax2.set_title("Validation Accuracy 对比")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / "combined_curves.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"合并曲线已保存: {save_path}")


def save_comparison_bar_chart(model_data: dict, output_dir: Path):
    """生成 4 模型对比柱状图（Accuracy / F1 / 参数量 / 训练时间）"""
    if not model_data:
        return

    model_names = sorted(model_data.keys())
    n = len(model_names)

    accs, f1s, params, times = [], [], [], []

    for name in model_names:
        data = model_data[name]
        report = data.get("report", {})
        log = data.get("log", [])
        config = data.get("config", {})

        accs.append(report.get("accuracy") or data.get("val_accuracy") or 0)
        f1s.append(report.get("macro_f1") or 0)
        params.append(data.get("total_params") or config.get("total_params") or 0)
        times.append(sum(r.get("time", 0) for r in log) if log else 0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # Accuracy
    ax = axes[0, 0]
    bars = ax.bar(model_names, accs, color=colors[:n])
    ax.set_ylabel("Accuracy")
    ax.set_title("Test Accuracy 对比")
    ax.set_ylim(min(accs) * 0.95 if accs else 0, max(accs) * 1.02 if accs else 1)
    for bar, val in zip(bars, accs):
        if val:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    # F1
    ax = axes[0, 1]
    bars = ax.bar(model_names, f1s, color=colors[:n])
    ax.set_ylabel("Macro F1-Score")
    ax.set_title("Macro F1-Score 对比")
    ax.set_ylim(min(f1s) * 0.95 if f1s else 0, max(f1s) * 1.02 if f1s else 1)
    for bar, val in zip(bars, f1s):
        if val:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    # 参数量
    ax = axes[1, 0]
    params_m = [p / 1e6 for p in params]  # 转为百万
    bars = ax.bar(model_names, params_m, color=colors[:n])
    ax.set_ylabel("参数量 (M)")
    ax.set_title("模型参数量对比")
    for bar, val in zip(bars, params_m):
        if val:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.2f}M", ha="center", va="bottom", fontsize=9)

    # 训练时间
    ax = axes[1, 1]
    times_min = [t / 60 for t in times]  # 转为分钟
    bars = ax.bar(model_names, times_min, color=colors[:n])
    ax.set_ylabel("训练时间 (分钟)")
    ax.set_title("总训练时间对比")
    for bar, val in zip(bars, times_min):
        if val:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f"{val:.1f}min", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    save_path = output_dir / "comparison_bar_chart.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"对比柱状图已保存: {save_path}")


def save_comparison_table(model_data: dict, output_dir: Path):
    """生成对比汇总表 (Markdown)"""
    if not model_data:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "model_comparison.md"

    model_names = sorted(model_data.keys())

    lines = []
    lines.append("# 模型对比汇总报告")
    lines.append("")
    lines.append(f"> 自动生成，共 {len(model_names)} 个模型")
    lines.append("")

    # ---- 总览表 ----
    lines.append("## 1. 总体性能对比")
    lines.append("")
    lines.append("| 模型 | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | 参数量 | 最佳 Epoch | 总训练时间 |")
    lines.append("|------|-------------|----------------|-------------|---------|--------|-----------|-----------|")

    for name in model_names:
        data = model_data[name]
        report = data.get("report", {})
        log = data.get("log", [])
        config = data.get("config", {})

        acc = report.get("accuracy") or data.get("val_accuracy") or "-"
        prec = report.get("macro_precision") or "-"
        rec = report.get("macro_recall") or "-"
        f1 = report.get("macro_f1") or "-"
        total_p = data.get("total_params") or config.get("total_params") or "-"
        best_ep = data.get("best_epoch", "-")
        total_time = sum(r.get("time", 0) for r in log) if log else 0

        acc_str = f"{acc:.4f}" if isinstance(acc, (int, float)) else str(acc)
        prec_str = f"{prec:.4f}" if isinstance(prec, (int, float)) else str(prec)
        rec_str = f"{rec:.4f}" if isinstance(rec, (int, float)) else str(rec)
        f1_str = f"{f1:.4f}" if isinstance(f1, (int, float)) else str(f1)
        params_str = f"{total_p:,}" if isinstance(total_p, (int, float)) else str(total_p)
        time_str = f"{total_time:.1f}s ({total_time/60:.1f}min)" if total_time else "-"

        lines.append(f"| {name} | {acc_str} | {prec_str} | {rec_str} | {f1_str} | {params_str} | {best_ep} | {time_str} |")

    lines.append("")

    # ---- 各类别指标对比 ----
    lines.append("## 2. 各类别 Accuracy 对比")
    lines.append("")

    # 收集每个模型的每类指标
    class_names = ["可回收物", "厨余垃圾", "有害垃圾", "其他垃圾"]
    lines.append("| 类别 | " + " | ".join(model_names) + " |")
    lines.append("|------|" + "|".join(["------"] * len(model_names)) + "|")

    for cname in class_names:
        row = f"| {cname} |"
        for name in model_names:
            data = model_data[name]
            report = data.get("report", {})
            per_class = report.get("per_class", {})
            val = per_class.get(cname, "-")
            row += f" {val:.4f} |" if isinstance(val, (int, float)) else f" {val} |"
        lines.append(row)

    lines.append("")

    # ---- 训练配置对比 ----
    lines.append("## 3. 训练超参数对比")
    lines.append("")

    hyper_keys = [
        ("optimizer", "优化器"),
        ("lr", "学习率"),
        ("batch_size", "批大小"),
        ("weight_decay", "权重衰减"),
        ("scheduler", "学习率策略"),
        ("two_stage", "两阶段训练"),
        ("stage1_epochs", "阶段1 Epoch数"),
        ("stage2_lr_factor", "阶段2 LR系数"),
        ("early_stopping_patience", "EarlyStopping耐心值"),
    ]

    lines.append("| 超参数 | " + " | ".join(model_names) + " |")
    lines.append("|--------|" + "|".join(["--------"] * len(model_names)) + "|")

    for key, label in hyper_keys:
        row = f"| {label} |"
        for name in model_names:
            config = model_data[name].get("config", {})
            train_cfg = config.get("train", config)
            val = train_cfg.get(key, "-")
            row += f" {val} |"
        lines.append(row)

    lines.append("")

    # ---- 数据增强对比 ----
    lines.append("## 4. 数据增强策略")
    lines.append("")
    aug_keys = [
        "random_horizontal_flip", "random_vertical_flip",
        "random_rotation", "random_erasing",
    ]
    lines.append("| 增强方式 | " + " | ".join(model_names) + " |")
    lines.append("|---------|" + "|".join(["---------"] * len(model_names)) + "|")

    for key in aug_keys:
        row = f"| {key} |"
        for name in model_names:
            config = model_data[name].get("config", {})
            aug_cfg = config.get("augmentation", {})
            val = aug_cfg.get(key, "-")
            row += f" {val} |"
        lines.append(row)

    lines.append("")

    # ---- 产出文件清单 ----
    lines.append("## 5. 各模型产出文件")
    lines.append("")
    for name in model_names:
        data = model_data[name]
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- 权重: `{data.get('checkpoint', '-')}`")
        lines.append(f"- 训练曲线: `outputs/figures/training_curve_{name}.png`")
        lines.append(f"- 配置快照: `logs/config_snapshot_{name}.yaml`")
        lines.append(f"- 评估报告: `outputs/reports/evaluation_report_{name}.md`")
        lines.append(f"- 混淆矩阵: `outputs/figures/confusion_matrix_{name}.png`")
        lines.append("")

    report_text = "\n".join(lines)
    save_path.write_text(report_text, encoding="utf-8")
    print(f"对比汇总表已保存: {save_path}")
    return report_text


# ---------- 主入口 ----------

def main():
    args = parse_args()

    log_dir = PROJECT_ROOT / args.log_dir
    report_dir = PROJECT_ROOT / args.report_dir
    ckpt_dir = PROJECT_ROOT / args.checkpoint_dir
    output_dir = PROJECT_ROOT / args.output_dir

    print("=" * 60)
    print("  多模型对比分析")
    print("=" * 60)

    # 收集数据
    model_data = gather_model_data(log_dir, report_dir, ckpt_dir)

    if not model_data:
        print("未发现任何模型数据，请确认已训练并评估模型。")
        sys.exit(1)

    print(f"发现 {len(model_data)} 个模型: {', '.join(sorted(model_data.keys()))}")
    print()

    # 生成对比表
    save_comparison_table(model_data, output_dir)

    # 生成合并曲线图
    save_combined_curves(model_data, output_dir)

    # 生成柱状对比图
    save_comparison_bar_chart(model_data, output_dir)

    print()
    print("=" * 60)
    print("  对比分析完成!")
    print(f"  输出目录: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
