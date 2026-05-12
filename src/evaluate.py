"""
评估引擎
支持 Accuracy/P/R/F1/混淆矩阵/错分样本分析
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)

# 中文字体配置（matplotlib 显示中文标签）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _get_chinese_font(size: int = 24):
    """获取支持中文的 PIL 字体"""
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]
    for fp in font_paths:
        p = Path(fp)
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size)
            except Exception:
                continue
    return ImageFont.load_default()


def _get_matplotlib_font():
    """获取 matplotlib 中文字体属性，确保热力图能正常显示中文"""
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]
    for fp in font_paths:
        p = Path(fp)
        if p.exists():
            try:
                return font_manager.FontProperties(fname=str(p))
            except Exception:
                continue
    return None


def _draw_labeled_sample(image: np.ndarray, true_name: str, pred_name: str) -> np.ndarray:
    """用 PIL 在错分样例图上叠加中英文标签，避免中文变成问号"""
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = _get_chinese_font(24)
    small_font = _get_chinese_font(22)

    lines = [
        (f"True: {true_name}", (220, 30, 30)),
        (f"Pred: {pred_name}", (37, 99, 235)),
    ]

    y = 10
    for text, color in lines:
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        draw.rounded_rectangle((8, y, 24 + w, y + h + 10), radius=10, fill=(255, 255, 255))
        draw.text((14, y + 4), text, font=font, fill=color)
        y += h + 14

    hint = "错分样例"
    bbox = draw.textbbox((0, 0), hint, font=small_font)
    draw.rounded_rectangle((8, image.shape[0] - 40, 18 + (bbox[2] - bbox[0]), image.shape[0] - 8), radius=8, fill=(255, 255, 255))
    draw.text((12, image.shape[0] - 36), hint, font=small_font, fill=(100, 116, 139))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    label_names: Dict[int, str],
) -> Dict:
    """
    完整评估模型性能

    Args:
        model: 已加载权重的模型
        dataloader: 测试集 DataLoader
        device: 计算设备
        label_names: 标签映射 {0: "可回收物", 1: "厨余垃圾", ...}

    Returns:
        {
            'accuracy': float,
            'precision': np.array,       # 每类 precision
            'recall': np.array,          # 每类 recall
            'f1': np.array,              # 每类 f1
            'confusion_matrix': np.array, # 4x4
            'per_class_accuracy': dict,
            'classification_report': str, # sklearn classification_report 文本
            'all_preds': np.array,       # 所有预测标签
            'all_labels': np.array,      # 所有真实标签
        }
    """
    model.eval()

    all_preds = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        _, predicted = outputs.max(1)

        all_preds.append(predicted.cpu().numpy())
        all_labels.append(labels.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    # 每类准确率（各类别预测正确的比例 / 该类别总数）
    per_class_accuracy = {}
    names_list = [label_names[i] for i in range(len(label_names))]
    for i, name in enumerate(names_list):
        class_total = cm[i].sum()
        class_correct = cm[i][i]
        per_class_accuracy[name] = float(class_correct / class_total) if class_total > 0 else 0.0

    # sklearn 完整报告
    report = classification_report(
        all_labels,
        all_preds,
        target_names=names_list,
        digits=4,
        zero_division=0,
    )

    logger.info(f"评估完成: Accuracy={accuracy:.4f}")
    logger.info(f"\n{report}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "per_class_accuracy": per_class_accuracy,
        "classification_report": report,
        "all_preds": all_preds,
        "all_labels": all_labels,
    }


def save_confusion_matrix(
    cm: np.ndarray,
    label_names: Dict[int, str],
    save_path: str,
    title: str = "混淆矩阵",
):
    """
    保存混淆矩阵热力图

    Args:
        cm: 混淆矩阵 (n_classes, n_classes)
        label_names: 标签映射 {0: "可回收物", ...}
        save_path: 保存路径
        title: 图表标题
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    names = [label_names[i] for i in range(len(label_names))]

    fig, ax = plt.subplots(figsize=(8, 6))

    # 归一化（按行归一化，表示各真实类别的预测分布）
    cm_norm = cm.astype(np.float64)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # 避免除零
    cm_norm = cm_norm / row_sums

    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 设置刻度
    font_prop = _get_matplotlib_font()

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=names,
        yticklabels=names,
    )
    if font_prop is not None:
        ax.set_title(title, fontproperties=font_prop)
        ax.set_ylabel("真实标签", fontproperties=font_prop)
        ax.set_xlabel("预测标签", fontproperties=font_prop)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(font_prop)
    else:
        ax.set_title(title)
        ax.set_ylabel("真实标签")
        ax.set_xlabel("预测标签")

    # 旋转 x 轴标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 在每个格子中显示数值（原始数量 + 百分比）
    thresh = 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = cm_norm[i, j] * 100
            text = f"{cm[i, j]}\n({pct:.1f}%)"
            ax.text(
                j, i, text,
                ha="center", va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
                fontsize=10,
                fontproperties=font_prop,
            )

    fig.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"混淆矩阵已保存: {save_path}")


def save_misclassified_samples(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    label_names: Dict[int, str],
    save_dir: str,
    max_samples: int = 20,
):
    """
    保存错分样本（原图 + 预测标签 + 真实标签）

    Args:
        model: 已加载权重的模型
        dataloader: 测试集 DataLoader（必须返回 (image_tensor, label)，且能回溯到文件路径）
        device: 计算设备
        label_names: 标签映射
        save_dir: 保存目录
        max_samples: 最多保存的错分样本数
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    names = [label_names[i] for i in range(len(label_names))]

    misclassified = []
    dataset = dataloader.dataset

    with torch.no_grad():
        for idx in range(len(dataset)):
            if len(misclassified) >= max_samples:
                break

            image_tensor, true_label = dataset[idx]
            # 添加 batch 维度
            input_tensor = image_tensor.unsqueeze(0).to(device)
            output = model(input_tensor)
            _, pred_label = output.max(1)
            pred_label = pred_label.item()

            if pred_label != true_label:
                # 获取图片路径（从 dataset 的 samples 列表中取）
                if hasattr(dataset, "samples"):
                    img_path = dataset.samples[idx][0]
                else:
                    img_path = None

                misclassified.append({
                    "img_path": img_path,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "index": idx,
                })

    if not misclassified:
        logger.info("没有错分样本！")
        return

    logger.info(f"发现 {len(misclassified)} 个错分样本，保存前 {min(len(misclassified), max_samples)} 个")

    for i, item in enumerate(misclassified):
        true_name = names[item["true_label"]]
        pred_name = names[item["pred_label"]]

        # 读取原图
        if item["img_path"] is not None and Path(item["img_path"]).exists():
            img = cv2.imread(str(item["img_path"]))
        else:
            # 如果无法获取原图路径，尝试从 tensor 反向生成
            img_tensor = dataset[item["index"]][0]
            img = _tensor_to_cv2(img_tensor)

        if img is None:
            continue

        # 缩放到合适显示尺寸
        h, w = img.shape[:2]
        scale = min(400 / max(h, w), 1.0)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

        # 用 PIL 渲染中文标签，避免 OpenCV 默认字体导致问号乱码
        img = _draw_labeled_sample(img, true_name, pred_name)

        save_path = save_dir / f"mis_{i:03d}_true{item['true_label']}_pred{item['pred_label']}.jpg"
        cv2.imwrite(str(save_path), img)

    logger.info(f"错分样本已保存到: {save_dir}")


def save_evaluation_report(
    results: Dict,
    label_names: Dict[int, str],
    model_name: str,
    save_path: str,
):
    """
    保存评估报告为 Markdown 文件

    Args:
        results: evaluate() 返回的结果字典
        label_names: 标签映射
        model_name: 模型名称
        save_path: 保存路径
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    names = [label_names[i] for i in range(len(label_names))]

    lines = []
    lines.append(f"# 评估报告 — {model_name}")
    lines.append("")
    lines.append("## 总体指标")
    lines.append("")
    lines.append(f"| 指标 | 值 |")
    lines.append(f"|------|-----|")
    lines.append(f"| Overall Accuracy | {results['accuracy']:.4f} |")
    lines.append(f"| Macro Precision | {results['precision'].mean():.4f} |")
    lines.append(f"| Macro Recall | {results['recall'].mean():.4f} |")
    lines.append(f"| Macro F1 | {results['f1'].mean():.4f} |")
    lines.append("")

    lines.append("## 各类别指标")
    lines.append("")
    lines.append("| 类别 | Precision | Recall | F1-Score | 类别准确率 |")
    lines.append("|------|-----------|--------|----------|-----------|")
    for i, name in enumerate(names):
        pca = results["per_class_accuracy"].get(name, 0.0)
        lines.append(
            f"| {name} | {results['precision'][i]:.4f} | "
            f"{results['recall'][i]:.4f} | {results['f1'][i]:.4f} | "
            f"{pca:.4f} |"
        )
    lines.append("")

    lines.append("## 混淆矩阵")
    lines.append("")
    lines.append("```")
    header = "真实\\预测\t" + "\t".join(names)
    lines.append(header)
    cm = results["confusion_matrix"]
    for i, name in enumerate(names):
        row = name + "\t" + "\t".join(str(cm[i][j]) for j in range(len(names)))
        lines.append(row)
    lines.append("```")
    lines.append("")

    lines.append("## Classification Report (sklearn)")
    lines.append("")
    lines.append("```")
    lines.append(results["classification_report"])
    lines.append("```")

    report_text = "\n".join(lines)
    save_path.write_text(report_text, encoding="utf-8")
    logger.info(f"评估报告已保存: {save_path}")


def _tensor_to_cv2(tensor: torch.Tensor) -> np.ndarray:
    """将 torchvision 输出的 tensor 转换为 BGR 格式的 numpy 数组"""
    # tensor: (3, H, W), normalized
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    # 反标准化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    # RGB → BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img
