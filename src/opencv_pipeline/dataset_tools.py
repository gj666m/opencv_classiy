"""
数据集工具模块
提供数据集扫描、损坏图片检测、模糊图片检测、样本可视化功能
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# 支持的图片扩展名
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


def scan_dataset(directory: str) -> Dict[str, int]:
    """
    扫描数据集，返回各文件夹图片数量统计

    Args:
        directory: 数据集根目录，每个子文件夹代表一个类别

    Returns:
        {类别名: 图片数量} 字典
    """
    directory = Path(directory)
    stats = {}

    for subdir in sorted(directory.iterdir()):
        if not subdir.is_dir():
            continue
        count = sum(
            1 for f in subdir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )
        if count > 0:
            stats[subdir.name] = count

    logger.info(f"扫描完成: {len(stats)} 个类别, 共 {sum(stats.values())} 张图片")
    return stats


def detect_corrupted_images(directory: str) -> List[str]:
    """
    检测损坏图片

    通过 OpenCV 读取验证图片完整性，对于 JPEG 文件额外检查 EOF 标记

    Args:
        directory: 数据集根目录

    Returns:
        损坏图片路径列表
    """
    directory = Path(directory)
    corrupted = []

    for img_path in sorted(directory.rglob('*')):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        # 尝试用 OpenCV 读取
        img = cv2.imread(str(img_path))
        if img is None:
            corrupted.append(str(img_path))
            logger.warning(f"损坏图片（无法读取）: {img_path}")
            continue

        # JPEG 额外检查：验证文件末尾的 EOI（End of Image）标记 0xFFD9
        if img_path.suffix.lower() in ('.jpg', '.jpeg'):
            with open(img_path, 'rb') as f:
                f.seek(-2, 2)
                eof = f.read(2)
                if eof != b'\xff\xd9':
                    corrupted.append(str(img_path))
                    logger.warning(f"损坏图片（JPEG 不完整）: {img_path}")

    logger.info(f"检测完成: {len(corrupted)} 张损坏图片")
    return corrupted


def detect_blurry_images(
    directory: str,
    threshold: float = 100.0,
    max_samples: Optional[int] = None
) -> List[Tuple[str, float]]:
    """
    使用拉普拉斯方差检测模糊图片
    方差越低，图片越模糊

    Args:
        directory: 数据集根目录
        threshold: 拉普拉斯方差阈值，低于此值视为模糊
        max_samples: 最大检测数量（None 表示全部检测）

    Returns:
        [(图片路径, 拉普拉斯方差)] 列表
    """
    directory = Path(directory)
    blurry = []

    all_images = [
        p for p in directory.rglob('*')
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if max_samples and len(all_images) > max_samples:
        all_images = random.sample(all_images, max_samples)

    for img_path in sorted(all_images):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        variance = cv2.Laplacian(img, cv2.CV_64F).var()
        if variance < threshold:
            blurry.append((str(img_path), float(variance)))

    blurry.sort(key=lambda x: x[1])
    logger.info(f"模糊检测完成: {len(blurry)} 张模糊图片（阈值={threshold}）")
    return blurry


def visualize_samples(
    directory: str,
    output_path: str = "outputs/figures/dataset_samples.png",
    num_per_class: int = 4,
    image_size: Tuple[int, int] = (128, 128),
    max_classes: int = 20
) -> str:
    """
    从每个类别随机抽样，拼成对比图保存

    Args:
        directory: 数据集根目录
        output_path: 保存路径
        num_per_class: 每类抽样数量
        image_size: 单张缩放尺寸
        max_classes: 最大显示类别数

    Returns:
        保存路径
    """
    directory = Path(directory)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    subdirs = sorted([
        d for d in directory.iterdir()
        if d.is_dir()
    ])[:max_classes]

    if not subdirs:
        logger.warning(f"未找到子目录: {directory}")
        return ""

    rows = []
    for subdir in subdirs:
        images = [
            p for p in subdir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not images:
            continue

        samples = random.sample(images, min(num_per_class, len(images)))
        row_imgs = []
        for img_path in samples:
            img = cv2.imread(str(img_path))
            if img is not None:
                img = cv2.resize(img, image_size)
                row_imgs.append(img)

        if row_imgs:
            row = np.hstack(row_imgs)
            rows.append((subdir.name, row))

    if not rows:
        logger.warning("没有可可视化的样本")
        return ""

    # 计算最大宽度
    max_w = max(row.shape[1] for _, row in rows)

    # 拼接所有行，添加类别名标签
    canvas_rows = []
    for name, row in rows:
        # 补齐宽度
        if row.shape[1] < max_w:
            pad = np.zeros((row.shape[0], max_w - row.shape[1], 3), dtype=np.uint8)
            row = np.hstack([row, pad])

        # 添加类别名（白色文字在黑色背景条上）
        label_bar = np.zeros((30, max_w, 3), dtype=np.uint8)
        cv2.putText(
            label_bar, name, (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )
        canvas_rows.append(label_bar)
        canvas_rows.append(row)

    canvas = np.vstack(canvas_rows)
    cv2.imwrite(str(output_path), canvas)
    logger.info(f"样本可视化已保存: {output_path}")
    return str(output_path)
