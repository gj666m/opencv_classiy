"""
OpenCV 图像 I/O 模块
提供图片读取、写入、批量路径加载功能
"""

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def read_image(path: str, color_mode: str = 'bgr') -> Optional[np.ndarray]:
    """
    读取图片

    Args:
        path: 图片路径
        color_mode: 颜色模式，支持 'bgr'（默认）、'rgb'、'gray'

    Returns:
        numpy array，读取失败返回 None
    """
    path = Path(path)
    if not path.exists():
        logger.warning(f"图片不存在: {path}")
        return None

    if color_mode == 'gray':
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    elif color_mode == 'rgb':
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:  # bgr
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if img is None:
        logger.warning(f"无法读取图片: {path}")

    return img


def write_image(path: str, image: np.ndarray) -> bool:
    """
    保存图片，自动创建父目录

    Args:
        path: 保存路径
        image: numpy array（BGR 格式）

    Returns:
        是否保存成功
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(path), image)
    if not success:
        logger.error(f"保存图片失败: {path}")
    return success


def batch_load(directory: str, extensions: Optional[List[str]] = None) -> List[Path]:
    """
    批量加载目录下所有图片路径

    Args:
        directory: 目录路径
        extensions: 文件扩展名列表，默认 ['.jpg', '.jpeg', '.png', '.bmp']

    Returns:
        图片路径列表（已排序）
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    directory = Path(directory)
    if not directory.exists():
        logger.warning(f"目录不存在: {directory}")
        return []

    ext_set = {e.lower() for e in extensions}
    paths = [
        p for p in directory.rglob('*')
        if p.is_file() and p.suffix.lower() in ext_set
    ]
    paths.sort()
    logger.info(f"从 {directory} 加载了 {len(paths)} 张图片")
    return paths
