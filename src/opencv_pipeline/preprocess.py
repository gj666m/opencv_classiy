"""
OpenCV 预处理模块
提供图像缩放、归一化、颜色空间转换、去噪、CLAHE 等预处理功能
所有函数接受 numpy array（BGR），返回 numpy array（BGR）
"""

import logging
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def resize(
    image: np.ndarray,
    size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    图像缩放

    Args:
        image: BGR 图像 (H, W, C)
        size: 目标尺寸 (height, width)

    Returns:
        缩放后的图像
    """
    return cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)


def normalize(image: np.ndarray) -> np.ndarray:
    """
    像素值归一化到 [0, 1]，再转回 uint8 范围
    用于 OpenCV 预处理管线中的视觉展示

    Args:
        image: BGR 图像，dtype=uint8

    Returns:
        归一化后的 uint8 图像
    """
    normalized = image.astype(np.float32) / 255.0
    return (normalized * 255).astype(np.uint8)


def convert_color(
    image: np.ndarray,
    target: str = 'rgb'
) -> np.ndarray:
    """
    颜色空间转换

    Args:
        image: BGR 图像
        target: 目标颜色空间，支持 'rgb'、'hsv'、'lab'、'bgr'

    Returns:
        转换后的图像
    """
    conversions = {
        'rgb': cv2.COLOR_BGR2RGB,
        'hsv': cv2.COLOR_BGR2HSV,
        'lab': cv2.COLOR_BGR2LAB,
        'bgr': None,  # 无需转换
    }

    if target not in conversions:
        raise ValueError(f"不支持的颜色空间: {target}，可选: {list(conversions.keys())}")

    code = conversions[target]
    if code is None:
        return image.copy()

    return cv2.cvtColor(image, code)


def denoise(
    image: np.ndarray,
    method: str = 'gaussian',
    ksize: int = 5
) -> np.ndarray:
    """
    图像去噪

    Args:
        image: BGR 图像
        method: 去噪方法，支持 'gaussian'（高斯模糊）、'median'（中值滤波）、'bilateral'（双边滤波）
        ksize: 核大小（奇数）

    Returns:
        去噪后的图像
    """
    if ksize % 2 == 0:
        ksize += 1  # 核大小必须为奇数

    if method == 'gaussian':
        return cv2.GaussianBlur(image, (ksize, ksize), 0)
    elif method == 'median':
        return cv2.medianBlur(image, ksize)
    elif method == 'bilateral':
        return cv2.bilateralFilter(image, d=ksize, sigmaColor=75, sigmaSpace=75)
    else:
        raise ValueError(f"不支持的去噪方法: {method}，可选: gaussian, median, bilateral")


def clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    自适应直方图均衡化（CLAHE），缓解光照不均

    对 LAB 颜色空间的 L 通道进行 CLAHE 处理

    Args:
        image: BGR 图像
        clip_limit: 对比度限制阈值
        grid_size: 网格大小

    Returns:
        CLAHE 处理后的 BGR 图像
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l = clahe_obj.apply(l)

    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def adjust_brightness(image: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """
    亮度调整

    Args:
        image: BGR 图像
        factor: 亮度因子，>1 变亮，<1 变暗

    Returns:
        调整后的图像
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def adjust_contrast(image: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """
    对比度调整

    Args:
        image: BGR 图像
        factor: 对比度因子，>1 增强对比度，<1 降低对比度

    Returns:
        调整后的图像
    """
    result = cv2.convertScaleAbs(image, alpha=factor, beta=0)
    return result


def canny_edge(
    image: np.ndarray,
    threshold1: float = 50,
    threshold2: float = 150,
) -> np.ndarray:
    """
    Canny 边缘检测

    Args:
        image: BGR 图像
        threshold1: 低阈值
        threshold2: 高阈值

    Returns:
        边缘图（单通道，白底黑线或黑底白线）
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1, threshold2)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def sobel_edge(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Sobel 边缘检测

    Args:
        image: BGR 图像
        ksize: Sobel 核大小（1, 3, 5, 7）

    Returns:
        边缘图（BGR 三通道）
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    return cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR)


def laplacian_edge(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Laplacian 边缘检测

    Args:
        image: BGR 图像
        ksize: 核大小

    Returns:
        边缘图（BGR 三通道）
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
    lap = np.clip(np.abs(lap), 0, 255).astype(np.uint8)
    return cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)


# 支持的预处理步骤名到函数的映射
_PREPROCESS_FUNCS = {
    'resize': resize,
    'normalize': normalize,
    'denoise': denoise,
    'clahe': clahe,
    'adjust_brightness': adjust_brightness,
    'adjust_contrast': adjust_contrast,
    'convert_color': convert_color,
    'canny_edge': canny_edge,
    'sobel_edge': sobel_edge,
    'laplacian_edge': laplacian_edge,
}


class PreprocessPipeline:
    """
    可配置的预处理管线，按顺序执行多个预处理步骤

    用法:
        pipe = PreprocessPipeline(['resize', 'denoise', 'clahe'])
        result = pipe(image)

    也可传入自定义参数:
        pipe = PreprocessPipeline([
            {'name': 'resize', 'size': (224, 224)},
            {'name': 'denoise', 'method': 'bilateral', 'ksize': 7},
            {'name': 'clahe', 'clip_limit': 3.0},
        ])
    """

    def __init__(self, steps: List[Union[str, dict]]):
        """
        Args:
            steps: 预处理步骤列表
                - 字符串形式: 使用默认参数，如 'resize'
                - 字典形式: 带参数，如 {'name': 'resize', 'size': (224, 224)}
        """
        self.steps = []
        for step in steps:
            if isinstance(step, str):
                if step not in _PREPROCESS_FUNCS:
                    raise ValueError(f"未知的预处理步骤: {step}，可选: {list(_PREPROCESS_FUNCS.keys())}")
                self.steps.append((_PREPROCESS_FUNCS[step], {}))
            elif isinstance(step, dict):
                name = step.pop('name')
                if name not in _PREPROCESS_FUNCS:
                    raise ValueError(f"未知的预处理步骤: {name}，可选: {list(_PREPROCESS_FUNCS.keys())}")
                self.steps.append((_PREPROCESS_FUNCS[name], step))
            else:
                raise TypeError(f"步骤类型错误: {type(step)}，期望 str 或 dict")

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        按顺序执行所有预处理步骤

        Args:
            image: BGR 图像

        Returns:
            处理后的图像
        """
        for func, kwargs in self.steps:
            image = func(image, **kwargs)
        return image
