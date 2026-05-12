"""
OpenCV 数据增强模块
提供翻转、旋转、亮度扰动、裁剪、噪声等数据增强功能
用于论文展示、离线增强和摄像头实时预处理
"""

import logging
import random
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def random_horizontal_flip(image: np.ndarray, p: float = 0.5) -> np.ndarray:
    """
    随机水平翻转

    Args:
        image: BGR 图像
        p: 翻转概率

    Returns:
        可能翻转后的图像
    """
    if random.random() < p:
        return cv2.flip(image, 1)  # 水平翻转
    return image


def random_vertical_flip(image: np.ndarray, p: float = 0.5) -> np.ndarray:
    """
    随机垂直翻转

    Args:
        image: BGR 图像
        p: 翻转概率

    Returns:
        可能翻转后的图像
    """
    if random.random() < p:
        return cv2.flip(image, 0)  # 垂直翻转
    return image


def random_rotation(image: np.ndarray, max_angle: int = 15) -> np.ndarray:
    """
    随机旋转（保持图像完整，用黑色填充空白区域）

    Args:
        image: BGR 图像
        max_angle: 最大旋转角度（正负范围）

    Returns:
        旋转后的图像
    """
    angle = random.uniform(-max_angle, max_angle)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # 计算旋转矩阵
    rotation_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 计算旋转后的图像边界
    cos_val = abs(rotation_mat[0, 0])
    sin_val = abs(rotation_mat[0, 1])
    new_w = int(h * sin_val + w * cos_val)
    new_h = int(h * cos_val + w * sin_val)

    # 调整旋转矩阵的平移量，使图像居中
    rotation_mat[0, 2] += (new_w - w) / 2
    rotation_mat[1, 2] += (new_h - h) / 2

    rotated = cv2.warpAffine(
        image, rotation_mat, (new_w, new_h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    # 裁剪回原始尺寸（居中裁剪）
    start_y = (new_h - h) // 2
    start_x = (new_w - w) // 2
    return rotated[start_y:start_y + h, start_x:start_x + w]


def random_brightness(image: np.ndarray, max_delta: float = 0.2) -> np.ndarray:
    """
    随机亮度扰动

    Args:
        image: BGR 图像
        max_delta: 最大亮度变化比例（0-1）

    Returns:
        亮度调整后的图像
    """
    delta = random.uniform(-max_delta, max_delta)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1.0 + delta), 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def random_crop(image: np.ndarray, crop_ratio: float = 0.9) -> np.ndarray:
    """
    随机裁剪后 resize 回原尺寸

    Args:
        image: BGR 图像
        crop_ratio: 裁剪区域占原图的比例（0-1）

    Returns:
        裁剪并缩放回原尺寸的图像
    """
    h, w = image.shape[:2]
    crop_h = int(h * crop_ratio)
    crop_w = int(w * crop_ratio)

    # 随机选择裁剪起点
    start_y = random.randint(0, h - crop_h)
    start_x = random.randint(0, w - crop_w)

    cropped = image[start_y:start_y + crop_h, start_x:start_x + crop_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def add_gaussian_noise(
    image: np.ndarray,
    mean: float = 0,
    std: float = 10
) -> np.ndarray:
    """
    添加高斯噪声

    Args:
        image: BGR 图像
        mean: 噪声均值
        std: 噪声标准差

    Returns:
        添加噪声后的图像
    """
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


# 增强函数注册表
_AUGMENT_FUNCS = {
    'random_horizontal_flip': random_horizontal_flip,
    'random_vertical_flip': random_vertical_flip,
    'random_rotation': random_rotation,
    'random_brightness': random_brightness,
    'random_crop': random_crop,
    'add_gaussian_noise': add_gaussian_noise,
}


class AugmentationPipeline:
    """
    可配置的数据增强管线

    用法一（传入 dict 配置）:
        config = {
            'random_horizontal_flip': {'p': 0.5},
            'random_rotation': {'max_angle': 15},
        }
        pipe = AugmentationPipeline(config)
        result = pipe(image)

    用法二（传入 list 步骤列表）:
        steps = [
            ('random_horizontal_flip', {'p': 0.5}),
            ('random_rotation', {'max_angle': 10}),
        ]
        pipe = AugmentationPipeline(steps)
        result = pipe(image)
    """

    def __init__(self, config: Union[Dict, List]):
        """
        Args:
            config: 增强配置
                - dict 形式: {'random_horizontal_flip': {'p': 0.5}, ...}
                - list 形式: [('random_horizontal_flip', {'p': 0.5}), ...]
        """
        self.steps = []

        if isinstance(config, dict):
            for name, kwargs in config.items():
                if name not in _AUGMENT_FUNCS:
                    logger.warning(f"未知的增强方法: {name}，已跳过")
                    continue
                # kwargs 为 True 表示使用默认参数
                if kwargs is True or kwargs is None:
                    kwargs = {}
                self.steps.append((_AUGMENT_FUNCS[name], kwargs))
        elif isinstance(config, list):
            for item in config:
                if isinstance(item, str):
                    name, kwargs = item, {}
                elif isinstance(item, (tuple, list)) and len(item) == 2:
                    name, kwargs = item
                else:
                    logger.warning(f"无效的增强配置项: {item}，已跳过")
                    continue

                if name not in _AUGMENT_FUNCS:
                    logger.warning(f"未知的增强方法: {name}，已跳过")
                    continue
                self.steps.append((_AUGMENT_FUNCS[name], kwargs))
        else:
            raise TypeError(f"config 类型错误: {type(config)}，期望 dict 或 list")

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        按顺序执行所有增强步骤

        Args:
            image: BGR 图像

        Returns:
            增强后的图像
        """
        for func, kwargs in self.steps:
            image = func(image, **kwargs)
        return image
