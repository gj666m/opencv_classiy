"""
OpenCV 结果绘制模块
在图像上叠加分类结果、置信度、FPS、概率条形图
使用 PIL 渲染中文文字，避免 OpenCV 中文乱码
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# 四大类对应的 BGR 颜色
CLASS_COLORS = {
    "可回收物": (0, 255, 0),     # 绿色
    "厨余垃圾": (0, 200, 255),   # 橙色
    "有害垃圾": (0, 0, 255),     # 红色
    "其他垃圾": (200, 200, 200), # 灰色
}

# 默认颜色（未知类别）
DEFAULT_COLOR = (255, 255, 255)

# 中文字体缓存
_cached_font = None
_cached_font_small = None


def _get_chinese_font(size: int = 24):
    """获取支持中文的 PIL 字体"""
    global _cached_font, _cached_font_small

    if size <= 18 and _cached_font_small is not None:
        return _cached_font_small
    if size > 18 and _cached_font is not None:
        return _cached_font

    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
        "C:/Windows/Fonts/simhei.ttf",     # 黑体
        "C:/Windows/Fonts/simsun.ttc",     # 宋体
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",  # Linux
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, size)
                if size <= 18:
                    _cached_font_small = font
                else:
                    _cached_font = font
                return font
            except Exception:
                continue

    return ImageFont.load_default()


def _cv2_to_pil(image: np.ndarray) -> Image.Image:
    """OpenCV BGR 图像 → PIL RGB 图像"""
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def _pil_to_cv2(image: Image.Image) -> np.ndarray:
    """PIL RGB 图像 → OpenCV BGR 图像"""
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def _put_chinese_text(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_size: int = 24,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
) -> np.ndarray:
    """
    在 OpenCV 图像上绘制中文文字（通过 PIL）

    Args:
        image: BGR 图像
        text: 要绘制的文字
        position: 文字左上角坐标 (x, y)
        font_size: 字体大小
        color: BGR 颜色
        thickness: 字体粗细（通过多次偏移绘制模拟）

    Returns:
        绘制了文字的图像
    """
    pil_img = _cv2_to_pil(image)
    draw = ImageDraw.Draw(pil_img)
    font = _get_chinese_font(font_size)

    # PIL 用 RGB 颜色
    rgb_color = (color[2], color[1], color[0])

    # 模拟粗体（多次偏移绘制）
    for dx in range(thickness):
        for dy in range(thickness):
            draw.text((position[0] + dx, position[1] + dy), text, font=font, fill=rgb_color)

    return _pil_to_cv2(pil_img)


def draw_result(
    image: np.ndarray,
    class_name: str,
    confidence: float,
    color: Optional[Tuple[int, int, int]] = None,
    thickness: int = 2
) -> np.ndarray:
    """
    在图像上叠加分类结果

    - 左上角显示：类别名称
    - 右上角显示：置信度百分比

    Args:
        image: BGR 图像（会在副本上绘制）
        class_name: 类别名称
        confidence: 置信度 (0-1)
        color: 文字颜色 (B, G, R)，默认根据类别自动选择
        thickness: 文字粗细

    Returns:
        绘制了结果的图像副本
    """
    result = image.copy()
    h, w = result.shape[:2]

    if color is None:
        color = CLASS_COLORS.get(class_name, DEFAULT_COLOR)

    # 自适应字体大小
    font_size = max(14, min(int(w / 800 * 24), 36))

    # 半透明背景条（左上角）— 类别名
    text = class_name
    font = _get_chinese_font(font_size)
    pil_tmp = _cv2_to_pil(np.zeros_like(result))
    draw_tmp = ImageDraw.Draw(pil_tmp)
    bbox = draw_tmp.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    overlay = result.copy()
    cv2.rectangle(overlay, (10, 10), (20 + tw, 20 + th + 8), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, result, 0.4, 0, result)

    result = _put_chinese_text(result, text, (15, 12), font_size, color, thickness)

    # 右上角置信度
    conf_text = f"{confidence * 100:.1f}%"
    bbox2 = draw_tmp.textbbox((0, 0), conf_text, font=font)
    cw = bbox2[2] - bbox2[0]
    ch = bbox2[3] - bbox2[1]

    overlay2 = result.copy()
    cv2.rectangle(overlay2, (w - cw - 25, 10), (w - 10, 20 + ch + 8), (0, 0, 0), -1)
    cv2.addWeighted(overlay2, 0.6, result, 0.4, 0, result)

    result = _put_chinese_text(result, conf_text, (w - cw - 20, 12), font_size, color, thickness)

    return result


def draw_fps(
    image: np.ndarray,
    fps: float,
    position: str = 'bottom_left',
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    在图像上显示 FPS

    Args:
        image: BGR 图像（会在副本上绘制）
        fps: 帧率数值
        position: 显示位置，'bottom_left' 或 'top_right'
        color: 文字颜色
        thickness: 文字粗细

    Returns:
        绘制了 FPS 的图像副本
    """
    result = image.copy()
    h, w = result.shape[:2]

    font_size = max(12, min(int(w / 1000 * 20), 28))
    text = f"FPS: {fps:.1f}"

    font = _get_chinese_font(font_size)
    pil_tmp = _cv2_to_pil(np.zeros_like(result))
    draw_tmp = ImageDraw.Draw(pil_tmp)
    bbox = draw_tmp.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    if position == 'bottom_left':
        org = (10, h - th - 15)
    else:
        org = (w - tw - 15, h - th - 15)

    # 半透明背景
    overlay = result.copy()
    x1, y1 = org[0] - 5, org[1] - 5
    x2, y2 = org[0] + tw + 5, org[1] + th + 5
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, result, 0.4, 0, result)

    result = _put_chinese_text(result, text, org, font_size, color, thickness)

    return result


def draw_probability_bar(
    image: np.ndarray,
    probabilities: Dict[str, float],
    label_names: Optional[List[str]] = None,
    bar_height: int = 40,
    max_bar_width: Optional[int] = None
) -> np.ndarray:
    """
    在图像底部绘制四分类概率条形图

    Args:
        image: BGR 图像（会在副本上绘制）
        probabilities: {类别名: 概率} 字典
        label_names: 类别名顺序，默认按 probabilities 的 key 顺序
        bar_height: 每个条形的高度
        max_bar_width: 条形图最大宽度（None 表示使用图像宽度的 60%）

    Returns:
        绘制了概率条的图像副本
    """
    result = image.copy()
    h, w = result.shape[:2]

    if label_names is None:
        label_names = list(probabilities.keys())

    num_classes = len(label_names)
    if num_classes == 0:
        return result

    if max_bar_width is None:
        max_bar_width = int(w * 0.6)

    total_bar_height = num_classes * bar_height + 20
    font_size = max(12, min(int(w / 1200 * 16), 20))

    # 半透明背景（底部）
    overlay = result.copy()
    cv2.rectangle(overlay, (0, h - total_bar_height), (w, h), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.7, result, 0.3, 0, result)

    # 绘制每个类别的概率条
    bar_x_start = 10
    bar_y_start = h - total_bar_height + 10

    for i, name in enumerate(label_names):
        prob = probabilities.get(name, 0.0)
        bar_color = CLASS_COLORS.get(name, DEFAULT_COLOR)

        y = bar_y_start + i * bar_height

        # 类别名
        result = _put_chinese_text(result, name, (bar_x_start, y + 2), font_size, (255, 255, 255))

        # 概率条
        text_width = 90  # 文字预留宽度
        bar_x = bar_x_start + text_width
        bar_w = int((w - bar_x - 60) * prob)
        bar_w = max(bar_w, 1)

        cv2.rectangle(result, (bar_x, y + 5), (bar_x + bar_w, y + bar_height - 5), bar_color, -1)

        # 百分比文字
        pct_text = f"{prob * 100:.1f}%"
        result = _put_chinese_text(result, pct_text, (bar_x + bar_w + 5, y + 2), font_size, bar_color)

    return result
