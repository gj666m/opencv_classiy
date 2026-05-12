"""
OpenCV 管线模块
包含图像 I/O、预处理、数据增强、摄像头采集、结果绘制、数据集工具
"""

from src.opencv_pipeline.augment import AugmentationPipeline
from src.opencv_pipeline.camera import CameraStream, FPSCounter
from src.opencv_pipeline.dataset_tools import (
    detect_blurry_images,
    detect_corrupted_images,
    scan_dataset,
    visualize_samples,
)
from src.opencv_pipeline.draw import (
    draw_fps,
    draw_probability_bar,
    draw_result,
)
from src.opencv_pipeline.io import batch_load, read_image, write_image
from src.opencv_pipeline.preprocess import PreprocessPipeline

__all__ = [
    # I/O
    "read_image",
    "write_image",
    "batch_load",
    # 预处理
    "PreprocessPipeline",
    # 数据增强
    "AugmentationPipeline",
    # 摄像头
    "CameraStream",
    "FPSCounter",
    # 结果绘制
    "draw_result",
    "draw_fps",
    "draw_probability_bar",
    # 数据集工具
    "scan_dataset",
    "detect_corrupted_images",
    "detect_blurry_images",
    "visualize_samples",
]
