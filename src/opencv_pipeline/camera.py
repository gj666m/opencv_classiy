"""
OpenCV 摄像头模块
提供摄像头视频流管理和 FPS 计数功能
"""

import logging
import time
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FPSCounter:
    """
    FPS 计数器
    使用滑动窗口计算实时帧率
    """

    def __init__(self, window_size: int = 30):
        """
        Args:
            window_size: 滑动窗口大小，取最近 N 帧计算平均 FPS
        """
        self.window_size = window_size
        self.timestamps: list = []
        self._fps: float = 0.0

    def tick(self):
        """每处理一帧调用一次，记录当前时间戳"""
        now = time.time()
        self.timestamps.append(now)

        # 只保留最近 window_size 个时间戳
        if len(self.timestamps) > self.window_size:
            self.timestamps = self.timestamps[-self.window_size:]

    def get_fps(self) -> float:
        """
        返回当前 FPS（基于最近 window_size 帧的平均值）

        Returns:
            当前帧率
        """
        if len(self.timestamps) < 2:
            return 0.0

        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed <= 0:
            return 0.0

        self._fps = (len(self.timestamps) - 1) / elapsed
        return self._fps

    def reset(self):
        """重置计数器"""
        self.timestamps.clear()
        self._fps = 0.0


class CameraStream:
    """
    摄像头视频流管理

    用法:
        cam = CameraStream(0)
        while True:
            frame = cam.read_frame()
            if frame is None:
                break
            # 处理帧...
        cam.release()
    """

    def __init__(
        self,
        camera_id: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None
    ):
        """
        Args:
            camera_id: 摄像头设备 ID（默认 0 为系统默认摄像头）
            width: 期望的画面宽度（None 表示使用摄像头默认值）
            height: 期望的画面高度
            fps: 期望的帧率
        """
        self.camera_id = camera_id
        self.cap: Optional[cv2.VideoCapture] = None
        self.fps_counter = FPSCounter()
        self._is_opened = False

        self._open(width, height, fps)

    def _open(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None
    ):
        """打开摄像头并设置参数"""
        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            logger.error(f"无法打开摄像头 (ID={self.camera_id})")
            self._is_opened = False
            return

        self._is_opened = True

        # 设置分辨率
        if width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps is not None:
            self.cap.set(cv2.CAP_PROP_FPS, fps)

        # 读取实际参数
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        logger.info(f"摄像头已打开: {actual_w}x{actual_h} @ {actual_fps:.1f}fps")

    @property
    def is_opened(self) -> bool:
        """摄像头是否已打开"""
        return self._is_opened and self.cap is not None and self.cap.isOpened()

    def read_frame(self) -> Optional[np.ndarray]:
        """
        读取一帧

        Returns:
            BGR numpy array，读取失败返回 None
        """
        if not self.is_opened:
            logger.warning("摄像头未打开")
            return None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            logger.warning("读取帧失败")
            return None

        self.fps_counter.tick()
        return frame

    def get_resolution(self) -> Tuple[int, int]:
        """
        获取当前摄像头分辨率

        Returns:
            (width, height)
        """
        if not self.is_opened:
            return (0, 0)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    def release(self):
        """释放摄像头资源"""
        if self.cap is not None:
            self.cap.release()
            self._is_opened = False
            logger.info("摄像头已释放")

    def __del__(self):
        self.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False
