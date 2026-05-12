"""
垃圾四分类数据集
PyTorch Dataset + DataLoader 工厂函数
支持 OpenCV 预处理管线（去噪 + CLAHE）
"""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.config import ProjectConfig
from src.opencv_pipeline.preprocess import denoise, clahe as clahe_fn

logger = logging.getLogger(__name__)

LABEL_MAP = {
    "recyclable": 0,
    "kitchen": 1,
    "hazardous": 2,
    "other": 3,
}

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


def get_train_transforms(input_size: int = 224, config: Optional[ProjectConfig] = None) -> transforms.Compose:
    """
    训练时数据增强 + 标准化

    Args:
        input_size: 输入图片尺寸
        config: 项目配置（用于读取增强参数）
    """
    transform_list = [
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
    ]

    if config is not None:
        aug = config.train.augmentation
        if aug.random_vertical_flip:
            transform_list.append(transforms.RandomVerticalFlip(p=0.5))
        if aug.random_rotation > 0:
            transform_list.append(transforms.RandomRotation(degrees=aug.random_rotation))
        if aug.color_jitter:
            cj = aug.color_jitter
            transform_list.append(transforms.ColorJitter(
                brightness=cj.get('brightness', 0),
                contrast=cj.get('contrast', 0),
                saturation=cj.get('saturation', 0),
            ))
    else:
        # 默认增强
        transform_list.extend([
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ])

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # RandomErasing 放在 ToTensor 之后
    if config is not None and config.train.augmentation.random_erasing:
        transform_list.append(transforms.RandomErasing(p=0.2))

    return transforms.Compose(transform_list)


def get_val_transforms(input_size: int = 224) -> transforms.Compose:
    """验证/测试时只做 resize + 标准化"""
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class WasteDataset(Dataset):
    """垃圾四分类数据集，支持 OpenCV 预处理管线"""

    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
        opencv_preprocess: Optional[Callable] = None,
    ):
        """
        Args:
            root_dir: 数据目录（如 data/split/train）
            transform: torchvision transforms
            opencv_preprocess: OpenCV 预处理函数（接收 BGR 图像，返回处理后的 BGR 图像）
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.opencv_preprocess = opencv_preprocess
        self.samples: List[Tuple[Path, int]] = []

        self._load_samples()

        if self.opencv_preprocess is not None:
            logger.info("OpenCV 预处理管线已启用（去噪 + CLAHE）")

    def _load_samples(self):
        """扫描目录，加载所有 (图片路径, 标签) 对"""
        self.samples = []

        for class_name, label in LABEL_MAP.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                logger.warning(f"类别目录不存在: {class_dir}")
                continue

            for img_path in class_dir.iterdir():
                if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((img_path, label))

        logger.info(f"从 {self.root_dir} 加载了 {len(self.samples)} 个样本")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        流程：
        1. OpenCV 读取图片（BGR）
        2. 可选：执行 opencv_preprocess
        3. BGR → RGB
        4. 转 PIL Image
        5. 执行 torchvision transform
        6. 返回 (image_tensor, label)
        """
        img_path, label = self.samples[idx]

        # OpenCV 读取（BGR）
        image = cv2.imread(str(img_path))
        if image is None:
            # 读取失败返回黑图
            image = np.zeros((224, 224, 3), dtype=np.uint8)
            logger.warning(f"无法读取图片: {img_path}")

        # 可选 OpenCV 预处理
        if self.opencv_preprocess is not None:
            image = self.opencv_preprocess(image)

        # BGR → RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # numpy → PIL
        from PIL import Image
        pil_image = Image.fromarray(image)

        # torchvision transform
        if self.transform is not None:
            image_tensor = self.transform(pil_image)
        else:
            image_tensor = transforms.ToTensor()(pil_image)

        return image_tensor, label


class OpenCVPreprocess:
    """可序列化的 OpenCV 预处理管线（支持 Windows 多进程）"""

    def __init__(self, config_dict: dict):
        self.do_denoise = config_dict.get("denoise", False)
        self.denoise_method = config_dict.get("denoise_method", "gaussian")
        self.denoise_ksize = config_dict.get("denoise_ksize", 3)
        self.do_clahe = config_dict.get("clahe", False)
        self.clahe_clip_limit = config_dict.get("clahe_clip_limit", 2.0)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """OpenCV 预处理：去噪 + CLAHE"""
        if self.do_denoise:
            image = denoise(image, method=self.denoise_method, ksize=self.denoise_ksize)
        if self.do_clahe:
            image = clahe_fn(image, clip_limit=self.clahe_clip_limit)
        return image


def build_opencv_preprocess(config: ProjectConfig) -> Optional[OpenCVPreprocess]:
    """
    根据配置构建 OpenCV 预处理管线

    Args:
        config: 项目配置

    Returns:
        OpenCVPreprocess 实例，如果未启用则返回 None
    """
    try:
        opencv_cfg = config.train.opencv_preprocess
    except AttributeError:
        return None

    if not isinstance(opencv_cfg, dict):
        return None

    if not opencv_cfg.get("enabled", False):
        return None

    return OpenCVPreprocess(opencv_cfg)


def create_dataloaders(
    config: ProjectConfig,
    phase: str = "train",
) -> Dict[str, DataLoader]:
    """
    根据配置创建 train/val/test DataLoader

    Args:
        config: 项目配置
        phase: 当前阶段，"train" 会同时返回 train + val DataLoader

    Returns:
        {"train": DataLoader, "val": DataLoader} 或 {"test": DataLoader}
    """
    split_dir = Path(config.data.split_dir)
    input_size = config.model.input_size
    batch_size = config.train.batch_size
    num_workers = config.data.num_workers

    dataloaders = {}

    if phase == "train":
        # 构建 OpenCV 预处理管线
        opencv_fn = build_opencv_preprocess(config)

        # 训练集（带增强 + OpenCV 预处理）
        train_dataset = WasteDataset(
            root_dir=str(split_dir / "train"),
            transform=get_train_transforms(input_size, config),
            opencv_preprocess=opencv_fn,
        )
        dataloaders["train"] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

        # 验证集（无增强，同样使用 OpenCV 预处理保持一致）
        val_dataset = WasteDataset(
            root_dir=str(split_dir / "val"),
            transform=get_val_transforms(input_size),
            opencv_preprocess=opencv_fn,
        )
        dataloaders["val"] = DataLoader(
            val_dataset,
            batch_size=config.val.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    elif phase == "test":
        # 测试集也使用 OpenCV 预处理
        opencv_fn = build_opencv_preprocess(config)
        test_dataset = WasteDataset(
            root_dir=str(split_dir / "test"),
            transform=get_val_transforms(input_size),
            opencv_preprocess=opencv_fn,
        )
        dataloaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return dataloaders
