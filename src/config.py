"""
配置加载器：读取 YAML 配置，返回 dataclass 实例
支持 base + 覆盖配置合并
"""

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass
class DataConfig:
    raw_dir: str = "data/trash_jpg/trash_jpg"
    processed_dir: str = "data/processed"
    split_dir: str = "data/split"
    meta_dir: str = "data/meta"
    num_workers: int = 4


@dataclass
class ModelConfig:
    name: str = "mobilenet_v3_small"
    num_classes: int = 4
    pretrained: bool = True
    freeze_backbone: bool = True
    input_size: int = 224


@dataclass
class LabelConfig:
    names: dict = field(default_factory=lambda: {
        0: "可回收物", 1: "厨余垃圾", 2: "有害垃圾", 3: "其他垃圾"
    })
    label_map: dict = field(default_factory=lambda: {
        "recyclable": 0, "kitchen": 1, "hazardous": 2, "other": 3
    })


@dataclass
class OutputConfig:
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    output_dir: str = "outputs"


@dataclass
class AugmentationConfig:
    random_horizontal_flip: bool = True
    random_vertical_flip: bool = True
    random_rotation: int = 15
    color_jitter: dict = field(default_factory=lambda: {
        "brightness": 0.2, "contrast": 0.2, "saturation": 0.2
    })
    random_erasing: bool = True
    gaussian_noise: bool = False


@dataclass
class EarlyStoppingConfig:
    patience: int = 10
    min_delta: float = 0.001


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 32
    lr: float = 0.001
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_epochs: int = 3
    two_stage: bool = True
    stage1_epochs: int = 15
    stage2_lr_factor: float = 0.1
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    opencv_preprocess: dict = field(default_factory=lambda: {
        "enabled": False,
        "denoise": True,
        "denoise_method": "gaussian",
        "denoise_ksize": 3,
        "clahe": True,
        "clahe_clip_limit": 2.0,
    })


@dataclass
class ValConfig:
    batch_size: int = 32


@dataclass
class InferConfig:
    checkpoint: str = "checkpoints/best_model.pth"
    device: str = "cpu"
    camera_id: int = 0
    confidence_threshold: float = 0.5


@dataclass
class ProjectConfig:
    """项目顶层配置"""
    name: str = "基于OpenCV的垃圾智能分类系统"
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    val: ValConfig = field(default_factory=ValConfig)
    infer: InferConfig = field(default_factory=InferConfig)


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并两个字典，override 中的值覆盖 base"""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _dict_to_dataclass(cls, data: dict):
    """将字典递归转换为 dataclass 实例"""
    if not isinstance(data, dict):
        return data

    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}

    for key, value in data.items():
        if key in field_types:
            field_type = field_types[key]
            # 检查字段类型是否是 dataclass
            if isinstance(value, dict) and hasattr(field_type, '__dataclass_fields__'):
                kwargs[key] = _dict_to_dataclass(field_type, value)
            else:
                # 基本类型自动转换（PyYAML 有时将 1e-4 解析为字符串）
                if isinstance(value, str) and field_type in (float, int):
                    try:
                        value = field_type(value)
                    except (ValueError, TypeError):
                        pass
                kwargs[key] = value

    return cls(**kwargs)


def load_config(config_path: str, base_path: str = "configs/base.yaml") -> ProjectConfig:
    """
    加载配置文件，与 base.yaml 合并后返回 ProjectConfig

    Args:
        config_path: 覆盖配置文件路径（如 configs/train.yaml）
        base_path: 基础配置文件路径

    Returns:
        ProjectConfig 实例
    """
    config_path = Path(config_path)
    base_path = Path(base_path)

    # 读取 base 配置
    base_data = {}
    if base_path.exists():
        with open(base_path, 'r', encoding='utf-8') as f:
            base_data = yaml.safe_load(f) or {}

    # 读取覆盖配置
    override_data = {}
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            override_data = yaml.safe_load(f) or {}

    # 合并配置
    merged = _deep_merge(base_data, override_data)

    return _dict_to_dataclass(ProjectConfig, merged)


def seed_everything(seed: int):
    """固定所有随机种子，确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
