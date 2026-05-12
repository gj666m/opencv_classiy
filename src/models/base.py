"""
模型注册基类
提供全局模型注册表和统一的模型构建接口
"""

import logging
from typing import Type

import torch.nn as nn

logger = logging.getLogger(__name__)

# 全局模型注册表
MODEL_REGISTRY: dict[str, Type[nn.Module]] = {}


def register_model(name: str):
    """
    模型注册装饰器

    用法:
        @register_model("mobilenet_v3_small")
        class MobileNetV3Small(nn.Module):
            ...
    """
    def decorator(cls: Type[nn.Module]):
        if name in MODEL_REGISTRY:
            logger.warning(f"模型 '{name}' 已存在，将被覆盖")
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def build_model(config) -> nn.Module:
    """
    根据配置构建模型

    Args:
        config: ProjectConfig 实例，需包含 config.model.name 和 config.model.num_classes

    Returns:
        构建好的模型实例

    Raises:
        ValueError: 未知的模型名称
    """
    name = config.model.name
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"未知的模型: '{name}'，可选: {list(MODEL_REGISTRY.keys())}"
        )
    logger.info(f"构建模型: {name}")
    return MODEL_REGISTRY[name](config)


def list_models() -> list[str]:
    """返回所有已注册的模型名称"""
    return list(MODEL_REGISTRY.keys())
