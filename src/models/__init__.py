"""
模型模块
通过注册表机制统一管理所有候选模型
"""

# 导入基类和注册机制
from src.models.base import MODEL_REGISTRY, build_model, list_models, register_model

# 导入各模型实现（触发 @register_model 装饰器注册）
from src.models.efficientnet import EfficientNetB0
from src.models.mobilenet_v3 import MobileNetV3Small
from src.models.resnet import ResNet18
from src.models.shufflenet import ShuffleNetV2

__all__ = [
    "MODEL_REGISTRY",
    "build_model",
    "list_models",
    "register_model",
    "MobileNetV3Small",
    "ResNet18",
    "EfficientNetB0",
    "ShuffleNetV2",
]
