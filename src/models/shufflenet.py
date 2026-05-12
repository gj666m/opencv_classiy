"""
ShuffleNetV2 模型
基于 torchvision 预训练权重，替换全连接层为 4 类垃圾分类
"""

import logging

import torch
import torch.nn as nn
import torchvision.models as models

from src.models.base import register_model

logger = logging.getLogger(__name__)


@register_model("shufflenet_v2")
class ShuffleNetV2(nn.Module):
    """
    ShuffleNetV2-x1.0 垃圾分类模型

    参数量: ~2.3M
    分类头替换位置: model.fc
    """

    def __init__(self, config):
        super().__init__()

        # 加载预训练模型
        weights = models.ShuffleNet_V2_X1_0_Weights.DEFAULT if config.model.pretrained else None
        self.backbone = models.shufflenet_v2_x1_0(weights=weights)

        # 获取全连接层输入维度
        in_features = self.backbone.fc.in_features  # 1024

        # 替换分类头
        self.backbone.fc = nn.Linear(in_features, config.model.num_classes)

        # 冻结 backbone
        if config.model.freeze_backbone:
            self._freeze_backbone()

        self._freeze_backbone_flag = config.model.freeze_backbone
        logger.info(
            f"ShuffleNetV2 初始化完成 "
            f"(pretrained={config.model.pretrained}, "
            f"freeze_backbone={config.model.freeze_backbone})"
        )

    def _freeze_backbone(self):
        """冻结除 fc 外的所有参数"""
        for name, param in self.backbone.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
        logger.info("Backbone 已冻结")

    def unfreeze_backbone(self):
        """解冻 backbone，用于第二阶段微调"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self._freeze_backbone_flag = False
        logger.info("Backbone 已解冻")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 (batch, 3, 224, 224)

        Returns:
            logits (batch, num_classes)
        """
        return self.backbone(x)
