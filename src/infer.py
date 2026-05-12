"""
推理引擎
提供垃圾分类推理器，支持单图推理和批量推理
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from src.config import load_config, ProjectConfig
from src.models import build_model
from src.opencv_pipeline.draw import draw_probability_bar, draw_result
from src.opencv_pipeline.io import read_image, write_image
from src.opencv_pipeline.preprocess import resize

logger = logging.getLogger(__name__)


class WasteClassifier:
    """垃圾分类推理器"""

    def __init__(
        self,
        config_path: str = "configs/train.yaml",
        checkpoint_path: str = "checkpoints/best_model.pth",
        device: str = "auto",
    ):
        """
        Args:
            config_path: 配置文件路径
            checkpoint_path: 模型权重路径
            device: 计算设备，'auto' 自动选择，'cpu' 或 'cuda'
        """
        # 1. 加载配置
        self.config = load_config(config_path)

        # 2. 设备
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        logger.info(f"推理设备: {self.device}")

        # 3. 加载 checkpoint 获取模型信息
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint 不存在: {ckpt_path}")

        checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if "config" in checkpoint:
            ckpt_config = checkpoint["config"]
            self.config.model.name = ckpt_config.get("model_name", self.config.model.name)
            self.config.model.num_classes = ckpt_config.get("num_classes", self.config.model.num_classes)
            self.config.model.input_size = ckpt_config.get("input_size", self.config.model.input_size)

        # 4. 构建模型并加载权重
        self.model = build_model(self.config)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info(f"模型加载完成: {self.config.model.name}")

        # 5. 标签映射
        self.label_names = self.config.labels.names  # {0: "可回收物", ...}
        self.label_name_list = [self.label_names[i] for i in range(len(self.label_names))]

        # 6. 推理时的 transform（与验证集一致）
        self.transform = transforms.Compose([
            transforms.Resize((self.config.model.input_size, self.config.model.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def predict_image(
        self,
        image_path_or_array: Union[str, np.ndarray],
    ) -> Dict:
        """
        单图推理

        Args:
            image_path_or_array: 图片路径或 BGR numpy 数组

        Returns:
            {
                'class_name': '可回收物',
                'class_id': 0,
                'confidence': 0.95,
                'probabilities': {'可回收物': 0.95, '厨余垃圾': 0.03, ...}
            }
        """
        # 1. OpenCV 读取图片
        if isinstance(image_path_or_array, str):
            image = cv2.imread(image_path_or_array)
            if image is None:
                raise ValueError(f"无法读取图片: {image_path_or_array}")
        else:
            image = image_path_or_array.copy()

        # 2. OpenCV 预处理：resize
        image_resized = resize(image, (self.config.model.input_size, self.config.model.input_size))

        # 3. BGR → RGB → PIL → tensor
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        from PIL import Image
        pil_image = Image.fromarray(image_rgb)
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        # 4. 模型推理
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred_class = probs.max(dim=1)

        pred_class = pred_class.item()
        confidence = confidence.item()

        # 5. 构建概率分布
        prob_values = probs.cpu().numpy()[0]
        probabilities = {}
        for i, name in enumerate(self.label_name_list):
            probabilities[name] = float(prob_values[i])

        return {
            "class_name": self.label_names[pred_class],
            "class_id": pred_class,
            "confidence": confidence,
            "probabilities": probabilities,
        }

    def predict_batch(
        self,
        image_paths: List[str],
        batch_size: int = 32,
    ) -> List[Dict]:
        """
        批量推理

        Args:
            image_paths: 图片路径列表
            batch_size: 批大小

        Returns:
            预测结果列表，每个元素与 predict_image 返回格式相同
        """
        results = []

        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start:start + batch_size]
            tensors = []

            for path in batch_paths:
                image = cv2.imread(path)
                if image is None:
                    image = np.zeros((224, 224, 3), dtype=np.uint8)
                image_resized = resize(image, (self.config.model.input_size, self.config.model.input_size))
                image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
                from PIL import Image
                pil_image = Image.fromarray(image_rgb)
                tensors.append(self.transform(pil_image))

            batch_tensor = torch.stack(tensors).to(self.device)

            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probs = torch.softmax(outputs, dim=1)
                confidences, pred_classes = probs.max(dim=1)

            for i in range(len(batch_paths)):
                pred_class = pred_classes[i].item()
                confidence = confidences[i].item()
                prob_values = probs[i].cpu().numpy()

                probabilities = {}
                for j, name in enumerate(self.label_name_list):
                    probabilities[name] = float(prob_values[j])

                results.append({
                    "image_path": batch_paths[i],
                    "class_name": self.label_names[pred_class],
                    "class_id": pred_class,
                    "confidence": confidence,
                    "probabilities": probabilities,
                })

        return results

    def predict_and_annotate(
        self,
        image_path: str,
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        单图推理并在图像上绘制结果（分类标签 + 概率条形图）

        Args:
            image_path: 图片路径
            save_path: 保存路径（None 则不保存）

        Returns:
            标注后的 BGR 图像
        """
        result = self.predict_image(image_path)
        image = cv2.imread(image_path)

        # 绘制分类结果
        image = draw_result(image, result["class_name"], result["confidence"])

        # 绘制概率条形图
        image = draw_probability_bar(image, result["probabilities"], self.label_name_list)

        # 保存
        if save_path is not None:
            write_image(save_path, image)
            logger.info(f"标注图已保存: {save_path}")

        return image
