"""
摄像头实时识别入口脚本

用法:
    D:\\anaconda\\python.exe scripts/infer_camera.py --checkpoint checkpoints/best_model.pth
    D:\\anaconda\\python.exe scripts/infer_camera.py --checkpoint checkpoints/best_model.pth --camera_id 0

按 Q 键退出
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch

from src.config import load_config
from src.models import build_model
from src.opencv_pipeline.camera import CameraStream, FPSCounter
from src.opencv_pipeline.draw import draw_result, draw_fps, draw_probability_bar
from src.opencv_pipeline.preprocess import resize
from torchvision import transforms


def parse_args():
    parser = argparse.ArgumentParser(description="垃圾智能分类 — 摄像头实时识别")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best_model.pth",
        help="模型权重文件路径"
    )
    parser.add_argument(
        "--config", type=str, default="configs/train.yaml",
        help="配置文件路径 (默认: configs/train.yaml)"
    )
    parser.add_argument(
        "--camera_id", type=int, default=0,
        help="摄像头设备 ID (默认: 0)"
    )
    parser.add_argument(
        "--confidence_threshold", type=float, default=0.5,
        help="置信度阈值，低于此值显示'未知' (默认: 0.5)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # 加载配置
    config = load_config(args.config)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"推理设备: {device}")

    # 加载 checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error(f"Checkpoint 不存在: {ckpt_path}")
        sys.exit(1)

    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if "config" in checkpoint:
        ckpt_config = checkpoint["config"]
        config.model.name = ckpt_config.get("model_name", config.model.name)
        config.model.num_classes = ckpt_config.get("num_classes", config.model.num_classes)
        config.model.input_size = ckpt_config.get("input_size", config.model.input_size)

    # 构建模型
    model = build_model(config)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()
    logger.info(f"模型加载完成: {config.model.name}")

    # 标签映射
    label_names = config.labels.names
    label_name_list = [label_names[i] for i in range(len(label_names))]

    # 推理 transform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config.model.input_size, config.model.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # 打开摄像头
    logger.info(f"打开摄像头 (ID={args.camera_id})...")
    cam = CameraStream(camera_id=args.camera_id)

    if not cam.is_opened:
        logger.error("无法打开摄像头，请检查设备连接")
        sys.exit(1)

    fps_counter = FPSCounter(window_size=30)

    logger.info("=" * 40)
    logger.info("摄像头实时识别已启动")
    logger.info("按 Q 键退出")
    logger.info("=" * 40)

    window_name = "Garbage Classification - Press Q to Quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            # 1. 读取帧
            frame = cam.read_frame()
            if frame is None:
                logger.warning("读取帧失败")
                break

            # 2. 复制一份用于显示
            display = frame.copy()

            # 3. OpenCV 预处理：resize → 归一化
            input_size = config.model.input_size
            frame_resized = cv2.resize(frame, (input_size, input_size))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

            # 4. 转 tensor → 模型推理
            tensor = transform(frame_rgb).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, pred_class = probs.max(dim=1)

            pred_class = pred_class.item()
            confidence = confidence.item()

            # 5. 构建概率分布
            prob_values = probs.cpu().numpy()[0]
            probabilities = {}
            for i, name in enumerate(label_name_list):
                probabilities[name] = float(prob_values[i])

            # 6. FPS
            fps_counter.tick()
            fps = fps_counter.get_fps()

            # 7. 绘制结果
            class_name = label_names[pred_class]
            if confidence < args.confidence_threshold:
                class_name = "识别中..."

            display = draw_result(display, class_name, confidence)
            display = draw_probability_bar(display, probabilities, label_name_list)
            display = draw_fps(display, fps)

            # 8. 显示
            cv2.imshow(window_name, display)

            # 9. 按 Q 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("用户按 Q 退出")
                break

    except KeyboardInterrupt:
        logger.info("键盘中断，退出")
    finally:
        cam.release()
        cv2.destroyAllWindows()
        logger.info("摄像头实时识别已停止")


if __name__ == "__main__":
    main()
