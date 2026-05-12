"""
单图识别入口脚本

用法:
    D:\\anaconda\\python.exe scripts/infer_image.py --image test.jpg --checkpoint checkpoints/best_model.pth
    D:\\anaconda\\python.exe scripts/infer_image.py --image test.jpg --checkpoint checkpoints/best_model.pth --save
    D:\\anaconda\\python.exe scripts/infer_image.py --image test.jpg --checkpoint checkpoints/best_model.pth --config configs/train.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.infer import WasteClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="垃圾智能分类 — 单图识别")
    parser.add_argument(
        "--image", type=str, required=True,
        help="待识别的图片路径"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best_model.pth",
        help="模型权重文件路径 (默认: checkpoints/best_model.pth)"
    )
    parser.add_argument(
        "--config", type=str, default="configs/train.yaml",
        help="配置文件路径 (默认: configs/train.yaml)"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="保存标注图到 outputs/predictions/"
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

    # 检查图片是否存在
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"错误: 图片不存在: {image_path}")
        sys.exit(1)

    # 初始化推理器
    print(f"加载模型: {args.checkpoint}")
    classifier = WasteClassifier(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
    )

    # 执行推理
    print(f"\n识别图片: {image_path.name}")
    print("-" * 40)

    result = classifier.predict_image(str(image_path))

    # 输出结果
    print(f"分类结果: {result['class_name']}")
    print(f"类别编号: {result['class_id']}")
    print(f"置信度:   {result['confidence']:.4f} ({result['confidence']*100:.1f}%)")
    print(f"\n各类别概率:")
    for name, prob in result["probabilities"].items():
        bar = "█" * int(prob * 30)
        print(f"  {name:>6s}: {prob:.4f} ({prob*100:5.1f}%) {bar}")

    # 保存标注图
    if args.save:
        output_dir = Path("outputs/predictions")
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"pred_{image_path.stem}_{result['class_name']}{image_path.suffix}"
        classifier.predict_and_annotate(str(image_path), str(save_path))
        print(f"\n标注图已保存: {save_path}")


if __name__ == "__main__":
    main()
