"""
评估入口脚本

用法:
    D:\\anaconda\\python.exe scripts/evaluate.py --config configs/train.yaml --checkpoint checkpoints/best_model.pth
    D:\\anaconda\\python.exe scripts/evaluate.py --checkpoint checkpoints/best_model.pth
    D:\\anaconda\\python.exe scripts/evaluate.py --checkpoint checkpoints/best_model.pth --split test
    D:\\anaconda\\python.exe scripts/evaluate.py --checkpoint checkpoints/best_model.pth --split val
"""

import argparse
import logging
import sys
from pathlib import Path

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.config import load_config
from src.dataset import create_dataloaders
from src.evaluate import (
    evaluate,
    save_confusion_matrix,
    save_evaluation_report,
    save_misclassified_samples,
)
from src.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="垃圾四分类模型评估")
    parser.add_argument(
        "--config", type=str, default="configs/train.yaml",
        help="训练配置文件路径 (默认: configs/train.yaml)"
    )
    parser.add_argument(
        "--base_config", type=str, default="configs/base.yaml",
        help="基础配置文件路径 (默认: configs/base.yaml)"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="模型权重文件路径 (如 checkpoints/best_model.pth)"
    )
    parser.add_argument(
        "--split", type=str, default="test",
        choices=["test", "val"],
        help="评估使用的数据集划分 (默认: test)"
    )
    parser.add_argument(
        "--max_misclassified", type=int, default=20,
        help="最多保存的错分样本数 (默认: 20)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="批大小覆盖"
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
    logger.info(f"加载配置: {args.config}")
    config = load_config(args.config, base_path=args.base_config)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 加载 checkpoint 获取模型信息
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error(f"Checkpoint 不存在: {ckpt_path}")
        sys.exit(1)

    logger.info(f"加载 checkpoint: {ckpt_path}")
    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    # 从 checkpoint 恢复模型配置
    if "config" in checkpoint:
        ckpt_config = checkpoint["config"]
        config.model.name = ckpt_config.get("model_name", config.model.name)
        config.model.num_classes = ckpt_config.get("num_classes", config.model.num_classes)
        config.model.input_size = ckpt_config.get("input_size", config.model.input_size)
        logger.info(f"从 checkpoint 恢复模型配置: {ckpt_config}")

    # 命令行覆盖
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size

    # 构建模型
    logger.info(f"构建模型: {config.model.name}")
    model = build_model(config)

    # 加载权重
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()

    logger.info(f"Checkpoint 信息: epoch={checkpoint.get('epoch', '?')}, "
                f"val_accuracy={checkpoint.get('val_accuracy', '?')}")

    # 构建数据加载器
    logger.info(f"加载 {args.split} 数据集...")
    phase = "test" if args.split == "test" else "train"  # train phase 同时返回 val
    dataloaders = create_dataloaders(config, phase=phase)

    if args.split == "test":
        dataloader = dataloaders["test"]
    else:
        dataloader = dataloaders["val"]

    logger.info(f"{args.split} 集: {len(dataloader.dataset)} 样本")

    # 标签名称映射
    label_names = config.labels.names

    # 执行评估
    logger.info("=" * 60)
    logger.info("开始评估...")
    logger.info("=" * 60)

    results = evaluate(model, dataloader, device, label_names)

    # 输出路径
    fig_dir = Path(config.output.output_dir) / "figures"
    report_dir = Path(config.output.output_dir) / "reports"
    model_name = config.model.name

    # 保存混淆矩阵
    cm_path = fig_dir / f"confusion_matrix_{model_name}.png"
    save_confusion_matrix(results["confusion_matrix"], label_names, cm_path, title=f"混淆矩阵 — {model_name}")

    # 保存错分样本
    mis_dir = fig_dir / "misclassified_samples"
    save_misclassified_samples(model, dataloader, device, label_names, mis_dir, max_samples=args.max_misclassified)

    # 保存评估报告
    report_path = report_dir / f"evaluation_report_{model_name}.md"
    save_evaluation_report(results, label_names, model_name, report_path)

    # 打印摘要
    logger.info("=" * 60)
    logger.info("评估完成！摘要:")
    logger.info(f"  模型: {model_name}")
    logger.info(f"  数据集: {args.split} ({len(dataloader.dataset)} 样本)")
    logger.info(f"  Overall Accuracy: {results['accuracy']:.4f}")
    logger.info(f"  Macro F1: {results['f1'].mean():.4f}")
    logger.info(f"  混淆矩阵: {cm_path}")
    logger.info(f"  错分样本: {mis_dir}")
    logger.info(f"  评估报告: {report_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
