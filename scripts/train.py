"""
训练入口脚本

用法:
    D:\\anaconda\\python.exe scripts/train.py --config configs/train.yaml
    D:\\anaconda\\python.exe scripts/train.py --config configs/train.yaml --model resnet18 --epochs 30 --lr 0.0005
"""

import argparse
import logging
import sys
from pathlib import Path

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.train import train


def parse_args():
    parser = argparse.ArgumentParser(description="垃圾四分类模型训练")
    parser.add_argument(
        "--config", type=str, default="configs/train.yaml",
        help="训练配置文件路径 (默认: configs/train.yaml)"
    )
    parser.add_argument(
        "--base_config", type=str, default="configs/base.yaml",
        help="基础配置文件路径 (默认: configs/base.yaml)"
    )
    # 命令行覆盖参数
    parser.add_argument("--model", type=str, default=None, help="模型名称覆盖")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数覆盖")
    parser.add_argument("--batch_size", type=int, default=None, help="批大小覆盖")
    parser.add_argument("--lr", type=float, default=None, help="学习率覆盖")
    parser.add_argument("--weight_decay", type=float, default=None, help="权重衰减覆盖")
    parser.add_argument("--optimizer", type=str, default=None, help="优化器覆盖")
    parser.add_argument("--num_workers", type=int, default=None, help="数据加载线程数覆盖")
    return parser.parse_args()


def apply_overrides(config, args):
    """将命令行参数覆盖到配置对象上"""
    if args.model is not None:
        config.model.name = args.model
    if args.epochs is not None:
        config.train.epochs = args.epochs
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
    if args.lr is not None:
        config.train.lr = args.lr
    if args.weight_decay is not None:
        config.train.weight_decay = args.weight_decay
    if args.optimizer is not None:
        config.train.optimizer = args.optimizer
    if args.num_workers is not None:
        config.data.num_workers = args.num_workers


def main():
    args = parse_args()

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 加载配置
    logger = logging.getLogger(__name__)
    logger.info(f"加载配置: {args.config} (base: {args.base_config})")
    config = load_config(args.config, base_path=args.base_config)

    # 命令行覆盖
    apply_overrides(config, args)

    logger.info("=" * 60)
    logger.info(f"项目: {config.name}")
    logger.info(f"模型: {config.model.name}")
    logger.info(f"Epochs: {config.train.epochs}")
    logger.info(f"Batch Size: {config.train.batch_size}")
    logger.info(f"学习率: {config.train.lr}")
    logger.info(f"优化器: {config.train.optimizer}")
    logger.info(f"两阶段训练: {config.train.two_stage}")
    logger.info(f"数据目录: {config.data.split_dir}")
    logger.info("=" * 60)

    # 开始训练
    result = train(config)

    logger.info("训练结果摘要:")
    logger.info(f"  最佳 Epoch: {result['best_epoch']}")
    logger.info(f"  最佳 Val Accuracy: {result['best_val_accuracy']:.4f}")
    logger.info(f"  总耗时: {result['total_time']:.1f}s")


if __name__ == "__main__":
    main()
