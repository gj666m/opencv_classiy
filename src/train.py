"""
训练引擎
支持两阶段训练、EarlyStopping、日志记录、训练曲线保存
"""

import csv
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader

from src.config import ProjectConfig
from src.dataset import create_dataloaders
from src.models import build_model

logger = logging.getLogger(__name__)


class EarlyStopping:
    """EarlyStopping 回调"""

    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def __call__(self, val_accuracy: float) -> bool:
        score = val_accuracy
        if self.best_score is None:
            self.best_score = score
            return False

        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            logger.info(
                f"EarlyStopping: {self.counter}/{self.patience} "
                f"(best={self.best_score:.4f}, current={score:.4f})"
            )
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """
    训练一个 epoch

    Returns:
        {"loss": avg_loss, "accuracy": avg_accuracy}
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # 前向
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 50 == 0:
            batch_acc = 100.0 * correct / total
            logger.info(
                f"  [{batch_idx + 1}/{len(dataloader)}] "
                f"loss={loss.item():.4f} acc={batch_acc:.2f}%"
            )

    avg_loss = running_loss / total
    avg_acc = correct / total
    return {"loss": avg_loss, "accuracy": avg_acc}


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    验证

    Returns:
        {"loss": avg_loss, "accuracy": avg_accuracy}
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    avg_acc = correct / total
    return {"loss": avg_loss, "accuracy": avg_acc}


def _build_optimizer(
    model: nn.Module,
    config: ProjectConfig,
    lr: Optional[float] = None,
) -> torch.optim.Optimizer:
    """构建优化器"""
    effective_lr = lr if lr is not None else config.train.lr
    params = filter(lambda p: p.requires_grad, model.parameters())

    if config.train.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=effective_lr, weight_decay=config.train.weight_decay)
    elif config.train.optimizer == "sgd":
        return torch.optim.SGD(params, lr=effective_lr, momentum=0.9, weight_decay=config.train.weight_decay)
    else:
        raise ValueError(f"未知优化器: {config.train.optimizer}")


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: ProjectConfig,
    num_epochs: int,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """构建学习率调度器"""
    if config.train.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif config.train.scheduler == "step":
        return StepLR(optimizer, step_size=10, gamma=0.1)
    elif config.train.scheduler == "plateau":
        return ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=5)
    return None


def _save_training_curve(
    history: list,
    save_path: Path,
    model_name: str,
):
    """保存训练曲线图（Loss + Accuracy + LR 三合一）"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_accuracy"] for h in history]
    val_acc = [h["val_accuracy"] for h in history]
    lr_vals = [h["lr"] for h in history]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

    # Loss 曲线
    ax1.plot(epochs, train_loss, "b-o", markersize=2, label="Train Loss")
    ax1.plot(epochs, val_loss, "r-o", markersize=2, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{model_name} - Loss Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy 曲线
    ax2.plot(epochs, train_acc, "b-o", markersize=2, label="Train Accuracy")
    ax2.plot(epochs, val_acc, "r-o", markersize=2, label="Val Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"{model_name} - Accuracy Curve")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # LR 曲线
    ax3.plot(epochs, lr_vals, "g-o", markersize=2, label="Learning Rate")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Learning Rate")
    ax3.set_title(f"{model_name} - LR Schedule")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    # 用科学计数法显示 LR
    ax3.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"训练曲线已保存: {save_path}")


def _save_log_csv(
    history: list,
    save_path: Path,
):
    """保存训练日志到 CSV"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    fieldnames = history[0].keys()
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    logger.info(f"训练日志已保存: {save_path}")


def _save_config_snapshot(
    config: ProjectConfig,
    save_path: Path,
    total_params: int,
    trainable_params: int,
):
    """保存训练配置快照到 logs/，用于实验追溯"""
    import yaml
    from dataclasses import asdict

    snapshot = {
        "model_name": config.model.name,
        "num_classes": config.model.num_classes,
        "input_size": config.model.input_size,
        "pretrained": config.model.pretrained,
        "freeze_backbone": config.model.freeze_backbone,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "train": {
            "epochs": config.train.epochs,
            "batch_size": config.train.batch_size,
            "lr": config.train.lr,
            "weight_decay": config.train.weight_decay,
            "optimizer": config.train.optimizer,
            "scheduler": config.train.scheduler,
            "warmup_epochs": config.train.warmup_epochs,
            "two_stage": config.train.two_stage,
            "stage1_epochs": config.train.stage1_epochs,
            "stage2_lr_factor": config.train.stage2_lr_factor,
            "early_stopping_patience": config.train.early_stopping.patience,
        },
        "augmentation": {
            "random_horizontal_flip": config.train.augmentation.random_horizontal_flip,
            "random_vertical_flip": config.train.augmentation.random_vertical_flip,
            "random_rotation": config.train.augmentation.random_rotation,
            "random_erasing": config.train.augmentation.random_erasing,
        },
    }

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(snapshot, f, default_flow_style=False, allow_unicode=True)
    logger.info(f"配置快照已保存: {save_path}")


def _log_gpu_info():
    """记录 GPU 环境信息"""
    logger.info("=" * 40 + " 环境信息 " + "=" * 40)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_mem = props.total_memory / (1024 ** 3)
            logger.info(
                f"GPU {i}: {props.name} | "
                f"显存: {total_mem:.1f} GB | "
                f"CUDA Compute: {props.major}.{props.minor}"
            )
        logger.info(f"CUDA 版本: {torch.version.cuda}")
        logger.info(f"PyTorch 版本: {torch.__version__}")
    else:
        logger.info("未检测到 GPU，使用 CPU 训练")
    logger.info("=" * 90)


def train(config: ProjectConfig) -> dict:
    """
    完整训练流程

    Args:
        config: 项目配置

    Returns:
        训练历史记录和最佳指标
    """
    from src.config import seed_everything

    seed_everything(config.seed)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 路径
    ckpt_dir = Path(config.output.checkpoint_dir)
    log_dir = Path(config.output.log_dir)
    fig_dir = Path(config.output.output_dir) / "figures"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # 数据加载
    logger.info("加载数据集...")
    dataloaders = create_dataloaders(config, phase="train")
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    logger.info(f"训练集: {len(train_loader.dataset)} 样本, 验证集: {len(val_loader.dataset)} 样本")

    # 构建模型
    logger.info(f"构建模型: {config.model.name}")
    model = build_model(config)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"总参数量: {total_params:,}, 可训练参数量: {trainable_params:,}")

    # 记录 GPU 环境信息
    _log_gpu_info()

    # 保存训练配置快照
    config_snapshot_path = log_dir / f"config_snapshot_{config.model.name}.yaml"
    _save_config_snapshot(config, config_snapshot_path, total_params, trainable_params)

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 训练配置
    num_epochs = config.train.epochs
    two_stage = config.train.two_stage
    stage1_epochs = config.train.stage1_epochs if two_stage else num_epochs

    # 初始阶段优化器和调度器
    optimizer = _build_optimizer(model, config)
    scheduler = _build_scheduler(optimizer, config, num_epochs)

    # EarlyStopping
    early_stopper = EarlyStopping(
        patience=config.train.early_stopping.patience,
        min_delta=config.train.early_stopping.min_delta,
    )

    # 训练循环
    history = []
    best_val_acc = 0.0
    best_epoch = 0

    logger.info(f"开始训练: 共 {num_epochs} epochs, 两阶段={two_stage}")
    if two_stage:
        logger.info(f"  阶段一: Epoch 1-{stage1_epochs} (冻结backbone)")
        logger.info(f"  阶段二: Epoch {stage1_epochs + 1}-{num_epochs} (解冻backbone)")

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # 两阶段切换
        if two_stage and epoch == stage1_epochs + 1:
            logger.info("=" * 50)
            logger.info(f"进入阶段二 (Epoch {epoch}): 解冻 backbone")
            logger.info("=" * 50)
            if hasattr(model, "unfreeze_backbone"):
                model.unfreeze_backbone()

            # 重建优化器（新学习率）
            stage2_lr = config.train.lr * config.train.stage2_lr_factor
            optimizer = _build_optimizer(model, config, lr=stage2_lr)
            scheduler = _build_scheduler(optimizer, config, num_epochs - stage1_epochs)
            logger.info(f"阶段二学习率: {stage2_lr}")

            trainable_now = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"解冻后可训练参数量: {trainable_now:,}")

        # 训练
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # 验证
        val_metrics = validate(model, val_loader, criterion, device)

        # 学习率调度
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics["accuracy"])
            else:
                scheduler.step()

        # 记录当前学习率
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start

        # 日志
        log_entry = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 4),
            "train_accuracy": round(train_metrics["accuracy"], 4),
            "val_loss": round(val_metrics["loss"], 4),
            "val_accuracy": round(val_metrics["accuracy"], 4),
            "lr": current_lr,
            "time": round(epoch_time, 2),
            "model": config.model.name,
            "total_params": total_params,
            "trainable_params": trainable_params if epoch == 1 or (two_stage and epoch == config.train.stage1_epochs + 1) else history[-1].get("trainable_params", trainable_params),
        }
        history.append(log_entry)

        logger.info(
            f"Epoch {epoch}/{num_epochs} ({epoch_time:.1f}s) | "
            f"Train Loss={train_metrics['loss']:.4f} Acc={train_metrics['accuracy']:.4f} | "
            f"Val Loss={val_metrics['loss']:.4f} Acc={val_metrics['accuracy']:.4f} | "
            f"LR={current_lr:.6f}"
        )

        # 保存最佳模型
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch
            best_path = ckpt_dir / f"best_{config.model.name}.pth"
            torch.save({
                "epoch": epoch,
                "model_name": config.model.name,
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": best_val_acc,
                "val_loss": val_metrics["loss"],
                "config": {
                    "model_name": config.model.name,
                    "num_classes": config.model.num_classes,
                    "input_size": config.model.input_size,
                },
            }, str(best_path))
            logger.info(f"  -> 保存最佳模型 (val_acc={best_val_acc:.4f})")

        # 每阶段结束保存最后一个 checkpoint
        if epoch == stage1_epochs or epoch == num_epochs:
            stage_path = ckpt_dir / f"checkpoint_epoch{epoch}.pth"
            torch.save({
                "epoch": epoch,
                "model_name": config.model.name,
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
            }, str(stage_path))

        # EarlyStopping 检查
        if early_stopper(val_metrics["accuracy"]):
            logger.info(f"EarlyStopping 触发，在 Epoch {epoch} 停止训练")
            break

    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"训练完成! 总耗时: {total_time:.1f}s")
    logger.info(f"最佳模型: Epoch {best_epoch}, Val Accuracy={best_val_acc:.4f}")
    logger.info("=" * 60)

    # 保存训练日志（每个模型单独一份 + 合并汇总）
    log_csv_per_model = log_dir / f"train_log_{config.model.name}.csv"
    _save_log_csv(history, log_csv_per_model)

    # 追加到合并日志（所有模型汇总）
    log_csv_all = log_dir / "train_log.csv"
    _save_log_csv(history, log_csv_all)

    # 保存训练曲线
    fig_path = fig_dir / f"training_curve_{config.model.name}.png"
    _save_training_curve(history, fig_path, config.model.name)

    return {
        "history": history,
        "best_val_accuracy": best_val_acc,
        "best_epoch": best_epoch,
        "total_time": total_time,
    }
