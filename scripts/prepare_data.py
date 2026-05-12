"""
数据整理脚本：将原始 245 类数据整理为四分类，并划分 train/val/test

功能：
1. 扫描 raw/ 下所有子文件夹
2. 根据文件夹名前缀映射到四大类
3. 将图片复制到 processed/{四大类}/ 下
4. 统计各类样本数
5. 按 8:1:1 划分 train/val/test
6. 保存 meta 信息

用法:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --raw_dir data/trash_jpg/trash_jpg --ratio 0.8 0.1 0.1
"""

import argparse
import json
import logging
import shutil
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}

# 文件夹名前缀 → 四大类英文名
PREFIX_MAP = {
    "可回收物": "recyclable",
    "厨余垃圾": "kitchen",
    "有害垃圾": "hazardous",
    "其他垃圾": "other",
}

LABEL_MAP = {
    "recyclable": 0,
    "kitchen": 1,
    "hazardous": 2,
    "other": 3,
}

LABEL_NAMES = {
    0: "可回收物",
    1: "厨余垃圾",
    2: "有害垃圾",
    3: "其他垃圾",
}


def parse_args():
    parser = argparse.ArgumentParser(description="数据整理与划分")
    parser.add_argument("--raw_dir", type=str, default="data/trash_jpg/trash_jpg",
                        help="原始数据集目录")
    parser.add_argument("--processed_dir", type=str, default="data/processed",
                        help="整理后的四分类数据目录")
    parser.add_argument("--split_dir", type=str, default="data/split",
                        help="划分后的数据目录")
    parser.add_argument("--meta_dir", type=str, default="data/meta",
                        help="元信息保存目录")
    parser.add_argument("--ratio", type=float, nargs=3, default=[0.8, 0.1, 0.1],
                        help="train/val/test 比例")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--copy", action="store_true", default=True,
                        help="复制文件（默认），否则使用符号链接")
    return parser.parse_args()


def merge_to_four_classes(raw_dir: Path, processed_dir: Path, meta_dir: Path) -> dict:
    """
    将 245 个子文件夹合并为 4 大类

    Returns:
        class_distribution: {大类英文名: 图片数量}
    """
    import random
    random.seed(42)

    logger.info(f"开始合并数据: {raw_dir} → {processed_dir}")

    class_distribution = defaultdict(int)
    class_details = defaultdict(list)  # 记录每个大类下有哪些子类
    skipped = []

    for subdir in sorted(raw_dir.iterdir()):
        if not subdir.is_dir():
            continue

        folder_name = subdir.name
        # 根据下划线分割，取前缀
        parts = folder_name.split("_", 1)
        prefix = parts[0] if parts else folder_name

        if prefix not in PREFIX_MAP:
            skipped.append(folder_name)
            logger.warning(f"跳过无法映射的文件夹: {folder_name}")
            continue

        class_name = PREFIX_MAP[prefix]
        target_dir = processed_dir / class_name
        target_dir.mkdir(parents=True, exist_ok=True)

        # 复制该子文件夹下所有图片
        count = 0
        for img_file in sorted(subdir.iterdir()):
            if not img_file.is_file():
                continue
            if img_file.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            # 目标文件名：子类名_原文件名，避免重名
            target_name = f"{folder_name}_{img_file.name}"
            target_path = target_dir / target_name
            shutil.copy2(str(img_file), str(target_path))
            count += 1

        class_distribution[class_name] += count
        class_details[class_name].append(folder_name)
        logger.info(f"  {prefix} → {class_name}: {folder_name} ({count} 张)")

    logger.info(f"合并完成:")
    for cls_name, count in sorted(class_distribution.items()):
        logger.info(f"  {LABEL_NAMES[LABEL_MAP[cls_name]]}({cls_name}): {count} 张")

    if skipped:
        logger.warning(f"跳过了 {len(skipped)} 个无法映射的文件夹: {skipped}")

    # 保存元信息
    meta_dir.mkdir(parents=True, exist_ok=True)

    with open(meta_dir / "class_distribution.json", 'w', encoding='utf-8') as f:
        json.dump({
            "distribution": dict(class_distribution),
            "total": sum(class_distribution.values()),
            "details": {k: v for k, v in class_details.items()}
        }, f, ensure_ascii=False, indent=2)

    with open(meta_dir / "label_map.json", 'w', encoding='utf-8') as f:
        json.dump({
            "label_map": LABEL_MAP,
            "label_names": LABEL_NAMES
        }, f, ensure_ascii=False, indent=2)

    return dict(class_distribution)


def split_dataset(
    processed_dir: Path,
    split_dir: Path,
    meta_dir: Path,
    ratio: list,
    seed: int
):
    """
    按 ratio 划分 train/val/test

    采用按类别分层抽样，保证各子集类别比例一致
    """
    import random
    random.seed(seed)

    train_ratio, val_ratio, test_ratio = ratio
    logger.info(f"开始划分数据集: ratio={ratio}, seed={seed}")

    split_info = {}
    total_stats = {"train": defaultdict(int), "val": defaultdict(int), "test": defaultdict(int)}

    for class_dir in sorted(processed_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        images = sorted([
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ])

        # 打乱
        random.shuffle(images)

        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val  # 剩余全部给 test

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        split_info[class_name] = {
            "total": n,
            "train": n_train,
            "val": n_val,
            "test": n_test,
        }

        for phase, file_list in splits.items():
            target_dir = split_dir / phase / class_name
            target_dir.mkdir(parents=True, exist_ok=True)

            for src_path in file_list:
                dst_path = target_dir / src_path.name
                shutil.copy2(str(src_path), str(dst_path))

            total_stats[phase][class_name] = len(file_list)

    logger.info("划分完成:")
    for phase in ["train", "val", "test"]:
        total = sum(total_stats[phase].values())
        detail = dict(total_stats[phase])
        logger.info(f"  {phase}: {total} 张 {detail}")

    # 保存划分信息
    with open(meta_dir / "split_info.json", 'w', encoding='utf-8') as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)

    # 保存处理日志
    with open(meta_dir / "preprocess_log.txt", 'w', encoding='utf-8') as f:
        f.write("数据整理日志\n")
        f.write("=" * 60 + "\n\n")
        for class_name, info in split_info.items():
            f.write(f"{LABEL_NAMES[LABEL_MAP[class_name]]}({class_name}): "
                    f"总数={info['total']}, "
                    f"训练={info['train']}, "
                    f"验证={info['val']}, "
                    f"测试={info['test']}\n")
        f.write(f"\n总计:\n")
        for phase in ["train", "val", "test"]:
            f.write(f"  {phase}: {sum(total_stats[phase].values())} 张\n")

    logger.info(f"元信息已保存到 {meta_dir}/")


def main():
    args = parse_args()

    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    split_dir = Path(args.split_dir)
    meta_dir = Path(args.meta_dir)

    # 检查原始数据目录
    if not raw_dir.exists():
        logger.error(f"原始数据目录不存在: {raw_dir}")
        return

    subdirs = [d for d in raw_dir.iterdir() if d.is_dir()]
    logger.info(f"发现 {len(subdirs)} 个子文件夹")

    # 步骤1：合并为四大类
    distribution = merge_to_four_classes(raw_dir, processed_dir, meta_dir)

    # 步骤2：划分数据集
    split_dataset(processed_dir, split_dir, meta_dir, args.ratio, args.seed)

    logger.info("数据处理全部完成！")


if __name__ == "__main__":
    main()
