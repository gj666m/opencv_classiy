"""
OpenCV 处理效果可视化演示
一键生成论文所需的 7 张 OpenCV 对比图

用法:
    D:\\anaconda\\python.exe scripts/generate_opencv_demo.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.opencv_pipeline.io import read_image, write_image
from src.opencv_pipeline.preprocess import (
    canny_edge,
    clahe,
    denoise,
    sobel_edge,
    laplacian_edge,
    resize,
    adjust_brightness,
    adjust_contrast,
)
from src.opencv_pipeline.augment import (
    random_horizontal_flip,
    random_vertical_flip,
    random_rotation,
    random_brightness,
    random_crop,
    add_gaussian_noise,
)

# 输出目录
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "opencv_demo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 数据集路径
DATA_DIR = PROJECT_ROOT / "data" / "trash_jpg" / "trash_jpg"

# 四大类各取一张样本
SAMPLE_IMAGES = {
    "可回收物": DATA_DIR / "可回收物_不锈钢制品" / "img_不锈钢制品_1.jpg",
    "厨余垃圾": DATA_DIR / "厨余垃圾_苹果" / "img_苹果_1.jpg",
    "有害垃圾": DATA_DIR / "有害垃圾_电池" / "img_电池_1.jpg",
    "其他垃圾": DATA_DIR / "其他垃圾_PE塑料袋" / "img_PE塑料袋_1.jpg",
}

# 统一显示尺寸
DISPLAY_SIZE = (224, 224)


def _load_samples():
    """加载所有样本图"""
    samples = {}
    for name, path in SAMPLE_IMAGES.items():
        img = read_image(str(path))
        if img is not None:
            samples[name] = img
            print(f"  加载: {name} — {img.shape}")
        else:
            print(f"  警告: 无法加载 {path}")
    return samples


def _make_label(image, text, font_size=16, color=(255, 255, 255)):
    """在图片顶部加一个中文文字标签条（用 PIL 渲染中文）"""
    from PIL import Image, ImageDraw, ImageFont
    import os

    h, w = image.shape[:2]
    bar = np.zeros((30, w, 3), dtype=np.uint8)

    # BGR → PIL
    pil_img = Image.fromarray(cv2.cvtColor(bar, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # 尝试加载中文字体，按优先级尝试
    font = None
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
        "C:/Windows/Fonts/simhei.ttf",     # 黑体
        "C:/Windows/Fonts/simsun.ttc",     # 宋体
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",  # Linux
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, font_size)
                break
            except Exception:
                continue
    if font is None:
        font = ImageFont.load_default()

    # PIL 用 RGB 颜色
    draw.text((5, 5), text, font=font, fill=(color[2], color[1], color[0]))

    # PIL → BGR
    bar = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return np.vstack([bar, image])


def _concat_grid(images_2d):
    """
    将二维列表的图片拼成网格
    images_2d: [[img1, img2, ...], [img3, img4, ...], ...]
    """
    rows = []
    for row in images_2d:
        # 统一每行中所有图片的高度
        max_h = max(img.shape[0] for img in row)
        padded = []
        for img in row:
            if img.shape[0] < max_h:
                pad = np.zeros((max_h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
                img = np.vstack([img, pad])
            padded.append(img)
        rows.append(np.hstack(padded))
    return np.vstack(rows)


# ============================================================
# 图 1: 预处理流水线总览
# ============================================================
def generate_pipeline_overview(samples):
    """4类各1张，每行: 原图→缩放→去噪→CLAHE→Canny边缘检测"""
    print("生成: pipeline_overview.png")

    grid = []
    step_names = ["原图", "缩放224x224", "高斯去噪", "CLAHE均衡", "Canny边缘"]

    for name, img in samples.items():
        row = []
        # 原图缩放到显示尺寸
        orig = resize(img, DISPLAY_SIZE)
        row.append(_make_label(orig, f"{name}-原图"))

        # 缩放
        resized = resize(img, DISPLAY_SIZE)
        row.append(_make_label(resized, "缩放"))

        # 高斯去噪
        denoised = denoise(resized, method='gaussian', ksize=5)
        row.append(_make_label(denoised, "高斯去噪"))

        # CLAHE
        clahe_img = clahe(resized)
        row.append(_make_label(clahe_img, "CLAHE"))

        # Canny
        edge = canny_edge(resized)
        edge = resize(edge, DISPLAY_SIZE)
        row.append(_make_label(edge, "Canny边缘"))

        grid.append(row)

    result = _concat_grid(grid)
    out_path = OUTPUT_DIR / "pipeline_overview.png"
    write_image(str(out_path), result)
    print(f"  已保存: {out_path}")


# ============================================================
# 图 2: 去噪方法对比
# ============================================================
def generate_denoise_comparison(samples):
    """原图 vs 高斯/中值/双边滤波"""
    print("生成: denoise_comparison.png")

    grid = []
    for name, img in samples.items():
        resized = resize(img, DISPLAY_SIZE)
        row = [
            _make_label(resized, f"{name}-原图"),
            _make_label(denoise(resized, 'gaussian', 5), "高斯滤波 k=5"),
            _make_label(denoise(resized, 'median', 5), "中值滤波 k=5"),
            _make_label(denoise(resized, 'bilateral', 9), "双边滤波 d=9"),
        ]
        grid.append(row)

    result = _concat_grid(grid)
    out_path = OUTPUT_DIR / "denoise_comparison.png"
    write_image(str(out_path), result)
    print(f"  已保存: {out_path}")


# ============================================================
# 图 3: CLAHE 效果对比
# ============================================================
def generate_clahe_comparison(samples):
    """原图 vs CLAHE (不同 clip_limit)"""
    print("生成: clahe_comparison.png")

    grid = []
    for name, img in samples.items():
        resized = resize(img, DISPLAY_SIZE)
        row = [
            _make_label(resized, f"{name}-原图"),
            _make_label(clahe(resized, clip_limit=2.0), "CLAHE clip=2.0"),
            _make_label(clahe(resized, clip_limit=4.0), "CLAHE clip=4.0"),
            _make_label(clahe(resized, clip_limit=8.0), "CLAHE clip=8.0"),
        ]
        grid.append(row)

    result = _concat_grid(grid)
    out_path = OUTPUT_DIR / "clahe_comparison.png"
    write_image(str(out_path), result)
    print(f"  已保存: {out_path}")


# ============================================================
# 图 4: 边缘检测对比
# ============================================================
def generate_edge_detection(samples):
    """原图 vs Canny / Sobel / Laplacian"""
    print("生成: edge_detection.png")

    grid = []
    for name, img in samples.items():
        resized = resize(img, DISPLAY_SIZE)
        row = [
            _make_label(resized, f"{name}-原图"),
            _make_label(canny_edge(resized), "Canny"),
            _make_label(sobel_edge(resized), "Sobel"),
            _make_label(laplacian_edge(resized), "Laplacian"),
        ]
        grid.append(row)

    result = _concat_grid(grid)
    out_path = OUTPUT_DIR / "edge_detection.png"
    write_image(str(out_path), result)
    print(f"  已保存: {out_path}")


# ============================================================
# 图 5: 颜色空间可视化
# ============================================================
def generate_color_space(samples):
    """BGR / HSV / LAB / 灰度 通道分解"""
    print("生成: color_space.png")

    grid = []
    for name, img in samples.items():
        resized = resize(img, DISPLAY_SIZE)

        # BGR 各通道
        b, g, r = cv2.split(resized)
        b_ch = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
        g_ch = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        r_ch = cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)

        # HSV 各通道
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        h_ch = cv2.cvtColor(hsv[:, :, 0], cv2.COLOR_GRAY2BGR)
        s_ch = cv2.cvtColor(hsv[:, :, 1], cv2.COLOR_GRAY2BGR)
        v_ch = cv2.cvtColor(hsv[:, :, 2], cv2.COLOR_GRAY2BGR)

        # 灰度
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        row = [
            _make_label(resized, f"{name}-BGR"),
            _make_label(h_ch, "H通道"),
            _make_label(s_ch, "S通道"),
            _make_label(v_ch, "V通道"),
            _make_label(gray_bgr, "灰度图"),
        ]
        grid.append(row)

    result = _concat_grid(grid)
    out_path = OUTPUT_DIR / "color_space.png"
    write_image(str(out_path), result)
    print(f"  已保存: {out_path}")


# ============================================================
# 图 6: 数据增强展示
# ============================================================
def generate_augmentation_showcase(samples):
    """原图 + 6种增强效果 3x3 网格（取2类展示）"""
    print("生成: augmentation_showcase.png")

    grid = []
    for name, img in list(samples.items())[:2]:  # 取2类
        resized = resize(img, DISPLAY_SIZE)

        # 为了确定性展示，固定随机种子的效果
        # 水平翻转
        flipped_h = cv2.flip(resized, 1)
        # 垂直翻转
        flipped_v = cv2.flip(resized, 0)
        # 旋转 15 度
        h, w = resized.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), 15, 1.0)
        rotated = cv2.warpAffine(resized, M, (w, h), borderValue=(0, 0, 0))
        # 亮度增强
        bright = adjust_brightness(resized, factor=1.4)
        # 亮度降低
        dark = adjust_brightness(resized, factor=0.6)
        # 高斯噪声
        noise = add_gaussian_noise(resized, std=20)

        row = [
            _make_label(resized, f"{name}-原图"),
            _make_label(flipped_h, "水平翻转"),
            _make_label(flipped_v, "垂直翻转"),
        ]
        grid.append(row)

        row2 = [
            _make_label(rotated, "旋转15°"),
            _make_label(bright, "亮度+40%"),
            _make_label(noise, "高斯噪声"),
        ]
        grid.append(row2)

    result = _concat_grid(grid)
    out_path = OUTPUT_DIR / "augmentation_showcase.png"
    write_image(str(out_path), result)
    print(f"  已保存: {out_path}")


# ============================================================
# 图 7: 亮度对比度调整
# ============================================================
def generate_brightness_contrast(samples):
    """原图 vs 亮/暗/高对比/低对比"""
    print("生成: brightness_contrast.png")

    grid = []
    for name, img in samples.items():
        resized = resize(img, DISPLAY_SIZE)
        row = [
            _make_label(resized, f"{name}-原图"),
            _make_label(adjust_brightness(resized, 1.5), "亮度 x1.5"),
            _make_label(adjust_brightness(resized, 0.5), "亮度 x0.5"),
            _make_label(adjust_contrast(resized, 1.5), "对比度 x1.5"),
            _make_label(adjust_contrast(resized, 0.5), "对比度 x0.5"),
        ]
        grid.append(row)

    result = _concat_grid(grid)
    out_path = OUTPUT_DIR / "brightness_contrast.png"
    write_image(str(out_path), result)
    print(f"  已保存: {out_path}")


# ============================================================
# 主入口
# ============================================================
def main():
    print("=" * 60)
    print("  OpenCV 处理效果可视化演示")
    print("=" * 60)

    # 加载样本
    print("\n加载样本图片...")
    samples = _load_samples()

    if not samples:
        print("错误: 未加载到任何样本图片")
        sys.exit(1)

    print(f"\n共加载 {len(samples)} 张样本，开始生成对比图...\n")

    # 生成所有对比图
    generate_pipeline_overview(samples)
    generate_denoise_comparison(samples)
    generate_clahe_comparison(samples)
    generate_edge_detection(samples)
    generate_color_space(samples)
    generate_augmentation_showcase(samples)
    generate_brightness_contrast(samples)

    print("\n" + "=" * 60)
    print(f"  全部完成! 共生成 7 张对比图")
    print(f"  输出目录: {OUTPUT_DIR}")
    print("=" * 60)

    # 列出所有生成的文件
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
