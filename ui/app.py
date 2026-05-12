"""
FastAPI 后端 — 垃圾智能分类系统 Web 界面
启动: python ui/app.py 或 start.bat
"""

import base64
import logging
import sys
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.infer import WasteClassifier
from src.opencv_pipeline.preprocess import denoise, clahe as clahe_enhance, resize

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("waste-classifier")

# ============================================================
# 四大类科普信息
# ============================================================
CATEGORY_INFO = {
    "可回收物": {
        "color": "#3B82F6",
        "bin_color": "蓝色",
        "icon": "♻️",
        "description": "适宜回收利用和资源化利用的生活废弃物",
        "examples": "纸张、塑料瓶、玻璃、金属罐、旧衣服、书本、纸箱",
        "tips": "投放前请清空容器内残留物，纸类尽量叠放整齐，瓶罐类倒空压扁",
    },
    "厨余垃圾": {
        "color": "#22C55E",
        "bin_color": "绿色",
        "icon": "🍂",
        "description": "居民日常生活中产生的有机类生活垃圾",
        "examples": "剩菜剩饭、果皮果核、茶叶渣、菜叶菜根、蛋壳、骨头",
        "tips": "投放前沥干水分，去除食品包装物，大骨头属于其他垃圾",
    },
    "有害垃圾": {
        "color": "#EF4444",
        "bin_color": "红色",
        "icon": "⚠️",
        "description": "对人体健康或自然环境造成直接或潜在危害的废弃物",
        "examples": "废电池、过期药品、废灯管、废油漆、杀虫剂、水银温度计",
        "tips": "投放时请注意轻放，易碎物请包裹后投放，液态物请密封",
    },
    "其他垃圾": {
        "color": "#6B7280",
        "bin_color": "灰色",
        "icon": "🗑️",
        "description": "除可回收物、厨余垃圾、有害垃圾以外的其他生活废弃物",
        "examples": "餐巾纸、烟蒂、陶瓷碎片、尘土、一次性餐具、污损塑料袋",
        "tips": "尽量沥干水分，难以辨识类别的生活垃圾投入其他垃圾容器",
    },
}

# ============================================================
# 模型管理
# ============================================================
AVAILABLE_MODELS = {
    "mobilenet_v3_small": {
        "name": "MobileNetV3-Small",
        "checkpoint": "checkpoints/best_mobilenet_v3_small.pth",
        "description": "轻量级模型，适合移动端部署",
    },
    "resnet18": {
        "name": "ResNet18",
        "checkpoint": "checkpoints/best_resnet18.pth",
        "description": "经典残差网络，平衡精度与速度",
    },
    "efficientnet_b0": {
        "name": "EfficientNet-B0",
        "checkpoint": "checkpoints/best_efficientnet_b0.pth",
        "description": "高效网络，参数利用率高",
    },
    "shufflenet_v2": {
        "name": "ShuffleNetV2",
        "checkpoint": "checkpoints/best_shufflenet_v2.pth",
        "description": "轻量化设计，推理速度快",
    },
}

# 默认 fallback checkpoint（本地测试用）
DEFAULT_CHECKPOINT = "checkpoints/best_model.pth"


class ModelManager:
    """管理多个模型的加载和切换"""

    def __init__(self):
        self._classifiers: Dict[str, WasteClassifier] = {}
        self._current_model: Optional[str] = None
        self._config = load_config("configs/train.yaml", "configs/base.yaml")

    def _find_available_model(self) -> Optional[str]:
        """找到第一个有 checkpoint 文件的模型"""
        for key, info in AVAILABLE_MODELS.items():
            if (PROJECT_ROOT / info["checkpoint"]).exists():
                return key
        if (PROJECT_ROOT / DEFAULT_CHECKPOINT).exists():
            return None  # 使用默认
        return None

    def get_classifier(self, model_key: Optional[str] = None) -> WasteClassifier:
        """获取指定模型的分类器（懒加载）"""
        if model_key is None:
            model_key = self._current_model

        # 如果指定了模型名且存在
        if model_key and model_key in AVAILABLE_MODELS:
            if model_key not in self._classifiers:
                info = AVAILABLE_MODELS[model_key]
                ckpt_path = PROJECT_ROOT / info["checkpoint"]
                if not ckpt_path.exists():
                    raise HTTPException(404, f"模型权重不存在: {info['checkpoint']}")
                logger.info(f"加载模型: {info['name']} <- {ckpt_path}")
                self._classifiers[model_key] = WasteClassifier(
                    config_path=str(PROJECT_ROOT / "configs/train.yaml"),
                    checkpoint_path=str(ckpt_path),
                )
            self._current_model = model_key
            return self._classifiers[model_key]

        # fallback: 使用默认 checkpoint
        if "default" not in self._classifiers:
            ckpt_path = PROJECT_ROOT / DEFAULT_CHECKPOINT
            if not ckpt_path.exists():
                raise HTTPException(404, "未找到任何可用的模型权重")
            logger.info(f"加载默认模型: {ckpt_path}")
            self._classifiers["default"] = WasteClassifier(
                config_path=str(PROJECT_ROOT / "configs/train.yaml"),
                checkpoint_path=str(ckpt_path),
            )
        self._current_model = None
        return self._classifiers["default"]

    def get_current_model_name(self) -> str:
        if self._current_model and self._current_model in AVAILABLE_MODELS:
            return AVAILABLE_MODELS[self._current_model]["name"]
        if "default" in self._classifiers:
            return "默认模型 (best_model.pth)"
        return "未加载"

    def list_models(self) -> list:
        """列出所有模型及其可用状态"""
        result = []
        for key, info in AVAILABLE_MODELS.items():
            exists = (PROJECT_ROOT / info["checkpoint"]).exists()
            is_current = (key == self._current_model)
            result.append({
                "key": key,
                "name": info["name"],
                "description": info["description"],
                "available": exists,
                "current": is_current,
            })
        # 默认模型
        default_exists = (PROJECT_ROOT / DEFAULT_CHECKPOINT).exists()
        if default_exists:
            result.append({
                "key": "default",
                "name": "默认模型",
                "description": "best_model.pth（本地测试权重）",
                "available": True,
                "current": self._current_model is None,
            })
        return result


# ============================================================
# FastAPI 应用
# ============================================================
app = FastAPI(title="垃圾智能分类系统", version="1.0")

# CORS（本地开发用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# 模型管理器
manager = ModelManager()


@app.get("/", response_class=HTMLResponse)
async def index():
    """返回主页面"""
    html_path = static_dir / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
):
    """
    图片上传分类
    接收图片文件，返回分类结果
    """
    # 读取图片
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(400, "无法解析图片文件，请上传有效的图片")

    # 获取分类器
    classifier = manager.get_classifier(model)

    # 推理
    result = classifier.predict_image(image)

    # 添加科普建议
    class_name = result["class_name"]
    if class_name in CATEGORY_INFO:
        info = CATEGORY_INFO[class_name]
        result["advice"] = (
            f"请投入{info['bin_color']}垃圾桶。{info['tips']}"
        )
        result["category_info"] = info
    else:
        result["advice"] = "无法识别该物品的类别"
        result["category_info"] = None

    # 生成标注图的 base64
    try:
        from src.opencv_pipeline.draw import draw_probability_bar, draw_result
        annotated = draw_result(image, result["class_name"], result["confidence"])
        annotated = draw_probability_bar(annotated, result["probabilities"], classifier.label_name_list)

        # 缩放到合理尺寸
        h, w = annotated.shape[:2]
        max_dim = 640
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            annotated = cv2.resize(annotated, (int(w * scale), int(h * scale)))

        # 编码为 base64
        _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        result["annotated_image"] = base64.b64encode(buffer).decode("utf-8")
    except Exception as e:
        logger.warning(f"标注图生成失败: {e}")
        result["annotated_image"] = None

    result["model_name"] = manager.get_current_model_name()

    return JSONResponse(content=result)


@app.get("/api/models")
async def list_models():
    """返回可用模型列表"""
    return JSONResponse(content={
        "models": manager.list_models(),
        "current": manager.get_current_model_name(),
    })


@app.post("/api/switch-model")
async def switch_model(model: str = Form(...)):
    """切换当前模型"""
    if model == "default":
        manager._current_model = None
        return {"status": "ok", "current": manager.get_current_model_name()}

    if model not in AVAILABLE_MODELS:
        raise HTTPException(400, f"未知模型: {model}")

    if not (PROJECT_ROOT / AVAILABLE_MODELS[model]["checkpoint"]).exists():
        raise HTTPException(404, f"模型权重不存在: {AVAILABLE_MODELS[model]['checkpoint']}")

    # 触发加载
    manager.get_classifier(model)
    return {"status": "ok", "current": manager.get_current_model_name()}


# ============================================================
# 辅助函数：numpy BGR 图像 → base64 字符串
# ============================================================
def _image_to_base64(image: np.ndarray, max_dim: int = 640, quality: int = 85) -> str:
    """将 BGR numpy 图像编码为 JPEG base64 字符串"""
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer).decode("utf-8")


# ============================================================
# 新增接口 1：四模型对比推理
# ============================================================
@app.post("/api/compare")
async def compare_models(file: UploadFile = File(...)):
    """
    四模型对比推理：上传一张图片，同时调用所有可用模型推理，
    返回每个模型的分类结果 + 推理耗时。
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(400, "无法解析图片文件，请上传有效的图片")

    results = []
    for model_key, model_info in AVAILABLE_MODELS.items():
        ckpt_path = PROJECT_ROOT / model_info["checkpoint"]
        if not ckpt_path.exists():
            continue  # 跳过没有权重文件的模型

        # 懒加载并推理
        classifier = manager.get_classifier(model_key)
        t0 = time.time()
        result = classifier.predict_image(image)
        t1 = time.time()

        result["model_key"] = model_key
        result["model_name"] = model_info["name"]
        result["inference_time_ms"] = round((t1 - t0) * 1000, 1)

        # 添加科普建议
        class_name = result["class_name"]
        if class_name in CATEGORY_INFO:
            info = CATEGORY_INFO[class_name]
            result["advice"] = f"请投入{info['bin_color']}垃圾桶。{info['tips']}"
        else:
            result["advice"] = "无法识别该物品的类别"

        results.append(result)

    if not results:
        raise HTTPException(404, "没有找到可用的模型权重文件")

    return JSONResponse(content={"results": results})


# ============================================================
# 新增接口 2：批量识别
# ============================================================
@app.post("/api/batch-predict")
async def batch_predict(files: List[UploadFile] = File(...)):
    """
    批量识别：支持多张图片同时上传，
    返回每张图片的分类结果 + 统计摘要。
    """
    if not files:
        raise HTTPException(400, "请至少上传一张图片")

    classifier = manager.get_classifier()
    results = []

    for file in files:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            results.append({
                "filename": file.filename,
                "error": "无法解析图片",
            })
            continue

        result = classifier.predict_image(image)
        result["filename"] = file.filename

        # 添加科普建议
        class_name = result["class_name"]
        if class_name in CATEGORY_INFO:
            info = CATEGORY_INFO[class_name]
            result["advice"] = f"请投入{info['bin_color']}垃圾桶。{info['tips']}"
        else:
            result["advice"] = "无法识别该物品的类别"

        results.append(result)

    # 计算统计摘要
    valid_results = [r for r in results if "error" not in r]
    category_counts = {}
    confidences = []
    low_confidence_count = 0

    for r in valid_results:
        cn = r["class_name"]
        category_counts[cn] = category_counts.get(cn, 0) + 1
        confidences.append(r["confidence"])
        if r["confidence"] < 0.70:
            low_confidence_count += 1

    total = len(valid_results)
    avg_confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0

    stats = {
        "total": total,
        "error_count": len(results) - total,
        "category_counts": category_counts,
        "average_confidence": avg_confidence,
        "low_confidence_count": low_confidence_count,
    }

    return JSONResponse(content={
        "results": results,
        "stats": stats,
        "model_name": manager.get_current_model_name(),
    })


# ============================================================
# 新增接口 3：OpenCV 预处理管线可视化
# ============================================================
@app.post("/api/opencv-pipeline")
async def opencv_pipeline(file: UploadFile = File(...)):
    """
    OpenCV 管线可视化：上传一张图片，
    逐步执行预处理并返回每一步的结果图（base64）。
    流程：原图 → 高斯去噪 → CLAHE 增强 → 模型输入（resize 224×224）
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(400, "无法解析图片文件，请上传有效的图片")

    steps = []

    # 步骤 1：原始图片（保留较大尺寸以显示细节差异）
    steps.append({
        "name": "原始图片",
        "description": "OpenCV 读取的 BGR 原始图片",
        "image_base64": _image_to_base64(image, max_dim=800),
        "params": None,
    })

    # 步骤 2：高斯去噪（使用 5×5 核，效果更直观）
    denoised = denoise(image, method="gaussian", ksize=5)
    steps.append({
        "name": "高斯去噪",
        "description": "使用 5×5 高斯核进行图像去噪，减少高频噪声",
        "image_base64": _image_to_base64(denoised, max_dim=800),
        "params": {"method": "gaussian", "ksize": 5, "sigma": 0},
    })

    # 步骤 3：CLAHE 增强（使用更强的 clip_limit 使对比度变化更明显）
    clahe_result = clahe_enhance(denoised, clip_limit=4.0, grid_size=(8, 8))
    steps.append({
        "name": "CLAHE 增强",
        "description": "自适应直方图均衡化（LAB 空间 L 通道），增强局部对比度",
        "image_base64": _image_to_base64(clahe_result, max_dim=800),
        "params": {"clipLimit": 4.0, "tileGridSize": [8, 8]},
    })

    # 步骤 4：模型输入（resize 到 224×224，尺寸明显缩小）
    resized = resize(clahe_result, (224, 224))
    steps.append({
        "name": "模型输入",
        "description": "缩放至 224×224 并归一化后，作为模型推理输入",
        "image_base64": _image_to_base64(resized, max_dim=800),
        "params": {"size": [224, 224]},
    })

    return JSONResponse(content={"steps": steps})


# ============================================================
# 启动入口
# ============================================================
if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("  垃圾智能分类系统 — Web 界面启动")
    logger.info(f"  访问地址: http://localhost:8080")
    logger.info("=" * 50)

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8080,
        log_level="info",
    )
