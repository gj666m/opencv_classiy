# 基于 OpenCV 的垃圾智能分类系统 — Claude Code 执行手册

> 本文档是 Claude Code 编写本毕设项目的唯一行动指南。
> 严格按照阶段顺序执行，每个阶段有明确的输入、输出和验收标准。

---

## 0. 项目元信息

| 项 | 值 |
|----|-----|
| 项目名称 | 基于OpenCV的垃圾智能分类系统 |
| 学生 | 郭靖 / 广州城市理工学院 / 人工智能2班 / 202210176064 |
| 指导教师 | 邓乃经 |
| Python | 3.12.4（`D:\anaconda\python.exe`） |
| 深度学习框架 | **PyTorch 2.x** |
| 图像处理 | **OpenCV 4.12**（已安装） |
| 参考数据集 | 项目根目录 `code/` 下原始代码，约 80,012 张图片 / 245 个细分类别 |
| 参考代码路径 | `E:/垃圾分类数据集和tf代码-8w张图片245个类(更新)/code/` |

---

## 1. 最终项目目录结构

```
waste-classification/                     ← 项目根目录
├── configs/
│   ├── base.yaml                         ← 全局基础配置
│   ├── train.yaml                        ← 训练专用配置（覆盖 base）
│   └── infer.yaml                        ← 推理专用配置
├── data/
│   ├── raw/                              ← 原始数据（245个子文件夹，从百度网盘下载）
│   ├── processed/                        ← 整理后的四分类数据
│   │   ├── recyclable/                   ← 可回收物（由 raw/可回收物_* 合并）
│   │   ├── kitchen/                      ← 厨余垃圾（由 raw/厨余垃圾_* 合并）
│   │   ├── hazardous/                    ← 有害垃圾（由 raw/有害垃圾_* 合并）
│   │   └── other/                        ← 其他垃圾（由 raw/其他垃圾_* 合并）
│   ├── split/                            ← 划分后的数据
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── meta/
│       ├── label_map.json                ← 标签映射 {0:recyclable, 1:kitchen, ...}
│       ├── class_distribution.json       ← 各类样本数统计
│       └── preprocess_log.txt            ← 数据处理日志
├── src/
│   ├── __init__.py
│   ├── config.py                         ← 配置加载器（读 YAML，返回 dataclass）
│   ├── dataset.py                        ← PyTorch Dataset + DataLoader 工厂
│   ├── opencv_pipeline/
│   │   ├── __init__.py
│   │   ├── io.py                         ← 图片读取、格式转换、路径批量加载
│   │   ├── preprocess.py                 ← Resize、Normalize、BGR↔RGB、去噪、CLAHE
│   │   ├── augment.py                    ← 翻转、旋转、亮度、裁剪、噪声增强
│   │   ├── camera.py                     ← 摄像头采集、帧控制、FPS 计算
│   │   ├── draw.py                       ← 分类结果文字/概率叠加到图像
│   │   └── dataset_tools.py              ← 批量清洗、统计、可视化抽样
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                       ← 模型基类/注册器
│   │   ├── mobilenet_v3.py               ← MobileNetV3-Small（主力）
│   │   ├── resnet.py                     ← ResNet18（基线）
│   │   ├── efficientnet.py               ← EfficientNet-B0（精度对照）
│   │   └── shufflenet.py                 ← ShuffleNetV2（实时性候选）
│   ├── train.py                          ← 训练引擎（单次训练主循环）
│   ├── evaluate.py                       ← 评估引擎（Accuracy/P/R/F1/混淆矩阵）
│   └── infer.py                          ← 推理引擎（单图推理接口）
├── scripts/
│   ├── prepare_data.py                   ← 数据整理：raw → processed → split
│   ├── train.py                          ← 训练入口脚本
│   ├── evaluate.py                       ← 评估入口脚本
│   ├── infer_image.py                    ← 单图识别入口脚本
│   ├── infer_camera.py                   ← 摄像头实时识别入口脚本
│   └── export_results.py                ← 导出实验结果汇总
├── ui/
│   ├── __init__.py
│   ├── app.py                            ← FastAPI 后端（7 个 API 端点）
│   └── static/
│       ├── index.html                    ← 4-Tab 主页面
│       ├── style.css                     ← 生态绿色主题样式
│       ├── script.js                     ← 主交互逻辑
│       ├── compare.js                    ← 模型对比模块
│       ├── batch.js                      ← 批量识别模块
│       ├── opencv-demo.js               ← OpenCV 管线演示
│       └── history.js                    ← 识别历史（localStorage）
├── checkpoints/                          ← 模型权重保存（.pth）
├── logs/                                 ← 训练日志（TensorBoard 或 CSV）
├── outputs/
│   ├── predictions/                      ← 单图识别结果图片
│   ├── figures/                          ← 训练曲线、混淆矩阵、热力图
│   └── reports/                          ← 实验汇总表（CSV/MD）
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 2. 技术栈与版本锁定

```
# requirements.txt
torch>=2.1.0
torchvision>=0.16.0
opencv-python>=4.8.0
numpy>=1.26.0
Pillow>=10.0.0
matplotlib>=3.8.0
PyYAML>=6.0
scikit-learn>=1.3.0       # 用于 Precision/Recall/F1/混淆矩阵
tensorboard>=2.15.0        # 训练日志可视化
tqdm>=4.66.0               # 进度条
```

安装命令：`D:\anaconda\python.exe -m pip install torch torchvision opencv-python numpy Pillow matplotlib PyYAML scikit-learn tensorboard tqdm`

---

## 3. 配置文件规范

### 3.1 configs/base.yaml

```yaml
# 全局基础配置
project:
  name: "基于OpenCV的垃圾智能分类系统"
  seed: 42

data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  split_dir: "data/split"
  meta_dir: "data/meta"
  num_workers: 4

model:
  name: "mobilenet_v3_small"   # 可选: mobilenet_v3_small, resnet18, efficientnet_b0, shufflenet_v2
  num_classes: 4
  pretrained: true
  freeze_backbone: true        # 初始训练冻结基座
  input_size: 224              # 输入图片尺寸 224x224

labels:
  0: "可回收物"
  1: "厨余垃圾"
  2: "有害垃圾"
  3: "其他垃圾"
  label_map:
    recyclable: 0
    kitchen: 1
    hazardous: 2
    other: 3

output:
  checkpoint_dir: "checkpoints"
  log_dir: "logs"
  output_dir: "outputs"
```

### 3.2 configs/train.yaml

```yaml
# 训练配置（覆盖 base）
train:
  epochs: 50
  batch_size: 32
  lr: 0.001
  weight_decay: 1e-4
  optimizer: "adamw"            # adamw / sgd
  scheduler: "cosine"          # cosine / step / plateau
  warmup_epochs: 3

  # 回调策略
  early_stopping:
    patience: 10
    min_delta: 0.001
  checkpoint:
    save_best: true
    monitor: "val_accuracy"

  # 数据增强（训练时）
  augmentation:
    random_horizontal_flip: true
    random_vertical_flip: true
    random_rotation: 15         # 度
    color_jitter:
      brightness: 0.2
      contrast: 0.2
      saturation: 0.2
    random_erasing: true
    gaussian_noise: false

val:
  batch_size: 32
```

### 3.3 configs/infer.yaml

```yaml
# 推理配置
infer:
  checkpoint: "checkpoints/best_model.pth"
  device: "cpu"                 # cpu / cuda
  camera_id: 0
  confidence_threshold: 0.5
```

---

## 4. 分阶段实施计划

### 阶段一：项目脚手架

**目标**：创建完整目录结构 + 配置文件 + 基础模块骨架

**产出文件清单**：
1. 所有目录（按第1节结构创建，含 `__init__.py`）
2. `configs/base.yaml`、`configs/train.yaml`、`configs/infer.yaml`
3. `src/config.py` — 配置加载器
4. `requirements.txt`
5. `.gitignore`
6. `README.md`

**src/config.py 设计要求**：
- 使用 `dataclass` 定义配置结构
- `load_config(path)` 函数：读取 YAML，合并 base + 覆盖配置，返回 dataclass 实例
- 支持从配置对象获取所有超参数

**验收标准**：
```python
from src.config import load_config
cfg = load_config("configs/train.yaml")
assert cfg.model.name == "mobilenet_v3_small"
assert cfg.model.num_classes == 4
assert cfg.labels.label_map["recyclable"] == 0
```

---

### 阶段二：数据处理管线

**目标**：将原始 245 类数据整理为四分类，并划分 train/val/test

**前置条件**：原始数据集已下载到 `data/raw/`（245 个子文件夹）

**2.1 数据整理脚本 `scripts/prepare_data.py`**

功能：
1. 扫描 `data/raw/` 下所有子文件夹
2. 根据文件夹名前缀映射到四大类：
   - `可回收物_*` → `recyclable`
   - `厨余垃圾_*` → `kitchen`
   - `有害垃圾_*` → `hazardous`
   - `其他垃圾_*` → `other`
3. 将图片复制（或符号链接）到 `data/processed/{四大类}/` 下
4. 统计各类样本数，保存到 `data/meta/class_distribution.json`
5. 生成 `data/meta/label_map.json`
6. 按 8:1:1 划分 train/val/test，保存到 `data/split/`

执行方式：
```bash
D:\anaconda\python.exe scripts/prepare_data.py --raw_dir data/raw --processed_dir data/processed --split_dir data/split --ratio 0.8 0.1 0.1
```

**2.2 OpenCV 数据工具 `src/opencv_pipeline/dataset_tools.py`**

功能：
1. `scan_dataset(directory)` — 扫描数据集，返回各文件夹图片数量统计
2. `detect_corrupted_images(directory)` — 检测损坏图片（参考原 jpeg2jpg.py 的 JPEG 校验逻辑）
3. `detect_blurry_images(directory, threshold=100)` — 使用拉普拉斯方差检测模糊图片
4. `visualize_samples(directory, num_per_class=4)` — 从每个类别随机抽样，拼成对比图保存

**2.3 OpenCV 图像 I/O `src/opencv_pipeline/io.py`**

功能：
1. `read_image(path, color_mode='bgr')` — 读取图片，支持 bgr/rgb/gray 三种模式
2. `write_image(path, image)` — 保存图片，自动创建父目录
3. `batch_load(directory, extensions=['.jpg', '.png', '.jpeg'])` — 批量加载目录下所有图片路径

**2.4 数据集类 `src/dataset.py`**

```python
class WasteDataset(torch.utils.data.Dataset):
    """垃圾四分类数据集"""

    def __init__(self, root_dir, transform=None, opencv_preprocess=None):
        """
        Args:
            root_dir: data/split/train 或 data/split/val 或 data/split/test
            transform: torchvision transforms（数据增强）
            opencv_preprocess: OpenCV 预处理函数列表（可选）
        """
        ...

    def __len__(self):
        ...

    def __getitem__(self, idx):
        """
        流程：
        1. OpenCV 读取图片（BGR）
        2. 可选：执行 opencv_preprocess 中的预处理链
        3. BGR → RGB
        4. 转 PIL Image
        5. 执行 torchvision transform
        6. 返回 (image_tensor, label)
        """
        ...
```

提供工厂函数：
```python
def create_dataloaders(config, phase="train"):
    """根据配置创建 train/val/test DataLoader"""
    ...
```

**验收标准**：
```bash
# 原始数据整理
python scripts/prepare_data.py --raw_dir data/raw ...
# 输出：data/processed/ 下有 4 个子文件夹
# 输出：data/split/ 下有 train/val/test 各含 4 个子文件夹
# 输出：data/meta/label_map.json 和 class_distribution.json

# 数据集加载测试
python -c "from src.dataset import create_dataloaders; dl = create_dataloaders(config, 'train'); print(next(iter(dl))[0].shape)"
# 预期输出: torch.Size([32, 3, 224, 224])
```

---

### 阶段三：OpenCV 预处理与增强模块

**目标**：完成论文中"OpenCV 图像处理"章节所需的全部功能

**3.1 预处理模块 `src/opencv_pipeline/preprocess.py`**

每个函数接受 numpy array（BGR），返回 numpy array（BGR）：

```python
def resize(image, size=(224, 224)):
    """cv2.resize 插值缩放"""

def normalize(image):
    """像素值归一化到 [0,1]"""

def convert_color(image, target='rgb'):
    """BGR ↔ RGB / HSV / LAB 转换"""

def denoise(image, method='gaussian', ksize=5):
    """去噪：高斯模糊 / 中值滤波 / 双边滤波"""

def clahe(image, clip_limit=2.0, grid_size=(8, 8)):
    """自适应直方图均衡化（CLAHE），缓解光照不均"""

def adjust_brightness(image, factor=1.0):
    """亮度调整"""

def adjust_contrast(image, factor=1.0):
    """对比度调整"""

class PreprocessPipeline:
    """可配置的预处理管线，按顺序执行多个预处理步骤"""
    def __init__(self, steps: list):
        """steps: ['resize', 'denoise', 'clahe', 'normalize']"""
        ...
    def __call__(self, image):
        for step in self.steps:
            image = step(image)
        return image
```

**3.2 数据增强模块 `src/opencv_pipeline/augment.py`**

```python
def random_horizontal_flip(image, p=0.5):
    """随机水平翻转"""

def random_vertical_flip(image, p=0.5):
    """随机垂直翻转"""

def random_rotation(image, max_angle=15):
    """随机旋转"""

def random_brightness(image, max_delta=0.2):
    """随机亮度扰动"""

def random_crop(image, crop_ratio=0.9):
    """随机裁剪后 resize 回原尺寸"""

def add_gaussian_noise(image, mean=0, std=10):
    """添加高斯噪声"""

class AugmentationPipeline:
    """可配置的数据增强管线"""
    def __init__(self, config: dict):
        ...
    def __call__(self, image):
        ...
```

**注意**：训练时的数据增强优先使用 `torchvision.transforms`（效率更高），但 `src/opencv_pipeline/augment.py` 必须实现同等功能，用于：
1. 论文中展示 OpenCV 增强前后的对比图
2. 数据预处理阶段的离线增强
3. 摄像头实时推理时的可选预处理

**3.3 摄像头模块 `src/opencv_pipeline/camera.py`**

```python
class CameraStream:
    """摄像头视频流管理"""
    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)
        self.fps_counter = FPSCounter()

    def read_frame(self):
        """读取一帧，返回 BGR numpy array"""

    def release(self):
        """释放摄像头"""

class FPSCounter:
    """FPS 计数器"""
    def __init__(self):
        ...
    def tick(self):
        """每处理一帧调用一次"""
    def get_fps(self):
        """返回当前 FPS"""
```

**3.4 结果绘制模块 `src/opencv_pipeline/draw.py`**

```python
def draw_result(image, class_name, confidence, color=(0, 255, 0)):
    """
    在图像上叠加分类结果
    - 左上角显示：类别名称
    - 右上角显示：置信度百分比
    - 底部显示：四分类概率条形图
    """

def draw_fps(image, fps):
    """左下角显示 FPS"""

def draw_probability_bar(image, probabilities, label_names):
    """底部绘制四分类概率条形图"""
```

**验收标准**：
```python
# 预处理管线测试
from src.opencv_pipeline.preprocess import PreprocessPipeline
pipe = PreprocessPipeline(['resize', 'denoise', 'clahe'])
result = pipe(cv2.imread("test.jpg"))
assert result.shape == (224, 224, 3)

# 摄像头读取测试（需要实际摄像头）
from src.opencv_pipeline.camera import CameraStream
cam = CameraStream(0)
frame = cam.read_frame()
assert frame is not None
cam.release()
```

---

### 阶段四：模型定义

**目标**：实现 4 个候选模型，支持统一接口切换

**4.1 模型基类 `src/models/base.py`**

```python
import torchvision.models as models
import torch.nn as nn

MODEL_REGISTRY = {}  # 全局模型注册表

def register_model(name):
    """模型注册装饰器"""
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def build_model(config):
    """根据配置中的 model.name 构建模型"""
    name = config.model.name
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}, available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](config)
```

**4.2 各模型实现要求**

所有模型统一接口：

```python
@register_model("mobilenet_v3_small")
class MobileNetV3Small(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 1. 加载 torchvision 预训练 MobileNetV3-Small
        # 2. 冻结 backbone（如果 config.model.freeze_backbone == True）
        # 3. 替换分类头为 nn.Linear(last_dim, 4)
        ...

    def forward(self, x):
        ...
        return logits  # shape: (batch, 4)
```

四个模型的结构要点：

| 模型 | torchvision 调用 | 分类头替换位置 | 预期参数量 |
|------|-----------------|-------------|----------|
| MobileNetV3-Small | `models.mobilenet_v3_s(weights='DEFAULT')` | `model.classifier[3]` | ~2.5M |
| ResNet18 | `models.resnet18(weights='DEFAULT')` | `model.fc` | ~11.7M |
| EfficientNet-B0 | `models.efficientnet_b0(weights='DEFAULT')` | `model.classifier[1]` | ~5.3M |
| ShuffleNetV2 | `models.shufflenet_v2_x1_0(weights='DEFAULT')` | `model.fc` | ~2.3M |

**4.3 模型训练/微调两阶段策略**

```
第一阶段（Epoch 1-15）：
  - freeze_backbone = True
  - 只训练分类头
  - lr = 0.001

第二阶段（Epoch 16-50）：
  - freeze_backbone = False
  - 全网络微调
  - lr = 0.0001（降低10倍）
  - 可选：差分学习率（backbone: 1e-5, classifier: 1e-3）
```

在 `configs/train.yaml` 中添加：
```yaml
train:
  two_stage: true
  stage1_epochs: 15
  stage2_lr_factor: 0.1     # 第二阶段学习率 = lr * factor
```

**验收标准**：
```python
from src.models.base import build_model
model = build_model(config)  # config.model.name = "mobilenet_v3_small"
dummy = torch.randn(2, 3, 224, 224)
output = model(dummy)
assert output.shape == (2, 4)
```

---

### 阶段五：训练引擎

**目标**：完成完整的训练循环，支持日志记录、模型保存、学习率调度

**5.1 训练引擎 `src/train.py`**

```python
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个 epoch，返回 avg_loss 和 avg_accuracy"""

def validate(model, dataloader, criterion, device):
    """验证，返回 val_loss, val_accuracy"""

def train(config):
    """
    完整训练流程：
    1. 加载配置
    2. 创建 DataLoader（train + val）
    3. 构建模型
    4. 定义损失函数（CrossEntropyLoss + 可选 label_smoothing）
    5. 定义优化器（AdamW）
    6. 定义学习率调度器（CosineAnnealingLR）
    7. 训练循环：
       - 支持 two_stage 策略（先冻骨干再解冻）
       - 每 epoch 记录 train_loss, train_acc, val_loss, val_acc
       - 保存 best_model（按 val_accuracy）
       - EarlyStopping
    8. 保存训练曲线图到 outputs/figures/
    9. 保存训练日志到 logs/（CSV 格式）
    """
```

**5.2 训练入口 `scripts/train.py`**

```bash
D:\anaconda\python.exe scripts/train.py --config configs/train.yaml
```

支持命令行覆盖参数：
```bash
D:\anaconda\python.exe scripts/train.py --config configs/train.yaml --model resnet18 --epochs 30 --lr 0.0005
```

**验收标准**：
```bash
# 使用少量数据快速验证训练流程（可手动创建小数据集）
python scripts/train.py --config configs/train.yaml
# 输出：checkpoints/best_model.pth
# 输出：outputs/figures/training_curve.png
# 输出：logs/train_log.csv
```

---

### 阶段六：评估引擎

**目标**：完整评估模型性能，输出论文所需的所有指标和图表

**6.1 评估引擎 `src/evaluate.py`**

```python
def evaluate(model, dataloader, device, label_names):
    """
    返回 dict:
    {
        'accuracy': float,
        'precision': np.array,       # 每类 precision
        'recall': np.array,          # 每类 recall
        'f1': np.array,              # 每类 f1
        'confusion_matrix': np.array, # 4x4
        'per_class_accuracy': dict,
        'classification_report': str  # sklearn classification_report 文本
    }
    """

def save_confusion_matrix(cm, label_names, save_path):
    """保存混淆矩阵热力图"""

def save_misclassified_samples(model, dataloader, device, label_names, save_dir, max_samples=20):
    """保存错分样本（原图 + 预测标签 + 真实标签）"""
```

**6.2 评估入口 `scripts/evaluate.py`**

```bash
D:\anaconda\python.exe scripts/evaluate.py --config configs/train.yaml --checkpoint checkpoints/best_model.pth
```

**验收标准**：
```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth
# 输出：outputs/figures/confusion_matrix.png
# 输出：outputs/figures/misclassified_samples/
# 输出：outputs/reports/evaluation_report.md
```

---

### 阶段七：推理引擎

**目标**：单图推理 + 摄像头实时推理

**7.1 推理核心 `src/infer.py`**

```python
class WasteClassifier:
    """垃圾分类推理器"""

    def __init__(self, config_path, checkpoint_path, device='cpu'):
        """
        1. 加载配置
        2. 构建模型
        3. 加载权重
        4. 初始化 OpenCV 预处理管线
        """

    def predict_image(self, image_path_or_array):
        """
        单图推理：
        1. OpenCV 读取图片
        2. OpenCV 预处理（resize → normalize）
        3. 转 tensor → 模型推理
        4. softmax → 返回 top1 类别 + 概率分布

        返回:
        {
            'class_name': '可回收物',
            'class_id': 0,
            'confidence': 0.95,
            'probabilities': {'可回收物': 0.95, '厨余垃圾': 0.03, ...}
        }
        """

    def predict_batch(self, image_paths):
        """批量推理"""
```

**7.2 单图识别入口 `scripts/infer_image.py`**

```bash
D:\anaconda\python.exe scripts/infer_image.py --image test.jpg --checkpoint checkpoints/best_model.pth
# 输出：分类结果 + 保存标注图到 outputs/predictions/
```

**7.3 摄像头实时识别入口 `scripts/infer_camera.py`**

```bash
D:\anaconda\python.exe scripts/infer_camera.py --checkpoint checkpoints/best_model.pth
# 按 Q 键退出
# 实时显示：画面 + 分类结果 + 置信度 + FPS
```

摄像头实时识别流程：
```
while True:
    1. cam.read_frame() → BGR 帧
    2. BGR 帧复制一份用于显示
    3. OpenCV 预处理：resize(224x224) → normalize
    4. 转 tensor → model.inference()
    5. softmax → 取 top1
    6. draw_result(显示帧, 类别名, 置信度)
    7. draw_fps(显示帧, fps)
    8. cv2.imshow()
    9. if cv2.waitKey(1) & 0xFF == ord('q'): break
```

**验收标准**：
```bash
# 单图
python scripts/infer_image.py --image test.jpg --checkpoint checkpoints/best_model.pth
# 控制台输出分类结果，图片保存到 outputs/predictions/

# 摄像头
python scripts/infer_camera.py --checkpoint checkpoints/best_model.pth
# 弹出窗口，实时识别，按 Q 退出
```

---

### 阶段八：Web 界面

**目标**：FastAPI + HTML Web 系统，4-Tab 多功能界面

**8.1 界面功能 `ui/app.py`**

使用 **FastAPI** 后端 + 原生 HTML/CSS/JS 前端。

界面结构（4-Tab 系统）：
```
┌──────────────────────────────────────────────────────────────┐
│  ♻ 垃圾智能分类系统     [首页] [智能识别] [批量识别] [OpenCV演示] │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Tab 1 - 首页：系统介绍 + 快速上传 + 四分类概览卡片             │
│                                                              │
│  Tab 2 - 智能识别：                                          │
│    ├── 单图识别（上传 → 标注图 + 概率分布 + 投放建议）          │
│    ├── 模型对比（四模型同时推理，对比表格）                     │
│    └── 识别历史（localStorage 保存最近 20 条）                 │
│                                                              │
│  Tab 3 - 批量识别：                                          │
│    ├── 多图上传 + 结果表格                                    │
│    ├── 统计摘要（总数/平均置信度/低置信度数）                   │
│    ├── ECharts 饼图 + 柱状图                                  │
│    └── CSV 导出                                               │
│                                                              │
│  Tab 4 - OpenCV 演示：                                       │
│    ├── 上传图片 → 预处理管线逐步可视化                         │
│    └── 原图 → 高斯去噪 → CLAHE → 模型输入                    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

功能：
1. **图片上传** → 拖拽/点击 → 自动推理 → 显示分类结果 + 置信度 + 概率分布
2. **模型切换** → 下拉选择 4 个模型，在线切换
3. **模型对比** → 上传一张图片同时调用所有模型推理，展示各模型结果和推理耗时
4. **批量识别** → 多图上传，自动生成统计摘要和 ECharts 图表
5. **OpenCV 管线演示** → 逐步展示预处理过程的可视化
6. **识别历史** → 自动保存最近 20 条，支持点击回看

**启动方式**：
```bash
D:\anaconda\python.exe ui/app.py
# 访问 http://localhost:8080
```

**验收标准**：
- 界面正常启动，4 个 Tab 切换正常
- 上传图片后正确显示分类结果和概率分布
- 模型切换和对比功能正常
- 批量识别和 CSV 导出功能正常
- OpenCV 管线可视化正常显示

---

### 阶段九：实验与论文支撑

**目标**：自动导出论文"实验分析"章节所需的全部材料

**9.1 实验脚本 `scripts/export_results.py`**

功能：
1. 对所有候选模型分别训练 + 评估
2. 汇总输出对比表：

| 模型 | Accuracy | Precision | Recall | F1 | 参数量 | 推理速度(ms) | 摄像头FPS |
|------|----------|-----------|--------|-----|-------|------------|----------|
| MobileNetV3-Small | ... | ... | ... | ... | ... | ... | ... |
| ResNet18 | ... | ... | ... | ... | ... | ... | ... |
| EfficientNet-B0 | ... | ... | ... | ... | ... | ... | ... |
| ShuffleNetV2 | ... | ... | ... | ... | ... | ... | ... |

3. 保存到 `outputs/reports/model_comparison.csv`
4. 自动生成对比柱状图保存到 `outputs/figures/`

**9.2 OpenCV 预处理对比实验**

在 `scripts/export_results.py` 中增加预处理对比功能：
- 无预处理 vs 仅 Resize vs Resize+去噪 vs Resize+CLAHE vs 全预处理
- 使用同一模型（MobileNetV3-Small）训练，比较准确率差异
- 保存对比表和对比图

**9.3 论文可交付材料清单**

| 论文章节 | 对应输出文件 |
|---------|------------|
| 系统总体架构图 | README.md 中的架构描述 |
| 数据集分布统计图 | `outputs/figures/class_distribution.png` |
| 预处理效果对比图 | `outputs/figures/preprocess_comparison.png` |
| 数据增强效果对比图 | `outputs/figures/augmentation_comparison.png` |
| 训练曲线图 | `outputs/figures/training_curve_{model}.png` |
| 混淆矩阵图 | `outputs/figures/confusion_matrix_{model}.png` |
| 模型对比表 | `outputs/reports/model_comparison.csv` |
| 预处理对比表 | `outputs/reports/preprocess_comparison.csv` |
| 错分样本展示 | `outputs/figures/misclassified_samples/` |
| 实时识别截图 | 运行 infer_camera.py 后手动截屏 |

---

## 5. 编码规范

1. **Python 版本**：3.12，使用 type hints
2. **import 顺序**：标准库 → 第三方库 → 本项目模块，各组之间空一行
3. **日志**：使用 Python 标准 `logging` 模块，不使用 `print()`
4. **路径**：所有路径使用 `pathlib.Path`，不使用字符串拼接
5. **随机种子**：统一在 `config.py` 中设置 `seed_everything(seed)` 函数
6. **中文注释**：代码注释使用中文，变量名/函数名使用英文
7. **模型权重**：保存为 `.pth`，使用 `torch.save(model.state_dict(), path)`
8. **配置覆盖**：命令行参数 > YAML 配置 > 默认值

---

## 6. 数据集迁移操作指南

### 当前状态

- 原始数据集需从百度网盘下载（见 `请看这里（新）.txt`）
- 下载后解压到 `data/raw/`，应包含 245 个子文件夹，命名格式 `{大类}_{小类}`

### 迁移步骤

```bash
# 步骤 1：下载并解压到 data/raw/
# 手动操作

# 步骤 2：运行数据整理脚本
D:\anaconda\python.exe scripts/prepare_data.py

# 步骤 3：验证数据
D:\anaconda\python.exe -c "
from src.opencv_pipeline.dataset_tools import scan_dataset
stats = scan_dataset('data/split/train')
print(stats)
"
# 预期输出：{'recyclable': N, 'kitchen': N, 'hazardous': N, 'other': N}
```

---

## 7. 执行优先级与依赖关系

```
阶段一（脚手架）
  ↓
阶段二（数据处理）  ← 需要数据集已下载到 data/raw/
  ↓
阶段三（OpenCV模块）+ 阶段四（模型定义）  ← 可并行
  ↓
阶段五（训练引擎）
  ↓
阶段六（评估引擎）
  ↓
阶段七（推理引擎）
  ↓
阶段八（Web界面）
  ↓
阶段九（实验导出）
```

**当前阻塞项**：数据集尚未下载。阶段二及之后的所有阶段都依赖数据集。
阶段一、阶段三（OpenCV 模块）、阶段四（模型定义）不依赖数据集，可以先行开发。

---

## 8. 关键注意事项

1. **OpenCV 必须是独立核心模块**，论文题目是"基于OpenCV的"，OpenCV 的工作量和独立性直接关系到毕设评分
2. **不要复用参考项目的 TensorFlow 代码**，框架不同，强行移植反而增加复杂度
3. **参考项目中可复用的文字内容**：四大类垃圾的分类说明（window_trash.py 中的四段中文描述）
4. **模型先跑通一个再扩展**：先跑通 MobileNetV3-Small 的完整流程，再添加其他候选模型
5. **所有实验结果必须可复现**：固定 seed，保存配置快照
6. **类别不均衡**：有害垃圾仅 ~5000 张（约 6%），训练时需关注其 recall，必要时使用加权损失
