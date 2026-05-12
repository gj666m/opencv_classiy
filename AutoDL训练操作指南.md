# AutoDL 远程训练操作指南

> 在 Cursor 中通过 Remote-SSH 连接 AutoDL，训练 4 个模型并下载结果回本地。
> 按顺序执行每一步即可，不需要额外查资料。

---

## 第一步：本地打包项目文件

在 Cursor 的**本地终端**中执行（确保当前目录是项目根目录）：

```bash
cd E:/垃圾分类数据集和tf代码-8w张图片245个类(更新)
```

打包代码文件（很小，秒完成）：

```bash
tar -czf src_package.tar.gz src/ configs/ scripts/ requirements.txt
```

打包数据集（约 8GB，需要几分钟）：

```bash
cd data
tar -czf ../data_split.tar.gz split/ meta/
cd ..
```

完成后项目根目录会多出两个文件：
- `src_package.tar.gz` — 我们写的代码和配置（src/、configs/、scripts/）
- `data_split.tar.gz` — 训练数据（data/split/、data/meta/）

---

## 第二步：注册 AutoDL 并租机器

1. 打开 https://www.autodl.com ，注册并登录
2. 充值 10~20 元（右上角）
3. 左侧菜单点「容器实例」→ 右上角「创建实例」
4. 选择配置：
   - GPU：选 **RTX 3090**（约 1.5 元/小时，性价比最高）
   - 镜像：选 **PyTorch 2.1.0** + **Python 3.10**
   - 数据盘：勾上
5. 点「创建」，等待状态变为「运行中」（约 30 秒）

---

## 第三步：记下 SSH 登录信息

实例开机后，在实例卡片上找到：
- **登录指令**：类似 `ssh -p 38909 root@region-1.autodl.com`
- **登录密码**：点「复制密码」

把这两个信息记下来，第四步要用。

---

## 第四步：Cursor 安装 Remote-SSH 插件

1. 打开 Cursor
2. 左侧点扩展图标（四个方块那个）
3. 搜索 `Remote-SSH`
4. 安装作者为 **Microsoft** 的那个插件

> 如果之前已装过，跳过此步。

---

## 第五步：Cursor 连接 AutoDL 服务器

1. 按 `Ctrl+Shift+P`
2. 输入 `remote ssh`
3. 选择 **「Remote-SSH: Connect to Host...」**
4. 把第三步的**登录指令粘贴进去**（注意去掉末尾空格！）
5. 弹窗选择操作系统 → 选 **Linux**
6. 输入第三步的**密码** → 回车
7. 等待连接（第一次会慢一些，要装服务端）

连接成功后，Cursor 左下角会显示绿色的 `>> SSH: region-xx.autodl.com`

---

## 第六步：上传文件到服务器

连上远程后，在 Cursor 菜单栏选「文件 → 打开文件夹」，输入 `/root`，回车。

### 6.1 上传代码包

把本地项目目录下的 `src_package.tar.gz` **直接拖拽**到 Cursor 左侧文件面板的 `/root` 目录下。

然后在 Cursor 的**远程终端**（菜单栏：终端 → 新建终端）中执行：

```bash
cd /root
tar -xzf src_package.tar.gz
```

### 6.2 上传数据包

同样把本地的 `data_split.tar.gz` **拖拽**到 `/root` 目录下。

> 如果拖拽很慢或失败，可以用 AutoDL 网页端的「公网网盘」上传，再在终端执行 `autodl-tmp` 命令移动过来。

然后在**远程终端**执行：

```bash
cd /root
tar -xzf data_split.tar.gz
```

### 6.3 验证文件完整性

```bash
ls -la /root/src/
ls -la /root/configs/
ls -la /root/data/split/train/
```

应该能看到：
- `src/` 下有 `train.py`、`evaluate.py`、`infer.py` 等
- `configs/` 下有 `base.yaml`、`train.yaml`、`infer.yaml`
- `data/split/train/` 下有 `recyclable/`、`kitchen/`、`hazardous/`、`other/` 四个文件夹

---

## 第七步：安装依赖

在**远程终端**执行：

```bash
pip install pyyaml scikit-learn tqdm
```

验证 PyTorch 和 GPU：

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

应该输出类似：
```
PyTorch 2.1.0
CUDA: True
GPU: NVIDIA GeForce RTX 3090
```

---

## 第八步：开守护进程（重要！）

SSH 连接断开后训练会中断，必须用 screen 守护：

```bash
# 创建一个名为 train 的会话
screen -S train

# 进入了 screen 终端，后续命令都在这里执行
```

> 提示：按 `Ctrl+A` 再按 `D` 可以暂时脱离（训练继续跑）
> 重新进入：`screen -r train`

---

## 第九步：开始训练

在 screen 终端里，依次训练 4 个模型：

```bash
cd /root
```

**模型 1：MobileNetV3-Small（主力模型）**

```bash
python scripts/train.py --config configs/train.yaml --model mobilenet_v3_small --epochs 50
```

**模型 2：ResNet18（对比实验）**

```bash
python scripts/train.py --config configs/train.yaml --model resnet18 --epochs 50
```

**模型 3：EfficientNet-B0（对比实验）**

```bash
python scripts/train.py --config configs/train.yaml --model efficientnet_b0 --epochs 50
```

**模型 4：ShuffleNetV2（对比实验）**

```bash
python scripts/train.py --config configs/train.yaml --model shufflenet_v2 --epochs 50
```

每个模型训练时会实时打印：
```
Epoch 1/50 (12.3s) | Train Loss=0.4521 Acc=0.8412 | Val Loss=0.3214 Acc=0.8724 | LR=0.001000
```

> 预计每个模型 50 epoch 约 15~25 分钟（RTX 3090）
> 4 个模型总共约 1~1.5 小时，花费约 2~3 元

---

## 第十步：训练完成后评估每个模型

```bash
cd /root

# 评估 MobileNetV3-Small
python scripts/evaluate.py --checkpoint checkpoints/best_mobilenet_v3_small.pth --split test

# 评估 ResNet18
python scripts/evaluate.py --checkpoint checkpoints/best_resnet18.pth --split test

# 评估 EfficientNet-B0
python scripts/evaluate.py --checkpoint checkpoints/best_efficientnet_b0.pth --split test

# 评估 ShuffleNetV2
python scripts/evaluate.py --checkpoint checkpoints/best_shufflenet_v2.pth --split test
```

评估结果会输出到 `outputs/` 目录下。

---

## 第十一步：打包结果并下载到本地

### 11.1 远程终端打包

```bash
cd /root
tar -czf training_results.tar.gz checkpoints/ logs/ outputs/
```

### 11.2 下载到本地

方式一：在 Cursor 文件面板中找到 `/root/training_results.tar.gz`，**右键 → 下载**

方式二：回到本地 Cursor 终端，用 SCP 下载（把端口号和地址换成你自己的）：

```bash
scp -P 38909 root@region-1.autodl.com:/root/training_results.tar.gz E:/垃圾分类数据集和tf代码-8w张图片245个类(更新)/
```

### 11.3 本地解压覆盖

在本地 Cursor 终端：

```bash
cd E:/垃圾分类数据集和tf代码-8w张图片245个类(更新)
tar -xzf training_results.tar.gz
```

这样 checkpoints、logs、outputs 就回到本地项目目录了。

---

## 第十二步：关闭 AutoDL 实例

回到 AutoDL 网页 → 容器实例 → 选择你的实例 → **关机**

> 关机后不再计费。下次需要训练再开机即可。

---

## 常见问题

**Q: 训练到一半 SSH 断了怎么办？**
A: 用 screen 守护的不会断。重新连上后执行 `screen -r train` 回到训练界面。

**Q: 怎么查看训练进度？**
A: 重新连上 SSH → `screen -r train` 即可看到实时日志。

**Q: 想只训练一个模型试试？**
A: 只跑第九步的第一条命令即可，其他跳过。

**Q: 磁盘空间不够怎么办？**
A: 把数据集放在 AutoDL 的 `/root/autodl-tmp/` 目录下（这是数据盘），不会占系统盘空间。需要修改配置文件中的路径：
```bash
# 修改 configs/base.yaml 中的 split_dir
sed -i 's|split_dir: "data/split"|split_dir: "/root/autodl-tmp/data/split"|' configs/base.yaml
```
或者在训练命令中通过参数覆盖。
