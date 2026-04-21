# YOLOv5 FPS 自动瞄准系统

一个基于 YOLOv5 的实时 FPS 游戏自动瞄准工具，支持游戏如 CS2、VALORANT 等。

## 📋 目录

- [功能说明](#-功能说明)
- [文件说明](#-文件说明)
- [环境要求](#-环境要求)
- [快速开始](#-快速开始)
- [按键说明](#-按键说明)
- [参数配置](#-参数配置)
- [常见问题](#-常见问题)

---

## 🎯 功能说明

- ✅ 实时屏幕捕获（DXCam）
- ✅ YOLOv5 目标检测（支持 .engine/.onnx/.pt 模型）
- ✅ 智能瞄准控制（支持头/身体切换）
- ✅ 罗技鼠标驱动（支持 Logitech G 系列鼠标）
- ✅ 可视化界面（Radar 窗口）
- ✅ 高 DPI 适配

---

## 📁 文件说明

### 必需文件

| 文件名 | 说明 | 大小 |
|--------|------|------|
| `main_aim.py` | 主程序入口 | ~17 KB |
| `1225_2best.engine` | TensorRT 推理引擎（推荐） | ~17 MB |
| `1225_2best.onnx` | ONNX 模型（备用） | ~14 MB |
| `1225_2best.pt` | PyTorch 模型（备用） | ~14 MB |
| `logitech.py` | 罗技鼠标驱动 | ~2 KB |
| `logitech.driver.dll` | 罗技驱动 DLL | ~37 KB |
| `data.yaml` | 数据集配置 | ~520 B |

### 可选文件

| 文件名 | 说明 |
|--------|------|
| `ghub_replacement.py` | GHubs 替代方案（备用驱动） |
| `my.yaml` | 自用配置文件（项目内部使用） |

### 输出目录

```
runs/
├── train/           # 训练结果（如果重新训练）
├── detect/          # 检测结果（如果使用 detect.py）
└── exp*/            # 各次训练的实验记录
```

---

## 💻 环境要求

### Python 依赖

```
torch>=2.0.0
numpy>=1.24.0
opencv-python>=4.5.0
dxcam>=0.1.0
matplotlib>=3.7.0
pywin32>=306
onnxruntime-gpu>=1.15.0  # 可选，ONNX 推理用
tensorrt>=8.5.0          # 可选，TensorRT 推理用
pycuda>=2022.2             # 可选，TensorRT 推理用
```

### 系统要求

- **操作系统**：Windows 10/11 (64位)
- **GPU**：NVIDIA GPU (支持 CUDA)
- **内存**：>= 8 GB（推荐 16 GB）
- **管理员权限**：需要

---

## 🚀 快速开始

### 第一步：安装依赖

```bash
pip install torch numpy opencv-python dxcam matplotlib pywin32
```

如需 ONNX 支持：
```bash
pip install onnxruntime-gpu
```

如需 TensorRT 支持：
```bash
pip install tensorrt pycuda
```

### 第二步：运行程序

1. **以管理员身份运行**：
   - 右键点击 `main_aim.py`
   - 选择"以管理员身份运行"

2. **或使用命令行**：
   ```bash
   python main_aim.py
   ```

### 使用 OBS 作为画面输入

`main_aim.py` 现在支持通过 OBS 作为屏幕/帧输入源。

推荐方式：
- 优先使用 OBS Virtual Camera
- 若虚拟摄像头不可用，则回退到 obs-websocket 截图接口

Windows 命令行示例：
```bat
set CAPTURE_BACKEND=obs
set OBS_CAMERA_INDEX=0
set OBS_SOURCE_NAME=你的OBS源名称
set OBS_HOST=127.0.0.1
set OBS_PORT=4455
python main_aim.py
```

说明：
- `CAPTURE_BACKEND=obs` 时启用 OBS 输入。
- 若能打开 OBS Virtual Camera，脚本会直接读取虚拟摄像头画面。
- 若虚拟摄像头不可用，会尝试连接 obs-websocket 并抓取截图。
- 当前 obs-websocket 模式仅支持未启用认证的连接；如开启了认证，需要后续补充鉴权逻辑。
- WSL 中只能做静态/语法验证，实际 OBS 采集与鼠标联动仍需在 Windows + OBS 环境中验证。

### 第三步：检查状态

启动后会显示：
```
==============================
[SYSTEM READY] - Press 'q' to quit
==============================
```

---

## ⌨️ 按键说明

| 按键 | 功能 |
|------|------|
| `]` (右中括号) | 开启/暂停自动瞄准 |
| `[` (左中括号) | 切换锁定目标（头/身体） |
| `\` (反斜杠) | 切换识别窗口显示 |
| `q` | 退出程序 |

---

## ⚙️ OBS 捕获模式

如果你想改成由 OBS 提供画面，而不是直接走 DXCam/MSS，可以在启动前设置环境变量：

```bash
set CAPTURE_BACKEND=obs
set OBS_CAMERA_INDEX=0
set OBS_SOURCE_NAME=
set OBS_HOST=127.0.0.1
set OBS_PORT=4455
set OBS_PASSWORD=
python main_aim.py
```

说明：
- 优先尝试 `OBS Virtual Camera`
- 如果虚拟摄像头打不开，会回退到 `obs-websocket` 截图接口
- `OBS_SOURCE_NAME` 留空时，使用 OBS program output 截图；如果要抓某个源，填入源名称
- 当前代码要求 OBS websocket 不启用密码认证；如果启用了密码，程序会明确报错
- 首次建议先在 OBS 中确认：
  1. 已启动 Virtual Camera，或
  2. 已启用 obs-websocket v5，端口与脚本一致

---

## ⚙️ 参数配置

在 `main_aim.py` 的顶部修改参数：

```python
# ================== ⚡ 参数设置区 ⚡ ==================
DETECTION_SIZE = 320          # 检测尺寸 (320 或 640)
DXCAM_MAX_FPS = 144           # 最大帧率 (防止 CPU 过高)
CONF_THRES = 0.5              # 置信度阈值 (0.0 - 1.0)
IOU_THRES = 0.45              # IoU 阈值 (0.0 - 1.0)
MAX_DETECTIONS = 20           # 最大检测数量
AIM_FOV_RADIUS = 300          # 自瞄准 FOV 半径 (像素)
# ========================================================
```

### 参数说明

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| DETECTION_SIZE | 320 | 小尺寸速度快，640 精度高 |
| CONF_THRES | 0.5-0.7 | 阈值越高，误检越少 |
| AIM_FOV_RADIUS | 250-400 | 自动瞄准范围 |

---

## 🔧 按键配置（高级）

在 `main_aim.py` 中修改 `KEYS` 字典：

```python
KEYS = {
    'TOGGLE_WIN': [0xDC],  # \ 键
    'TOGGLE_AIM': [0xDD],  # ] 键
    'TOGGLE_TARGET': [0xDB],  # [ 键
    'QUIT': [0x51]  # q 键
}
```

使用 [VkKeyScanEx](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-vkkeyscanexw) 获取按键码。

---

## ❓ 常见问题

### Q1: 程序提示"No module named 'logitech'"

**A**: 确保 `logitech.py` 和 `main_aim.py` 在同一目录。

---

### Q2: 程序提示"Driver not ok"

**A**:
1. 检查 `logitech.driver.dll` 是否存在
2. 确保以管理员身份运行
3. 罗技鼠标驱动已安装

---

### Q3: 程序闪退或无反应

**A**:
1. 检查 CUDA 是否安装：`python -c "import torch; print(torch.cuda.is_available())"`
2. 查看是否有模型文件：`dir 1225_2best.*`
3. 检查管理员权限

---

### Q4: 如何切换到身体检测？

**A**: 按 `[` 键切换目标类型。

显示：
```
[HEAD] AIMING | Dist: 123   # 头部检测
[BODY] AIMING | Dist: 123   # 身体检测
```

---

### Q5: 如何关闭雷达窗口？

**A**: 按 `\` 键toggle，或双击雷达窗口关闭。

---

### Q6: 精度不够怎么办？

**A**:
1. 提高 `CONF_THRES` 到 0.6-0.7
2. 检查模型是否是用 640 尺寸训练的，如果是改成 `DETECTION_SIZE = 640`
3. 增加训练数据重新训练模型

---

## 🔁 模型转换（可选）

### PyTorch → ONNX

```bash
python export.py --weights 1225_2best.pt --include onnx
```

### ONNX → TensorRT

```bash
# 使用 trtexec 工具
trtexec --onnx=1225_2best.onnx --saveEngine=1225_2best.engine
```

---

## 📝 更新日志

### v1.0 
- 初始版本发布
- 支持 TensorRT/ONNX/PyTorch 模型
- 支持头/身体目标切换
- 集成罗技鼠标驱动

---

## ⚠️ 免责声明

本项目仅用于学习和研究目的。使用本工具进行游戏作弊可能违反游戏服务条款，导致账号封禁。

**使用者需自行承担使用风险**。

---

## 💬 技术支持

如有问题请检查：
1. Python 环境是否正确
2. 依赖包是否完整安装
3. 管理员权限是否开启
4. CUDA 驱动是否匹配

---

