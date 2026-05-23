# YOLOv5 FPS Auto-Aim System

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11-lightgrey)]()

A real-time FPS game auto-aim tool based on YOLOv5. Supports TensorRT / ONNX / PyTorch backends, OBS Virtual Camera input, and Logitech mouse control.

> English · [中文说明](#-功能说明)

**Disclaimer:** This project is for educational and research purposes only. Using this tool may violate game ToS and result in account bans. Use at your own risk.

## 📋 目录

- [功能说明](#-功能说明)
- [项目文件](#-项目文件)
- [环境要求](#-环境要求)
- [快速开始](#-快速开始)
- [按键说明](#-按键说明)
- [参数配置](#-参数配置)
- [常见问题](#-常见问题)

---

## 🎯 功能说明

- ✅ 实时屏幕捕获 — OBS Virtual Camera（主力）/ DXCam / MSS 多后端
- ✅ YOLOv5 目标检测 — TensorRT (.engine) / ONNX (.onnx) / PyTorch (.pt) 自动回退
- ✅ 智能瞄准控制 — 连续灵敏度曲线 + 自适应平滑滤波，支持头/身体切换
- ✅ 罗技鼠标驱动 — Logitech G 系列（DLL 驱动）/ GHUB 替代方案（SendInput 模拟）
- ✅ 可视化窗口 — Radar 实时检测画框 + FOV 圈 + 瞄准线
- ✅ 高 DPI 适配 + 管理员权限自动提升
- ✅ 命令行 + GUI 双版本

---

## 📁 项目文件

### 核心脚本

| 文件名 | 说明 |
|--------|------|
| `main_aim.py` | **主程序入口（推荐）** — 命令行版，OBS 优先，代码干净修复版 |
| `main_aim_gui.py` | GUI 版本 — 带图形操作界面，DXCam/MSS 后端 |
| `main_aim_optimized.py` | 优化版 v2.0 — 多线程异步捕获 + CUDA NMS，RTX 4060 60+ FPS |
| `build_engine.py` | TensorRT 引擎构建工具 — ONNX → .engine 转换 |
| `test_main_aim_runtime.py` | 运行时测试 — pytest 验证主流程参数传递 |

### 模型文件

| 文件名 | 说明 | 大小 |
|--------|------|------|
| `1225_2best.engine` | TensorRT 推理引擎（推荐，FP16） | ~17 MB |
| `1225_2best_v10.engine` | TensorRT 10.x 引擎（备用） | ~17 MB |
| `1225_2best.onnx` | ONNX 模型（CPU 回退） | ~14 MB |
| `1225_2best.pt` | PyTorch 权重（训练/微调用） | ~14 MB |

### 鼠标驱动

| 文件名 | 说明 |
|--------|------|
| `logitech.py` | 罗技 DLL 驱动封装（主力） |
| `logitech.driver.dll` | 罗技驱动 DLL (GHUB 依赖) |
| `ghub_replacement.py` | GHUB 替代方案 — 纯 SendInput 模拟，无需罗技驱动 |

### 配置文件

| 文件名 | 说明 |
|--------|------|
| `data.yaml` | YOLOv5 训练数据集配置（body/head 两类） |
| `my.yaml` | 本地数据处理配置（与本项目自瞄无关） |
| `.gitignore` | Git 忽略规则 |

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
pip install torch numpy opencv-python dxcam matplotlib pywin32 onnxruntime pytest
```

如需 TensorRT 支持（推荐，速度最快）：
```bash
pip install tensorrt pycuda
```

### 第二步：启动 OBS Virtual Camera

本系统通过 OBS Virtual Camera 获取游戏画面：
1. 打开 OBS Studio
2. 添加游戏源（Game Capture / Window Capture）
3. 点击「启动虚拟摄像机」

### 第三步：运行程序

以**管理员身份**运行（鼠标驱动需要系统级权限）：

```bash
python main_aim.py
```

可选：指定摄像头索引
```bash
python main_aim.py --obs-camera-index 0
```

### 第四步：检查状态

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

## ⚙️ OBS 捕获模式（高级）

当前 `main_aim.py` 默认使用 OBS Virtual Camera 作为输入。如需调整摄像头索引：

```bash
# 命令行参数
python main_aim.py --obs-camera-index 0

# 或环境变量
set OBS_CAMERA_INDEX=0
python main_aim.py
```

GUI 版 (`main_aim_gui.py`) 和优化版 (`main_aim_optimized.py`) 使用 DXCam (DXGI 硬件捕获)，不依赖 OBS。

---

## ⚙️ 参数配置

在 `main_aim.py` 顶部修改：

```python
# ---- 模型/检测参数 ----
DETECTION_SIZE = 320       # 检测尺寸 (320 或 640)
CONF_THRES = 0.5           # 置信度阈值 (0.0 - 1.0)
IOU_THRES = 0.45           # NMS IOU 阈值
MAX_DETECTIONS = 20        # 最大检测数
AIM_FOV_RADIUS = 300       # 自瞄 FOV 半径 (像素)

# ---- 性能参数 ----
DXCAM_MAX_FPS = 144        # 目标最大 FPS
BATCH_SIZE = 1             # 批量推理 (1=实时)
BATCH_MAX_LATENCY_MS = 8   # 批量最大等待时间 (毫秒)
```

### 关键参数说明

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| DETECTION_SIZE | 320 | 小尺寸速度快，640 精度高（需模型匹配） |
| CONF_THRES | 0.5-0.7 | 阈值越高，误检越少 |
| AIM_FOV_RADIUS | 250-400 | 自瞄范围（像素），越大越容易误锁 |
| DXCAM_MAX_FPS | 144 | 目标帧率上限，降低可减少 CPU 占用 |

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



## 支持创作

如果这个项目对你有帮助，欢迎请我喝杯咖啡 ☕

<div align="center">
  <table>
    <tr>
      <td align="center"><img src="donate/wechat.jpg" width="200" alt="微信赞赏"/><br/>微信赞赏</td>
      <td align="center"><img src="donate/alipay.jpg" width="200" alt="支付宝赞赏"/><br/>支付宝赞赏</td>
    </tr>
  </table>
</div>
