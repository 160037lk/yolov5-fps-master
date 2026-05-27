# OBS Auto Aim — 云端 GPU 推理自瞄系统

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11-lightgrey)]()

基于 YOLO 多模型集成的云端 GPU 实时自瞄系统。本地 DXCam 截图 → SSH 隧道传输 → 阿里云 RTX 5880 推理 → 本地鼠标控制。

**免责声明：** 本项目仅供学习和研究使用。使用本工具可能违反游戏服务条款并导致账号封禁，使用者自行承担风险。

---

## 架构

```
本地 PC (Windows)                          云端 (阿里云 RTX 5880 48GB)
─────────────────                          ─────────────────────────────
DXCam 屏幕截图 (DirectX)                   cloud_server_ensemble.py
    ↓ JPEG 编码                               ↓ ONNX CUDA 推理
TCP → [4B大小][JPEG数据] ── SSH 隧道 ──→  Dawan v11s 4尺度 + V5/V8
    ↓                                       ↓ NMS 融合 (同目标取最高置信度)
TCP ← [1B数量][N×6×float32] ←── SSH 隧道 ── 返回检测框
    ↓
AimController → Interception 内核驱动 → 鼠标移动
```

- **截图**: DXCam (DXGI Desktop Duplication API)，~1ms 延迟
- **传输**: SSH 端口转发 (本地 29999 → 云端 9999)，绕过阿里云防火墙
- **推理**: 多模型集成 + NMS 融合，~67ms/帧
- **鼠标**: Interception 内核驱动（通用），SendInput 自动回退

---

## 文件说明

### 云端推理 (当前版本)

| 文件 | 说明 |
|------|------|
| `main_aim_cloud.py` | **云端客户端** — DXCam 截图 → 云端推理 → 鼠标控制 |
| `cloud_server_ensemble.py` | **云端服务器** — 多模型集成推理 (Dawan v11s 4尺度 + V5/V8 NMS融合) |
| `start.bat` | **一键启动** — SSH 隧道 + 客户端 |
| `run_cloud_tunnel.bat` | SSH 隧道模式启动脚本 |

### 鼠标驱动

| 文件 | 说明 |
|------|------|
| `interception.dll` | Interception 内核驱动 DLL (x64，通用所有鼠标) |
| `install-interception.exe` | Interception 驱动安装器 |
| `logitech.py` | 罗技 DLL 驱动封装 (仅罗技鼠标) |
| `logitech.driver.dll` | 罗技驱动 DLL |
| `ghub_replacement.py` | SendInput 模拟方案 (无需硬件驱动) |

### 配置文件

| 文件 | 说明 |
|------|------|
| `config.json` | 瞄准参数配置 (FOV、速度、热键等) |

### 本地推理 (旧版)

| 文件 | 说明 |
|------|------|
| `main_aim.py` | 本地推理主程序 (OBS Virtual Camera) |
| `main_aim_gui.py` | GUI 版本 |
| `main_aim_optimized.py` | 优化版 v2.0 (多线程 + CUDA NMS) |
| `build_engine.py` | TensorRT 引擎构建工具 |
| `1225_2best.engine` | TensorRT 推理引擎 (FP16) |
| `1225_2best.onnx` | ONNX 模型 (CPU 回退) |
| `1225_2best.pt` | PyTorch 权重 |

---

## 环境要求

### 本地 PC

- Windows 10/11 (64位)
- Python 3.8+
- NVIDIA GPU (可选，云端推理不需要)
- 管理员权限

### Python 依赖

```bash
pip install opencv-python numpy dxcam mss pywin32
```

### 云端服务器

- 阿里云 GPU 实例 (RTX 5880 48GB)
- Python 3.8+
- ONNX Runtime CUDA
- 模型文件: Dawan v11s 4尺度

---

## 快速开始

### 第一步：部署云端服务器

将 `cloud_server_ensemble.py` 和模型文件上传到云服务器：

```bash
# 上传文件
scp -P 53414 cloud_server_ensemble.py root@8.160.149.149:/root/deploy/

# 启动服务器 (4个 Dawan 模型集成)
ssh -p 53414 root@8.160.149.149
cd /root/deploy
python3 cloud_server_ensemble.py --port 9999 --models dawan --dawan-dir /root/deploy --obs-dir /root/deploy
```

### 第二步：配置 SSH 密钥

将 `cloud_rsa` 私钥放到 `C:\Users\<用户名>\.ssh\` 目录下。

### 第三步：启动客户端

**方式一：一键启动**
```
双击 start.bat
```

**方式二：手动启动**
```bash
# 终端1: 启动 SSH 隧道
ssh -o StrictHostKeyChecking=no -o IdentitiesOnly=yes -i "%USERPROFILE%\.ssh\cloud_rsa" -N -L 29999:127.0.0.1:9999 root@8.160.149.149 -p 53414

# 终端2: 运行客户端
python main_aim_cloud.py --host 127.0.0.1 --port 29999 --target head --mouse-method interception
```

### 第四步：检查状态

启动后显示：
```
[Capture] DXCam initialized (1920x1080)
[Cloud] Connected to 127.0.0.1:29999
[Mouse] Interception kernel driver (device=12)
==================================================
[SYSTEM READY] Cloud: 127.0.0.1:29999
  FOV: 9999px | Speed: 0.5
  Target: HEAD
  [] Toggle Aim  [] Toggle Target  [\] Radar  [Q] Quit
==================================================
```

---

## 按键说明

| 按键 | 功能 |
|------|------|
| `]` | 开启/关闭自瞄 |
| `[` | 切换目标（头/身体） |
| `\` | 显示/隐藏雷达窗口 |
| `Q` | 退出程序 |

---

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `8.160.149.149` | 云端服务器 IP |
| `--port` | `9999` | 云端服务器端口 |
| `--capture-size` | `640` | 截图区域大小 (像素) |
| `--jpeg-quality` | `75` | JPEG 压缩质量 (越低越快) |
| `--fov` | `55` | 自瞄 FOV 半径 (默认全屏) |
| `--speed` | `0.5` | 瞄准速度 |
| `--track-speed` | `0.015` | 跟踪速度 |
| `--target` | `head` | 目标类型 (`head` / `body`) |
| `--mouse-method` | `interception` | 鼠标驱动 (`interception` / `kmbox` / `logitech`) |
| `--tunnel` | 否 | 自动建立 SSH 隧道 |
| `--tunnel-host` | `127.0.0.1` | 隧道本地地址 |
| `--tunnel-port` | `19999` | 隧道本地端口 |

---

## 通信协议

客户端与服务端之间使用 TCP 二进制协议：

**请求 (Client → Server):**
```
[4 bytes: JPEG 大小 uint32 LE] [N bytes: JPEG 图像数据]
```

**响应 (Server → Client):**
```
[1 byte: 检测数量 uint8] [N × 24 bytes: 每个检测 6×float32]
```
每个检测 = `[x1, y1, x2, y2, conf, cls]` (24 字节)

---

## 推理策略

云端服务器采用多模型集成 + NMS 融合策略：

1. 所有模型独立推理同一帧
2. 将所有检测结果汇总
3. 同目标（IoU > 阈值）取置信度最高的
4. 返回融合后的检测框

支持的模型:
- `dawan` — Dawan v11s 4尺度 (320/416/512/640)，速度与精度兼顾（推荐）
- `v5` — YOLOv5
- `v8` — YOLOv8

---

## 常见问题

### Q1: 连接云端失败

- 检查 SSH 隧道是否正常: 新开终端执行 `ssh -p 53414 root@8.160.149.149` 测试
- 确认云端服务器进程在运行: `ss -tlnp | grep 9999`
- 检查云服务器防火墙和阿里云安全组

### Q2: Interception 驱动安装失败 (Windows 11)

以管理员身份运行:
```bash
install-interception.exe /install
```
如果仍失败（Win11 常见），程序会自动回退到 SendInput 模式。

### Q3: 雷达窗口没有检测框

- 确认云端服务器正常运行且返回检测结果
- 确认 SSH 隧道连接稳定
- 查看控制台输出的 `Dets:` 数量

### Q4: 鼠标不移动

- 确认以管理员身份运行
- 尝试 `--mouse-method sendinput`（纯 API 模式）
- Interception 驱动在 Win11 上可能不兼容，会自动回退

---

## 支持创作

如果这个项目对你有帮助，欢迎请我喝杯咖啡

<div align="center">
  <table>
    <tr>
      <td align="center"><img src="donate/wechat.jpg" width="200" alt="微信赞赏"/><br/>微信赞赏</td>
      <td align="center"><img src="donate/alipay.jpg" width="200" alt="支付宝赞赏"/><br/>支付宝赞赏</td>
    </tr>
  </table>
</div>
