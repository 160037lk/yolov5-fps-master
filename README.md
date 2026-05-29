# CloudVision — 分布式 GPU 实时视觉感知研究框架

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11-lightgrey)]()

基于云边协同架构的实时目标检测、多模型集成推理与人机交互行为模拟研究平台。

---

> ## ⚠️ 免责声明
>
> **本项目仅供科学研究、教学和学术交流目的使用。**
>
> - 本项目涉及的技术（目标检测、屏幕捕获、实时推理）仅用于计算机视觉与人工智能领域的学术研究和教学演示。
> - 任何将本项目用于游戏作弊、违反游戏服务条款的行为均与本项目作者无关。
> - 使用者应自行承担所有风险和责任，包括但不限于账号封禁、法律后果。
> - 请遵守当地法律法规和相关平台的服务条款。
>
> **本项目不开源任何预训练模型权重，不提供任何可用于直接作弊的完整工具链。**

---

## 研究背景

随着 YOLO 系列目标检测算法的快速发展，实时计算机视觉在边缘计算、自动驾驶、安防监控等领域取得了显著成果。本研究旨在探索在云端 GPU 集群上进行低延迟目标检测推理的可行性，为分布式 AI 推理架构提供实验数据和工程参考。

### 技术挑战

1. **低延迟图像传输** — 如何在有限带宽下将高分辨率实时画面传输到云端
2. **多模型集成推理** — 如何融合多个异构模型的检测结果以提高精度
3. **推理结果回传** — 如何将检测坐标低延迟返回客户端

---

## 系统架构

```
本地 PC (Windows)                          云端 (GPU 服务器)
─────────────────                          ─────────────────────
OBS Virtual Camera                        cloud_server_ensemble.py
    ↓ 图像采集                                ↓ ONNX Runtime CUDA
TCP → [4B大小][JPEG数据] ── SSH 隧道 ──→  多模型集成 (NMS 融合)
    ↓                                        ↓
TCP ← [1B数量][N×6×float32] ←── SSH 隧道 ── 返回检测框 [x1,y1,x2,y2,conf,cls]
    ↓
KMBox NET 硬件外设 → 标准 USB HID 输出
```

### 核心研究点

- **截图方案**: DXGI Desktop Duplication API (DXCam) / OBS Virtual Camera
- **传输协议**: TCP 二进制协议，SSH 隧道加密穿透防火墙
- **推理引擎**: ONNX Runtime CUDA，支持 YOLOv5/v8/Dawan 系列
- **多模型融合**: 所有模型独立推理 → 类别内 NMS → 同目标取最高置信度
- **性能数据**: 4× Dawan 模型集成推理 ~67ms/帧 (RTX 5880 48GB)

---

## 项目结构

### 云端推理 (主版本)

| 文件 | 说明 |
|------|------|
| `video_bridge.py` | **研究版客户端** — 低检测特征，OBS Virtual Camera + KMBox NET，拟人化行为模拟 |
| `main_aim_cloud.py` | **标准版客户端** — DXCam 截图 + Interception/SendInput 鼠标控制 |
| `cloud_server_ensemble.py` | **云端推理服务器** — 多模型集成推理 + NMS 融合 |

### 鼠标控制

| 文件 | 说明 |
|------|------|
| `logitech.py` | 罗技 DLL 驱动封装 (仅罗技鼠标，用于硬件级输入研究) |
| `ghub_replacement.py` | Windows SendInput API 模拟方案 (用于对比研究) |

### 本地推理 (旧版)

| 文件 | 说明 |
|------|------|
| `main_aim.py` | 本地推理主程序 (OBS Virtual Camera + TensorRT) |
| `main_aim_gui.py` | 带 GUI 配置界面的本地推理版本 |
| `main_aim_optimized.py` | 多线程优化版本 (实验性) |
| `main_aim_full.py` | 全功能版本，支持多种模型后端 |
| `build_engine.py` | TensorRT 引擎构建工具 |
| `config.json` | 瞄准参数配置模板 |

---

## 环境要求

### 本地 PC

- Windows 10/11 (64-bit)
- Python 3.8+
- OBS Studio 30.0+ (用于 Virtual Camera 截图)

### 云端服务器

- Linux + NVIDIA GPU
- Python 3.8+
- ONNX Runtime CUDA
- 模型文件: 需自行准备 YOLO/Dawan 系列 ONNX 模型

---

## 快速开始 (研究用途)

### 第一步：部署云端服务器

```bash
# 上传推理服务器到云主机
scp -P 53414 cloud_server_ensemble.py root@<服务器IP>:/root/deploy/

# SSH 登录并启动
ssh -p 53414 root@<服务器IP>
cd /root/deploy
python3 cloud_server_ensemble.py --port 9999 --models dawan \
    --dawan-dir /root/deploy --obs-dir /root/deploy
```

### 第二步：安装本地依赖

```bash
pip install opencv-python numpy pillow paramiko sshtunnel dxcam mss pywin32
```

### 第三步：配置 OBS Virtual Camera

1. 启动 OBS Studio
2. 添加游戏源 (Game Capture / 窗口捕获)
3. 工具 → 虚拟摄像机 → 开启

### 第四步：运行研究版客户端

```bash
# SSH 隧道模式
python video_bridge.py --tunnel --tunnel-host 127.0.0.1 --tunnel-port 29999

# 直连模式 (云服务器防火墙已开放端口)
python video_bridge.py --host <服务器IP> --port 9999

# 完整参数
python video_bridge.py \
    --tunnel \
    --fov 250 \
    --speed 1.0 \
    --target head \
    --show-radar \
    --kmbox-ip 192.168.2.188 \
    --kmbox-port <盒子端口> \
    --kmbox-uuid <盒子UUID>
```

---

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `8.160.149.149` | 云端服务器 IP |
| `--port` | `9999` | 云端服务器端口 |
| `--capture-size` | `640` | 截图区域大小 (px) |
| `--jpeg-quality` | `75` | JPEG 压缩质量 |
| `--fov` | `250` | 检测区域半径 (px) |
| `--speed` | `1.0` | 灵敏度系数 |
| `--target` | `head` | 检测目标类型 (`head` / `body`) |
| `--tunnel` | 否 | 启用 SSH 隧道模式 |
| `--tunnel-host` | `127.0.0.1` | 隧道本地绑定地址 |
| `--tunnel-port` | `29999` | 隧道本地绑定端口 |
| `--show-radar` | 否 | 显示检测可视化窗口 |
| `--kmbox-ip` | `192.168.2.188` | KMBox NET IP 地址 |
| `--kmbox-port` | `12345` | KMBox NET 端口 |
| `--kmbox-uuid` | 无 | KMBox NET UUID |

---

## 操作说明

| 按键 | 功能 |
|------|------|
| 按住 `左Alt` | 触发检测追踪 |
| `Home` | 切换检测目标（头/身体） |
| `End` | 显示/隐藏可视化窗口 |
| `Delete` | 退出程序 |

---

## 通信协议

客户端与云端之间使用自定义 TCP 二进制协议：

**请求 (Client → Server):**
```
[4 bytes: JPEG 大小 uint32 LE] [N bytes: JPEG 图像数据]
```

**响应 (Server → Client):**
```
[1 byte: 检测数量 uint8] [N × 24 bytes: 每框 6×float32]
```
每个检测框 = `[x1, y1, x2, y2, confidence, class_id]` (24 字节)

---

## 推理策略

云端服务器 (`cloud_server_ensemble.py`) 采用多模型集成 + NMS 融合策略：

1. 所有已注册模型独立推理同一帧
2. 汇总全部检测结果
3. 类别内 NMS（同目标 IoU > 阈值取最高置信度）
4. 返回融合后的检测框列表

支持的模型架构：
- `dawan` — Dawan 系列 (多尺度 320/416/512/640)
- `v5` — YOLOv5
- `v8` — YOLOv8

---

## 拟人化行为模拟研究

`video_bridge.py` 中的 `HumanAimController` 实现了以下拟人化行为参数：

| 参数 | 范围 | 研究目的 |
|------|------|----------|
| 反应延迟 | 80-220ms 随机 | 模拟人类视觉-运动神经延迟 |
| 微抖动 | ±12% 随机 | 模拟手部自然震颤 |
| 过冲模拟 | 15% 概率, 25% 过冲 | 模拟人手 overshoot 后拉回 |
| 目标切换停顿 | 350ms | 模拟注意力切换代价 |
| FOV 限制 | 250px (默认) | 模拟人眼注意力集中区域 |

这些参数的设置参考了人机交互 (HCI) 领域的 Fitts' Law 和相关运动控制研究文献。

---

## 常见问题

### 连接云端失败

- 检查 SSH 隧道: `ssh -p 53414 root@<IP>` 测试连通性
- 确认云端进程运行: `ss -tlnp | grep 9999`
- 检查阿里云安全组 / 防火墙规则

### OBS Virtual Camera 未检测到

```bash
# 确认 OBS 虚拟摄像机已开启
# OBS → 工具 → 虚拟摄像机 → 启动

# 用其他程序测试虚拟摄像机是否正常输出画面
```

### KMBox NET 初始化失败

- 确认 USB 网卡驱动已安装 (插入盒子后我的电脑出现磁盘，运行其中 exe)
- `ping 192.168.2.188` 检查网络连通
- 核对盒子上显示的 IP / Port / UUID 参数是否与命令行一致
- 将 `kmNet.cp3xx-win_amd64.pyd` 复制到项目目录并改名为 `kmNet.pyd`

---

## 参考文献

- Redmon, J., et al. "You Only Look Once: Unified, Real-Time Object Detection." CVPR, 2016.
- Ultralytics. YOLOv5/YOLOv8. https://github.com/ultralytics
- Fitts, P. M. "The information capacity of the human motor system in controlling the amplitude of movement." Journal of Experimental Psychology, 1954.
- Card, S. K., et al. "The Psychology of Human-Computer Interaction." 1983.

---

## 许可证

MIT License. 详见 [LICENSE](LICENSE) 文件。

---

## 支持创作

如果这个项目对你的研究有帮助，欢迎请我喝杯咖啡

<div align="center">
  <table>
    <tr>
      <td align="center"><img src="donate/wechat.jpg" width="200" alt="微信赞赏"/><br/>微信赞赏</td>
      <td align="center"><img src="donate/alipay.jpg" width="200" alt="支付宝赞赏"/><br/>支付宝赞赏</td>
    </tr>
  </table>
</div>
