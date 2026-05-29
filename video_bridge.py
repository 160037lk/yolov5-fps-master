#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
video_bridge — 低检测风险云端推理客户端
==========================================

架构:
  OBS Virtual Camera → 云端 GPU → KMBox NET 硬件鼠标

检测风险优化:
  · 无需管理员权限
  · OBS Virtual Camera 截图（OBS 在 ACE 白名单）
  · KMBox NET 硬件级鼠标（无软件驱动痕迹）
  · 拟人化瞄准算法（随机延迟、微抖动、过冲模拟）
  · FOV 限制（模拟人眼注意力区域）
  · 必须按住扳机键才自瞄
  · paramiko 内联 SSH 隧道（无外部 ssh.exe 进程）
  · 雷达窗口默认关闭
  · 干净命名、无敏感字符串

依赖:
  pip install opencv-python numpy pillow paramiko sshtunnel dxcam
  KMBox NET: 将 kmNet.cp3xx-win_amd64.pyd 改名 kmNet.pyd 放本目录

使用:
  # 直连模式
  python video_bridge.py --host 8.160.149.149 --port 9999

  # SSH 隧道模式 (通过 53414 端口穿透)
  python video_bridge.py --tunnel --tunnel-host 127.0.0.1 --tunnel-port 29999

热键:
  按住 [左Alt]    自瞄
  [Home]           切换 头/身体
  [End]            开/关 雷达窗口
  [Delete]         退出
"""

import argparse
import io
import os
import random
import socket
import struct
import sys
import time
import traceback
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import win32api


# ═══════════════════════════════════════════════════════════════
#  全局常量
# ═══════════════════════════════════════════════════════════════

# 热键 (全部用虚拟键码 vk_code)
VK_TRIGGER   = 0xA4   # 左Alt — 按住自瞄
VK_SWITCH    = 0x24   # Home — 切换头/身体
VK_RADAR     = 0x23   # End  — 开关雷达
VK_QUIT      = 0x2E   # Delete — 退出

# 拟人化参数
HUMAN_REACTION_MIN_MS = 80    # 最小反应延迟
HUMAN_REACTION_MAX_MS = 220   # 最大反应延迟
HUMAN_JITTER_PCT      = 0.12  # 移动量随机抖动 ±12%
HUMAN_OVERSHOOT_PCT   = 0.15  # 过冲概率 15%
HUMAN_OVERSHOOT_RATIO = 0.25  # 过冲比例 (多移 25%)
HUMAN_MISS_CHANCE      = 0.05  # 故意打偏概率 5%
HUMAN_SWITCH_DELAY_MS  = 350   # 切换目标后短暂停

# 帧率统计
FPS_WINDOW = 30


# ═══════════════════════════════════════════════════════════════
#  1. OBS Virtual Camera 捕获
# ═══════════════════════════════════════════════════════════════

def find_obs_camera(max_index=10):
    """查找 OBS Virtual Camera 设备索引 (DirectShow)。"""
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            # 读一帧测试
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                # 尝试获取设备名 (Windows DirectShow)
                try:
                    backend = cap.getBackendName() if hasattr(cap, 'getBackendName') else ''
                except Exception:
                    backend = ''
                cap.release()
                # OBS Virtual Camera 通常不是物理摄像头
                if frame.shape[1] >= 640 and frame.shape[0] >= 360:
                    return i, frame.shape
            else:
                cap.release()
    return None, None


class VideoCapture:
    """OBS Virtual Camera 截图，DXCam 回退。"""

    def __init__(self, width=640, height=640, prefer_obs=True):
        self.cap_width = width
        self.cap_height = height
        self.backend = None
        self.cap = None
        self._camera = None

        # 优先尝试 OBS Virtual Camera
        if prefer_obs:
            idx, shape = find_obs_camera()
            if idx is not None:
                self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self.backend = "obs-cam"
                h, w = shape[:2]
                print(f"[Cam] OBS Virtual Camera #{idx} ({w}x{h})")
                return

        # 回退: DXCam
        try:
            import dxcam
            self._camera = dxcam.create(output_color="BGR")
            test = self._camera.grab()
            if test is not None:
                self.backend = "dxcam"
                print(f"[Cam] DXCam ({test.shape[1]}x{test.shape[0]})")
                return
        except Exception as e:
            print(f"[Cam] DXCam unavailable: {e}")

        # 最后回退: MSS
        try:
            import mss
            self._sct = mss.mss()
            self._monitor = self._sct.monitors[1]
            self.backend = "mss"
            print(f"[Cam] MSS (monitor {self._monitor})")
        except Exception:
            raise RuntimeError(
                "No capture backend available.\n"
                "  Option 1: Start OBS → Tools → VirtualCam → Start\n"
                "  Option 2: pip install dxcam\n"
                "  Option 3: pip install mss"
            )

    def read(self):
        """读取一帧，返回中心裁剪的 BGR numpy array。"""
        if self.backend == "obs-cam":
            ret, frame = self.cap.read()
            if not ret or frame is None:
                return None
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2
            half = self.cap_width // 2
            x1, y1 = max(0, cx - half), max(0, cy - half)
            x2, y2 = min(w, cx + half), min(h, cy + half)
            return frame[y1:y2, x1:x2]

        elif self.backend == "dxcam":
            frame = self._camera.grab()
            if frame is None:
                return None
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2
            half = self.cap_width // 2
            x1, y1 = max(0, cx - half), max(0, cy - half)
            x2, y2 = min(w, cx + half), min(h, cy + half)
            return frame[y1:y2, x1:x2]

        elif self.backend == "mss":
            img = np.array(self._sct.grab(self._monitor), dtype=np.uint8)
            h, w = img.shape[:2]
            cx, cy = w // 2, h // 2
            half = self.cap_width // 2
            x1, y1 = max(0, cx - half), max(0, cy - half)
            x2, y2 = min(w, cx + half), min(h, cy + half)
            return img[y1:y2, x1:x2]

        return None

    def release(self):
        if self.cap:
            self.cap.release()
        if self._camera:
            try:
                self._camera.release()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════
#  2. 云端推理客户端 (TCP + JPEG)
# ═══════════════════════════════════════════════════════════════

class InferenceClient:
    """TCP 连接云端推理服务器。"""

    def __init__(self, host, port, jpeg_quality=75):
        self.host = host
        self.port = port
        self.jpeg_quality = jpeg_quality
        self.sock = None
        self._enc = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(5)
        self.sock.connect((self.host, self.port))
        self.sock.settimeout(2)
        print(f"[Net] connected {self.host}:{self.port}")

    def infer(self, img_bgr):
        try:
            ok, buf = cv2.imencode('.jpg', img_bgr, self._enc)
            if not ok:
                return np.empty((0, 6), dtype=np.float32)
            data = buf.tobytes()
            self.sock.sendall(struct.pack('<I', len(data)) + data)
            count = self._recv(1)[0]
            if count == 0:
                return np.empty((0, 6), dtype=np.float32)
            raw = self._recv(count * 24)
            return np.frombuffer(raw, dtype=np.float32).reshape(count, 6)
        except Exception:
            return np.empty((0, 6), dtype=np.float32)

    def _recv(self, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("disconnected")
            buf.extend(chunk)
        return bytes(buf)

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None


# ═══════════════════════════════════════════════════════════════
#  3. SSH 隧道 (paramiko 内联，无外部进程)
# ═══════════════════════════════════════════════════════════════

def start_ssh_tunnel(local_host, local_port, remote_host, remote_port,
                     ssh_host, ssh_port, ssh_user, ssh_key_path):
    """通过 sshtunnel 建立 SSH 端口转发。无外部 ssh.exe 进程。"""
    from sshtunnel import SSHTunnelForwarder

    tunnel = SSHTunnelForwarder(
        (ssh_host, ssh_port),
        ssh_username=ssh_user,
        ssh_pkey=ssh_key_path,
        remote_bind_address=(remote_host, remote_port),
        local_bind_address=(local_host, local_port),
        set_keepalive=30,
    )
    tunnel.start()
    print(f"[Tunnel] {local_host}:{local_port} → {ssh_host}:{ssh_port} → "
          f"{remote_host}:{remote_port}")
    return tunnel


# ═══════════════════════════════════════════════════════════════
#  4. KMBox NET 硬件鼠标控制
# ═══════════════════════════════════════════════════════════════

class HwMouse:
    """
    KMBox NET 硬件鼠标控制器。

    要求: 将 kmNet.cp3xx-win_amd64.pyd 复制到本目录，
          改名为 kmNet.pyd（如果已安装到 site-packages 则不用）。
    协议: 私有二进制协议，通过专用 USB 网卡通信。
    """

    def __init__(self, ip="192.168.2.188", port=None, uuid=None):
        self.ready = False
        try:
            import kmNet
            self._km = kmNet
        except ImportError:
            print("[Mouse] ERROR: kmNet.pyd not found.")
            print("  1. 将 KMBox NET 附带光盘/下载链接中的 kmNet.cp3xx-win_amd64.pyd")
            print("     复制到本目录，改名为 kmNet.pyd")
            print("  2. 或者: pip install kmNet  (如果有官方 pip 包)")
            self._km = None
            return

        # 盒子显示屏上会显示 IP / Port / UUID
        self.ip = ip
        self.port = port or "12345"   # 默认端口，实际以显示屏为准
        self.uuid = uuid or "00000000-0000-0000-0000-000000000000"

        try:
            self._km.init(self.ip, str(self.port), self.uuid)
            self.ready = True
            print(f"[Mouse] KMBox NET ready ({self.ip}:{self.port})")
        except Exception as e:
            print(f"[Mouse] KMBox NET init failed: {e}")
            print(f"  请确认: ping {self.ip} 是否通？盒子上 IP/Port/UUID 是否正确？")

    def move_auto(self, x, y, duration_ms=15):
        """拟人轨迹移动 (KMBox 内置贝塞尔曲线)。"""
        if not self.ready:
            return
        try:
            self._km.move_auto(int(x), int(y), int(duration_ms))
        except Exception:
            pass

    def move(self, x, y):
        """即时移动 (用于微调)。"""
        if not self.ready:
            return
        try:
            self._km.move(int(x), int(y))
        except Exception:
            pass

    def click(self, down=True):
        """鼠标左键 (用于自动开枪，可选)。"""
        if not self.ready:
            return
        try:
            self._km.left(1 if down else 0)
        except Exception:
            pass

    def close(self):
        pass


# ═══════════════════════════════════════════════════════════════
#  5. 拟人化瞄准控制器
# ═══════════════════════════════════════════════════════════════

class HumanAimController:
    """
    拟人化瞄准 — 降低 Replay 行为分析检测风险。

    核心优化:
      · 反应延迟: 80-220ms 随机，模拟人类视觉-运动延迟
      · 微抖动: 每帧移动量 ±12% 随机波动
      · 过冲模拟: 15% 概率多移 25%，然后拉回（模拟人手 overshoot）
      · 故意打偏: 5% 概率偏移 (模拟人类瞄准误差)
      · 切换停顿: 换目标后 350ms 内不移动
      · FOV 限制: 默认 250px (模拟注意力集中区域)
      · 仅扳机触发: 按住左Alt才自瞄
    """

    def __init__(self, mouse: HwMouse):
        self.mouse = mouse
        self.target_type = 1          # 0=身体, 1=头
        self.fov = 250                # FOV 半径 (px)
        self.speed = 1.0              # 基础灵敏度
        self.kp = 0.4                 # 移动比例系数

        # 内部状态
        self._lock_id = None          # 当前锁定目标身份
        self._lock_frames = 0         # 连续锁定帧数
        self._last_seen = {}          # {target_id: timestamp}
        self._switch_cooldown_until = 0
        self._reaction_until = 0
        self._overshoot_remaining = (0, 0)
        self._rng = random.Random()

    def set_target_type(self, t):
        self.target_type = t
        self._lock_id = None
        self._lock_frames = 0

    def set_fov(self, fov):
        self.fov = fov

    def set_speed(self, speed):
        self.speed = speed

    def _make_id(self, det):
        """用检测框粗略位置生成临时 ID。"""
        x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
        return (x1 // 5, y1 // 5, x2 // 5, y2 // 5)

    def _should_react(self, same_target):
        now = time.monotonic()
        if now < self._switch_cooldown_until:
            return False
        if now < self._reaction_until:
            return False
        # 切换目标时加反应延迟
        if not same_target:
            delay = self._rng.uniform(HUMAN_REACTION_MIN_MS, HUMAN_REACTION_MAX_MS) / 1000.0
            self._reaction_until = now + delay
            self._switch_cooldown_until = now + HUMAN_SWITCH_DELAY_MS / 1000.0
            return False
        return True

    def process(self, detections, frame_w, frame_h, trigger_held):
        """
        处理一帧。
        返回: (has_target, distance, display_info)
        """
        center_x, center_y = frame_w / 2, frame_h / 2
        now = time.monotonic()

        # 清理过期目标
        expired = [tid for tid, ts in self._last_seen.items()
                   if now - ts > 3.0]
        for tid in expired:
            del self._last_seen[tid]

        # ── 如果没有按扳机键，不瞄准 ──
        if not trigger_held:
            self._lock_id = None
            self._lock_frames = 0
            self._reaction_until = 0
            self._switch_cooldown_until = 0
            return False, 0, "wait"

        # ── 筛选候选目标 ──
        candidates = []
        for d in detections:
            cls = int(d[5])
            if cls != self.target_type:
                continue
            tx = (d[0] + d[2]) / 2
            ty = (d[1] + d[3]) / 2
            dx, dy = tx - center_x, ty - center_y
            dist = (dx * dx + dy * dy) ** 0.5
            if dist > self.fov:
                continue
            tid = self._make_id(d)
            conf = float(d[4])
            # 综合分: 距离优先 + 置信度
            score = dist + (1 - conf) * 80
            candidates.append((score, dx, dy, tid, conf))

        if not candidates:
            self._lock_id = None
            self._lock_frames = 0
            return False, 0, "scan"

        # ── 选择目标 ──
        candidates.sort(key=lambda x: x[0])
        best = candidates[0]
        _, dx, dy, tid, conf = best
        same_target = (tid == self._lock_id)
        self._last_seen[tid] = now

        # ── 反应延迟检查 ──
        if not self._should_react(same_target):
            return True, (dx * dx + dy * dy) ** 0.5, "react"

        self._lock_id = tid
        self._lock_frames += 1

        # ── 过冲修正 ──
        if self._overshoot_remaining != (0, 0):
            ox, oy = self._overshoot_remaining
            # 反向拉回 60%
            pullback_x = -ox * 0.6
            pullback_y = -oy * 0.6
            self._overshoot_remaining = (0, 0)
            self.mouse.move_auto(int(pullback_x), int(pullback_y), 8)
            # 然后在原目标基础上继续
            dx += pullback_x * 0.3
            dy += pullback_y * 0.3

        # ── 故意打偏 ──
        if self._rng.random() < HUMAN_MISS_CHANCE:
            miss_x = self._rng.uniform(-12, 12)
            miss_y = self._rng.uniform(-15, 5)  # 偏上偏下
            dx += miss_x
            dy += miss_y

        # ── 计算移动量 ──
        move_x = dx * self.kp * self.speed
        move_y = dy * self.kp * self.speed

        # ── 微抖动 ±12% ──
        jitter = 1.0 + self._rng.uniform(-HUMAN_JITTER_PCT, HUMAN_JITTER_PCT)
        move_x *= jitter
        move_y *= jitter

        # ── 过冲模拟 (15% 概率) ──
        if self._rng.random() < HUMAN_OVERSHOOT_PCT:
            overshoot = 1.0 + HUMAN_OVERSHOOT_RATIO
            self._overshoot_remaining = (
                int(move_x * HUMAN_OVERSHOOT_RATIO),
                int(move_y * HUMAN_OVERSHOOT_RATIO)
            )
            move_x *= overshoot
            move_y *= overshoot

        # ── 误差很小时不移动 (模拟手部死区) ──
        if abs(move_x) < 0.5 and abs(move_y) < 0.5:
            return True, (dx * dx + dy * dy) ** 0.5, "lock"

        # ── 发送到 KMBox NET (硬件贝塞尔曲线) ──
        dur = max(5, min(25, int((abs(move_x) + abs(move_y)) * 0.5)))
        self.mouse.move_auto(int(move_x), int(move_y), dur)

        dist = (dx * dx + dy * dy) ** 0.5
        status = "lock" if self._lock_frames > 5 else "track"
        return True, dist, status

    def reset(self):
        self._lock_id = None
        self._lock_frames = 0
        self._reaction_until = 0
        self._switch_cooldown_until = 0
        self._overshoot_remaining = (0, 0)


# ═══════════════════════════════════════════════════════════════
#  6. 辅助工具
# ═══════════════════════════════════════════════════════════════

def debounce(vk_code, last_time, interval=0.3):
    now = time.time()
    if now - last_time < interval:
        return False, last_time
    if win32api.GetAsyncKeyState(vk_code) & 0x8000:
        time.sleep(0.05)
        return True, now
    return False, last_time


def draw_radar(frame, detections, center, fov, aim_controller):
    """绘制雷达覆盖层 (可选，开发调试用)。"""
    display = frame.copy()
    cv2.circle(display, center, fov, (0, 255, 0), 1)
    cv2.circle(display, center, 3, (0, 255, 255), -1)
    for d in detections:
        x1, y1, x2, y2 = map(int, d[:4])
        cls = int(d[5])
        color = (0, 255, 0) if cls == 1 else (255, 165, 0)
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 1)
    return display


# ═══════════════════════════════════════════════════════════════
#  7. 主循环
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Video Bridge")
    parser.add_argument("--host", default="8.160.149.149")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--capture-size", type=int, default=640)
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("--fov", type=int, default=250, help="Aim FOV radius (px)")
    parser.add_argument("--speed", type=float, default=1.0, help="Aim sensitivity")
    parser.add_argument("--target", choices=["head", "body"], default="head")
    parser.add_argument("--tunnel", action="store_true")
    parser.add_argument("--tunnel-host", default="127.0.0.1")
    parser.add_argument("--tunnel-port", type=int, default=29999)
    parser.add_argument("--show-radar", action="store_true", help="Show radar window")
    parser.add_argument("--kmbox-ip", default="192.168.2.188")
    parser.add_argument("--kmbox-port", type=int, default=12345)
    parser.add_argument("--kmbox-uuid", default=None, help="KMBox UUID (see box display)")
    args = parser.parse_args()

    CAP = args.capture_size

    # ── 初始化视频捕获 ──
    print("[Init] video capture...")
    cap = VideoCapture(width=CAP, height=CAP, prefer_obs=True)
    print(f"[Init] backend: {cap.backend}")

    # ── SSH 隧道 ──
    tunnel = None
    if args.tunnel:
        try:
            key_path = os.path.expanduser("~/.ssh/cloud_rsa")
            if not os.path.exists(key_path):
                print(f"[!] SSH key not found: {key_path}")
                cap.release()
                return
            tunnel = start_ssh_tunnel(
                local_host=args.tunnel_host,
                local_port=args.tunnel_port,
                remote_host="127.0.0.1",
                remote_port=9999,
                ssh_host=args.host,
                ssh_port=53414,
                ssh_user="root",
                ssh_key_path=key_path,
            )
            connect_host = args.tunnel_host
            connect_port = args.tunnel_port
        except Exception as e:
            print(f"[Fatal] SSH tunnel failed: {e}")
            cap.release()
            return
    else:
        connect_host = args.host
        connect_port = args.port

    # ── 初始化云端推理 ──
    print(f"[Init] connecting {connect_host}:{connect_port}...")
    cloud = InferenceClient(connect_host, connect_port, args.jpeg_quality)
    try:
        cloud.connect()
    except Exception as e:
        print(f"[Fatal] connection failed: {e}")
        cap.release()
        if tunnel:
            tunnel.stop()
        return

    # ── 初始化 KMBox NET 鼠标 ──
    mouse = HwMouse(
        ip=args.kmbox_ip,
        port=args.kmbox_port,
        uuid=args.kmbox_uuid,
    )
    if not mouse.ready:
        print("[!] KMBox NET not ready — 鼠标功能不可用")
        print("    推理和雷达仍可工作，用于测试流程")

    # ── 初始化瞄准控制器 ──
    aim = HumanAimController(mouse)
    aim.set_fov(args.fov)
    aim.set_speed(args.speed)
    aim.set_target_type(1 if args.target == "head" else 0)

    # ── 雷达窗口 ──
    radar_visible = args.show_radar
    if radar_visible:
        cv2.namedWindow("radar", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("radar", CAP, CAP)

    # ── 状态 ──
    frame_count = 0
    fps_hist = deque(maxlen=FPS_WINDOW)
    t_switch = t_radar = 0

    # ── 启动信息 ──
    print()
    print("=" * 48)
    print(f"  target: {'HEAD' if aim.target_type == 1 else 'BODY'}")
    print(f"  fov: {aim.fov}px  speed: {args.speed}")
    print(f"  mouse: {'KMBox NET' if mouse.ready else 'NONE'}")
    print(f"  [Alt] aim  [Home] switch  [End] radar  [Del] quit")
    print("=" * 48)
    print()

    try:
        while True:
            t0 = time.time()

            # ── 截图 ──
            frame = cap.read()
            if frame is None or frame.size == 0:
                time.sleep(0.001)
                continue
            fh, fw = frame.shape[:2]

            # ── 热键 ──
            if win32api.GetAsyncKeyState(VK_QUIT) & 0x8000:
                break

            pressed, t_switch = debounce(VK_SWITCH, t_switch)
            if pressed:
                aim.set_target_type(1 - aim.target_type)
                tp = "HEAD" if aim.target_type == 1 else "BODY"
                print(f"\n[switch] → {tp}")

            pressed, t_radar = debounce(VK_RADAR, t_radar)
            if pressed:
                radar_visible = not radar_visible
                if not radar_visible:
                    cv2.destroyAllWindows()
                else:
                    cv2.namedWindow("radar", cv2.WINDOW_NORMAL)

            # ── 扳机键 ──
            trigger_held = bool(win32api.GetAsyncKeyState(VK_TRIGGER) & 0x8000)

            # ── 云端推理 ──
            detections = cloud.infer(frame)

            # ── 瞄准 ──
            frame_count += 1
            has_target, tgt_dist, status = aim.process(
                detections, fw, fh, trigger_held
            )

            # ── FPS ──
            dt = time.time() - t0
            fps_hist.append(1.0 / dt if dt > 0 else 0)

            if frame_count % 15 == 0:
                avg_fps = sum(fps_hist) / len(fps_hist) if fps_hist else 0
                tp_short = "H" if aim.target_type == 1 else "B"
                sig = "🎯" if trigger_held else "  "
                sys.stdout.write(
                    f"\r{sig} [{tp_short}] {status:<6} "
                    f"fps:{int(avg_fps):>3}  dt:{len(detections):>2}   "
                )
                sys.stdout.flush()

            # ── 雷达 ──
            if radar_visible:
                center = (fw // 2, fh // 2)
                radar = draw_radar(frame, detections, center, aim.fov, aim)
                cv2.imshow("radar", radar)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cloud.close()
        mouse.close()
        cv2.destroyAllWindows()
        if tunnel:
            tunnel.stop()
        print("\n[Exit] done.")


if __name__ == "__main__":
    main()
