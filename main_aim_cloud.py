#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cloud GPU Auto-Aim Client
=========================
本地 DXCam 截图 → 云端 GPU 推理 → 本地鼠标控制。

架构:
  本地 PC                             云端 (RTX 5880, 48GB)
  ────────                            ──────────────────────
  DXCam 截图                          cloud_server_onnx.py
    ↓ 压缩 JPEG                           ↓ ONNX CUDA 推理
  TCP → [jpeg] ──────────────────────→ 检测目标
    ↓                                    ↓
  TCP ← [detections] ←──────────────── 返回坐标
    ↓
  AimController → 鼠标移动

协议 (binary):
  Client → Server: [4B: jpeg_size uint32 LE] [jpeg_size B: JPEG]
  Server → Client: [1B: num_detections uint8] [N × 24B: float32 × 6]

使用:
  python main_aim_cloud.py --host 8.160.149.149 --port 9999
"""

import argparse
import ctypes
import io
import os
import socket
import struct
import sys
import time
import traceback
import warnings
from collections import deque

import cv2
import numpy as np
import win32api

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
#  DXCam (DirectX) 屏幕捕获
# ═══════════════════════════════════════════════════════════════

try:
    import dxcam
    HAS_DXCAM = True
except ImportError:
    HAS_DXCAM = False
    print("[!] dxcam not installed. Install: pip install dxcam")

try:
    import mss
    HAS_MSS = True
except ImportError:
    HAS_MSS = False


class DirectXCapture:
    """使用 DXCam (DXGI Desktop Duplication API) 捕获屏幕。"""

    def __init__(self, capture_width=640, capture_height=640):
        self.width = capture_width
        self.height = capture_height
        self.camera = None
        self.backend = None

        if HAS_DXCAM:
            try:
                self.camera = dxcam.create(output_color="BGR")
                test = self.camera.grab()
                if test is not None:
                    self.backend = "dxcam"
                    print(f"[Capture] DXCam initialized ({test.shape[1]}x{test.shape[0]})")
                    return
            except Exception as e:
                print(f"[Capture] DXCam init failed: {e}")

        if HAS_MSS:
            self.sct = mss.mss()
            self.monitor = self.sct.monitors[1]
            self.backend = "mss"
            print(f"[Capture] MSS initialized (monitor: {self.monitor})")
        else:
            raise RuntimeError("Neither DXCam nor MSS available. Install: pip install dxcam mss")

    def grab_center(self):
        """捕获屏幕中心区域，返回 BGR numpy array。"""
        if self.backend == "dxcam":
            frame = self.camera.grab()
            if frame is None:
                return None
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2
            half = self.width // 2
            x1, y1 = max(0, cx - half), max(0, cy - half)
            x2, y2 = min(w, cx + half), min(h, cy + half)
            return frame[y1:y2, x1:x2]
        elif self.backend == "mss":
            img = np.array(self.sct.grab(self.monitor), dtype=np.uint8)
            h, w = img.shape[:2]
            cx, cy = w // 2, h // 2
            half = self.width // 2
            x1, y1 = max(0, cx - half), max(0, cy - half)
            x2, y2 = min(w, cx + half), min(h, cy + half)
            return img[y1:y2, x1:x2]
        return None

    def release(self):
        if self.backend == "dxcam" and self.camera:
            try:
                self.camera.release()
            except Exception:
                pass
        print("[Capture] Released")


# ═══════════════════════════════════════════════════════════════
#  云端推理客户端
# ═══════════════════════════════════════════════════════════════

class CloudInferenceClient:
    """通过 TCP 连接云端 GPU 推理服务器。"""

    def __init__(self, host, port, jpeg_quality=85):
        self.host = host
        self.port = port
        self.jpeg_quality = jpeg_quality
        self.sock = None
        self._encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(5)
        self.sock.connect((self.host, self.port))
        self.sock.settimeout(2)
        print(f"[Cloud] Connected to {self.host}:{self.port}")

    def infer(self, img_bgr):
        """
        发送一帧 BGR 图像到云端，返回检测结果。
        返回: (N, 6) numpy array [x1, y1, x2, y2, conf, cls]
        """
        try:
            # JPEG 编码
            ok, jpeg_buf = cv2.imencode('.jpg', img_bgr, self._encode_params)
            if not ok:
                return np.empty((0, 6), dtype=np.float32)
            jpeg_data = jpeg_buf.tobytes()

            # 发送: [4B size][JPEG data]
            header = struct.pack('<I', len(jpeg_data))
            self.sock.sendall(header + jpeg_data)

            # 接收: [1B count][N × 24B detections]
            count_byte = self._recv_exact(1)
            count = count_byte[0]

            if count == 0:
                return np.empty((0, 6), dtype=np.float32)

            det_data = self._recv_exact(count * 24)  # 6 × float32 = 24 bytes
            dets = np.frombuffer(det_data, dtype=np.float32).reshape(count, 6)
            return dets
        except (socket.timeout, ConnectionError, OSError):
            return np.empty((0, 6), dtype=np.float32)

    def _recv_exact(self, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Server disconnected")
            buf.extend(chunk)
        return bytes(buf)

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None
        print("[Cloud] Disconnected")


# ═══════════════════════════════════════════════════════════════
#  鼠标驱动 (KmBox NET / Logitech 驱动)
# ═══════════════════════════════════════════════════════════════

INTERCEPTION_MOUSE_MOVE_RELATIVE = 0x0001
INTERCEPTION_FILTER_MOUSE_ALL = 0xFFFF

class InterceptionMouseStroke(ctypes.Structure):
    _fields_ = [("flags", ctypes.c_ushort),
                ("button", ctypes.c_ushort),
                ("state", ctypes.c_ushort),
                ("rolling", ctypes.c_ushort),
                ("x", ctypes.c_long),
                ("y", ctypes.c_long),
                ("information", ctypes.c_ushort)]


class MouseDriver:
    """Interception 内核驱动 (通用所有品牌鼠标) / KmBox / Logitech。"""

    def __init__(self, method="interception", kmbox_ip="192.168.2.3", kmbox_port=12349):
        self.method = method
        self.kmbox_sock = None
        self.kmbox_addr = (kmbox_ip, kmbox_port)
        self.dll = None
        self._ictx = None
        self._idevice = 0

        if method == "interception":
            self._init_interception()
        elif method == "kmbox":
            self._init_kmbox()
        elif method == "logitech":
            self._init_logitech()

    # ── Interception (通用内核驱动) ──

    def _init_interception(self):
        import os as _os
        dll_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  'interception.dll')
        if not _os.path.exists(dll_path):
            raise FileNotFoundError(f"interception.dll not found: {dll_path}")

        self._idll = ctypes.CDLL(dll_path)
        self._ictx = self._idll.interception_create_context()
        if not self._ictx:
            raise RuntimeError("interception_create_context failed")

        # 直接用设备 ID (拦截驱动不设过滤器也能发，设备号通常是 12)
        # 先尝试标准 mouse filter 等待设备，失败则用默认值
        self._idevice = 12
        print(f"[Mouse] Interception kernel driver (device={self._idevice})")

    def _init_kmbox(self):
        self.kmbox_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.kmbox_sock.settimeout(0.001)
        print(f"[Mouse] KmBox NET ({self.kmbox_addr[0]}:{self.kmbox_addr[1]})")

    def _init_logitech(self):
        import os as _os
        dll_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  'logitech.driver.dll')
        if not _os.path.exists(dll_path):
            raise FileNotFoundError(f"Logitech DLL not found: {dll_path}")
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        devnull = open(os.devnull, 'w')
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            self.dll = ctypes.CDLL(dll_path)
            self.dll.device_open()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            devnull.close()
        print("[Mouse] Logitech driver loaded")

    # ── 移动 / 点击 ──

    def move(self, x, y):
        """相对移动。"""
        if self.method == "interception" and self._ictx:
            try:
                stroke = InterceptionMouseStroke()
                stroke.flags = INTERCEPTION_MOUSE_MOVE_RELATIVE
                stroke.x = int(x)
                stroke.y = int(y)
                self._idll.interception_send(self._ictx, self._idevice,
                                              ctypes.pointer(stroke), 1)
                return
            except OSError:
                self.method = "sendinput"
        if self.method == "sendinput":
            self._sendinput_move(int(x), int(y))
        elif self.method == "kmbox":
            try:
                self.kmbox_sock.sendto(f"move({int(x)},{int(y)})\n".encode(),
                                       self.kmbox_addr)
            except Exception:
                pass
        elif self.dll:
            self.dll.moveR(int(x), int(y))

    def _sendinput_move(self, x, y):
        """SendInput 相对移动 (备用)。"""
        extra = ctypes.c_ulong(0)
        class _MI(ctypes.Structure):
            _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long),
                        ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong),
                        ("time", ctypes.c_ulong),
                        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]
        class _II(ctypes.Union):
            _fields_ = [("mi", _MI)]
        class _IN(ctypes.Structure):
            _fields_ = [("type", ctypes.c_ulong), ("ii", _II)]
        ii = _II()
        ii.mi = _MI(int(x), int(y), 0, 0x0001, 0, ctypes.pointer(extra))
        cmd = _IN(ctypes.c_ulong(0), ii)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(cmd), ctypes.sizeof(cmd))

    def click(self):
        """鼠标左键。"""
        if self.method == "interception" and self._ictx:
            stroke = InterceptionMouseStroke()
            stroke.flags = 0x0002  # INTERCEPTION_MOUSE_LEFT_BUTTON_DOWN
            self._idll.interception_send(self._ictx, self._idevice,
                                          ctypes.pointer(stroke), 1)
            time.sleep(0.01)
            stroke.flags = 0x0004  # INTERCEPTION_MOUSE_LEFT_BUTTON_UP
            self._idll.interception_send(self._ictx, self._idevice,
                                          ctypes.pointer(stroke), 1)
        elif self.method == "kmbox":
            try:
                self.kmbox_sock.sendto(b"click(left)\n", self.kmbox_addr)
            except Exception:
                pass
        elif self.dll:
            self.dll.mouse_event(0x0002, 0, 0, 0, 0)
            time.sleep(0.01)
            self.dll.mouse_event(0x0004, 0, 0, 0, 0)

    def close(self):
        if self._ictx:
            try:
                self._idll.interception_destroy_context(self._ictx)
            except Exception:
                pass
        if self.kmbox_sock:
            self.kmbox_sock.close()
        if self.dll:
            try:
                self.dll.device_close()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════
#  瞄准控制器 (简化版，完整版见 main_aim_full.py)
# ═══════════════════════════════════════════════════════════════

class AimController:
    """本地瞄准逻辑: 选择目标 → 计算移动 → 驱动鼠标。"""

    def __init__(self, mouse):
        self.mouse = mouse
        self.current_target_type = 1  # 0=身体, 1=头
        self.aim_fov = 55             # FOV 半径 (步枪)
        self.static_speed = 0.5
        self.tracking_speed = 0.015
        self.last_move_x = 0
        self.last_move_y = 0
        self.is_locking = False
        self.last_tx, self.last_ty = 0, 0

    def set_fov(self, fov):
        self.aim_fov = fov

    def set_speed(self, static_speed, tracking_speed):
        self.static_speed = static_speed
        self.tracking_speed = tracking_speed

    def select_target(self, detections, frame_center, frame_size):
        """
        从云端返回的检测结果中选择目标。
        detections: (N, 6) [x1, y1, x2, y2, conf, cls] — 已在原始图像坐标中
        这里的图像是裁剪后的 640x640 中心区域
        """
        candidates = []
        center_x, center_y = frame_center
        fw, fh = frame_size

        for d in detections:
            x1, y1, x2, y2, conf, cls = d[:6]

            # 类别过滤
            if int(cls) != self.current_target_type:
                continue

            # 计算目标中心
            tx, ty = (x1 + x2) / 2, (y1 + y2) / 2
            dx, dy = tx - center_x, ty - center_y
            dist = (dx ** 2 + dy ** 2) ** 0.5

            # FOV 过滤
            if dist > self.aim_fov:
                continue

            # 加权打分
            weight = dist * 0.6 + (1 - conf) * 100
            if self.is_locking:
                last_dist = ((dx - self.last_tx) ** 2 + (dy - self.last_ty) ** 2) ** 0.5
                weight = weight * 0.4 + last_dist * 0.6

            candidates.append((weight, dx, dy))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            _, best_dx, best_dy = candidates[0]
            self.is_locking = True
            self.last_tx, self.last_ty = best_dx, best_dy
            return (best_dx, best_dy)
        else:
            self.is_locking = False
            return None

    def calculate_movement(self, error_x, error_y):
        distance = (error_x ** 2 + error_y ** 2) ** 0.5
        sensitivity = self.static_speed * 2.4
        if self.is_locking and distance > 5:
            sensitivity *= (1.0 + self.tracking_speed * 50)
        accel = (min(1.0, distance / 120.0)) ** 1.8
        move_x = error_x * sensitivity * accel
        move_y = error_y * sensitivity * accel
        smooth = 0.2 * (1.0 - 0.5 * min(1.0, distance / 100.0))
        filtered_x = self.last_move_x * smooth + move_x * (1 - smooth)
        filtered_y = self.last_move_y * smooth + move_y * (1 - smooth)
        self.last_move_x = filtered_x
        self.last_move_y = filtered_y
        return filtered_x, filtered_y

    def reset(self):
        self.last_move_x = 0
        self.last_move_y = 0
        self.is_locking = False

    def process(self, detections, frame_w, frame_h, aim_key_held):
        """
        处理一帧: 选目标 → 算移动 → 移鼠标。
        返回 (has_target, distance, status_msg)
        """
        center = (frame_w / 2, frame_h / 2)
        target = self.select_target(detections, center, (frame_w, frame_h))

        if target is None:
            return False, 0, "Scanning"

        dx, dy = target
        dist = (dx ** 2 + dy ** 2) ** 0.5

        if aim_key_held:
            move_x, move_y = self.calculate_movement(dx, dy)
            self.mouse.move(int(move_x), int(move_y))

        return True, dist, "AIMING"


# ═══════════════════════════════════════════════════════════════
#  热键 / 工具
# ═══════════════════════════════════════════════════════════════

def is_key_pressed(vk_code):
    return bool(win32api.GetAsyncKeyState(vk_code) & 0x8000)


def check_key_toggle(vk_code, last_time, debounce=0.3):
    now = time.time()
    if now - last_time < debounce:
        return False, last_time
    if is_key_pressed(vk_code):
        time.sleep(0.05)
        return True, now
    return False, last_time


# ═══════════════════════════════════════════════════════════════
#  主循环
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Cloud GPU Auto-Aim Client")
    parser.add_argument("--host", default="8.160.149.149", help="Cloud server IP")
    parser.add_argument("--port", type=int, default=9999, help="Cloud server port")
    parser.add_argument("--capture-size", type=int, default=640, help="Capture region size")
    parser.add_argument("--jpeg-quality", type=int, default=75, help="JPEG quality (lower=faster)")
    parser.add_argument("--fov", type=int, default=55, help="Aim FOV radius (pixels)")
    parser.add_argument("--speed", type=float, default=0.5, help="Aim speed")
    parser.add_argument("--track-speed", type=float, default=0.015, help="Tracking speed")
    parser.add_argument("--target", choices=["head", "body"], default="head", help="Target type")
    parser.add_argument("--mouse-method", choices=["interception", "kmbox", "logitech"], default="interception",
                        help="Mouse control method")
    parser.add_argument("--kmbox-ip", default="192.168.2.3", help="KmBox IP address")
    parser.add_argument("--kmbox-port", type=int, default=12349, help="KmBox UDP port")
    parser.add_argument("--tunnel", action="store_true",
                        help="Auto SSH tunnel through port 53414 (bypasses firewall)")
    parser.add_argument("--tunnel-host", default="127.0.0.1", help="Tunnel local endpoint")
    parser.add_argument("--tunnel-port", type=int, default=19999, help="Tunnel local port")
    args = parser.parse_args()

    # 管理员权限
    if not ctypes.windll.shell32.IsUserAnAdmin():
        print("[!] Admin required. Restarting...")
        script = os.path.abspath(sys.argv[0])
        ret = ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable,
                                                   f'"{script}" {" ".join(sys.argv[1:])}',
                                                   None, 1)
        if ret <= 32:
            print("[!] Failed to get admin. Exiting.")
        return

    # DPI
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

    CAP_SIZE = args.capture_size

    # 初始化 DirectX 捕获
    print("[Init] DirectX screen capture...")
    capture = DirectXCapture(capture_width=CAP_SIZE, capture_height=CAP_SIZE)
    print(f"[Init] Backend: {capture.backend}")

    # ── 云端推理 ──
    tunnel_proc = None

    if args.tunnel:
        import subprocess
        print(f"[Tunnel] SSH tunnel {args.tunnel_host}:{args.tunnel_port} -> cloud:9999...")
        ssh_cmd = [
            "ssh", "-o", "StrictHostKeyChecking=no",
            "-o", "IdentitiesOnly=yes",
            "-i", os.path.expanduser("~/.ssh/cloud_rsa"),
            "-N", "-L", f"{args.tunnel_host}:{args.tunnel_port}:127.0.0.1:9999",
            f"root@{args.host}", "-p", "53414"
        ]
        tunnel_proc = subprocess.Popen(ssh_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)
        connect_host = args.tunnel_host
        connect_port = args.tunnel_port
    else:
        connect_host = args.host
        connect_port = args.port

    print(f"[Init] Connecting to {connect_host}:{connect_port}...")
    cloud = CloudInferenceClient(connect_host, connect_port, jpeg_quality=args.jpeg_quality)
    try:
        cloud.connect()
    except Exception as e:
        print(f"[Fatal] Cannot connect to cloud: {e}")
        print("  Use --tunnel to SSH through port 53414")
        capture.release()
        if tunnel_proc:
            tunnel_proc.terminate()
        return

    # 初始化鼠标
    mouse = MouseDriver(method=args.mouse_method,
                        kmbox_ip=args.kmbox_ip,
                        kmbox_port=args.kmbox_port)
    aim_ctrl = AimController(mouse)
    aim_ctrl.aim_fov = args.fov if args.fov != 55 else 9999  # 默认全屏FOV
    aim_ctrl.set_speed(args.speed, args.track_speed)
    aim_ctrl.current_target_type = 1 if args.target == "head" else 0

    # 状态
    aim_enabled = True
    show_window = True
    t_aim, t_target, t_win = 0, 0, 0
    frame_count = 0
    fps_history = deque(maxlen=30)

    # 热键
    HK_TOGGLE_AIM = 0xDD     # ]
    HK_TOGGLE_TARGET = 0xDB  # [
    HK_TOGGLE_WIN = 0xDC     # \
    HK_QUIT = 0x51           # Q
    HK_AIM = 0xA4            # 左Alt (自瞄键)

    # 窗口
    if show_window:
        cv2.namedWindow("Cloud Radar", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Cloud Radar", CAP_SIZE, CAP_SIZE)
        cv2.setWindowProperty("Cloud Radar", cv2.WND_PROP_TOPMOST, 1)

    print("\n" + "=" * 50)
    print(f"[SYSTEM READY] Cloud: {connect_host}:{connect_port}")
    print(f"  FOV: {aim_ctrl.aim_fov}px | Speed: {args.speed}")
    print(f"  Target: {'HEAD' if aim_ctrl.current_target_type == 1 else 'BODY'}")
    print(f"  []] Toggle Aim  [[] Toggle Target  [\\] Radar  [Q] Quit")
    print("=" * 50 + "\n")

    try:
        while True:
            loop_start = time.time()

            # ── DirectX 截图 ──
            frame = capture.grab_center()
            if frame is None or frame.size == 0:
                time.sleep(0.001)
                continue
            fh, fw = frame.shape[:2]

            # ── 热键 ──
            if is_key_pressed(HK_QUIT):
                break
            pressed, t_aim = check_key_toggle(HK_TOGGLE_AIM, t_aim)
            if pressed:
                aim_enabled = not aim_enabled
                print(f"\n[Aim] {'ON' if aim_enabled else 'OFF'}")
            pressed, t_target = check_key_toggle(HK_TOGGLE_TARGET, t_target)
            if pressed:
                aim_ctrl.current_target_type = 1 - aim_ctrl.current_target_type
                print(f"\n[Target] {'HEAD' if aim_ctrl.current_target_type == 1 else 'BODY'}")
            pressed, t_win = check_key_toggle(HK_TOGGLE_WIN, t_win)
            if pressed:
                show_window = not show_window
                if show_window:
                    cv2.namedWindow("Cloud Radar", cv2.WINDOW_NORMAL)
                else:
                    cv2.destroyAllWindows()

            # ── 推理 ──
            detections = cloud.infer(frame)

            # ── 瞄准 ──
            frame_count += 1
            if aim_enabled:
                has_target, target_dist, status = aim_ctrl.process(
                    detections, fw, fh, True
                )
            else:
                aim_ctrl.reset()
                has_target, target_dist, status = False, 0, "DISABLED"

            # ── FPS ──
            loop_time = time.time() - loop_start
            fps_history.append(1.0 / loop_time if loop_time > 0 else 0)

            if frame_count % 15 == 0:
                avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0
                tgt = "HEAD" if aim_ctrl.current_target_type == 1 else "BODY"
                sys.stdout.write(
                    f"\r[{tgt}] {status} | "
                    f"FPS: {int(avg_fps)} | "
                    f"Dets: {len(detections)}   "
                )
                sys.stdout.flush()

            # ── 雷达窗口 ──
            if show_window:
                display = frame.copy()
                center = (fw // 2, fh // 2)
                cv2.circle(display, center, aim_ctrl.aim_fov, (0, 255, 0), 1)
                for d in detections:
                    x1, y1, x2, y2, conf, cls = d[:6]
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    color = (0, 255, 0) if int(cls) == 1 else (255, 0, 0)
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                if has_target:
                    cv2.line(display, center,
                             (int(center[0] + aim_ctrl.last_tx),
                              int(center[1] + aim_ctrl.last_ty)),
                             (0, 255, 255), 2)
                cv2.imshow("Cloud Radar", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        if cloud:
            cloud.close()
        cv2.destroyAllWindows()
        if tunnel_proc:
            tunnel_proc.terminate()
            print("\n[Tunnel] Closed")
        print("\n[Exit] Clean shutdown.")


if __name__ == "__main__":
    main()
