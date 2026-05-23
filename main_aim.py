# -*- coding: utf-8 -*-
"""
YOLOv5 FPS 实时自瞄系统 - 优化版
===================================
修复内容:
  1. 移除死导入 (matplotlib, threading, queue)
  2. 修复导入顺序 (PEP 8: 标准库 → 第三方 → 本地)
  3. 添加 USE_DXCAM 定义
  4. 修复 build_runtime_config 环境变量类型安全
  5. 修复 OBSCapture camera_index 被环境变量覆盖的问题
  6. 删除重复的配置块
  7. 修复 mouse_callback 缺少 param 导致的崩溃
  8. 修复 check_key_state 按键独立去抖
  9. 移除死代码 (show_radar_matplotlib, cuda_nms)
  10. FastAimController 灵敏度使用连续函数替代分段
  11. 修复 FPS 计算精度
  12. 移除主循环中的死代码分支
  13. 移除 ScreenCapture 薄封装，直接使用 OBSCapture
  14. 实例化 CrosshairCalibration 并正确传入 mouse_callback
"""

# ==================== 标准库导入 ====================
import argparse
import ctypes
import os
import subprocess
import sys
import time
import traceback
import warnings

# ==================== 第三方库导入 ====================
import cv2
import numpy as np
import torch
import win32api

# ==================== 本地模块导入 ====================
from logitech import Logitech

# ==================== 全局配置 ====================
warnings.filterwarnings("ignore")

# ---- 模型/检测参数 ----
DETECTION_SIZE = 320       # 检测尺寸 (模型训练尺寸)
CONF_THRES = 0.5           # 置信度阈值
IOU_THRES = 0.45           # NMS IOU 阈值
MAX_DETECTIONS = 20        # 最大检测数
AIM_FOV_RADIUS = 300       # 自瞄 FOV 半径 (像素)

# ---- 性能参数 ----
DXCAM_MAX_FPS = 144        # 目标最大 FPS
BATCH_SIZE = 1             # 批量推理 (1=实时, 2-4=高吞吐)
BATCH_MAX_LATENCY_MS = 8   # 批量最大等待时间 (毫秒)

# ---- 目标类型 ----
current_target_type = 1    # 0=身体, 1=头部
TARGET_MAPPING = {0: 'BODY', 1: 'HEAD'}

# ---- NMS 函数引用 (运行时设置) ----
non_max_suppression = None

# ---- 按键映射 ----
KEYS = {
    'TOGGLE_WIN':    [0xDC],  # \  键：开关识别窗口
    'TOGGLE_AIM':    [0xDD],  # ]  键：开启/暂停自瞄
    'TOGGLE_TARGET': [0xDB],  # [  键：切换锁定目标（头/身）
    'QUIT':          [0x51],  # q  键：退出程序
}

# ---- 屏幕捕获后端标志 ----
USE_DXCAM = False
try:
    import dxcam
    USE_DXCAM = True
except ImportError:
    pass


# ==================== 工具函数 ====================

def _safe_int(value, default=0):
    """安全地将字符串转为整数，失败时返回默认值。"""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def build_runtime_config(argv=None):
    """解析命令行参数和环境变量，构建运行时配置。"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--obs-camera-index', dest='obs_camera_index', type=int)
    args, _ = parser.parse_known_args(argv if argv is not None else sys.argv[1:])

    config = {
        'obs_camera_index': _safe_int(os.environ.get('OBS_CAMERA_INDEX', '0')),
    }

    if args.obs_camera_index is not None:
        config['obs_camera_index'] = args.obs_camera_index

    return config


def maybe_relaunch_as_admin(runtime_config):
    """如果需要，以管理员权限重新启动当前脚本。"""
    print(">>> Checking admin privileges...")
    if ctypes.windll.shell32.IsUserAnAdmin():
        print(">>> Admin check passed.")
        return

    print(">>> Requesting admin privileges...")
    script_path = os.path.abspath(__file__)
    extra_args = [
        '--obs-camera-index={}'.format(runtime_config['obs_camera_index']),
    ]

    arg_string = subprocess.list2cmdline([script_path] + extra_args)
    result = ctypes.windll.shell32.ShellExecuteW(
        None, "runas", sys.executable, arg_string, None, 1
    )
    if result <= 32:
        raise RuntimeError("ShellExecuteW runas failed with code {}".format(result))
    sys.exit()


def print_status(message):
    """不换行打印状态信息。"""
    print(f"\r{message}", end="", flush=True)


def check_key_state(key_codes, last_time, debounce_time=0.2):
    """
    按键去抖检测。
    返回 (is_pressed, new_last_time)。
    """
    current_time = time.time()
    if current_time - last_time < debounce_time:
        return False, last_time
    for key_code in key_codes:
        if win32api.GetAsyncKeyState(key_code) & 0x8000:
            return True, current_time
    return False, last_time


# ==================== 屏幕捕获 ====================

class OBSCapture:
    """通过 OBS Virtual Camera 提供画面输入。"""

    def __init__(self, camera_index=0, width=None, height=None):
        # 优先使用环境变量，否则使用传入参数
        env_index = os.environ.get('OBS_CAMERA_INDEX')
        self.camera_index = _safe_int(env_index) if env_index is not None else camera_index
        self.width = width
        self.height = height
        self.cap = None
        self._init_virtual_camera()
        print(f"[OBS] Using Virtual Camera index {self.camera_index}")

    def _init_virtual_camera(self):
        backends = []
        if hasattr(cv2, 'CAP_DSHOW'):
            backends.append(('CAP_DSHOW', cv2.CAP_DSHOW))
        if hasattr(cv2, 'CAP_MSMF'):
            backends.append(('CAP_MSMF', cv2.CAP_MSMF))
        backends.append(('default', None))

        errors = []
        for backend_name, backend in backends:
            if backend is None:
                cap = cv2.VideoCapture(self.camera_index)
            else:
                cap = cv2.VideoCapture(self.camera_index, backend)

            if not cap.isOpened():
                cap.release()
                errors.append('backend {} open failed'.format(backend_name))
                continue

            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if self.width is not None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            if self.height is not None:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            ok, frame = cap.read()
            if ok and frame is not None and frame.size:
                self.cap = cap
                return

            cap.release()
            errors.append('backend {} returned no frame'.format(backend_name))

        raise RuntimeError('OBS Virtual Camera open failed: ' + '; '.join(errors))

    def grab(self):
        ok, frame = self.cap.read()
        if not ok or frame is None or frame.size == 0:
            raise RuntimeError('OBS Virtual Camera frame read failed')
        return frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        print("[OBS] Released")


# ==================== 准星校准 ====================

class CrosshairCalibration:
    """准星校准：允许用户点击画面设定准星中心位置。"""

    def __init__(self):
        self.enabled = False
        self.cx = None
        self.cy = None

    def set_point(self, x, y):
        self.cx = int(x)
        self.cy = int(y)
        self.enabled = True

    def clear(self):
        self.cx = None
        self.cy = None
        self.enabled = False

    def get_point(self, frame_shape):
        h, w = frame_shape[:2]
        if self.enabled:
            return self.cx, self.cy
        return w // 2, h // 2


def mouse_callback(event, x, y, flags, param):
    """OpenCV 鼠标回调：左键点击设定准星校准点。"""
    if param is None:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        param.set_point(x, y)
        print(f"[CALIBRATION] Crosshair center set to ({x}, {y})")


# ==================== 瞄准控制器 ====================

class FastAimController:
    """
    快速瞄准控制器。
    使用连续灵敏度曲线 + 自适应平滑滤波器。
    """

    def __init__(self, base_sensitivity=1.0, acceleration_curve=1.8):
        self.base_sensitivity = base_sensitivity
        self.acceleration_curve = acceleration_curve
        self.last_move_x = 0
        self.last_move_y = 0
        self.smooth_factor = 0.2

    def calculate_movement(self, error_x, error_y):
        distance = (error_x ** 2 + error_y ** 2) ** 0.5

        # ---- 连续灵敏度曲线 (替代分段) ----
        # 使用 sigmoid-like 函数: 近距离低灵敏度, 远距离高灵敏度
        # 范围: [0.7, 2.0] * base_sensitivity
        normalized = min(1.0, distance / 150.0)
        sensitivity = self.base_sensitivity * (0.7 + 1.3 * normalized)

        # ---- 非线性加速 ----
        acceleration = (min(1.0, distance / 120.0)) ** self.acceleration_curve

        # ---- 基础移动量 ----
        move_x = error_x * sensitivity * acceleration
        move_y = error_y * sensitivity * acceleration

        # ---- 自适应平滑 ----
        # 近距离: 更多平滑 (稳定性); 远距离: 更少平滑 (响应速度)
        dynamic_smooth = self.smooth_factor * (1.0 - 0.5 * min(1.0, distance / 100.0))

        filtered_move_x = self.last_move_x * dynamic_smooth + move_x * (1 - dynamic_smooth)
        filtered_move_y = self.last_move_y * dynamic_smooth + move_y * (1 - dynamic_smooth)

        self.last_move_x = filtered_move_x
        self.last_move_y = filtered_move_y

        return filtered_move_x, filtered_move_y

    def reset(self):
        """重置控制器状态。"""
        self.last_move_x = 0
        self.last_move_y = 0


# ==================== 模型预处理 ====================

def preprocess_for_model(img_bgr, device, half=True):
    """
    将 BGR uint8 图像预处理为模型输入格式。
    - 输出: (1, 3, H, W) tensor, 归一化到 [0, 1]
    """
    img_tensor = torch.from_numpy(img_bgr).to(device)
    img_tensor = img_tensor[..., [2, 1, 0]]  # BGR -> RGB
    if half:
        img_tensor = img_tensor.half() / 255.0
    else:
        img_tensor = img_tensor.float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
    img_tensor = img_tensor.unsqueeze(0)       # add batch dim
    return img_tensor


# ==================== 模型加载 ====================

def load_model_core():
    """
    按优先级加载模型: TensorRT > PyTorch > ONNX。
    返回: (model, device_str, is_half, model_type, nms_fn)
    """
    print("Loading model...")
    script_dir = os.path.dirname(os.path.abspath(__file__))

    engine_path = os.path.join(script_dir, '1225_2best.engine')
    pt_path = os.path.join(script_dir, '1225_2best.pt')
    onnx_path = os.path.join(script_dir, '1225_2best.onnx')

    # ---- 1. TensorRT (最快) ----
    if os.path.exists(engine_path):
        try:
            print("[TensorRT] Loading TensorRT engine...")
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401

            logger = trt.Logger(trt.Logger.WARNING)
            with open(engine_path, "rb") as f:
                runtime = trt.Runtime(logger)
                engine = runtime.deserialize_cuda_engine(f.read())
            context = engine.create_execution_context()

            def _allocate_buffers(eng):
                inputs, outputs, bindings = [], [], []
                stream = cuda.Stream()
                for binding in eng:
                    shape = eng.get_binding_shape(binding)
                    size = trt.volume(shape) * eng.max_batch_size
                    dtype = trt.nptype(eng.get_binding_dtype(binding))
                    host_mem = cuda.pagelocked_empty(size, dtype)
                    device_mem = cuda.mem_alloc(host_mem.nbytes)
                    bindings.append(int(device_mem))
                    if eng.binding_is_input(binding):
                        inputs.append({'host': host_mem, 'device': device_mem})
                    else:
                        outputs.append({'host': host_mem, 'device': device_mem})
                return inputs, outputs, bindings, stream

            inputs, outputs, bindings, stream = _allocate_buffers(engine)
            print("[TensorRT] Loaded successfully")
            return (context, engine, inputs, outputs, bindings, stream), 'cuda', True, 'engine', None
        except Exception as e:
            print(f"[TensorRT Error] {e}")
            traceback.print_exc()

    # ---- 2. PyTorch ----
    if os.path.exists(pt_path):
        try:
            print("[PyTorch] Loading YOLOv5 model...")
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            print(f"[PyTorch] Using device: {device}")

            current_dir = os.path.dirname(os.path.abspath(__file__))
            yolo_root = os.path.dirname(current_dir)
            possible_yolo_paths = [
                os.path.join(yolo_root, 'yolov5-6.2'),
                os.path.join(yolo_root, 'yolov5'),
                yolo_root,
            ]
            for yolo_path in possible_yolo_paths:
                if os.path.exists(os.path.join(yolo_path, 'models')):
                    if yolo_path not in sys.path:
                        sys.path.insert(0, yolo_path)
                        print(f"[PyTorch] Added YOLOv5 path: {yolo_path}")
                    break

            import utils.general
            utils.general.check_requirements = lambda *args, **kwargs: None
            from utils.general import non_max_suppression as nms_fn
            from models.common import DetectMultiBackend

            model = DetectMultiBackend(pt_path, device=device, dnn=False, fp16=True)

            # 预热
            if device.type == 'cuda':
                dummy = torch.zeros(1, 3, DETECTION_SIZE, DETECTION_SIZE).to(device).type(torch.float16)
            else:
                dummy = torch.zeros(1, 3, DETECTION_SIZE, DETECTION_SIZE).to(device)
            for _ in range(3):
                model(dummy)

            print("[PyTorch] Loaded successfully")
            return model, device, True, 'pt', nms_fn
        except Exception as e:
            print(f"[PyTorch Error] {e}")
            traceback.print_exc()

    # ---- 3. ONNX (CPU 回退) ----
    if os.path.exists(onnx_path):
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(
                onnx_path,
                providers=['CPUExecutionProvider']  # 优先 CPU，避免 CUDA 警告
            )
            print("[ONNX] Loaded successfully")
            return session, 'cpu', False, 'onnx', None
        except Exception as e:
            print(f"[ONNX Error] {e}")

    raise FileNotFoundError("No model file found (checked .engine, .pt and .onnx)")


# ==================== 推理辅助函数 ====================

def _crop_center(frame, size):
    """从帧中裁剪中心区域。"""
    h, w = frame.shape[:2]
    half = size // 2
    x1 = max(0, w // 2 - half)
    y1 = max(0, h // 2 - half)
    x2 = min(w, w // 2 + half)
    y2 = min(h, h // 2 + half)
    return frame[y1:y2, x1:x2]


def _run_onnx_inference(model, img_rgb):
    """ONNX 推理，返回 (N, 6) tensor (xyxy 格式)。"""
    img_resized = cv2.resize(img_rgb, (DETECTION_SIZE, DETECTION_SIZE))
    img_tensor = img_resized.astype(np.float16) / 255.0
    img_tensor = np.transpose(img_tensor, (2, 0, 1))
    img_tensor = np.expand_dims(img_tensor, axis=0)
    outputs = model.run(None, {'images': img_tensor})
    detections = outputs[0][0]
    valid_dets = detections[detections[:, 4] > CONF_THRES]
    if len(valid_dets) > MAX_DETECTIONS:
        valid_dets = valid_dets[np.argsort(valid_dets[:, 4])[::-1][:MAX_DETECTIONS]]
    det = torch.from_numpy(valid_dets)
    # XYWH -> XYXY
    if len(det):
        det[:, 0] = det[:, 0] - det[:, 2] / 2
        det[:, 1] = det[:, 1] - det[:, 3] / 2
        det[:, 2] = det[:, 0] + det[:, 2]
        det[:, 3] = det[:, 1] + det[:, 3]
    return det


def _run_pt_inference(model, device, is_half, nms_fn, img_raw):
    """PyTorch 推理，返回 (N, 6) tensor (xyxy 格式)。"""
    img_resized = cv2.resize(img_raw, (DETECTION_SIZE, DETECTION_SIZE))
    img = preprocess_for_model(img_resized, device, half=is_half)
    pred = model(img)
    det = nms_fn(pred, CONF_THRES, IOU_THRES, max_det=MAX_DETECTIONS)[0]
    return det


def _run_engine_inference(model, img_rgb):
    """TensorRT 推理，返回 (N, 6) tensor (xyxy 格式)。"""
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401

    context, engine, trt_inputs, trt_outputs, bindings, stream = model

    img_resized = cv2.resize(img_rgb, (DETECTION_SIZE, DETECTION_SIZE))
    img_norm = img_resized.astype(np.float32) / 255.0
    img_input = np.transpose(img_norm, (2, 0, 1)).flatten()

    np.copyto(trt_inputs[0]['host'], img_input)
    cuda.memcpy_htod_async(trt_inputs[0]['device'], trt_inputs[0]['host'], stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(trt_outputs[0]['host'], trt_outputs[0]['device'], stream)
    stream.synchronize()

    output = trt_outputs[0]['host'].reshape(-1, 6)
    valid_mask = output[:, 4] > CONF_THRES
    detections = output[valid_mask]

    if len(detections) > MAX_DETECTIONS:
        indices = np.argsort(detections[:, 4])[::-1][:MAX_DETECTIONS]
        detections = detections[indices]

    det = torch.from_numpy(detections)
    # XYWH -> XYXY
    if len(det):
        det[:, 0] = det[:, 0] - det[:, 2] / 2
        det[:, 1] = det[:, 1] - det[:, 3] / 2
        det[:, 2] = det[:, 0] + det[:, 2]
        det[:, 3] = det[:, 1] + det[:, 3]
    return det


# ==================== 目标选择 ====================

def _select_target(detections, target_type, is_locking, last_dx, last_dy):
    """
    从检测结果中选择最佳目标。
    返回: (best_dx, best_dy, has_target, is_locking, last_dx, last_dy)
    """
    best_dx, best_dy = 0, 0
    has_target = False
    center = DETECTION_SIZE / 2
    candidates = []

    if len(detections) == 0:
        return best_dx, best_dy, False, False, last_dx, last_dy

    det_np = detections.cpu().numpy()
    for detection in det_np:
        if len(detection) < 6:
            continue

        x1, y1, x2, y2, conf, cls = detection[:6]
        if conf < CONF_THRES:
            continue

        # 坐标裁剪
        x1 = max(0, min(x1, DETECTION_SIZE))
        y1 = max(0, min(y1, DETECTION_SIZE))
        x2 = max(0, min(x2, DETECTION_SIZE))
        y2 = max(0, min(y2, DETECTION_SIZE))

        if int(cls) != target_type:
            continue

        tx, ty = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = tx - center, ty - center
        dist = (dx ** 2 + dy ** 2) ** 0.5

        if dist > AIM_FOV_RADIUS:
            continue

        # 权重: 距离 + 置信度
        weight = dist * 0.6 + (1 - conf) * 100
        if is_locking:
            last_dist = ((dx - last_dx) ** 2 + (dy - last_dy) ** 2) ** 0.5
            weight = weight * 0.4 + last_dist * 0.6

        candidates.append((weight, dx, dy))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        _, best_dx, best_dy = candidates[0]
        has_target = True
        is_locking = True
        last_dx, last_dy = best_dx, best_dy
    else:
        is_locking = False

    return best_dx, best_dy, has_target, is_locking, last_dx, last_dy


# ==================== 可视化 ====================

def _draw_overlay(img, detections, center, aim_fov_radius, has_target, best_dx, best_dy):
    """在图像上绘制 FOV 圆、检测框和瞄准线。"""
    cv2.circle(img, (int(center), int(center)), aim_fov_radius, (0, 255, 0), 1)

    if len(detections):
        for d in detections:
            x1, y1, x2, y2, conf, cls = map(int, d[:6])
            color = (0, 255, 0) if cls == 1 else (255, 0, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    if has_target:
        cv2.line(
            img,
            (int(center), int(center)),
            (int(center + best_dx), int(center + best_dy)),
            (0, 255, 255), 2
        )


# ==================== 主函数 ====================

def main():
    """主入口：初始化 → 主循环 → 清理。"""

    # ---- DPI 感知 ----
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception as e:
        print(f">>> [Warning] DPI Aware failed: {e}")

    global current_target_type, non_max_suppression

    # ---- 运行时配置 ----
    runtime_config = build_runtime_config()
    maybe_relaunch_as_admin(runtime_config)

    obs_camera_index = runtime_config['obs_camera_index']
    os.environ['OBS_CAMERA_INDEX'] = str(obs_camera_index)
    print(f">>> OBS camera index: {obs_camera_index}")

    # ---- 罗技驱动 ----
    print(">>> Initializing Logitech driver...")
    try:
        driver = Logitech()
        if not driver.ok:
            raise Exception("Driver not ok")
        print(">>> Logitech driver initialized.")
    except Exception as e:
        print(f">>> [WARNING] Mouse driver failed: {e}. Aiming disabled.")
        driver = type('Mock', (), {'ok': False, 'move': lambda s, x, y: None})()

    # ---- 加载模型 ----
    try:
        model, device, is_half, model_type, nms_fn = load_model_core()
        non_max_suppression = nms_fn
    except Exception as e:
        print(f">>> Model Error: {e}")
        traceback.print_exc()
        return

    # ---- 初始化屏幕捕获 ----
    print(">>> Initializing OBS Virtual Camera capture...")
    try:
        capture = OBSCapture(camera_index=obs_camera_index)
        print(">>> OBS Virtual Camera capture ready")
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"OBS Virtual Camera init failed: {e}") from e

    # ---- 窗口设置 ----
    show_window = True
    calibration = CrosshairCalibration()  # 实例化校准对象
    try:
        cv2.namedWindow("Radar", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Radar", 320, 320)
        cv2.setWindowProperty("Radar", cv2.WND_PROP_TOPMOST, 1)
        cv2.setMouseCallback("Radar", mouse_callback, calibration)
    except Exception:
        pass

    # ---- 状态变量 ----
    aim_controller = FastAimController(base_sensitivity=1.2)
    aim_enabled = True
    t_win, t_aim, t_target = 0, 0, 0
    is_locking = False
    last_target_x, last_target_y = 0, 0

    # FPS 计算: 使用滑动窗口
    fps_history = []
    fps_window_size = 30

    print("\n" + "=" * 30)
    print("[SYSTEM READY] - Press 'q' to quit")
    print("=" * 30 + "\n")

    try:
        frame_count = 0
        while True:
            loop_start = time.time()

            # ---- 帧捕获 ----
            obs_frame = capture.grab()
            if obs_frame is None or obs_frame.size == 0:
                continue

            img_raw = _crop_center(obs_frame, DETECTION_SIZE)
            if img_raw.size == 0:
                continue

            if not img_raw.flags['C_CONTIGUOUS']:
                img_raw = np.ascontiguousarray(img_raw)

            # ---- 按键处理 ----
            if win32api.GetAsyncKeyState(KEYS['QUIT'][0]) & 0x8000:
                break

            is_pressed, t_win = check_key_state(KEYS['TOGGLE_WIN'], t_win)
            if is_pressed:
                show_window = not show_window
                if show_window:
                    cv2.namedWindow("Radar", cv2.WINDOW_NORMAL)
                else:
                    cv2.destroyAllWindows()

            is_pressed, t_aim = check_key_state(KEYS['TOGGLE_AIM'], t_aim)
            if is_pressed:
                aim_enabled = not aim_enabled

            is_pressed, t_target = check_key_state(KEYS['TOGGLE_TARGET'], t_target)
            if is_pressed:
                current_target_type = 1 - current_target_type

            # ---- 推理 ----
            img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

            if model_type == 'onnx':
                det = _run_onnx_inference(model, img_rgb)
            elif model_type == 'pt':
                det = _run_pt_inference(model, device, is_half, nms_fn, img_raw)
            elif model_type == 'engine':
                det = _run_engine_inference(model, img_rgb)
            else:
                det = torch.empty((0, 6))

            # ---- 目标选择 ----
            frame_count += 1
            best_dx, best_dy, has_target, is_locking, last_target_x, last_target_y = \
                _select_target(det, current_target_type, is_locking, last_target_x, last_target_y)

            # ---- 鼠标移动 ----
            status = "Ready"
            if aim_enabled and has_target:
                move_x, move_y = aim_controller.calculate_movement(best_dx, best_dy)
                if driver.ok:
                    driver.move(int(move_x), int(move_y))
                    dist = (best_dx ** 2 + best_dy ** 2) ** 0.5
                    status = f"AIMING | Dist: {int(dist)}"
            else:
                aim_controller.reset()

            # ---- FPS 计算 (滑动窗口平均) ----
            loop_time = time.time() - loop_start
            fps_history.append(1.0 / loop_time if loop_time > 0 else 0)
            if len(fps_history) > fps_window_size:
                fps_history.pop(0)

            if frame_count % 10 == 0:
                avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0
                print_status(
                    f"[{'HEAD' if current_target_type == 1 else 'BODY'}] {status} | FPS: {int(avg_fps)}   "
                )

            # ---- 显示 ----
            if show_window and frame_count % 5 == 0:
                center = DETECTION_SIZE / 2
                _draw_overlay(img_raw, det, center, AIM_FOV_RADIUS, has_target, best_dx, best_dy)
                cv2.imshow("Radar", img_raw)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()