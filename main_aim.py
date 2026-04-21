import warnings

warnings.filterwarnings("ignore")
import torch
import numpy as np
import time
import os
import sys
import ctypes
import cv2
import win32api

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import traceback
import subprocess
import argparse
from logitech import Logitech
import threading
import queue


def build_runtime_config(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--obs-camera-index', dest='obs_camera_index', type=int)
    args, _ = parser.parse_known_args(argv if argv is not None else sys.argv[1:])

    config = {
        'obs_camera_index': int(os.environ.get('OBS_CAMERA_INDEX', '0')),
    }

    if args.obs_camera_index is not None:
        config['obs_camera_index'] = args.obs_camera_index

    return config


def maybe_relaunch_as_admin(runtime_config):
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
    result = ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, arg_string, None, 1)
    if result <= 32:
        raise RuntimeError("ShellExecuteW runas failed with code {}".format(result))
    sys.exit()


# ================== ⚡ 屏幕捕获类 ⚡ ==================
class OBSCapture:
    """
    通过 OBS Virtual Camera 提供画面输入。
    """

    def __init__(self, camera_index=0, width=None, height=None):
        self.camera_index = int(os.environ.get('OBS_CAMERA_INDEX', camera_index))
        self.width = width
        self.height = height
        self.cap = None
        self._init_virtual_camera()
        print(f"[OBS] Using Virtual Camera index {self.camera_index}")

    def _init_virtual_camera(self):
        backends = []
        if hasattr(cv2, 'CAP_DSHOW'):
            backends.append(cv2.CAP_DSHOW)
        if hasattr(cv2, 'CAP_MSMF'):
            backends.append(cv2.CAP_MSMF)
        backends.append(None)

        errors = []
        for backend in backends:
            if backend is None:
                cap = cv2.VideoCapture(self.camera_index)
                backend_name = 'default'
            else:
                cap = cv2.VideoCapture(self.camera_index, backend)
                backend_name = str(backend)

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


class ScreenCapture:
    """
    OBS Virtual Camera 捕获封装类
    """

    def __init__(self, target_fps=144, obs_camera_index=0):
        self.target_fps = target_fps
        self.obs_capture = OBSCapture(camera_index=obs_camera_index)

    def grab(self):
        return self.obs_capture.grab()

    def release(self):
        self.obs_capture.release()
        print("[OBS] Released")


# ========================================================

# ================== ⚡ 参数设置区 ⚡ ==================
# [注意] 如果你的模型是用 640 训练的，这里改成 640
DETECTION_SIZE = 320

# 限制最大FPS，防止CPU占用过高
DXCAM_MAX_FPS = 144

# 批量推理设置 (设为1禁用批量推理，设为2-4启用)
# 注意: 批量推理会增加延迟，实时瞄准建议保持为1
BATCH_SIZE = 1  # 建议值: 1(实时) 或 2-4(高吞吐)
BATCH_MAX_LATENCY_MS = 8  # 最大等待时间(毫秒)

CONF_THRES = 0.5
IOU_THRES = 0.45
MAX_DETECTIONS = 20
AIM_FOV_RADIUS = 300
# ========================================================

# 初始目标: 1=头 (可以通过按键切换)
current_target_type = 1
target_mapping = {0: 'BODY', 1: 'HEAD'}

non_max_suppression = None
# [注意] 如果你的模型是用 640 训练的，这里改成 640
DETECTION_SIZE = 320

# 限制最大FPS，防止CPU占用过高
DXCAM_MAX_FPS = 144

# 批量推理设置 (设为1禁用批量推理，设为2-4启用)
# 注意: 批量推理会增加延迟，实时瞄准建议保持为1
BATCH_SIZE = 1  # 建议值: 1(实时) 或 2-4(高吞吐)
BATCH_MAX_LATENCY_MS = 8  # 最大等待时间(毫秒)

CONF_THRES = 0.5
IOU_THRES = 0.45
MAX_DETECTIONS = 20
AIM_FOV_RADIUS = 300
# ========================================================

# 初始目标: 1=头 (可以通过按键切换)
current_target_type = 1
target_mapping = {0: 'BODY', 1: 'HEAD'}

non_max_suppression = None

# 按键表
KEYS = {
    'TOGGLE_WIN': [0xDC],  # \ 键：开关识别窗口
    'TOGGLE_AIM': [0xDD],  # ] 键：开启/暂停自瞄
    'TOGGLE_TARGET': [0xDB],  # [ 键：切换锁定目标（头/身）
    'QUIT': [0x51]  # q 键：退出程序
}


def print_status(message):
    print(f"\r{message}", end="", flush=True)


class CrosshairCalibration:
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


class FastAimController:
    def __init__(self, base_sensitivity=1.0, acceleration_curve=1.8):
        self.base_sensitivity = base_sensitivity
        self.acceleration_curve = acceleration_curve
        self.last_move_x = 0
        self.last_move_y = 0
        self.smooth_factor = 0.2  # 平滑因子，平衡响应速度和稳定性

    def calculate_movement(self, error_x, error_y):
        # 计算距离
        distance = (error_x**2 + error_y**2)**0.5
        
        # 根据距离调整灵敏度（远距离时更快，近距离更精确）
        if distance > 100:
            sensitivity = self.base_sensitivity * 2.0  # 远距离大幅提速
        elif distance > 50:
            sensitivity = self.base_sensitivity * 1.5
        elif distance > 20:
            sensitivity = self.base_sensitivity * 1.1
        else:
            # 关键优化：近距离时使用更精细的控制
            sensitivity = self.base_sensitivity * 0.7  # 极近距离降低灵敏度以提高精度
            
        # 应用非线性加速曲线
        normalized_distance = min(1.0, distance / 120.0)  # 调整归一化范围
        acceleration = normalized_distance ** self.acceleration_curve
        
        # 计算基础移动量
        move_x = error_x * sensitivity * acceleration
        move_y = error_y * sensitivity * acceleration
        
        # 应用智能平滑滤波器
        # 在远距离时减少平滑以提高响应速度，在近距离时增加平滑以提高稳定性
        dynamic_smooth = self.smooth_factor * (0.5 + 0.5 * (distance / 100.0)) if distance < 100 else 0.1
        
        # 应用低通滤波器减少抖动，但保持快速响应
        filtered_move_x = self.last_move_x * dynamic_smooth + move_x * (1 - dynamic_smooth)
        filtered_move_y = self.last_move_y * dynamic_smooth + move_y * (1 - dynamic_smooth)
        
        # 更新历史值
        self.last_move_x = filtered_move_x
        self.last_move_y = filtered_move_y
        
        # 一帧定位，直接返回计算结果
        return filtered_move_x, filtered_move_y

    def reset(self):
        """重置控制器状态"""
        self.last_move_x = 0
        self.last_move_y = 0


def mouse_callback(event, x, y, flags, param):
    calibration = param
    if event == cv2.EVENT_LBUTTONDOWN:
        calibration.set_point(x, y)
        print(f"[CALIBRATION] Crosshair center set to ({x}, {y})")


def show_radar_matplotlib(img):
    global show_window, fig, ax, im
    if 'fig' not in globals():
        plt.ion()
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title('Radar')
        ax.axis('off')
        close_ax = plt.axes([0.8, 0.9, 0.1, 0.05])
        close_button = Button(close_ax, 'Close')
        close_button.on_clicked(lambda event: plt.close(fig))
        plt.show()
    else:
        im.set_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.draw()
        plt.pause(0.001)


def check_key_state(key_codes, last_time, debounce_time=0.2):
    current_time = time.time()
    if current_time - last_time < debounce_time:
        return False, last_time
    for key_code in key_codes:
        if win32api.GetAsyncKeyState(key_code) & 0x8000:
            return True, current_time
    return False, last_time


# ================== ⚡ GPU 预处理函数 ⚡ ==================
def preprocess_gpu(img_bgr, device, half=True):
    """
    GPU 预处理函数
    - 将 BGR uint8 图像在 GPU 上转换为 RGB float32/float16
    - 输出格式: (1, 3, H, W)
    """
    # 上传 BGR 到 GPU
    img_tensor = torch.from_numpy(img_bgr).to(device)
    # BGR -> RGB (使用索引重排)
    img_tensor = img_tensor[..., [2, 1, 0]]
    # uint8 -> float + normalize
    if half:
        img_tensor = img_tensor.half() / 255.0
    else:
        img_tensor = img_tensor.float() / 255.0
    # HWC -> CHW
    img_tensor = img_tensor.permute(2, 0, 1)
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor


# ================== ⚡ CUDA NMS 函数 ⚡ ==================
def cuda_nms(boxes, scores, iou_thres=0.45, max_det=300):
    """
    CUDA NMS 使用 torchvision.ops.nms
    boxes: tensor [N, 4] in xyxy format
    scores: tensor [N]
    """
    try:
        from torchvision.ops import nms
        indices = nms(boxes, scores, iou_thres)
        if len(indices) > max_det:
            indices = indices[:max_det]
        return indices
    except Exception as e:
        # 回退到 CPU NMS
        return None


# ========================================================

def load_model_core():
    print("Loading model...")
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 按优先级：TensorRT > PyTorch > ONNX (TensorRT 最快)
    engine_path = os.path.join(script_dir, '1225_2best.engine')
    pt_path = os.path.join(script_dir, '1225_2best.pt')
    onnx_path = os.path.join(script_dir, '1225_2best.onnx')

    # 1. 优先使用 TensorRT (最快，GPU 优化)
    if os.path.exists(engine_path):
        try:
            print("[TensorRT] Loading TensorRT engine...")
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit

            logger = trt.Logger(trt.Logger.WARNING)
            with open(engine_path, "rb") as f:
                runtime = trt.Runtime(logger)
                engine = runtime.deserialize_cuda_engine(f.read())
            context = engine.create_execution_context()

            # 分配 CUDA 缓冲区 (只分配一次，复用)
            def allocate_buffers(engine):
                inputs = []
                outputs = []
                bindings = []
                stream = cuda.Stream()

                for binding in engine:
                    size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
                    dtype = trt.nptype(engine.get_binding_dtype(binding))
                    # 分配主机和设备内存
                    host_mem = cuda.pagelocked_empty(size, dtype)
                    device_mem = cuda.mem_alloc(host_mem.nbytes)
                    bindings.append(int(device_mem))
                    if engine.binding_is_input(binding):
                        inputs.append({'host': host_mem, 'device': device_mem})
                    else:
                        outputs.append({'host': host_mem, 'device': device_mem})
                return inputs, outputs, bindings, stream

            inputs, outputs, bindings, stream = allocate_buffers(engine)

            print("[TensorRT] Loaded successfully")
            return (context, engine, inputs, outputs, bindings, stream), 'cuda', True, 'engine', None
        except Exception as e:
            print(f"[TensorRT Error] {e}")
            traceback.print_exc()

    # 2. 备用 PyTorch (GPU 速度快，依赖少)
    if os.path.exists(pt_path):
        try:
            print("[PyTorch] Loading YOLOv5 model...")
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            print(f"[PyTorch] Using device: {device}")

            current_dir = os.path.dirname(os.path.abspath(__file__))
            yolo_root = os.path.dirname(current_dir)
            # 尝试多个可能的 YOLOv5 路径
            possible_yolo_paths = [
                os.path.join(yolo_root, 'yolov5-6.2'),  # yolov5-6.2 目录
                os.path.join(yolo_root, 'yolov5'),      # yolov5 目录
                yolo_root,                               # 父目录
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

    # 3. 最后尝试 ONNX (CPU 模式)
    if os.path.exists(onnx_path):
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            print("[ONNX] Loaded successfully")
            return session, 'cuda', False, 'onnx', None
        except Exception as e:
            print(f"[ONNX Error] {e}")

    raise FileNotFoundError("No model file found (checked .engine, .pt and .onnx)")


def main():
    # === 【关键修改1】强制开启高DPI感知，解决分辨率不匹配 ===
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception as e:
        print(f">>> [Warning] DPI Aware failed: {e}")

    global show_window, current_target_type

    runtime_config = build_runtime_config()
    maybe_relaunch_as_admin(runtime_config)

    capture_backend = runtime_config['capture_backend']
    obs_source_name = runtime_config['obs_source_name']
    obs_camera_index = runtime_config['obs_camera_index']
    obs_host = runtime_config['obs_host']
    obs_port = runtime_config['obs_port']
    obs_password = runtime_config['obs_password']

    os.environ['CAPTURE_BACKEND'] = capture_backend
    os.environ['OBS_SOURCE_NAME'] = obs_source_name
    os.environ['OBS_CAMERA_INDEX'] = str(obs_camera_index)
    os.environ['OBS_HOST'] = obs_host
    os.environ['OBS_PORT'] = str(obs_port)
    os.environ['OBS_PASSWORD'] = obs_password

    print(f">>> Capture backend requested: {capture_backend}")
    if capture_backend == 'obs':
        print(
            f">>> OBS settings: host={obs_host}:{obs_port}, camera_index={obs_camera_index}, "
            f"source={'program output' if not obs_source_name else obs_source_name}"
        )

    # 3. 初始化罗技驱动
    print(">>> Initializing Logitech driver...")
    try:
        driver = Logitech()
        if not driver.ok: raise Exception("Driver not ok")
        print(">>> Logitech driver initialized.")
    except Exception as e:
        print(f">>> [WARNING] Mouse driver failed: {e}. Aiming disabled.")
        driver = type('Mock', (), {'ok': False, 'move': lambda s, x, y: None})()

    # 4. 加载模型
    try:
        model, device, is_half, model_type, nms_fn = load_model_core()
        global non_max_suppression
        non_max_suppression = nms_fn
        if model_type == 'pt':
            # 预热（已在加载时执行，跳过）
            pass
    except Exception as e:
        print(f">>> Model Error: {e}")
        traceback.print_exc()
        return

    # 5. 初始化屏幕捕获 (支持 OBS / dxcam / mss)
    print(">>> Initializing screen capture...")
    try:
        capture = ScreenCapture(
            target_fps=DXCAM_MAX_FPS,
            backend=capture_backend,
            obs_source_name=obs_source_name,
            obs_camera_index=obs_camera_index,
            obs_host=obs_host,
            obs_port=obs_port,
            obs_password=obs_password,
        )
        active_backend = 'obs' if capture.obs_capture else ('dxcam' if capture.camera else 'mss')
        print(f">>> Capture backend ready: {active_backend}")
    except Exception as e:
        print(f">>> Screen Capture Init Failed: {e}")
        return

    # 窗口设置
    show_window = True
    try:
        cv2.namedWindow("Radar", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Radar", 320, 320)
        cv2.setWindowProperty("Radar", cv2.WND_PROP_TOPMOST, 1)
        cv2.setMouseCallback("Radar", mouse_callback)
    except:
        pass

    # 变量初始化
    aim_controller = FastAimController(base_sensitivity=1.2)  # 提高基础灵敏度
    aim_enabled = True
    t_win, t_aim, t_target = 0, 0, 0
    prev_time = time.time()

    # 锁定逻辑变量
    is_locking = False
    last_target_x, last_target_y = 0, 0

    print("\n" + "=" * 30)
    print("[SYSTEM READY] - Press 'q' to quit")
    print("=" * 30 + "\n")

    try:
        frame_count = 0
        while True:
            # === 获取裁剪后的帧 (使用 dxcam 或 mss) ===
            if capture.obs_capture:
                obs_frame = capture.grab()
                if obs_frame is None or obs_frame.size == 0:
                    continue
                h_scr, w_scr = obs_frame.shape[:2]
                center_x_scr, center_y_scr = w_scr // 2, h_scr // 2
                half_size = DETECTION_SIZE // 2
                x1_crop = max(0, center_x_scr - half_size)
                y1_crop = max(0, center_y_scr - half_size)
                x2_crop = min(w_scr, center_x_scr + half_size)
                y2_crop = min(h_scr, center_y_scr + half_size)
                img_raw = obs_frame[y1_crop:y2_crop, x1_crop:x2_crop]
            elif USE_DXCAM and capture.camera:
                # dxcam 直接捕获中心区域
                region = capture.camera.region
                screen_w, screen_h = region[2], region[3]
                center_x, center_y = screen_w // 2, screen_h // 2
                half_size = DETECTION_SIZE // 2
                crop_region = (
                    center_x - half_size,
                    center_y - half_size,
                    center_x + half_size,
                    center_y + half_size
                )
                img_raw = capture.camera.grab(crop_region)
                if img_raw is None:
                    continue
            else:
                # mss 捕获全屏再裁剪
                full_img = capture.grab()
                if full_img is None:
                    continue

                # 手动裁剪中心区域
                h_scr, w_scr = full_img.shape[:2]
                center_x_scr, center_y_scr = w_scr // 2, h_scr // 2
                half_size = DETECTION_SIZE // 2

                x1_crop = max(0, center_x_scr - half_size)
                y1_crop = max(0, center_y_scr - half_size)
                x2_crop = min(w_scr, center_x_scr + half_size)
                y2_crop = min(h_scr, center_y_scr + half_size)

                img_raw = full_img[y1_crop:y2_crop, x1_crop:x2_crop]

            if img_raw.size == 0:
                continue

            # === 修复 OpenCV 数组连续性问题 ===
            if not img_raw.flags['C_CONTIGUOUS']:
                img_raw = np.ascontiguousarray(img_raw)

            # === 1. 按键处理 ===
            if win32api.GetAsyncKeyState(KEYS['QUIT'][0]) & 0x8000: break

            is_pressed, t_win = check_key_state(KEYS['TOGGLE_WIN'], t_win)
            if is_pressed:
                show_window = not show_window
                if show_window:
                    cv2.namedWindow("Radar", cv2.WINDOW_NORMAL)
                else:
                    cv2.destroyAllWindows()

            is_pressed, t_aim = check_key_state(KEYS['TOGGLE_AIM'], t_aim)
            if is_pressed: aim_enabled = not aim_enabled

            is_pressed, t_target = check_key_state(KEYS['TOGGLE_TARGET'], t_target)
            if is_pressed: current_target_type = 1 - current_target_type

            # === 2. 推理 ===
            img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

            # === 【关键修改4】防止 ONNX 模式下检测到人时闪退 ===
            detection_size = DETECTION_SIZE

            if model_type == 'onnx':
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
                det[:, 0] = det[:, 0] - det[:, 2] / 2
                det[:, 1] = det[:, 1] - det[:, 3] / 2
                det[:, 2] = det[:, 0] + det[:, 2]
                det[:, 3] = det[:, 1] + det[:, 3]

            elif model_type == 'pt':
                # 使用 GPU 预处理 (跳过 CPU 的 BGR->RGB 转换)
                img_resized = cv2.resize(img_raw, (DETECTION_SIZE, DETECTION_SIZE))
                img = preprocess_gpu(img_resized, device, half=is_half)
                pred = model(img)
                det = non_max_suppression(pred, CONF_THRES, IOU_THRES, max_det=MAX_DETECTIONS)[0]

            elif model_type == 'engine':
                # TensorRT 推理
                import pycuda.driver as cuda
                import pycuda.autoinit

                context, engine, trt_inputs, trt_outputs, bindings, stream = model

                # 预处理 - resize + normalize + transpose
                img_resized = cv2.resize(img_rgb, (DETECTION_SIZE, DETECTION_SIZE))
                img_norm = img_resized.astype(np.float32) / 255.0
                img_input = np.transpose(img_norm, (2, 0, 1)).flatten()

                # 复制输入到页锁定内存
                np.copyto(trt_inputs[0]['host'], img_input)

                # 异步传输到 GPU 并执行推理
                cuda.memcpy_htod_async(trt_inputs[0]['device'], trt_inputs[0]['host'], stream)
                context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
                cuda.memcpy_dtoh_async(trt_outputs[0]['host'], trt_outputs[0]['device'], stream)
                stream.synchronize()

                # 解析输出
                output = trt_outputs[0]['host']
                # YOLOv5 输出格式: [batch, num_detections, 6] 或 [num_detections, 6]
                # 6 = [x, y, w, h, conf, cls]
                output = output.reshape(-1, 6)

                # 过滤低置信度
                valid_mask = output[:, 4] > CONF_THRES
                detections = output[valid_mask]

                # 限制最大检测数
                if len(detections) > MAX_DETECTIONS:
                    indices = np.argsort(detections[:, 4])[::-1][:MAX_DETECTIONS]
                    detections = detections[indices]

                # 转换为 torch tensor (与 PyTorch 分支保持一致)
                det = torch.from_numpy(detections)

                # XYWH -> XYXY
                if len(det) > 0:
                    det[:, 0] = det[:, 0] - det[:, 2] / 2
                    det[:, 1] = det[:, 1] - det[:, 3] / 2
                    det[:, 2] = det[:, 0] + det[:, 2]
                    det[:, 3] = det[:, 1] + det[:, 3]

            # === 3. 目标选择 logic ===
            frame_count += 1
            best_dx, best_dy = 0, 0
            has_target = False
            center = DETECTION_SIZE / 2
            candidates = []

            if len(det):
                det = det.cpu().numpy()
                for detection in det:
                    if len(detection) >= 6:
                        # 统一格式处理
                        x1, y1, x2, y2, conf, cls = detection[:6]

                        if conf < CONF_THRES: continue

                        # 坐标限制
                        x1 = max(0, min(x1, detection_size))
                        y1 = max(0, min(y1, detection_size))
                        x2 = max(0, min(x2, detection_size))
                        y2 = max(0, min(y2, detection_size))

                        if int(cls) == current_target_type:
                            tx, ty = (x1 + x2) / 2, (y1 + y2) / 2
                            dx, dy = tx - center, ty - center
                            dist = (dx ** 2 + dy ** 2) ** 0.5

                            if dist > AIM_FOV_RADIUS: continue

                            # 权重计算
                            weight = dist * 0.6 + (1 - conf) * 100
                            if is_locking:
                                last_dist = ((dx - last_target_x) ** 2 + (dy - last_target_y) ** 2) ** 0.5
                                weight = weight * 0.4 + last_dist * 0.6

                            candidates.append((weight, dx, dy, x1, y1, x2, y2, cls))

            if candidates:
                candidates.sort(key=lambda x: x[0])
                _, best_dx, best_dy, b_x1, b_y1, b_x2, b_y2, _ = candidates[0]
                has_target = True
                is_locking = True
                last_target_x, last_target_y = best_dx, best_dy
            else:
                is_locking = False

            # === 4. 鼠标移动 ===
            status = "Ready"
            if aim_enabled and has_target:
                # 使用新的快速瞄准控制器
                move_x, move_y = aim_controller.calculate_movement(best_dx, best_dy)

                if driver.ok:
                    driver.move(int(move_x), int(move_y))
                    dist = (best_dx ** 2 + best_dy ** 2) ** 0.5
                    status = f"AIMING | Dist: {int(dist)}"
            else:
                # 重置控制器状态
                aim_controller.reset()

            # 每10帧更新一次 FPS 显示，减少 IO 开销
            if frame_count % 10 == 0:
                print_status(
                    f"[{'HEAD' if current_target_type == 1 else 'BODY'}] {status} | FPS: {int(1 / (time.time() - prev_time)) if time.time() > prev_time else 0}   ")
                prev_time = time.time()

            # === 5. 显示优化 - 每5帧显示一次 ===
            if show_window and frame_count % 5 == 0:
                # 画FOV
                cv2.circle(img_raw, (int(center), int(center)), AIM_FOV_RADIUS, (0, 255, 0), 1)

                # 画框
                if len(det):
                    for d in det:
                        x1, y1, x2, y2, conf, cls = map(int, d[:6])
                        c = (0, 255, 0) if cls == 1 else (255, 0, 0)
                        cv2.rectangle(img_raw, (x1, y1), (x2, y2), c, 2)

                if has_target:
                    cv2.line(img_raw, (int(center), int(center)), (int(center + best_dx), int(center + best_dy)),
                             (0, 255, 255), 2)

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