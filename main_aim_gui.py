#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOv5 FPS 自动瞄准系统 - GUI 版本
带图形界面的版本，方便操作
"""

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

# 尝试导入 dxcam (DXGI 硬件捕获), 否则回退到 mss
USE_DXCAM = False
try:
    import dxcam
    USE_DXCAM = True
    print("[INFO] DXCam available")
except ImportError:
    pass

try:
    import mss
    print("[INFO] Using mss for screen capture")
except ImportError:
    mss = None
    print("[ERROR] Neither dxcam nor mss is available!")

import traceback
from logitech import Logitech

# ================== ⚡ 屏幕捕获类 ⚡ ==================
class ScreenCapture:
    """
    屏幕捕获封装类
    - 优先使用 dxcam (DXGI 硬件捕获)
    - 回退到 mss
    """

    def __init__(self, target_fps=144):
        self.target_fps = target_fps
        self.camera = None
        self.sct = None
        self.monitor = None
        self.using_dxcam = False

        if USE_DXCAM:
            try:
                self.camera = dxcam.create()
                # 测试捕获一帧，确保真正能工作
                test_frame = self.camera.grab()
                if test_frame is not None:
                    self.using_dxcam = True
                    print(f"[DXCam] Created and tested OK (region: {self.camera.region})")
                else:
                    raise Exception("DXCam test grab returned None")
            except Exception as e:
                print(f"[DXCam] Failed: {e}, falling back to mss")
                try:
                    if self.camera:
                        self.camera.release()
                except:
                    pass
                self.camera = None
                self._init_mss()
        else:
            self._init_mss()

    def _init_mss(self):
        """初始化 mss"""
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]
        print("[MSS] Initialized")

    def grab(self, region=None):
        """捕获屏幕"""
        if self.using_dxcam and self.camera:
            # DXCam 捕获
            if region:
                return self.camera.grab(region)
            return self.camera.grab()
        elif self.sct:
            # MSS 捕获 + 裁剪
            screenshot = self.sct.grab(self.monitor)
            img = np.array(screenshot, dtype=np.uint8)
            if region:
                x, y, w, h = region
                return img[y:y+h, x:x+w]
            return img
        return None

    def release(self):
        """释放资源"""
        if self.using_dxcam and self.camera:
            try:
                self.camera.stop()
                print("[DXCam] Released")
            except:
                pass
        elif self.sct:
            try:
                del self.sct
                print("[MSS] Released")
            except:
                pass

# ========================================================

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


# ================== ⚡ 参数设置区 ⚡ ==================
DETECTION_SIZE = 320
DXCAM_MAX_FPS = 144
CONF_THRES = 0.5
IOU_THRES = 0.45
MAX_DETECTIONS = 20
AIM_FOV_RADIUS = 300
# ========================================================

current_target_type = 1  # 0=身体, 1=头部

# ================== GUI 界面类 ==================
import tkinter as tk
from tkinter import ttk, messagebox


class AimApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv5 FPS Auto-Aim v1.0")
        self.root.geometry("400x550")
        self.root.resizable(False, False)

        # 设置窗口图标（使用简单颜色）
        self.root.configure(bg='#1e1e2e')

        # 变量
        self.aim_enabled = tk.BooleanVar(value=True)
        self.show_window = tk.BooleanVar(value=True)
        self.model_type = tk.StringVar(value="Unknown")
        self.status_text = tk.StringVar(value="Ready")
        self.fps_text = tk.StringVar(value="0 FPS")
        self.target_type = tk.StringVar(value="HEAD")

        # 初始化组件
        self.create_widgets()

        # 窗口置顶
        self.root.attributes('-topmost', True)

    def create_widgets(self):
        # 标题
        title_frame = tk.Frame(self.root, bg='#1e1e2e', pady=15)
        title_frame.pack(fill=tk.X)

        tk.Label(title_frame, text="🎯 YOLOv5 FPS Auto-Aim",
                font=('Microsoft YaHei', 16, 'bold'),
                fg='#00d4ff', bg='#1e1e2e').pack()
        tk.Label(title_frame, text="智能瞄准系统 - GUI 版本",
                font=('Microsoft YaHei', 9),
                fg='#888', bg='#1e1e2e').pack()

        # 状态卡片
        status_frame = tk.LabelFrame(self.root, text="系统状态",
                                    bg='#2a2a3a', fg='#fff',
                                    font=('Microsoft YaHei', 10),
                                    padx=10, pady=10, borderwidth=2)
        status_frame.pack(fill=tk.X, padx=10, pady=10)

        # 模型状态
        tk.Label(status_frame, text="模型类型:", font=('Microsoft YaHei', 9),
                bg='#2a2a3a', fg='#ccc').grid(row=0, column=0, sticky='w')
        tk.Label(status_frame, textvariable=self.model_type, font=('Microsoft YaHei', 9, 'bold'),
                bg='#2a2a3a', fg='#00d4ff').grid(row=0, column=1, sticky='w', padx=10)

        # FPS 显示
        tk.Label(status_frame, text="运行 FPS:", font=('Microsoft YaHei', 9),
                bg='#2a2a3a', fg='#ccc').grid(row=1, column=0, sticky='w')
        tk.Label(status_frame, textvariable=self.fps_text, font=('Microsoft YaHei', 9, 'bold'),
                bg='#2a2a3a', fg='#00ff88').grid(row=1, column=1, sticky='w', padx=10)

        # 目标类型
        tk.Label(status_frame, text="当前目标:", font=('Microsoft YaHei', 9),
                bg='#2a2a3a', fg='#ccc').grid(row=2, column=0, sticky='w')
        tk.Label(status_frame, textvariable=self.target_type, font=('Microsoft YaHei', 9, 'bold'),
                bg='#2a2a3a', fg='#ff0066').grid(row=2, column=1, sticky='w', padx=10)

        # 状态消息
        tk.Label(status_frame, text="状态消息:", font=('Microsoft YaHei', 9),
                bg='#2a2a3a', fg='#ccc').grid(row=3, column=0, sticky='w', pady=(5,0))
        tk.Label(status_frame, textvariable=self.status_text, font=('Microsoft YaHei', 9),
                bg='#2a2a3a', fg='#aaaaaa', width=30, anchor='w').grid(row=3, column=1, sticky='w', padx=10, pady=(5,0))

        # 控制面板
        control_frame = tk.LabelFrame(self.root, text="控制面板",
                                     bg='#2a2a3a', fg='#fff',
                                     font=('Microsoft YaHei', 10),
                                     padx=10, pady=10, borderwidth=2)
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        # 自瞄准开关
        tk.Label(control_frame, text="自动瞄准:", font=('Microsoft YaHei', 9),
                bg='#2a2a3a', fg='#ccc').grid(row=0, column=0, sticky='w')
        self.aim_toggle_btn = tk.Button(control_frame, text="开启",
                                       command=self.toggle_aim,
                                       bg='#00d4ff', fg='#fff', activebackground='#00a8cc',
                                       font=('Microsoft YaHei', 9, 'bold'),
                                       width=10, height=1, border=0)
        self.aim_toggle_btn.grid(row=0, column=1, sticky='w', padx=10)

        # 目标类型切换
        tk.Label(control_frame, text="目标类型:", font=('Microsoft YaHei', 9),
                bg='#2a2a3a', fg='#ccc').grid(row=1, column=0, sticky='w', pady=(5,0))
        self.target_toggle_btn = tk.Button(control_frame, text=" HEAD ",
                                          command=self.toggle_target,
                                          bg='#ff0066', fg='#fff', activebackground='#cc0055',
                                          font=('Microsoft YaHei', 9, 'bold'),
                                          width=10, height=1, border=0)
        self.target_toggle_btn.grid(row=1, column=1, sticky='w', padx=10, pady=(5,0))

        # 预热模型按钮
        self.warmup_btn = tk.Button(control_frame, text="预热模型",
                                   command=self.warmup_model,
                                   bg='#ffaa00', fg='#fff', activebackground='#cc8800',
                                   font=('Microsoft YaHei', 9),
                                   width=10, height=1, border=0)
        self.warmup_btn.grid(row=2, column=1, sticky='w', padx=10, pady=(5,0))

        # 按键说明
        hint_frame = tk.LabelFrame(self.root, text="快捷键说明",
                                  bg='#2a2a3a', fg='#fff',
                                  font=('Microsoft YaHei', 10),
                                  padx=10, pady=10, borderwidth=2)
        hint_frame.pack(fill=tk.X, padx=10, pady=5)

        hints = [
            ("F1", "切换自动瞄准"),
            ("F2", "切换目标类型"),
            ("F3", "显示/隐藏窗口"),
            ("F4", "退出程序"),
        ]

        for i, (key, desc) in enumerate(hints):
            hint_row = tk.Frame(hint_frame, bg='#2a2a3a')
            hint_row.pack(fill=tk.X, pady=3)
            tk.Label(hint_row, text=f"[{key}]", font=('Microsoft YaHei', 9, 'bold'),
                    bg='#2a2a3a', fg='#00d4ff', width=8, anchor='w').pack(side=tk.LEFT)
            tk.Label(hint_row, text=desc, font=('Microsoft YaHei', 9),
                    bg='#2a2a3a', fg='#aaa', anchor='w').pack(side=tk.LEFT)

        # 参数设置
        param_frame = tk.LabelFrame(self.root, text="参数设置",
                                   bg='#2a2a3a', fg='#fff',
                                   font=('Microsoft YaHei', 10),
                                   padx=10, pady=10, borderwidth=2)
        param_frame.pack(fill=tk.X, padx=10, pady=5)

        # 置信度阈值
        tk.Label(param_frame, text="置信度阈值:", font=('Microsoft YaHei', 9),
                bg='#2a2a3a', fg='#ccc').grid(row=0, column=0, sticky='w')
        self.conf_scale = tk.Scale(param_frame, from_=0.1, to=1.0, resolution=0.05,
                                  orient=tk.HORIZONTAL, bg='#2a2a3a', fg='#fff',
                                  highlightthickness=0, command=self.update_conf)
        self.conf_scale.set(CONF_THRES * 100)
        self.conf_scale.grid(row=0, column=1, sticky='ew', padx=5)

        # FOV 半径
        tk.Label(param_frame, text="FOV 半径:", font=('Microsoft YaHei', 9),
                bg='#2a2a3a', fg='#ccc').grid(row=1, column=0, sticky='w', pady=(5,0))
        self.fov_scale = tk.Scale(param_frame, from_=100, to=500, orient=tk.HORIZONTAL,
                                 bg='#2a2a3a', fg='#fff', highlightthickness=0, command=self.update_fov)
        self.fov_scale.set(AIM_FOV_RADIUS)
        self.fov_scale.grid(row=1, column=1, sticky='ew', padx=5, pady=(5,0))

        # 底部按钮
        bottom_frame = tk.Frame(self.root, bg='#1e1e2e')
        bottom_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Button(bottom_frame, text="重启程序",
                 command=self.restart_program,
                 bg='#ff4444', fg='#fff', activebackground='#cc3333',
                 font=('Microsoft YaHei', 9),
                 width=12, height=2, border=0).pack(side=tk.LEFT, padx=5)

        tk.Button(bottom_frame, text="退出程序",
                 command=self.quit_program,
                 bg='#ff6666', fg='#fff', activebackground='#cc5555',
                 font=('Microsoft YaHei', 9, 'bold'),
                 width=12, height=2, border=0).pack(side=tk.LEFT, padx=5)

    def toggle_aim(self):
        self.aim_enabled.set(not self.aim_enabled.get())
        self.update_aim_button()

    def toggle_target(self):
        global current_target_type
        current_target_type = 1 - current_target_type
        self.target_type.set("HEAD" if current_target_type == 1 else "BODY")
        self.update_target_button()

    def warmup_model(self):
        if hasattr(self, 'warmup_callback'):
            self.warmup_callback()

    def update_aim_button(self):
        if self.aim_enabled.get():
            self.aim_toggle_btn.config(text="已开启", bg='#00ff88', fg='#006600')
        else:
            self.aim_toggle_btn.config(text="已暂停", bg='#ffaa00', fg='#664400')

    def update_target_button(self):
        if current_target_type == 1:
            self.target_toggle_btn.config(text=" HEAD ", bg='#ff0066', fg='#fff')
        else:
            self.target_toggle_btn.config(text=" BODY ", bg='#0088ff', fg='#fff')

    def update_conf(self, val):
        global CONF_THRES
        CONF_THRES = int(val) / 100.0

    def update_fov(self, val):
        global AIM_FOV_RADIUS
        AIM_FOV_RADIUS = int(val)

    def restart_program(self):
        if messagebox.askyesno("确认", "确定要重启程序吗？"):
            python = sys.executable
            os.execv(python, [python] + sys.argv)

    def quit_program(self):
        if messagebox.askyesno("确认", "确定要退出程序吗？"):
            self.root.destroy()
            sys.exit()


# ================== 核心功能 ==================

def load_model_core():
    """加载模型（按优先级：TensorRT > PyTorch > ONNX）"""
    print(">>> Loading model...")
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 按优先级：TensorRT > PyTorch > ONNX (TensorRT 最快)
    engine_path = os.path.join(script_dir, '1225_2best_v10.engine')  # TensorRT 10.x
    engine_path_old = os.path.join(script_dir, '1225_2best.engine')  # 旧版本兼容
    pt_path = os.path.join(script_dir, '1225_2best.pt')
    onnx_path = os.path.join(script_dir, '1225_2best.onnx')

    # 1. 优先使用 TensorRT (最快，GPU 优化)
    if os.path.exists(engine_path):
        try:
            print(">>> [TensorRT] Loading TensorRT engine...")
            # TensorRT 10.x 使用 tensorrt_bindings
            try:
                import tensorrt_bindings as trt
            except ImportError:
                import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit

            logger = trt.Logger(trt.Logger.WARNING)
            # 检测 TensorRT 版本
            try:
                import tensorrt_bindings
                trt_version = tensorrt_bindings.__version__
            except:
                trt_version = '8.x'
            is_trt10 = str(trt_version).startswith('10')

            with open(engine_path, "rb") as f:
                runtime = trt.Runtime(logger)
                engine = runtime.deserialize_cuda_engine(f.read())
            context = engine.create_execution_context()

            # 分配 CUDA 缓冲区 (只分配一次，复用) - 适配 TensorRT 8.x 和 10.x
            def allocate_buffers(engine, is_trt10):
                inputs = []
                outputs = []
                bindings = []
                stream = cuda.Stream()

                if is_trt10:
                    # TensorRT 10.x API
                    for i in range(engine.num_io_tensors):
                        name = engine.get_tensor_name(i)
                        mode = engine.get_tensor_mode(name)
                        shape = engine.get_tensor_shape(name)
                        size = trt.volume(shape)
                        dtype = trt.nptype(engine.get_tensor_dtype(name))
                        # 分配主机和设备内存
                        host_mem = cuda.pagelocked_empty(size, dtype)
                        device_mem = cuda.mem_alloc(host_mem.nbytes)
                        bindings.append(int(device_mem))
                        if mode == trt.TensorIOMode.INPUT:
                            inputs.append({'host': host_mem, 'device': device_mem, 'name': name})
                        else:
                            outputs.append({'host': host_mem, 'device': device_mem, 'name': name})
                else:
                    # TensorRT 8.x API
                    for binding in engine:
                        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
                        dtype = trt.nptype(engine.get_binding_dtype(binding))
                        # 分配主机和设备内存
                        host_mem = cuda.pagelocked_empty(size, dtype)
                        device_mem = cuda.mem_alloc(host_mem.nbytes)
                        bindings.append(int(device_mem))
                        if engine.binding_is_input(binding):
                            inputs.append({'host': host_mem, 'device': device_mem, 'name': binding})
                        else:
                            outputs.append({'host': host_mem, 'device': device_mem, 'name': binding})
                return inputs, outputs, bindings, stream, is_trt10

            inputs, outputs, bindings, stream, _ = allocate_buffers(engine, is_trt10)

            print(f">>> [TensorRT] Loaded successfully (v{trt_version})")
            return (context, engine, inputs, outputs, bindings, stream, is_trt10), 'cuda', True, 'engine', True, None
        except Exception as e:
            print(f">>> [TensorRT Error] {e}")
            import traceback
            traceback.print_exc()

    # 2. 备用 PyTorch (GPU 速度快，依赖少)
    if os.path.exists(pt_path):
        try:
            print(">>> [PyTorch] Loading YOLOv5 model...")
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            print(f">>> [PyTorch] Using device: {device}")

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
                        print(f">>> [PyTorch] Added YOLOv5 path: {yolo_path}")
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

            print(">>> [PyTorch] Loaded successfully")
            return model, device, True, 'pt', True, nms_fn
        except Exception as e:
            print(f">>> [PyTorch Error] {e}")
            import traceback
            traceback.print_exc()

    # 3. 最后尝试 ONNX (CPU 模式)
    if os.path.exists(onnx_path):
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            print(">>> [ONNX] Loaded successfully (CPU mode)")
            return session, 'cpu', False, 'onnx', True, None
        except Exception as e:
            print(f">>> [ONNX Error] {e}")

    raise FileNotFoundError("No model file found or all formats failed to load")


class AimController:
    """瞄准控制器"""
    def __init__(self, base_sensitivity=1.2):
        self.base_sensitivity = base_sensitivity
        self.last_move_x = 0
        self.last_move_y = 0
        self.smooth_factor = 0.2

    def calculate_movement(self, error_x, error_y):
        distance = (error_x**2 + error_y**2)**0.5

        if distance > 100:
            sensitivity = self.base_sensitivity * 2.0
        elif distance > 50:
            sensitivity = self.base_sensitivity * 1.5
        elif distance > 20:
            sensitivity = self.base_sensitivity * 1.1
        else:
            sensitivity = self.base_sensitivity * 0.7

        normalized_distance = min(1.0, distance / 120.0)
        acceleration = normalized_distance ** 1.8

        move_x = error_x * sensitivity * acceleration
        move_y = error_y * sensitivity * acceleration

        dynamic_smooth = 0.1 if distance >= 100 else 0.2

        filtered_move_x = self.last_move_x * dynamic_smooth + move_x * (1 - dynamic_smooth)
        filtered_move_y = self.last_move_y * dynamic_smooth + move_y * (1 - dynamic_smooth)

        self.last_move_x = filtered_move_x
        self.last_move_y = filtered_move_y

        return filtered_move_x, filtered_move_y

    def reset(self):
        self.last_move_x = 0
        self.last_move_y = 0


# ================== 主程序 ==================

def main():
    # 检查管理员权限（临时注释掉，先调试）
    # if not ctypes.windll.shell32.IsUserAnAdmin():
    #     script_path = os.path.abspath(__file__)
    #     ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, f'"{script_path}"', None, 1)
    #     return

    # 先定义退出回调，以便异常时可以调用
    capture = None
    root = None
    app = None

    def on_closing():
        try:
            if capture:
                capture.release()
        except:
            pass
        try:
            if root:
                cv2.destroyAllWindows()
        except:
            pass
        try:
            if root:
                root.destroy()
        except:
            pass
        sys.exit(0)

    # 初始化 GUI
    try:
        print(">>> Starting GUI...")
        root = tk.Tk()
        app = AimApp(root)

        # 加载模型
        model, device, is_half, model_type, success, nms_fn = load_model_core()
        app.model_type.set(model_type.upper() if model_type != 'pt' else 'PyTorch')

        # 初始化屏幕录制（使用 dxcam 或 mss）
        try:
            capture = ScreenCapture(target_fps=DXCAM_MAX_FPS)
            app.status_text.set("屏幕录制初始化成功 (" + ("dxcam" if USE_DXCAM else "mss") + ")")
        except Exception as cam_err:
            print(f">>> [Screen Error] {cam_err}")
            app.status_text.set(f"屏幕初始化失败: {str(cam_err)[:50]}...")
            capture = None

        # 初始化控制器
        aim_controller = AimController(base_sensitivity=1.2)

        # 初始化罗技驱动
        driver = Logitech()
        if not driver.ok:
            app.status_text.set("鼠标驱动警告: 使用模拟模式")

        # 更新按钮状态
        app.update_aim_button()
        app.update_target_button()

        # 核心变量
        global current_target_type
        aim_enabled = True
        show_window = True
        is_locking = False
        last_target_x, last_target_y = 0, 0
        prev_time = time.time()

        app.status_text.set("系统就绪 - 按 F1 开始瞄准")

        # 构建控制回调
        def warmup_model():
            app.status_text.set("模型已在加载时预热完成")

        app.warmup_callback = warmup_model

        if root:
            root.protocol("WM_DELETE_WINDOW", on_closing)

        # 注册热键
        KEYS = {
            'TOGGLE_AIM': 0x70,      # F1
            'TOGGLE_TARGET': 0x71,   # F2
            'TOGGLE_WIN': 0x72,      # F3
            'QUIT': 0x73             # F4
        }

        # 主循环 - 最大性能模式，不处理 GUI 事件
        frame_count = 0
        prev_time = time.time()

        while True:
            # 1. 获取帧（使用 dxcam 或 mss）
            frame_count += 1
            if capture is not None:
                try:
                    if capture.using_dxcam and capture.camera:
                        # dxcam 直接捕获中心区域
                        try:
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
                        except Exception as dxcam_err:
                            print(f"[DXCam Error] {dxcam_err}, switching to mss")
                            capture.using_dxcam = False
                            capture._init_mss()
                            full_img = capture.grab()
                            if full_img is None:
                                continue
                            # 裁剪...
                            h_scr, w_scr = full_img.shape[:2]
                            center_x_scr, center_y_scr = w_scr // 2, h_scr // 2
                            half_size = DETECTION_SIZE // 2
                            x1_crop = max(0, center_x_scr - half_size)
                            y1_crop = max(0, center_y_scr - half_size)
                            x2_crop = min(w_scr, center_x_scr + half_size)
                            y2_crop = min(h_scr, center_y_scr + half_size)
                            img_raw = full_img[y1_crop:y2_crop, x1_crop:x2_crop]
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

                    if not img_raw.flags['C_CONTIGUOUS']:
                        img_raw = np.ascontiguousarray(img_raw)
                except Exception as e:
                    full_img = np.zeros((DETECTION_SIZE, DETECTION_SIZE, 3), dtype=np.uint8)
                    time.sleep(0.01)
                    continue
            else:
                full_img = np.zeros((DETECTION_SIZE, DETECTION_SIZE, 3), dtype=np.uint8)
                time.sleep(0.01)
                continue

            # 4. 检测热键（直接轮询，不依赖 GUI）
            if win32api.GetAsyncKeyState(KEYS['QUIT']) & 0x8000:
                on_closing()
                break
            if win32api.GetAsyncKeyState(KEYS['TOGGLE_WIN']) & 0x8000:
                time.sleep(0.3)
                show_window = not show_window
                if show_window:
                    cv2.namedWindow("Radar", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Radar", 320, 320)
                    cv2.setWindowProperty("Radar", cv2.WND_PROP_TOPMOST, 1)
                else:
                    cv2.destroyWindow("Radar")
            if win32api.GetAsyncKeyState(KEYS['TOGGLE_AIM']) & 0x8000:
                time.sleep(0.3)
                aim_enabled = not aim_enabled
            if win32api.GetAsyncKeyState(KEYS['TOGGLE_TARGET']) & 0x8000:
                time.sleep(0.3)
                current_target_type = 1 - current_target_type

            # 模型推理
            det = []
            detection_size = DETECTION_SIZE

            if model_type == 'onnx':
                img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
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
                det = nms_fn(pred, CONF_THRES, IOU_THRES, max_det=MAX_DETECTIONS)[0] if nms_fn else []

            elif model_type == 'engine':
                # TensorRT 推理 - 适配 8.x 和 10.x
                import pycuda.driver as cuda

                context, engine, trt_inputs, trt_outputs, bindings, stream, is_trt10 = model

                # ⚡ 性能分析计时
                t_start = time.perf_counter()

                # 预处理 - resize + normalize + transpose
                img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (DETECTION_SIZE, DETECTION_SIZE))
                img_norm = img_resized.astype(np.float32) / 255.0
                img_input = np.transpose(img_norm, (2, 0, 1)).flatten()

                t_preprocess = time.perf_counter()

                # 复制输入到页锁定内存
                np.copyto(trt_inputs[0]['host'], img_input)

                if is_trt10:
                    # TensorRT 10.x API
                    for inp in trt_inputs:
                        context.set_tensor_address(inp['name'], int(inp['device']))
                    for out in trt_outputs:
                        context.set_tensor_address(out['name'], int(out['device']))

                    cuda.memcpy_htod_async(trt_inputs[0]['device'], trt_inputs[0]['host'], stream)
                    context.execute_async_v3(stream_handle=stream.handle)
                    cuda.memcpy_dtoh_async(trt_outputs[0]['host'], trt_outputs[0]['device'], stream)
                else:
                    cuda.memcpy_htod_async(trt_inputs[0]['device'], trt_inputs[0]['host'], stream)
                    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
                    cuda.memcpy_dtoh_async(trt_outputs[0]['host'], trt_outputs[0]['device'], stream)
                stream.synchronize()

                t_inference = time.perf_counter()

                # 解析输出
                output = trt_outputs[0]['host']
                output = output.reshape(-1, 6)

                # 过滤低置信度
                valid_mask = output[:, 4] > CONF_THRES
                detections = output[valid_mask]

                # 限制最大检测数
                if len(detections) > MAX_DETECTIONS:
                    indices = np.argsort(detections[:, 4])[::-1][:MAX_DETECTIONS]
                    detections = detections[indices]

                # 转换为 torch tensor
                det = torch.from_numpy(detections)

                # XYWH -> XYXY
                if len(det) > 0:
                    det[:, 0] = det[:, 0] - det[:, 2] / 2
                    det[:, 1] = det[:, 1] - det[:, 3] / 2
                    det[:, 2] = det[:, 0] + det[:, 2]
                    det[:, 3] = det[:, 1] + det[:, 3]

                t_postprocess = time.perf_counter()

                # 每100帧打印性能分析
                if frame_count % 100 == 0:
                    total = (t_postprocess - t_start) * 1000
                    pre = (t_preprocess - t_start) * 1000
                    inf = (t_inference - t_preprocess) * 1000
                    post = (t_postprocess - t_inference) * 1000
                    print(f"[Perf] Total:{total:.1f}ms Pre:{pre:.1f}ms Infer:{inf:.1f}ms Post:{post:.1f}ms")

            # 目标选择
            best_dx, best_dy = 0, 0
            has_target = False
            center = DETECTION_SIZE / 2

            if len(det):
                det = det.cpu().numpy()
                candidates = []

                for detection in det:
                    if len(detection) >= 6:
                        x1, y1, x2, y2, conf, cls = detection[:6]

                        if conf < CONF_THRES:
                            continue

                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(DETECTION_SIZE, x2), min(DETECTION_SIZE, y2)

                        if int(cls) == current_target_type:
                            tx, ty = (x1 + x2) / 2, (y1 + y2) / 2
                            dx, dy = tx - center, ty - center
                            dist = (dx ** 2 + dy ** 2) ** 0.5

                            if dist > AIM_FOV_RADIUS:
                                continue

                            weight = dist * 0.6 + (1 - conf) * 100
                            if is_locking:
                                last_dist = ((dx - last_target_x) ** 2 + (dy - last_target_y) ** 2) ** 0.5
                                weight = weight * 0.4 + last_dist * 0.6

                            candidates.append((weight, dx, dy))

                if candidates:
                    candidates.sort(key=lambda x: x[0])
                    best_dx, best_dy = candidates[0][1], candidates[0][2]
                    has_target = True
                    is_locking = True
                    last_target_x, last_target_y = best_dx, best_dy
                else:
                    is_locking = False

            # 鼠标移动
            if aim_enabled and has_target:
                move_x, move_y = aim_controller.calculate_movement(best_dx, best_dy)
                if driver.ok:
                    driver.move(int(move_x), int(move_y))
                dist = (best_dx ** 2 + best_dy ** 2) ** 0.5
                # 只每50帧更新一次状态文本
                if frame_count % 50 == 0 and root:
                    root.after(0, app.status_text.set, f"追踪目标 | 距离: {int(dist)}px")
            else:
                aim_controller.reset()
                if not has_target and frame_count % 100 == 0 and root:
                    root.after(0, app.status_text.set, "扫描目标...")

            # FPS 计算 - 每100帧计算一次（减少 GUI 更新）
            if frame_count % 100 == 0:
                current_time = time.time()
                fps = int(100 / (current_time - prev_time)) if current_time > prev_time else 0
                # 使用 after 异步更新 GUI，避免阻塞
                if root:
                    root.after(0, app.fps_text.set, f"{fps} FPS")
                prev_time = current_time

            # 显示窗口 - 每5帧显示一次
            if show_window and frame_count % 5 == 0:
                cv2.circle(img_raw, (int(center), int(center)), AIM_FOV_RADIUS, (0, 255, 0), 1)

                if len(det):
                    for d in det:
                        x1, y1, x2, y2 = map(int, d[:4])
                        c = (0, 255, 0) if d[5] == 1 else (255, 0, 0)
                        cv2.rectangle(img_raw, (x1, y1), (x2, y2), c, 2)

                if has_target:
                    cv2.line(img_raw, (int(center), int(center)),
                            (int(center + best_dx), int(center + best_dy)),
                            (0, 255, 255), 2)

                cv2.imshow("Radar", img_raw)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    on_closing()
                    break

    except Exception as e:
        print(f"程序发生错误: {e}")
        traceback.print_exc()
        try:
            on_closing()
        except:
            pass


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
        input("按回车键退出...")
