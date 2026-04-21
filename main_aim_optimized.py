#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOv5 FPS 自动瞄准系统 - 优化版 v2.0
目标：RTX 4060 上 60+ FPS
优化点：TensorRT优先、GPU预处理、CUDA NMS、异步捕获
"""

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import numpy as np
import time
import os
import sys
import ctypes
import cv2
import win32api
import mss
from threading import Thread, Queue
from logitech import Logitech

# ================== ⚡ 参数设置区 ⚡ ==================
DETECTION_SIZE = 320          # 检测尺寸
CONF_THRES = 0.5              # 置信度阈值
IOU_THRES = 0.45              # IoU 阈值
MAX_DETECTIONS = 20           # 最大检测数量
AIM_FOV_RADIUS = 300          # 自瞄准 FOV 半径

# 性能参数
MAX_QUEUE_SIZE = 2            # 帧队列大小（防止堆积）
USE_CUDA_NMS = True           # 使用 CUDA NMS
WARMUP_FRAMES = 10            # 预热帧数
# =====================================================

current_target_type = 1       # 0=身体, 1=头部

# 按键表
KEYS = {
    'TOGGLE_WIN': 0xDC,      # \ 键
    'TOGGLE_AIM': 0xDD,      # ] 键
    'TOGGLE_TARGET': 0xDB,   # [ 键
    'QUIT': 0x51             # q 键
}


class TensorRTInference:
    """TensorRT 推理封装 - 优化版"""
    
    def __init__(self, engine_path, device_id=0):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # 加载引擎
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        # 获取输入输出绑定
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        
        # 获取输入尺寸
        self.input_shape = self.engine.get_tensor_shape(self.input_name)
        self.batch_size = self.input_shape[0]
        self.input_h = self.input_shape[2]
        self.input_w = self.input_shape[3]
        
        # 获取输出尺寸
        self.output_shape = self.engine.get_tensor_shape(self.output_name)
        
        # 分配 GPU 内存
        self.d_input = cuda.mem_alloc(self.batch_size * 3 * self.input_h * self.input_w * np.dtype(np.float16).itemsize)
        self.d_output = cuda.mem_alloc(self.batch_size * self.output_shape[1] * self.output_shape[2] * np.dtype(np.float16).itemsize)
        
        # 绑定张量
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))
        
        # 预分配页锁定内存（加速 CPU-GPU 传输）
        self.h_input = cuda.pagelocked_empty((self.batch_size, 3, self.input_h, self.input_w), dtype=np.float16)
        self.h_output = cuda.pagelocked_empty((self.batch_size, self.output_shape[1], self.output_shape[2]), dtype=np.float16)
        
        print(f">>> [TensorRT] Initialized: {self.input_h}x{self.input_w}, Batch={self.batch_size}")
    
    def infer(self, img_np):
        """单帧推理"""
        import pycuda.driver as cuda
        
        # 预处理在 GPU 上完成，这里直接拷贝
        cuda.memcpy_htod_async(self.d_input, img_np, self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()
        
        return self.h_output.copy()
    
    def warmup(self, num_warmup=10):
        """预热"""
        dummy = np.zeros((self.batch_size, 3, self.input_h, self.input_w), dtype=np.float16)
        for _ in range(num_warmup):
            self.infer(dummy)
        print(f">>> [TensorRT] Warmup complete ({num_warmup} frames)")


class GPUPreprocessor:
    """GPU 预处理 - BGR->RGB, Normalize, Permute"""
    
    def __init__(self, device='cuda:0'):
        self.device = torch.device(device)
        # 预分配 GPU 缓冲区
        self.gpu_buffer = torch.empty((320, 320, 3), dtype=torch.uint8, device=self.device)
        self.output_buffer = torch.empty((1, 3, 320, 320), dtype=torch.float16, device=self.device)
    
    def preprocess(self, img_bgr, target_size=320):
        """
        输入: BGR numpy array (H, W, 3) uint8
        输出: CHW float16 tensor on GPU (1, 3, 320, 320)
        """
        # 上传到 GPU
        img_tensor = torch.from_numpy(img_bgr).to(self.device, non_blocking=True)
        
        # BGR -> RGB (使用索引，比 cvtColor 快)
        img_rgb = img_tensor[:, :, [2, 1, 0]]
        
        # Resize (如果尺寸不对)
        if img_rgb.shape[0] != target_size or img_rgb.shape[1] != target_size:
            img_rgb = img_rgb.permute(2, 0, 1).unsqueeze(0).float()
            img_rgb = torch.nn.functional.interpolate(img_rgb, size=(target_size, target_size), mode='bilinear', align_corners=False)
            img_rgb = img_rgb.squeeze(0).permute(1, 2, 0).to(torch.uint8)
        
        # Normalize: /255.0
        img_norm = img_rgb.float() / 255.0
        
        # HWC -> CHW
        img_chw = img_norm.permute(2, 0, 1)
        
        # Add batch dimension and convert to float16
        img_batched = img_chw.unsqueeze(0).half()
        
        return img_batched.cpu().numpy()  # 返回 numpy 供 TensorRT 使用


class FastAimController:
    """快速瞄准控制器"""
    
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


def cuda_nms(detections, conf_thres=0.5, iou_thres=0.45, max_det=20):
    """
    CUDA NMS using torchvision
    detections: [N, 6] tensor [x1, y1, x2, y2, conf, cls]
    """
    try:
        from torchvision.ops import nms
        
        if len(detections) == 0:
            return []
        
        # 置信度过滤
        mask = detections[:, 4] > conf_thres
        detections = detections[mask]
        
        if len(detections) == 0:
            return []
        
        # 分离坐标和分数
        boxes = detections[:, :4]
        scores = detections[:, 4]
        
        # CUDA NMS
        keep = nms(boxes, scores, iou_thres)
        
        # 限制数量
        if len(keep) > max_det:
            keep = keep[:max_det]
        
        return detections[keep]
    except Exception as e:
        print(f">>> [CUDA NMS Error] {e}, falling back to CPU")
        return cpu_nms(detections, conf_thres, iou_thres, max_det)


def cpu_nms(detections, conf_thres=0.5, iou_thres=0.45, max_det=20):
    """CPU NMS fallback"""
    import torchvision
    if len(detections) == 0:
        return []
    
    mask = detections[:, 4] > conf_thres
    detections = detections[mask]
    
    if len(detections) == 0:
        return []
