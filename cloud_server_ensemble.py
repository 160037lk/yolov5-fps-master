#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Model Ensemble Inference Server
=====================================
加载全部 8 个 ONNX 模型，多尺度/多版本集成推理，NMS 融合。

模型分组:
  Dawan v11s (7类): 256, 320, 416, 640 — 同一架构多尺度
  V5 (4类):       pro(320), std(256)
  V8 (4类):       pro(320), std(256)

融合策略: 所有模型 → 坐标统一 → 全量 NMS (同目标取置信度最高)

协议:
  Client → Server: [4B: jpeg_size uint32 LE] [jpeg_size B: JPEG]
  Server → Client: [1B: num_detections uint8] [N × 24B: float32 × 6]
    det 格式: [x1, y1, x2, y2, conf, cls]

用法:
  python cloud_server_ensemble.py --port 9999
  python cloud_server_ensemble.py --port 9999 --models dawan --conf 0.3
"""

import argparse
import io
import os
import socket
import struct
import sys
import time
from collections import deque

import cv2
import numpy as np
import onnxruntime as ort


# ═══════════════════════════════════════════════════════════════
#  模型输出解码器
# ═══════════════════════════════════════════════════════════════

def nms(dets, iou_thres=0.5):
    """纯 numpy NMS，按置信度排序，同目标保留最高分。"""
    if len(dets) == 0:
        return np.empty((0, 6), dtype=np.float32)

    x1 = dets[:, 0]; y1 = dets[:, 1]; x2 = dets[:, 2]; y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]

    return dets[keep]


def decode_v8_dawan(raw, input_size, orig_h, orig_w, conf_thres):
    """
    解码 YOLOv8 / Dawan v11s 格式输出。
    输出: [1, 4+C, N]  →  ch0-3: cx,cy,w,h (像素)  ch4+: 类别分数 (已 sigmoid)
    """
    raw = raw[0]  # (4+C, N)
    cx = raw[0]; cy = raw[1]; w = raw[2]; h = raw[3]
    scores = raw[4:]  # (C, N)

    cls_ids = scores.argmax(axis=0)
    confs = scores.max(axis=0)

    mask = confs > conf_thres
    if mask.sum() == 0:
        return np.empty((0, 6), dtype=np.float32)

    cx = cx[mask]; cy = cy[mask]
    w = w[mask]; h = h[mask]
    confs = confs[mask]
    cls_ids = cls_ids[mask]

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    # 缩放到原始图像坐标 (输入尺寸 → 原图)
    scale_x = orig_w / input_size
    scale_y = orig_h / input_size
    x1 *= scale_x; y1 *= scale_y
    x2 *= scale_x; y2 *= scale_y

    dets = np.stack([x1, y1, x2, y2, confs, cls_ids.astype(np.float32)], axis=1)
    return dets


def decode_v5(raw, input_size, orig_h, orig_w, conf_thres):
    """
    解码 YOLOv5 格式输出。
    输出: [1, N, 9] → ch0-3: cx,cy,w,h (像素) ch4: obj (sigmoid) ch5-8: cls (sigmoid)
    置信度 = obj * max(cls_scores)
    """
    raw = raw[0]  # (N, 9)
    cx = raw[:, 0]; cy = raw[:, 1]; w = raw[:, 2]; h = raw[:, 3]
    obj = raw[:, 4]
    cls_scores = raw[:, 5:9]  # (N, 4)

    cls_max = cls_scores.max(axis=1)
    cls_ids = cls_scores.argmax(axis=1)
    confs = obj * cls_max

    mask = confs > conf_thres
    if mask.sum() == 0:
        return np.empty((0, 6), dtype=np.float32)

    cx = cx[mask]; cy = cy[mask]
    w = w[mask]; h = h[mask]
    confs = confs[mask]
    cls_ids = cls_ids[mask]

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    scale_x = orig_w / input_size
    scale_y = orig_h / input_size
    x1 *= scale_x; y1 *= scale_y
    x2 *= scale_x; y2 *= scale_y

    dets = np.stack([x1, y1, x2, y2, confs, cls_ids.astype(np.float32)], axis=1)
    return dets


# ═══════════════════════════════════════════════════════════════
#  模型包装
# ═══════════════════════════════════════════════════════════════

class ONNXModel:
    """单个 ONNX 模型包装器。"""

    def __init__(self, path, input_size, model_type, device="cuda"):
        self.path = path
        self.input_size = input_size
        self.model_type = model_type  # 'v8', 'dawan', 'v5'
        self.name = os.path.basename(path)

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(path, sess_options=sess_opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.device = self.session.get_providers()[0]
        self.warmup()

    def warmup(self, n=3):
        c, h, w = 3, self.input_size, self.input_size
        dummy = np.random.randn(1, c, h, w).astype(np.float32)
        for _ in range(n):
            self.session.run(None, {self.input_name: dummy})

    def preprocess(self, bgr):
        """BGR 图像 → CHW float32 张量，resize + 归一化。"""
        img = cv2.resize(bgr, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # HWC → CHW
        return np.expand_dims(img, 0)

    def infer(self, bgr, orig_h, orig_w, conf_thres=0.3):
        """完整推理流水线: 预处理 → 推理 → 后处理。"""
        inp = self.preprocess(bgr)
        raw = self.session.run(None, {self.input_name: inp})

        if self.model_type in ('v8', 'dawan'):
            dets = decode_v8_dawan(raw[0], self.input_size, orig_h, orig_w, conf_thres)
        elif self.model_type == 'v5':
            dets = decode_v5(raw[0], self.input_size, orig_h, orig_w, conf_thres)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        # 每模型内部先做一次 NMS
        dets = nms(dets, iou_thres=0.6)
        return dets


# ═══════════════════════════════════════════════════════════════
#  集成推理引擎
# ═══════════════════════════════════════════════════════════════

class EnsembleEngine:
    """加载全部模型，并行推理后 NMS 融合。"""

    def __init__(self, configs, device="cuda", conf_thres=0.3, iou_thres=0.5):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.models = []
        self.fps_history = deque(maxlen=50)

        print(f"\n{'='*60}")
        print(f"Multi-Model Ensemble Engine")
        print(f"  Device: {device.upper()}")
        print(f"  Conf threshold: {conf_thres}")
        print(f"  IoU threshold: {iou_thres}")
        print(f"{'='*60}")

        for cfg in configs:
            path = cfg['path']
            if not os.path.exists(path):
                print(f"  [SKIP] Not found: {path}")
                continue
            try:
                m = ONNXModel(path, cfg['size'], cfg['type'], device)
                self.models.append(m)
                print(f"  [LOADED] {cfg['type']:8s} {cfg['size']}x{cfg['size']:4d}  "
                      f"on {m.device}  ({m.name})")
            except Exception as e:
                print(f"  [FAILED] {os.path.basename(path)}: {e}")

        print(f"\n  Total: {len(self.models)}/{len(configs)} models loaded\n")

    def infer(self, bgr):
        """对单帧运行所有模型，融合结果。"""
        h, w = bgr.shape[:2]
        all_dets = []

        for m in self.models:
            t0 = time.perf_counter()
            dets = m.infer(bgr, h, w, self.conf_thres)
            t = (time.perf_counter() - t0) * 1000
            self.fps_history.append(t)
            if len(dets) > 0:
                all_dets.append(dets)

        if not all_dets:
            return np.empty((0, 6), dtype=np.float32)

        pooled = np.concatenate(all_dets, axis=0)
        # 跨模型 NMS: 同目标保留置信度最高的那个模型的检测结果
        fused = nms(pooled, iou_thres=self.iou_thres)
        return fused

    def stats(self):
        if not self.fps_history:
            return 0, 0
        times = np.array(self.fps_history)
        return times.mean(), times.sum()


# ═══════════════════════════════════════════════════════════════
#  TCP 服务器
# ═══════════════════════════════════════════════════════════════

class TCPServer:
    """二进制 TCP 推理服务。"""

    def __init__(self, engine, port=9999):
        self.engine = engine
        self.port = port
        self.sock = None

    def start(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('0.0.0.0', self.port))
        self.sock.listen(1)
        print(f"[Server] Listening on 0.0.0.0:{self.port}")

    def serve_forever(self):
        self.start()
        frame_count = 0

        while True:
            print(f"[Server] Waiting for connection...")
            conn, addr = self.sock.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"[Server] Connected from {addr[0]}:{addr[1]}")

            try:
                while True:
                    # 读取: [4B size][JPEG data]
                    header = self._recv_exact(conn, 4)
                    if header is None:
                        break
                    jpeg_size = struct.unpack('<I', header)[0]
                    if jpeg_size == 0 or jpeg_size > 10 * 1024 * 1024:
                        continue
                    jpeg_data = self._recv_exact(conn, jpeg_size)
                    if jpeg_data is None:
                        break

                    # JPEG 解码
                    frame = cv2.imdecode(np.frombuffer(jpeg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        # 返回空结果
                        conn.sendall(b'\x00')
                        continue

                    # 集成推理
                    dets = self.engine.infer(frame)
                    num = min(len(dets), 255)
                    # 按置信度降序排列
                    if num > 0:
                        dets = dets[dets[:, 4].argsort()[::-1]]
                        dets_top = dets[:num]
                    else:
                        dets_top = np.empty((0, 6), dtype=np.float32)

                    # 返回: [1B count][N × 24B float32×6]
                    payload = dets_top.astype(np.float32).tobytes()
                    conn.sendall(struct.pack('B', num) + payload)

                    frame_count += 1
                    if frame_count % 30 == 0:
                        avg, total = self.engine.stats()
                        print(f"\r[Frame {frame_count}] "
                              f"Models: {len(self.engine.models)} | "
                              f"Detections: {num} | "
                              f"Avg infer: {avg:.1f}ms | "
                              f"Total: {total:.1f}ms   ", end="", flush=True)

            except (ConnectionResetError, ConnectionError, OSError):
                pass
            finally:
                conn.close()
                print(f"\n[Server] Client disconnected")

    @staticmethod
    def _recv_exact(conn, n):
        buf = bytearray()
        while len(buf) < n:
            try:
                chunk = conn.recv(n - len(buf))
            except socket.timeout:
                return None
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)


# ═══════════════════════════════════════════════════════════════
#  模型配置
# ═══════════════════════════════════════════════════════════════

def discover_models(dawan_dir, obs_dir):
    """从目录自动发现模型，返回配置列表。"""
    configs = []

    # Dawan v11s 系列 (7 类, 同一架构多尺度) — 支持两种命名
    dawan_names = [
        ("Dawan_0121_v11s_256.onnx", 256),
        ("Dawan_0121_v11s_320.onnx", 320),
        ("Dawan_0121_v11s_416.onnx", 416),
        ("Dawan_0121_v11s_640.onnx", 640),
        # 云端可能用短命名
        ("dawan_v11s_256.onnx", 256),
        ("dawan_v11s_320.onnx", 320),
        ("dawan_v11s_416.onnx", 416),
        ("dawan_v11s_640.onnx", 640),
    ]
    seen = set()
    for fname, size in dawan_names:
        path = os.path.join(dawan_dir, fname)
        key = f"{size}"
        if os.path.exists(path) and key not in seen:
            configs.append({"path": path, "size": size, "type": "dawan"})
            seen.add(key)

    # V5/V8 系列 (4 类) — 在 obs_dir 子目录或当前目录
    for subdir in [obs_dir, obs_dir + "/obs_models", dawan_dir]:
        for vname, vsize, vtype in [
            ("V5-pro.onnx", 320, "v5"),
            ("V5-std.onnx", 256, "v5"),
            ("V8-pro.onnx", 320, "v8"),
            ("V8-std.onnx", 256, "v8"),
        ]:
            path = os.path.join(subdir, vname)
            if os.path.exists(path):
                configs.append({"path": path, "size": vsize, "type": vtype})

    return configs


def get_model_preset(preset, dawan_dir, obs_dir):
    """根据预设筛选模型。"""
    all_models = discover_models(dawan_dir, obs_dir)
    if preset == "all":
        return all_models
    elif preset == "dawan":
        return [m for m in all_models if m["type"] == "dawan"]
    elif preset == "dawan_v8":
        return [m for m in all_models if m["type"] in ("dawan", "v8")]
    return all_models


def main():
    parser = argparse.ArgumentParser(description="Multi-Model Ensemble Inference Server")
    parser.add_argument("--port", type=int, default=9999, help="TCP port")
    parser.add_argument("--models", choices=["all", "dawan", "dawan_v8"], default="dawan")
    parser.add_argument("--dawan-dir", default=os.getcwd(),
                        help="Directory with Dawan model files")
    parser.add_argument("--obs-dir", default=os.getcwd(),
                        help="Directory with V5/V8 model files")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for NMS")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--no-server", action="store_true",
                        help="Just load models and exit (dry-run)")
    args = parser.parse_args()

    # 自动发现 + 预设筛选
    configs = get_model_preset(args.models, args.dawan_dir, args.obs_dir)

    # 初始化引擎
    engine = EnsembleEngine(configs, device=args.device,
                            conf_thres=args.conf, iou_thres=args.iou)

    if len(engine.models) == 0:
        print("[FATAL] No models loaded. Check paths.")
        return 1

    if args.no_server:
        print("[Dry-run] Models loaded successfully. Exiting.")
        return 0

    # 启动服务
    server = TCPServer(engine, port=args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Server] Shutting down.")
    finally:
        if server.sock:
            server.sock.close()


if __name__ == "__main__":
    main()
