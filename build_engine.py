#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
用 TensorRT 10.x 生成 engine 文件
"""

import os
import sys
import argparse

def build_engine(onnx_path, engine_path, fp16=True, workspace=512):
    """
    从 ONNX 构建 TensorRT engine
    """
    try:
        import tensorrt_bindings as trt
    except ImportError:
        import tensorrt as trt

    print(f">>> Building TensorRT engine from: {onnx_path}")
    print(f">>> Output: {engine_path}")
    print(f">>> FP16: {fp16}")

    # 创建 logger
    logger = trt.Logger(trt.Logger.INFO)

    # 创建 builder
    builder = trt.Builder(logger)

    # 创建 network
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )

    # 创建 parser
    parser = trt.OnnxParser(network, logger)

    # 解析 ONNX
    with open(onnx_path, 'rb') as f:
        onnx_data = f.read()

    if not parser.parse(onnx_data):
        print(">>> ERROR: ONNX parse failed!")
        for i in range(parser.num_errors):
            print(f"  Error {i}: {parser.get_error(i)}")
        return False

    print(f">>> ONNX parsed successfully")
    print(f">>> Network inputs: {network.num_inputs}")
    print(f">>> Network outputs: {network.num_outputs}")

    # 创建 builder config
    config = builder.create_builder_config()

    # TensorRT 10.x 使用不同的 API
    try:
        # 尝试 10.x API
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace * (1 << 20))
    except:
        # 回退到 8.x API
        config.max_workspace_size = workspace * (1 << 20)

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print(">>> Enabled FP16 mode")

    # 构建 engine
    print(">>> Building engine (this may take a while)...")

    # TensorRT 10.x 使用 build_serialized_network
    try:
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            print(">>> ERROR: Engine build failed!")
            return False
    except AttributeError:
        # 回退到 8.x API
        engine = builder.build_engine(network, config)
        if engine is None:
            print(">>> ERROR: Engine build failed!")
            return False
        serialized_engine = engine.serialize()

    # 保存 engine
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)

    print(f">>> Engine saved to: {engine_path}")
    print(f">>> Engine size: {os.path.getsize(engine_path) / (1024*1024):.2f} MB")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build TensorRT engine from ONNX')
    parser.add_argument('--onnx', default='1225_2best.onnx', help='ONNX model path')
    parser.add_argument('--engine', default='1225_2best.engine', help='Output engine path')
    parser.add_argument('--fp16', action='store_true', default=True, help='Enable FP16')
    parser.add_argument('--workspace', type=int, default=512, help='Workspace size in MB')

    args = parser.parse_args()

    success = build_engine(args.onnx, args.engine, args.fp16, args.workspace)
    sys.exit(0 if success else 1)
