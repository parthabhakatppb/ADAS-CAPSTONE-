"""
Export Hybrid ADAS Model to ONNX and Build TensorRT Engines (FP32, FP16, INT8).
PyTorch 2.x+ compatible, supports dynamic shapes and trtexec build pipeline.

Usage:
    # 1. Export PyTorch checkpoint to ONNX
    python scripts/export_onnx_tensorrt.py \
        --weights runs/phase1_hybrid_s/best.pt \
        --variant hybrid_s \
        --output runs/phase1_hybrid_s/hybrid_s.onnx

    # 2. Generate trtexec build commands for FP32, FP16, and INT8 engines
    python scripts/export_onnx_tensorrt.py \
        --onnx runs/phase1_hybrid_s/hybrid_s.onnx \
        --build_trt
"""
import argparse
import os
from pathlib import Path
import time

import torch
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.hybrid_model import build_model_variant
from configs.phase1_classes import NUM_PHASE1_CLASSES


def export_onnx(
    model,
    output_path,
    img_size=640,
    batch_size=1,
    dynamic=False,
    opset_version=17,
):
    """
    Exports Hybrid ADAS model to ONNX.
    """
    model.eval()
    dummy_input = torch.randn(batch_size, 3, img_size, img_size)

    input_names = ["images"]
    output_names = ["cls_logits", "boxes", "anchors", "strides"]

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting PyTorch model to ONNX: {output_path} (opset {opset_version})...")

    if dynamic:
        dynamic_axes = {
            "images": {0: "batch", 2: "height", 3: "width"},
            "cls_logits": {0: "batch", 1: "num_anchors"},
            "boxes": {0: "batch", 1: "num_anchors"},
        }
    else:
        dynamic_axes = None

    try:
        # PyTorch export with dynamo fallback for PyTorch 2.0+
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
        print(f"ONNX export successful: {output_path} ({Path(output_path).stat().st_size / 1e6:.2f} MB)")
    except Exception as e:
        print(f"Standard ONNX export encountered error: {e}")
        print("Retrying with legacy exporter flags...")
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
        )
        print(f"Legacy ONNX export successful: {output_path}")

    return output_path


def generate_tensorrt_scripts(onnx_path, img_size=640, output_dir=None):
    """
    Generates commands and calibration instructions for NVIDIA TensorRT (trtexec / ModelOpt).
    """
    onnx_file = Path(onnx_path)
    base_name = onnx_file.stem
    out_dir = Path(output_dir or onnx_file.parent)

    engine_fp32 = out_dir / f"{base_name}_fp32.engine"
    engine_fp16 = out_dir / f"{base_name}_fp16.engine"
    engine_int8 = out_dir / f"{base_name}_int8.engine"
    calib_cache = out_dir / f"{base_name}_int8_calib.cache"

    print("\n" + "=" * 70)
    print("NVIDIA TensorRT Deployment Commands (trtexec / ModelOpt)")
    print("=" * 70)

    print(f"\n1. Build TensorRT FP32 Engine:")
    print(f"   trtexec --onnx={onnx_file} --saveEngine={engine_fp32} --memPoolSize=workspace:2048")

    print(f"\n2. Build TensorRT FP16 Engine (Target 20-30+ FPS):")
    print(f"   trtexec --onnx={onnx_file} --saveEngine={engine_fp16} --fp16 --memPoolSize=workspace:2048")

    print(f"\n3. Build TensorRT INT8 Engine (with Calibration Cache):")
    print(f"   trtexec --onnx={onnx_file} --saveEngine={engine_int8} --int8 --calib={calib_cache} --memPoolSize=workspace:2048")

    print(f"\n4. Hardware-Synchronized Latency & FPS Benchmark:")
    print(f"   trtexec --loadEngine={engine_fp16} --warmUp=200 --duration=10 --useCudaGraph")
    print("=" * 70 + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None, help="Path to PyTorch checkpoint .pt")
    ap.add_argument("--variant", default="hybrid_s", choices=["cnn_baseline", "hybrid_n", "hybrid_s", "hybrid_m"])
    ap.add_argument("--num_classes", type=int, default=NUM_PHASE1_CLASSES)
    ap.add_argument("--img_size", type=int, default=640)
    ap.add_argument("--output", default="runs/phase1_hybrid_s/hybrid_s.onnx")
    ap.add_argument("--onnx", default=None, help="Existing ONNX file for TensorRT instructions")
    ap.add_argument("--dynamic", action="store_true", help="Enable dynamic batch and spatial shapes")
    ap.add_argument("--build_trt", action="store_true", help="Print TensorRT commands")
    args = ap.parse_args()

    device = "cpu"
    if args.onnx:
        generate_tensorrt_scripts(args.onnx, img_size=args.img_size)
        return

    # Build model
    model = build_model_variant(
        args.variant,
        num_classes=args.num_classes,
        use_drivable_head=False,
        use_weather_head=False,
    ).to(device)

    if args.weights and Path(args.weights).exists():
        ckpt = torch.load(args.weights, map_location=device)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state_dict)
        print(f"Loaded weights from {args.weights}")

    onnx_path = export_onnx(
        model,
        args.output,
        img_size=args.img_size,
        batch_size=1,
        dynamic=args.dynamic,
    )

    generate_tensorrt_scripts(onnx_path, img_size=args.img_size)


if __name__ == "__main__":
    main()
