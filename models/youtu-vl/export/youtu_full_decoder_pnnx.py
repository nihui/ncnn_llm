#!/usr/bin/env python3
"""Convert an already traced 40-layer Youtu decoder without loading PyTorch weights.

Keeping pnnx in a separate process is important on 24 GB machines: the traced
graph is about 17 GB and conversion peaks near 24 GB by itself.  Loading the
4B checkpoint in the same process makes a reproducible conversion impossible.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


STEM = "youtu_decoder_full_manual_kv_sdpa_ncnn"


def shape_arg(shape: tuple[int, ...], dtype: str = "f32") -> str:
    return "[" + ",".join(str(value) for value in shape) + "]" + dtype


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--torchscript", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pnnx", required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument("--cache-length", type=int, default=33)
    parser.add_argument("--layers", type=int, default=40)
    args = parser.parse_args()

    torchscript = Path(args.torchscript).resolve()
    output_dir = Path(args.output_dir).resolve()
    pnnx = Path(args.pnnx).resolve()
    if not torchscript.is_file():
        raise SystemExit(f"TorchScript not found: {torchscript}")
    if not pnnx.is_file():
        raise SystemExit(f"pnnx not found: {pnnx}")
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / STEM

    shapes = [
        shape_arg((1, 1, 2560)),
        shape_arg((1, args.cache_length + 1)),
        shape_arg((1, 1, 64)),
        shape_arg((1, 1, 64)),
    ]
    for _ in range(args.layers):
        shapes.append(shape_arg((32, args.cache_length, 192)))
        shapes.append(shape_arg((32, args.cache_length, 128)))

    command = [
        str(pnnx),
        str(torchscript),
        "inputshape=" + ",".join(shapes),
        f"pnnxparam={prefix}.pnnx.param",
        f"pnnxbin={prefix}.pnnx.bin",
        f"pnnxpy={prefix}_pnnx.py",
        f"pnnxonnx={prefix}.pnnx.onnx",
        f"ncnnparam={prefix}.ncnn.param",
        f"ncnnbin={prefix}.ncnn.bin",
        f"ncnnpy={prefix}_ncnn.py",
        f"fp16={1 if args.precision == 'fp16' else 0}",
    ]
    print("[pnnx] " + " ".join(command), flush=True)
    subprocess.run(command, check=True)

    raw_param = Path(f"{prefix}.ncnn.param")
    ncnn_bin = Path(f"{prefix}.ncnn.bin")
    patched_param = Path(f"{prefix}.ncnn.patched.param")
    if not raw_param.is_file() or not ncnn_bin.is_file() or ncnn_bin.stat().st_size == 0:
        raise SystemExit("pnnx returned success without complete ncnn artifacts")

    patcher = Path(__file__).resolve().parent / "patch_ncnn_param.py"
    patch_command = [
        sys.executable,
        str(patcher),
        "--fix-cache-concat-axis-3d",
        "--rename-full-decoder-io",
        "--fuse-sdpa-kv-cache",
        "--merge-kv-cache-input",
        "--reshape-all-binary-adds",
        str(raw_param),
        str(patched_param),
    ]
    print("[patch] " + " ".join(patch_command), flush=True)
    subprocess.run(patch_command, check=True)

    patched_text = patched_param.read_text(encoding="utf-8")
    residual = [name for name in ("pnnx.", "Tensor.", "aten::") if name in patched_text]
    if residual:
        raise SystemExit(f"unsupported residual operators in {patched_param}: {residual}")
    print(f"[ok] param={patched_param} bin={ncnn_bin} bytes={ncnn_bin.stat().st_size}")


if __name__ == "__main__":
    main()
