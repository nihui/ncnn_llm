"""Private helpers shared by the production dynamic vision exporter."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Sequence

from _subprocess_utils import run_utf8
from youtu_vl_reference import die


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_NPZ = ROOT / "assets" / "youtu_vl_test" / "vision_embeds_smoke" / "coco_cats_describe.npz"
DEFAULT_OUTPUT_DIR = ROOT / "assets" / "youtu_vl_export"


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == ROOT.name:
        return (ROOT.parent / path).resolve()
    return (ROOT / path).resolve()


def tensor_shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(x) for x in shape]


def pnnx_dtype_name(torch, tensor) -> str:
    dtype = tensor.dtype
    if dtype is torch.float32:
        return "f32"
    if dtype is torch.float64:
        return "f64"
    if dtype is torch.float16:
        return "f16"
    if dtype is torch.bfloat16:
        return "bf16"
    if dtype is torch.int64 or dtype is torch.long:
        return "i64"
    if dtype is torch.int32 or dtype is torch.int:
        return "i32"
    if dtype is torch.bool:
        return "bool"
    die(f"unsupported pnnx input dtype: {dtype}")


def _shape_arg(shape: Sequence[int], dtype_name: str) -> str:
    return "[" + ",".join(str(int(x)) for x in shape) + "]" + dtype_name


def resolve_pnnx_path(pnnx_arg: str) -> Path | None:
    if pnnx_arg:
        path = resolve_path(pnnx_arg)
        if not path.exists():
            die(f"pnnx not found: {path}")
        return path
    found = shutil.which("pnnx")
    if found:
        return Path(found).resolve()
    venv_pnnx = ROOT / "venv" / "bin" / "pnnx"
    return venv_pnnx if venv_pnnx.exists() else None


def run_pnnx(*, pnnx_path: Path, torchscript_path: Path, example_inputs,
             output_prefix: Path, extra_args: Sequence[str], torch, fp16: bool = False) -> dict[str, Any]:
    inputshape = ",".join(
        _shape_arg(tensor_shape(tensor) or [], pnnx_dtype_name(torch, tensor))
        for tensor in example_inputs
    )
    cmd = [
        str(pnnx_path),
        str(torchscript_path),
        f"inputshape={inputshape}",
        f"pnnxparam={output_prefix}.pnnx.param",
        f"pnnxbin={output_prefix}.pnnx.bin",
        f"pnnxpy={output_prefix}_pnnx.py",
        f"pnnxonnx={output_prefix}.pnnx.onnx",
        f"ncnnparam={output_prefix}.ncnn.param",
        f"ncnnbin={output_prefix}.ncnn.bin",
        f"ncnnpy={output_prefix}_ncnn.py",
        f"fp16={1 if fp16 else 0}",
    ]
    cmd.extend(extra_args)
    print("[pnnx] " + " ".join(cmd))
    result = run_utf8(cmd, capture_output=True, check=False)
    produced = [
        str(path)
        for path in [
            Path(f"{output_prefix}.pnnx.param"),
            Path(f"{output_prefix}.pnnx.bin"),
            Path(f"{output_prefix}_pnnx.py"),
            Path(f"{output_prefix}.pnnx.onnx"),
            Path(f"{output_prefix}.ncnn.param"),
            Path(f"{output_prefix}.ncnn.bin"),
            Path(f"{output_prefix}_ncnn.py"),
        ]
        if path.exists()
    ]
    return {
        "command": cmd,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "produced": produced,
        "passed": result.returncode == 0,
    }


def diff_stats(np, actual, expected) -> dict[str, float]:
    if actual.shape != expected.shape:
        die(f"shape mismatch actual={actual.shape} expected={expected.shape}")
    diff = np.abs(actual.astype("float32") - expected.astype("float32"))
    return {
        "max_abs": float(diff.max()) if diff.size else 0.0,
        "mean_abs": float(diff.mean()) if diff.size else 0.0,
        "rms": float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0,
    }
