#!/usr/bin/env python3
"""Prepare Youtu-VL text subnets for pnnx/ncnn conversion.

This is the P2D entry script.  It deliberately starts with the stable text
boundaries proven by tools/test_youtu_text_wrapper_parity.py:

  input_ids -> model.model.embed_tokens
  hidden_states -> model.lm_head

The full decoder path is recorded in the export manifest, but is not traced by
default because YoutuMLAttention, RoPE and KV cache need a dedicated wrapper
before pnnx conversion is meaningful.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = ROOT / "assets" / "youtu_text_test" / "dumps" / "metadata.json"
DEFAULT_OUTPUT_DIR = ROOT / "assets" / "youtu_text_export"


def die(message: str, code: int = 2) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def import_runtime_deps():
    try:
        import numpy as np
        import torch
        import transformers
        from transformers import AutoModelForCausalLM
        from safetensors import safe_open
    except ModuleNotFoundError as exc:
        missing = exc.name or "a required module"
        die(
            "missing Python dependency "
            f"`{missing}`. Activate the project venv or install runtime deps:\n"
            "  pip install -U torch transformers accelerate numpy safetensors",
            code=3,
        )
    return np, torch, transformers, AutoModelForCausalLM, safe_open


def install_pydensecrf_import_stub() -> bool:
    try:
        import pydensecrf.densecrf  # type: ignore  # noqa: F401
        import pydensecrf.utils  # type: ignore  # noqa: F401
        return False
    except ModuleNotFoundError:
        pass

    def unavailable(*_args, **_kwargs):
        raise RuntimeError(
            "pydensecrf is not installed. It is not needed for text export, "
            "but is required for dense CRF post-processing."
        )

    package = types.ModuleType("pydensecrf")
    densecrf = types.ModuleType("pydensecrf.densecrf")
    utils = types.ModuleType("pydensecrf.utils")

    class DenseCRF2D:  # pragma: no cover - should not be used here
        def __init__(self, *_args, **_kwargs):
            unavailable()

    densecrf.DenseCRF2D = DenseCRF2D
    densecrf.DIAG_KERNEL = 0
    densecrf.NORMALIZE_SYMMETRIC = 0
    utils.unary_from_softmax = unavailable

    package.densecrf = densecrf
    package.utils = utils
    sys.modules.setdefault("pydensecrf", package)
    sys.modules.setdefault("pydensecrf.densecrf", densecrf)
    sys.modules.setdefault("pydensecrf.utils", utils)
    return True


def resolve_project_path(path_like: str | Path, base: Path = ROOT) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        die(f"file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def parse_dtype(torch, name: str):
    normalized = name.lower()
    if normalized == "auto":
        return "auto"
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32", "float"}:
        return torch.float32
    die(f"unsupported dtype: {name}")


def resolve_torch_device(torch, device_name: str):
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def first_parameter_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return getattr(model, "device", "cpu")


def tensor_shape(value: Any) -> Optional[List[int]]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(x) for x in shape]


def shape_arg(shape: Sequence[int], dtype_name: str = "f32") -> str:
    return "[" + ",".join(str(int(x)) for x in shape) + "]" + dtype_name


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
    if dtype is torch.int16:
        return "i16"
    if dtype is torch.int8:
        return "i8"
    if dtype is torch.uint8:
        return "u8"
    if dtype is torch.bool:
        return "bool"
    die(f"unsupported pnnx input dtype: {dtype}")


class EmbedTokensWrapper:
    """Created lazily after torch import to keep this file import-light."""


class LmHeadWrapper:
    """Created lazily after torch import to keep this file import-light."""


def make_wrappers(torch):
    class _EmbedTokensWrapper(torch.nn.Module):
        def __init__(self, embed_tokens):
            super().__init__()
            self.embed_tokens = embed_tokens

        def forward(self, input_ids):
            return self.embed_tokens(input_ids)

    class _LmHeadWrapper(torch.nn.Module):
        def __init__(self, lm_head):
            super().__init__()
            self.lm_head = lm_head

        def forward(self, hidden_states):
            return self.lm_head(hidden_states)

    return _EmbedTokensWrapper, _LmHeadWrapper


def make_minimal_modules(torch):
    class _TextBackbone(torch.nn.Module):
        def __init__(self, embed_tokens):
            super().__init__()
            self.embed_tokens = embed_tokens

    class _TiedLmHead(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = torch.nn.Parameter(weight, requires_grad=False)
            self.bias = None

        def forward(self, hidden_states):
            weight = self.weight.to(dtype=hidden_states.dtype)
            return torch.matmul(hidden_states, weight.t())

    class _MinimalYoutuTextModel(torch.nn.Module):
        def __init__(self, config, embed_weight):
            super().__init__()
            self.config = config
            self.model = _TextBackbone(torch.nn.Embedding.from_pretrained(embed_weight, freeze=True))
            self.lm_head = _TiedLmHead(self.model.embed_tokens.weight)

    return _MinimalYoutuTextModel


def select_sample(metadata: Mapping[str, Any], sample_id: str) -> Mapping[str, Any]:
    samples = metadata.get("samples", [])
    if not samples:
        die("metadata has no samples")
    if not sample_id:
        return samples[0]
    for sample in samples:
        if sample.get("sample_id") == sample_id:
            return sample
    die(f"sample not found in metadata: {sample_id}")


def resolve_npz_path(sample_meta: Mapping[str, Any], metadata_path: Path) -> Path:
    raw = sample_meta.get("npz_path")
    if not isinstance(raw, str) or not raw:
        die(f"sample {sample_meta.get('sample_id')} has no npz_path")
    path = Path(raw)
    if path.is_absolute():
        return path
    return (metadata_path.parent / path).resolve()


def resolve_model_arg(args, metadata: Mapping[str, Any]):
    model_arg = args.model or metadata.get("model", {}).get("path_or_id") or metadata.get("model", {}).get("hf_model_id")
    if not model_arg:
        die("no model path/id provided")
    return resolve_project_path(model_arg) if isinstance(model_arg, str) and not model_arg.startswith("tencent/") else model_arg


def load_config_namespace(model_path: Path) -> SimpleNamespace:
    config_path = model_path / "config.json"
    if not config_path.exists():
        die(f"config.json not found for safetensors loading: {config_path}")
    return SimpleNamespace(**load_json(config_path))


def find_safetensor_for_key(model_path: Path, key: str, safe_open) -> Path:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        index = load_json(index_path)
        shard_name = index.get("weight_map", {}).get(key)
        if shard_name:
            shard = model_path / shard_name
            if shard.exists():
                with safe_open(shard, framework="pt", device="cpu") as f:
                    if key in f.keys():
                        return shard

    for shard in sorted(model_path.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as f:
            if key in f.keys():
                return shard
    die(f"tensor `{key}` not found in safetensors under {model_path}")


def load_tensor_from_safetensors(model_path: Path, key: str, safe_open):
    shard = find_safetensor_for_key(model_path, key, safe_open)
    print(f"[load] tensor {key}: {shard}")
    with safe_open(shard, framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def load_minimal_text_model(args, metadata: Mapping[str, Any], targets: Sequence[str]):
    np, torch, transformers, AutoModelForCausalLM, safe_open = import_runtime_deps()
    model_path = resolve_model_arg(args, metadata)
    if isinstance(model_path, str):
        die("safetensors weight loading requires a local model directory, not a remote model id")
    config = load_config_namespace(model_path)
    embed_weight = load_tensor_from_safetensors(model_path, "model.embed_tokens.weight", safe_open)
    device = resolve_torch_device(torch, args.device)
    dtype = parse_dtype(torch, args.dtype)
    if dtype != "auto":
        embed_weight = embed_weight.to(dtype=dtype)
    elif "embed_tokens" in targets and embed_weight.dtype is torch.bfloat16:
        # pnnx 20260716 writes a truncated ncnn Embed payload when the traced
        # constant remains BF16 but fp16=1 is requested.  Trace this boundary
        # from FP32 and let pnnx perform the supported FP32 -> FP16 conversion.
        embed_weight = embed_weight.float()
    embed_weight = embed_weight.to(device)

    MinimalModel = make_minimal_modules(torch)
    model = MinimalModel(config, embed_weight).eval()
    print(f"[load] minimal text modules on {device}; targets={','.join(targets)}")
    return np, torch, transformers, model_path, model, device, False


def load_full_model(args, metadata: Mapping[str, Any]):
    np, torch, transformers, AutoModelForCausalLM, _safe_open = import_runtime_deps()
    pydensecrf_stubbed = install_pydensecrf_import_stub()

    model_path = resolve_model_arg(args, metadata)

    dtype = parse_dtype(torch, args.dtype)
    kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": dtype,
    }
    if args.attn_implementation != "none":
        kwargs["attn_implementation"] = args.attn_implementation
    if args.device_map != "none":
        kwargs["device_map"] = args.device_map

    print(f"[load] model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(str(model_path), **kwargs).eval()
    if args.device_map == "none":
        device = resolve_torch_device(torch, args.device)
        print(f"[load] moving model to {device}")
        model = model.to(device)
    else:
        device = first_parameter_device(model)

    return np, torch, transformers, model_path, model, device, pydensecrf_stubbed


def load_export_model(args, metadata: Mapping[str, Any], targets: Sequence[str]):
    if args.weight_loading == "full":
        return load_full_model(args, metadata)
    if args.weight_loading == "safetensors":
        return load_minimal_text_model(args, metadata, targets)

    if set(targets).issubset({"embed_tokens", "lm_head"}):
        return load_minimal_text_model(args, metadata, targets)
    return load_full_model(args, metadata)


def module_summary(model) -> Dict[str, Any]:
    cfg = model.config
    embed_weight = model.model.embed_tokens.weight
    lm_head_weight = model.lm_head.weight
    return {
        "model_class": type(model).__name__,
        "text_decoder_class": type(model.model).__name__,
        "embed_tokens_class": type(model.model.embed_tokens).__name__,
        "lm_head_class": type(model.lm_head).__name__,
        "embed_tokens_weight_shape": tensor_shape(embed_weight),
        "embed_tokens_weight_dtype": str(embed_weight.dtype),
        "lm_head_weight_shape": tensor_shape(lm_head_weight),
        "lm_head_weight_dtype": str(lm_head_weight.dtype),
        "lm_head_has_bias": getattr(model.lm_head, "bias", None) is not None,
        "weights_tied": embed_weight.data_ptr() == lm_head_weight.data_ptr(),
        "config": {
            "model_type": getattr(cfg, "model_type", None),
            "vocab_size": int(getattr(cfg, "vocab_size", 0)),
            "hidden_size": int(getattr(cfg, "hidden_size", 0)),
            "num_hidden_layers": int(getattr(cfg, "num_hidden_layers", 0)),
            "num_attention_heads": int(getattr(cfg, "num_attention_heads", 0)),
            "qk_head_dim": int(getattr(cfg, "qk_head_dim", 0)),
            "v_head_dim": int(getattr(cfg, "v_head_dim", 0)),
            "qk_rope_head_dim": int(getattr(cfg, "qk_rope_head_dim", 0)),
            "rope_theta": int(getattr(cfg, "rope_theta", 0)),
            "rope_interleave": bool(getattr(cfg, "rope_interleave", False)),
        },
        "boundaries": {
            "embed_tokens": {
                "status": "traceable",
                "input": "input_ids int64 [batch, seq]",
                "output": "inputs_embeds float [batch, seq, hidden_size]",
            },
            "lm_head": {
                "status": "traceable",
                "input": "hidden_states float [batch, seq, hidden_size]",
                "output": "logits float [batch, seq, vocab_size]",
            },
            "decoder": {
                "status": "planned",
                "reason": "YoutuMLAttention cache/RoPE wrapper must be fixed before pnnx conversion.",
                "input": "inputs_embeds + attention_mask + past_key_values",
                "output": "last_hidden_state + updated past_key_values",
            },
        },
    }


def trace_module(torch, module, example_inputs, output_path: Path, check_trace: bool) -> Dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    module.eval()
    with torch.inference_mode():
        traced = torch.jit.trace(module, example_inputs, check_trace=check_trace)
        traced.save(str(output_path))
        outputs = traced(*example_inputs)
    return {
        "torchscript": str(output_path),
        "example_input_shapes": [tensor_shape(x) for x in example_inputs],
        "example_input_dtypes": [str(x.dtype) for x in example_inputs],
        "example_input_pnnx_dtypes": [pnnx_dtype_name(torch, x) for x in example_inputs],
        "example_output_shape": tensor_shape(outputs),
        "example_output_dtype": str(outputs.dtype),
        "bytes": output_path.stat().st_size,
    }


def resolve_pnnx_path(pnnx_arg: str) -> Optional[Path]:
    if not pnnx_arg:
        found = shutil.which("pnnx")
        return Path(found).resolve() if found else None
    path = resolve_project_path(pnnx_arg)
    if not path.exists():
        die(f"pnnx executable not found: {path}")
    if not path.is_file():
        die(f"pnnx path is not a file: {path}")
    return path


def run_pnnx(
    *,
    pnnx_path: Path,
    torchscript_path: Path,
    input_shapes: Sequence[Sequence[int]],
    input_dtypes: Sequence[str],
    output_prefix: Path,
    fp16: bool,
    extra_args: Sequence[str],
) -> Dict[str, Any]:
    inputshape = ",".join(shape_arg(shape, dtype) for shape, dtype in zip(input_shapes, input_dtypes))
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
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
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
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "produced": produced,
        "passed": proc.returncode == 0,
    }


def parse_targets(values: Sequence[str]) -> List[str]:
    allowed = {"embed_tokens", "lm_head"}
    if not values:
        return ["embed_tokens", "lm_head"]

    targets: List[str] = []
    for value in values:
        for part in value.split(","):
            target = part.strip()
            if not target:
                continue
            if target == "all":
                return ["embed_tokens", "lm_head"]
            if target not in allowed:
                die(f"unsupported target `{target}`; allowed: {sorted(allowed)}")
            if target not in targets:
                targets.append(target)
    return targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Youtu-VL text-only pnnx/ncnn export artifacts."
    )
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA), help="Path to P2B metadata.json.")
    parser.add_argument("--model", default="", help="Local HF model dir or model id. Defaults to metadata model.path_or_id.")
    parser.add_argument("--output-dir", default="", help="Output directory. Defaults to assets/youtu_text_export[/fp16].")
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16"], help="ncnn weight storage precision.")
    parser.add_argument("--sample", default="en_short", help="Sample id used for trace shapes.")
    parser.add_argument("--target", action="append", default=[], help="Export target: embed_tokens, lm_head, or all. Can repeat.")
    parser.add_argument("--trace", action="store_true", help="Actually write TorchScript .pt files.")
    parser.add_argument("--run-pnnx", action="store_true", help="Run pnnx after tracing. Requires --trace.")
    parser.add_argument(
        "--allow-large-lm-head-pnnx",
        action="store_true",
        help="Allow lm_head pnnx conversion. This may create very large temporary files and can OOM.",
    )
    parser.add_argument("--pnnx", default="", help="Path to pnnx executable. If omitted, PATH is searched.")
    parser.add_argument("--pnnx-arg", action="append", default=[], help="Extra key=value argument passed to pnnx.")
    parser.add_argument("--check-trace", action="store_true", help="Enable torch.jit.trace check_trace.")
    parser.add_argument(
        "--weight-loading",
        default="auto",
        choices=["auto", "safetensors", "full"],
        help="auto/safetensors loads only text boundary weights when possible; full loads the HF model.",
    )
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--device", default="auto", help="Used when --device-map none. Use 'auto', 'cuda:0', or 'cpu'.")
    parser.add_argument("--device-map", default="none")
    parser.add_argument("--attn-implementation", default="eager")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_pnnx and not args.trace:
        die("--run-pnnx requires --trace")

    metadata_path = resolve_project_path(args.metadata)
    metadata = load_json(metadata_path)
    sample = select_sample(metadata, args.sample)
    npz_path = resolve_npz_path(sample, metadata_path)

    output_dir = resolve_project_path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR / ("fp16" if args.precision == "fp16" else "")
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = parse_targets(args.target)
    if args.run_pnnx and "lm_head" in targets and not args.allow_large_lm_head_pnnx:
        die(
            "refusing to run pnnx for lm_head without --allow-large-lm-head-pnnx. "
            "The full vocab projection creates very large temporary files and has OOMed on this machine."
        )
    np, torch, transformers, model_path, model, device, pydensecrf_stubbed = load_export_model(args, metadata, targets)
    arrays = np.load(npz_path)

    input_ids = torch.from_numpy(arrays["input_ids"]).long().to(device)
    hidden_states = torch.from_numpy(arrays["prefill_last_hidden_state"]).to(device=device)

    EmbedWrapper, LmHeadWrapper = make_wrappers(torch)
    summary = module_summary(model)
    artifacts: Dict[str, Any] = {}
    pnnx_path = resolve_pnnx_path(args.pnnx) if args.run_pnnx else None

    if "embed_tokens" in targets:
        target_info: Dict[str, Any] = {
            "status": "planned",
            "example_input_shape": tensor_shape(input_ids),
            "expected_output_shape": tensor_shape(torch.empty((*input_ids.shape, summary["config"]["hidden_size"]))),
        }
        if args.trace:
            traced = trace_module(
                torch,
                EmbedWrapper(model.model.embed_tokens),
                (input_ids,),
                output_dir / "youtu_embed_tokens.pt",
                args.check_trace,
            )
            target_info.update({"status": "traced", **traced})
            if args.run_pnnx and pnnx_path is not None:
                target_info["pnnx"] = run_pnnx(
                    pnnx_path=pnnx_path,
                    torchscript_path=Path(traced["torchscript"]),
                    input_shapes=[traced["example_input_shapes"][0]],
                    input_dtypes=[traced["example_input_pnnx_dtypes"][0]],
                    output_prefix=output_dir / "youtu_embed_tokens",
                    fp16=args.precision == "fp16",
                    extra_args=args.pnnx_arg,
                )
                if not target_info["pnnx"]["passed"]:
                    raise SystemExit(target_info["pnnx"]["returncode"] or 1)
                if args.precision == "fp16":
                    ncnn_bin = output_dir / "youtu_embed_tokens.ncnn.bin"
                    expected_bytes = 4 + int(model.model.embed_tokens.weight.numel()) * 2
                    actual_bytes = ncnn_bin.stat().st_size if ncnn_bin.exists() else 0
                    if actual_bytes != expected_bytes:
                        die(
                            "invalid fp16 Embed payload: "
                            f"expected_bytes={expected_bytes} actual_bytes={actual_bytes} path={ncnn_bin}"
                        )
                    target_info["pnnx"]["validated_ncnn_bin_bytes"] = actual_bytes
        artifacts["embed_tokens"] = target_info

    if "lm_head" in targets:
        target_info = {
            "status": "planned",
            "example_input_shape": tensor_shape(hidden_states),
            "expected_output_shape": [int(hidden_states.shape[0]), int(hidden_states.shape[1]), summary["config"]["vocab_size"]],
        }
        if args.trace:
            traced = trace_module(
                torch,
                LmHeadWrapper(model.lm_head),
                (hidden_states,),
                output_dir / "youtu_lm_head.pt",
                args.check_trace,
            )
            target_info.update({"status": "traced", **traced})
            if args.run_pnnx and pnnx_path is not None:
                target_info["pnnx"] = run_pnnx(
                    pnnx_path=pnnx_path,
                    torchscript_path=Path(traced["torchscript"]),
                    input_shapes=[traced["example_input_shapes"][0]],
                    input_dtypes=[traced["example_input_pnnx_dtypes"][0]],
                    output_prefix=output_dir / "youtu_lm_head",
                    fp16=args.precision == "fp16",
                    extra_args=args.pnnx_arg,
                )
                if not target_info["pnnx"]["passed"]:
                    raise SystemExit(target_info["pnnx"]["returncode"] or 1)
        artifacts["lm_head"] = target_info

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "metadata": str(metadata_path),
        "sample_id": sample.get("sample_id"),
        "npz_path": str(npz_path),
        "model": {
            "path_or_id": str(model_path),
            "device": str(device),
            "dtype_arg": args.dtype,
            "attn_implementation": args.attn_implementation,
        },
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "pydensecrf_stubbed": pydensecrf_stubbed,
        },
        "pnnx": {
            "requested": bool(args.run_pnnx),
            "path": str(pnnx_path) if pnnx_path is not None else None,
            "precision": args.precision,
            "fp16": args.precision == "fp16",
            "extra_args": args.pnnx_arg,
        },
        "summary": summary,
        "artifacts": artifacts,
        "decoder_note": (
            "Decoder export is intentionally not traced here. Use the P2C split "
            "parity report to design a stable decoder/cache wrapper first."
        ),
    }

    manifest_path = output_dir / "export_manifest.json"
    write_json(manifest_path, manifest)
    print(f"[write] {manifest_path}")
    for name, info in artifacts.items():
        print(f"[{info['status']}] {name}")


if __name__ == "__main__":
    main()
