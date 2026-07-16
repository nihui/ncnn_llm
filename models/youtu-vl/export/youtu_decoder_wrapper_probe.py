#!/usr/bin/env python3
"""Probe Youtu-VL decoder wrapper parity against saved text-only dumps.

This script is a PyTorch-side staging tool for the decoder export work.  It
does not write TorchScript or ncnn files.  Instead, it proves that the decoder
can be driven from the flattened tensors stored by youtu_text_reference.py:

  prefill: inputs_embeds + attention_mask + cache_position -> last_hidden_state + cache
  decode:  inputs_embeds + attention_mask + cache_position + cache -> last_hidden_state + cache

Once these checks are stable, the same boundary can be turned into a traceable
wrapper for pnnx/ncnn conversion.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


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
        from transformers.cache_utils import DynamicCache
    except ModuleNotFoundError as exc:
        missing = exc.name or "a required module"
        die(
            "missing Python dependency "
            f"`{missing}`. Activate the project venv or install runtime deps:\n"
            "  pip install -U torch transformers accelerate numpy safetensors",
            code=3,
        )
    return np, torch, transformers, AutoModelForCausalLM, DynamicCache


def install_pydensecrf_import_stub() -> bool:
    try:
        import pydensecrf.densecrf  # type: ignore  # noqa: F401
        import pydensecrf.utils  # type: ignore  # noqa: F401
        return False
    except ModuleNotFoundError:
        pass

    def unavailable(*_args, **_kwargs):
        raise RuntimeError(
            "pydensecrf is not installed. It is not needed for text decoder "
            "probes, but is required for dense CRF post-processing."
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


def first_parameter_dtype(model):
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return None


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
    if isinstance(model_arg, str) and not model_arg.startswith("tencent/"):
        return resolve_project_path(model_arg)
    return model_arg


def tensor_shape(value: Any) -> Optional[List[int]]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(x) for x in shape]


def tensor_from_np(torch, arrays, name: str, *, device, dtype=None):
    if name not in arrays:
        die(f"npz is missing required array `{name}`")
    tensor = torch.from_numpy(arrays[name])
    if dtype is not None and tensor.is_floating_point():
        tensor = tensor.to(dtype=dtype)
    return tensor.to(device)


def cache_layers_from_npz(torch, arrays, prefix: str, layer_count: int, *, device, dtype) -> List[Tuple[Any, Any]]:
    layers = []
    for layer_idx in range(layer_count):
        key_name = f"{prefix}_layer{layer_idx:02d}_key"
        value_name = f"{prefix}_layer{layer_idx:02d}_value"
        key = tensor_from_np(torch, arrays, key_name, device=device, dtype=dtype)
        value = tensor_from_np(torch, arrays, value_name, device=device, dtype=dtype)
        layers.append((key, value))
    return layers


def cache_to_layers(cache: Any) -> List[Tuple[Any, Any]]:
    if cache is None:
        return []
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return list(zip(cache.key_cache, cache.value_cache))
    layers = []
    try:
        iterable = list(cache)
    except TypeError:
        return layers
    for item in iterable:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            layers.append((item[0], item[1]))
    return layers


def dynamic_cache_from_layers(DynamicCache, layers: Sequence[Tuple[Any, Any]]):
    try:
        return DynamicCache(ddp_cache_data=list(layers))
    except TypeError:
        cache = DynamicCache()
        for layer_idx, (key, value) in enumerate(layers):
            cache.update(key, value, layer_idx)
        return cache


def cache_to_flat_tuple(cache: Any) -> Tuple[Any, ...]:
    flat: List[Any] = []
    for key, value in cache_to_layers(cache):
        flat.extend([key, value])
    return tuple(flat)


def make_flat_wrappers(torch, DynamicCache, layer_count: int):
    class DecoderPrefillFlatWrapper(torch.nn.Module):
        def __init__(self, text_model):
            super().__init__()
            self.text_model = text_model

        def forward(self, inputs_embeds, attention_mask, cache_position):
            outputs = self.text_model(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                use_cache=True,
                output_hidden_states=False,
                output_attentions=False,
            )
            return (outputs.last_hidden_state, *cache_to_flat_tuple(outputs.past_key_values))

    class DecoderDecodeFlatWrapper(torch.nn.Module):
        def __init__(self, text_model):
            super().__init__()
            self.text_model = text_model

        def forward(self, inputs_embeds, attention_mask, cache_position, *flat_cache):
            layers = []
            for layer_idx in range(layer_count):
                base = layer_idx * 2
                layers.append((flat_cache[base], flat_cache[base + 1]))
            past = dynamic_cache_from_layers(DynamicCache, layers)
            outputs = self.text_model(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past,
                cache_position=cache_position,
                use_cache=True,
                output_hidden_states=False,
                output_attentions=False,
            )
            return (outputs.last_hidden_state, *cache_to_flat_tuple(outputs.past_key_values))

    return DecoderPrefillFlatWrapper, DecoderDecodeFlatWrapper


def make_4d_causal_mask(torch, attention_mask, inputs_embeds, cache_position):
    dtype = torch.float32
    device = inputs_embeds.device
    batch_size = inputs_embeds.shape[0]
    sequence_length = inputs_embeds.shape[1]
    target_length = attention_mask.shape[-1]
    min_dtype = torch.finfo(dtype).min

    causal_mask = torch.full(
        (sequence_length, target_length),
        fill_value=min_dtype,
        dtype=dtype,
        device=device,
    )
    if sequence_length != 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
    causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
    causal_mask = causal_mask.clone()

    mask_length = attention_mask.shape[-1]
    padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(device)
    padding_mask = padding_mask == 0
    causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
        padding_mask, min_dtype
    )
    return causal_mask


def prepare_flat_attention_mask(torch, attention_mask, inputs_embeds, cache_position, mask_mode: str):
    if mask_mode == "2d":
        return attention_mask
    if mask_mode == "4d":
        return make_4d_causal_mask(torch, attention_mask, inputs_embeds, cache_position)
    die(f"unsupported mask mode: {mask_mode}")


def diff_stats(torch, actual, expected) -> Dict[str, Any]:
    actual_f = actual.detach().float().cpu()
    expected_f = expected.detach().float().cpu()
    diff = actual_f - expected_f
    abs_diff = diff.abs()
    return {
        "shape": tensor_shape(actual),
        "expected_shape": tensor_shape(expected),
        "max_abs": float(abs_diff.max().item()) if abs_diff.numel() else 0.0,
        "mean_abs": float(abs_diff.mean().item()) if abs_diff.numel() else 0.0,
        "rms": float(torch.sqrt(torch.mean(diff * diff)).item()) if diff.numel() else 0.0,
    }


def print_stats(name: str, stats: Mapping[str, Any], atol: float) -> bool:
    passed = float(stats["max_abs"]) <= atol
    state = "PASS" if passed else "FAIL"
    print(
        f"[{state}] {name}: shape={stats['shape']} "
        f"max_abs={stats['max_abs']:.6g} mean_abs={stats['mean_abs']:.6g} rms={stats['rms']:.6g}"
    )
    return passed


def summarize_cache_diff(torch, actual_cache, expected_layers, *, layer_limit: int = 3) -> Dict[str, Any]:
    actual_layers = cache_to_layers(actual_cache)
    if len(actual_layers) != len(expected_layers):
        return {
            "layer_count": len(actual_layers),
            "expected_layer_count": len(expected_layers),
            "max_abs": float("inf"),
            "worst": "layer-count-mismatch",
            "samples": [],
        }

    worst_name = ""
    worst_max = -1.0
    samples = []
    for layer_idx, ((actual_key, actual_value), (expected_key, expected_value)) in enumerate(
        zip(actual_layers, expected_layers)
    ):
        for suffix, actual, expected in (
            ("key", actual_key, expected_key),
            ("value", actual_value, expected_value),
        ):
            stats = diff_stats(torch, actual, expected)
            current = float(stats["max_abs"])
            name = f"layer{layer_idx:02d}_{suffix}"
            if current > worst_max:
                worst_max = current
                worst_name = name
            if layer_idx < layer_limit:
                samples.append({"name": name, **stats})

    return {
        "layer_count": len(actual_layers),
        "expected_layer_count": len(expected_layers),
        "max_abs": worst_max,
        "worst": worst_name,
        "samples": samples,
    }


def print_cache_summary(name: str, summary: Mapping[str, Any], atol: float) -> bool:
    passed = float(summary["max_abs"]) <= atol
    state = "PASS" if passed else "FAIL"
    print(
        f"[{state}] {name}: layers={summary['layer_count']} "
        f"max_abs={summary['max_abs']:.6g} worst={summary['worst']}"
    )
    for sample in summary.get("samples", []):
        print(
            f"  sample {sample['name']}: shape={sample['shape']} "
            f"max_abs={sample['max_abs']:.6g} mean_abs={sample['mean_abs']:.6g}"
        )
    return passed


def compare_flat_outputs(
    torch,
    name: str,
    actual_outputs: Sequence[Any],
    expected_hidden,
    expected_cache_layers: Sequence[Tuple[Any, Any]],
    args,
) -> bool:
    ok = print_stats(f"{name}.last_hidden_state", diff_stats(torch, actual_outputs[0], expected_hidden), args.atol)
    actual_cache = []
    flat_cache = actual_outputs[1:]
    for layer_idx in range(len(expected_cache_layers)):
        base = layer_idx * 2
        actual_cache.append((flat_cache[base], flat_cache[base + 1]))
    cache_summary = summarize_cache_diff(torch, actual_cache, expected_cache_layers, layer_limit=args.cache_sample_layers)
    ok = print_cache_summary(f"{name}.output_cache", cache_summary, args.cache_atol) and ok
    return ok


def trace_wrapper(torch, wrapper, example_inputs: Tuple[Any, ...], output_path: Path, check_trace: bool) -> Dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper.eval()
    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, example_inputs, check_trace=check_trace)
        traced.save(str(output_path))
        outputs = traced(*example_inputs)
    return {
        "path": str(output_path),
        "bytes": output_path.stat().st_size,
        "output_count": len(outputs) if isinstance(outputs, tuple) else 1,
    }


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


def shape_arg(torch, tensor) -> str:
    shape = ",".join(str(int(x)) for x in tensor.shape)
    return f"[{shape}]{pnnx_dtype_name(torch, tensor)}"


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
    torch,
    *,
    pnnx_path: Path,
    torchscript_path: Path,
    example_inputs: Tuple[Any, ...],
    output_prefix: Path,
    fp16: bool,
    extra_args: Sequence[str],
) -> Dict[str, Any]:
    inputshape = ",".join(shape_arg(torch, tensor) for tensor in example_inputs)
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


def load_model(args, metadata: Mapping[str, Any]):
    np, torch, transformers, AutoModelForCausalLM, DynamicCache = import_runtime_deps()
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

    return np, torch, transformers, DynamicCache, model, device, pydensecrf_stubbed


def run_prefill_probe(torch, DynamicCache, model, arrays, layer_count: int, *, device, dtype, args) -> bool:
    inputs_embeds = tensor_from_np(torch, arrays, "decoder_prefill_inputs_embeds", device=device, dtype=dtype)
    attention_mask = tensor_from_np(torch, arrays, "decoder_prefill_attention_mask", device=device)
    cache_position = tensor_from_np(torch, arrays, "decoder_prefill_cache_position", device=device)
    expected_hidden = tensor_from_np(torch, arrays, "decoder_prefill_last_hidden_state", device=device, dtype=torch.float32)
    expected_cache = cache_layers_from_npz(
        torch, arrays, "decoder_prefill_output_cache", layer_count, device=device, dtype=dtype
    )

    with torch.inference_mode():
        outputs = model.model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            use_cache=True,
            output_hidden_states=False,
            output_attentions=False,
        )

    ok = print_stats("prefill.last_hidden_state", diff_stats(torch, outputs.last_hidden_state, expected_hidden), args.atol)
    cache_summary = summarize_cache_diff(torch, outputs.past_key_values, expected_cache, layer_limit=args.cache_sample_layers)
    ok = print_cache_summary("prefill.output_cache", cache_summary, args.cache_atol) and ok
    return ok


def run_prefill_flat_probe(torch, DynamicCache, model, arrays, layer_count: int, *, device, dtype, args) -> Tuple[bool, Optional[Tuple[Any, Tuple[Any, ...]]]]:
    inputs_embeds = tensor_from_np(torch, arrays, "decoder_prefill_inputs_embeds", device=device, dtype=dtype)
    attention_mask = tensor_from_np(torch, arrays, "decoder_prefill_attention_mask", device=device)
    cache_position = tensor_from_np(torch, arrays, "decoder_prefill_cache_position", device=device)
    attention_mask = prepare_flat_attention_mask(torch, attention_mask, inputs_embeds, cache_position, args.mask_mode)
    if args.float_cache_position:
        cache_position = cache_position.float()
    expected_hidden = tensor_from_np(torch, arrays, "decoder_prefill_last_hidden_state", device=device, dtype=torch.float32)
    expected_cache = cache_layers_from_npz(
        torch, arrays, "decoder_prefill_output_cache", layer_count, device=device, dtype=dtype
    )

    PrefillWrapper, _DecodeWrapper = make_flat_wrappers(torch, DynamicCache, layer_count)
    wrapper = PrefillWrapper(model.model).eval()
    example_inputs = (inputs_embeds, attention_mask, cache_position)
    with torch.inference_mode():
        outputs = wrapper(*example_inputs)

    ok = compare_flat_outputs(torch, "prefill_flat", outputs, expected_hidden, expected_cache, args)
    return ok, (wrapper, example_inputs)


def run_decode_probe(torch, DynamicCache, model, arrays, layer_count: int, *, device, dtype, args) -> bool:
    inputs_embeds = tensor_from_np(torch, arrays, "decoder_decode_step0_inputs_embeds", device=device, dtype=dtype)
    attention_mask = tensor_from_np(torch, arrays, "decoder_decode_step0_attention_mask", device=device)
    cache_position = tensor_from_np(torch, arrays, "decoder_decode_step0_cache_position", device=device)
    expected_hidden = tensor_from_np(
        torch, arrays, "decoder_decode_step0_last_hidden_state", device=device, dtype=torch.float32
    )

    input_cache_layers = cache_layers_from_npz(
        torch, arrays, "decoder_decode_step0_input_cache", layer_count, device=device, dtype=dtype
    )
    expected_cache_layers = cache_layers_from_npz(
        torch, arrays, "decoder_decode_step0_output_cache", layer_count, device=device, dtype=dtype
    )
    past = dynamic_cache_from_layers(DynamicCache, input_cache_layers)

    with torch.inference_mode():
        outputs = model.model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past,
            cache_position=cache_position,
            use_cache=True,
            output_hidden_states=False,
            output_attentions=False,
        )

    ok = print_stats("decode_step0.last_hidden_state", diff_stats(torch, outputs.last_hidden_state, expected_hidden), args.atol)
    cache_summary = summarize_cache_diff(torch, outputs.past_key_values, expected_cache_layers, layer_limit=args.cache_sample_layers)
    ok = print_cache_summary("decode_step0.output_cache", cache_summary, args.cache_atol) and ok
    return ok


def run_decode_flat_probe(torch, DynamicCache, model, arrays, layer_count: int, *, device, dtype, args) -> Tuple[bool, Optional[Tuple[Any, Tuple[Any, ...]]]]:
    inputs_embeds = tensor_from_np(torch, arrays, "decoder_decode_step0_inputs_embeds", device=device, dtype=dtype)
    attention_mask = tensor_from_np(torch, arrays, "decoder_decode_step0_attention_mask", device=device)
    cache_position = tensor_from_np(torch, arrays, "decoder_decode_step0_cache_position", device=device)
    attention_mask = prepare_flat_attention_mask(torch, attention_mask, inputs_embeds, cache_position, args.mask_mode)
    if args.float_cache_position:
        cache_position = cache_position.float()
    expected_hidden = tensor_from_np(
        torch, arrays, "decoder_decode_step0_last_hidden_state", device=device, dtype=torch.float32
    )
    input_cache_layers = cache_layers_from_npz(
        torch, arrays, "decoder_decode_step0_input_cache", layer_count, device=device, dtype=dtype
    )
    expected_cache = cache_layers_from_npz(
        torch, arrays, "decoder_decode_step0_output_cache", layer_count, device=device, dtype=dtype
    )

    _PrefillWrapper, DecodeWrapper = make_flat_wrappers(torch, DynamicCache, layer_count)
    wrapper = DecodeWrapper(model.model).eval()
    flat_input_cache = tuple(tensor for layer in input_cache_layers for tensor in layer)
    example_inputs = (inputs_embeds, attention_mask, cache_position, *flat_input_cache)
    with torch.inference_mode():
        outputs = wrapper(*example_inputs)

    ok = compare_flat_outputs(torch, "decode_step0_flat", outputs, expected_hidden, expected_cache, args)
    return ok, (wrapper, example_inputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Youtu-VL decoder wrapper parity against text dump tensors.")
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA), help="Path to text dump metadata.json.")
    parser.add_argument("--model", default="", help="Local HF model dir or model id. Defaults to metadata model.path_or_id.")
    parser.add_argument("--sample", default="en_short", help="Sample id in metadata.json.")
    parser.add_argument("--mode", default="both", choices=["prefill", "decode", "both"], help="Which decoder path to probe.")
    parser.add_argument(
        "--wrapper-mode",
        default="flat",
        choices=["direct", "flat", "both"],
        help="direct uses HF DynamicCache directly; flat uses trace-friendly tensor inputs/outputs.",
    )
    parser.add_argument(
        "--mask-mode",
        default="4d",
        choices=["2d", "4d"],
        help="flat wrapper attention mask form. 4d precomputes the causal mask before tracing.",
    )
    parser.add_argument(
        "--float-cache-position",
        action="store_true",
        help="Pass cache_position as float32 in flat wrappers to avoid Tensor.to in pnnx/ncnn.",
    )
    parser.add_argument("--trace", action="store_true", help="Trace flat wrappers to TorchScript after parity checks pass.")
    parser.add_argument("--run-pnnx", action="store_true", help="Run pnnx after tracing flat wrappers. Requires --trace.")
    parser.add_argument("--pnnx", default="", help="Path to pnnx executable. If omitted, PATH is searched.")
    parser.add_argument("--pnnx-arg", action="append", default=[], help="Extra key=value argument passed to pnnx.")
    parser.add_argument("--pnnx-fp16", action="store_true", help="Ask pnnx to write fp16 ncnn weights.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for traced decoder wrappers.")
    parser.add_argument("--check-trace", action="store_true", help="Enable torch.jit.trace check_trace.")
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--device", default="auto", help="Used when --device-map none. Use 'auto', 'cuda:0', or 'cpu'.")
    parser.add_argument("--device-map", default="none")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--atol", type=float, default=1e-3, help="Max absolute tolerance for hidden-state checks.")
    parser.add_argument("--cache-atol", type=float, default=1e-3, help="Max absolute tolerance for cache checks.")
    parser.add_argument("--cache-sample-layers", type=int, default=2, help="How many cache layers to print in detail.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_pnnx and not args.trace:
        die("--run-pnnx requires --trace")
    metadata_path = resolve_project_path(args.metadata)
    metadata = load_json(metadata_path)
    sample = select_sample(metadata, args.sample)
    npz_path = resolve_npz_path(sample, metadata_path)
    if not npz_path.exists():
        die(f"npz not found: {npz_path}")

    np, torch, transformers, DynamicCache, model, device, pydensecrf_stubbed = load_model(args, metadata)
    arrays = np.load(npz_path)
    model_dtype = first_parameter_dtype(model) or torch.float32
    layer_count = int(metadata.get("model", {}).get("num_hidden_layers") or 0)
    if layer_count <= 0:
        die("metadata model.num_hidden_layers is missing or invalid")

    print(f"[runtime] torch={torch.__version__} transformers={transformers.__version__} pydensecrf_stubbed={pydensecrf_stubbed}")
    print(f"[probe] sample={sample.get('sample_id')} npz={npz_path}")
    print(f"[probe] device={device} model_dtype={model_dtype} layers={layer_count}")

    ok = True
    trace_jobs: List[Tuple[str, Any, Tuple[Any, ...]]] = []

    if args.wrapper_mode in {"direct", "both"} and args.mode in {"prefill", "both"}:
        ok = run_prefill_probe(torch, DynamicCache, model, arrays, layer_count, device=device, dtype=model_dtype, args=args) and ok
    if args.wrapper_mode in {"direct", "both"} and args.mode in {"decode", "both"}:
        ok = run_decode_probe(torch, DynamicCache, model, arrays, layer_count, device=device, dtype=model_dtype, args=args) and ok

    if args.wrapper_mode in {"flat", "both"} and args.mode in {"prefill", "both"}:
        passed, trace_data = run_prefill_flat_probe(
            torch, DynamicCache, model, arrays, layer_count, device=device, dtype=model_dtype, args=args
        )
        ok = passed and ok
        if trace_data is not None:
            wrapper, example_inputs = trace_data
            trace_jobs.append(("youtu_decoder_prefill.pt", wrapper, example_inputs))
    if args.wrapper_mode in {"flat", "both"} and args.mode in {"decode", "both"}:
        passed, trace_data = run_decode_flat_probe(
            torch, DynamicCache, model, arrays, layer_count, device=device, dtype=model_dtype, args=args
        )
        ok = passed and ok
        if trace_data is not None:
            wrapper, example_inputs = trace_data
            trace_jobs.append(("youtu_decoder_decode_step0.pt", wrapper, example_inputs))

    if not ok:
        raise SystemExit(1)

    if args.trace:
        output_dir = resolve_project_path(args.output_dir)
        pnnx_path = resolve_pnnx_path(args.pnnx) if args.run_pnnx else None
        for filename, wrapper, example_inputs in trace_jobs:
            info = trace_wrapper(torch, wrapper, example_inputs, output_dir / filename, args.check_trace)
            print(f"[trace] {info['path']} bytes={info['bytes']} outputs={info['output_count']}")
            if pnnx_path is not None:
                prefix = output_dir / Path(filename).with_suffix("").name
                pnnx_result = run_pnnx(
                    torch,
                    pnnx_path=pnnx_path,
                    torchscript_path=Path(info["path"]),
                    example_inputs=example_inputs,
                    output_prefix=prefix,
                    fp16=args.pnnx_fp16,
                    extra_args=args.pnnx_arg,
                )
                print(f"[pnnx] returncode={pnnx_result['returncode']} produced={len(pnnx_result['produced'])}")
                if pnnx_result["stdout"]:
                    print("[pnnx:stdout]")
                    print(pnnx_result["stdout"])
                if pnnx_result["stderr"]:
                    print("[pnnx:stderr]")
                    print(pnnx_result["stderr"])
                if not pnnx_result["passed"]:
                    raise SystemExit(pnnx_result["returncode"] or 1)


if __name__ == "__main__":
    main()
