#!/usr/bin/env python3
"""Check ncnn decoder decode-step output against saved PyTorch dump tensors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = ROOT / "assets" / "youtu_text_test" / "dumps" / "metadata.json"
DEFAULT_PARAM = ROOT / "assets" / "youtu_text_export" / "youtu_decoder_decode_step0.ncnn.param"
DEFAULT_BIN = ROOT / "assets" / "youtu_text_export" / "youtu_decoder_decode_step0.ncnn.bin"


def die(message: str, code: int = 2) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def import_runtime_deps():
    try:
        import ncnn
        import numpy as np
    except ModuleNotFoundError as exc:
        missing = exc.name or "a required module"
        die(
            "missing Python dependency "
            f"`{missing}`. Activate the project venv or install runtime deps:\n"
            "  pip install -U ncnn numpy",
            code=3,
        )
    return ncnn, np


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


def diff_stats(np, actual, expected) -> Dict[str, Any]:
    actual_f = np.asarray(actual, dtype=np.float32)
    expected_f = np.asarray(expected, dtype=np.float32)
    if actual_f.shape != expected_f.shape:
        return {
            "shape": list(actual_f.shape),
            "expected_shape": list(expected_f.shape),
            "max_abs": float("inf"),
            "mean_abs": float("inf"),
            "rms": float("inf"),
        }
    diff = actual_f - expected_f
    abs_diff = np.abs(diff)
    return {
        "shape": list(actual_f.shape),
        "expected_shape": list(expected_f.shape),
        "max_abs": float(abs_diff.max()) if abs_diff.size else 0.0,
        "mean_abs": float(abs_diff.mean()) if abs_diff.size else 0.0,
        "rms": float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0,
    }


def print_stats(name: str, stats: Mapping[str, Any], atol: float) -> bool:
    passed = float(stats["max_abs"]) <= atol
    state = "PASS" if passed else "FAIL"
    print(
        f"[{state}] {name}: shape={stats['shape']} expected={stats['expected_shape']} "
        f"max_abs={stats['max_abs']:.6g} mean_abs={stats['mean_abs']:.6g} rms={stats['rms']:.6g}"
    )
    return passed


def make_4d_decode_mask(np, arrays):
    inputs_embeds = arrays["decoder_decode_step0_inputs_embeds"]
    attention_mask = arrays["decoder_decode_step0_attention_mask"]
    cache_position = arrays["decoder_decode_step0_cache_position"]

    batch_size = inputs_embeds.shape[0]
    sequence_length = inputs_embeds.shape[1]
    target_length = attention_mask.shape[-1]
    min_dtype = np.finfo(np.float32).min
    causal_mask = np.full((sequence_length, target_length), min_dtype, dtype=np.float32)
    if sequence_length != 1:
        causal_mask = np.triu(causal_mask, k=1)
    causal_mask *= (np.arange(target_length, dtype=np.int64) > cache_position.reshape(-1, 1)).astype(np.float32)
    causal_mask = np.broadcast_to(causal_mask[None, None, :, :], (batch_size, 1, sequence_length, target_length)).copy()

    padding_mask = causal_mask[:, :, :, :target_length] + attention_mask[:, None, None, :].astype(np.float32)
    padding_mask = padding_mask == 0
    causal_mask[:, :, :, :target_length] = np.where(
        padding_mask,
        np.float32(min_dtype),
        causal_mask[:, :, :, :target_length],
    )
    return causal_mask


def input_arrays(np, arrays, layer_count: int, mask_mode: str) -> List[Any]:
    if mask_mode == "2d":
        attention_mask = arrays["decoder_decode_step0_attention_mask"].astype("int32")
    elif mask_mode == "4d":
        attention_mask = make_4d_decode_mask(np, arrays)
    else:
        die(f"unsupported mask mode: {mask_mode}")

    inputs = [
        arrays["decoder_decode_step0_inputs_embeds"],
        attention_mask,
        arrays["decoder_decode_step0_cache_position"].astype("int32"),
    ]
    for layer_idx in range(layer_count):
        inputs.append(arrays[f"decoder_decode_step0_input_cache_layer{layer_idx:02d}_key"])
        inputs.append(arrays[f"decoder_decode_step0_input_cache_layer{layer_idx:02d}_value"])
    return inputs


def expected_output(arrays, output_index: int):
    if output_index == 0:
        return arrays["decoder_decode_step0_last_hidden_state"]
    cache_index = output_index - 1
    layer_idx = cache_index // 2
    suffix = "key" if cache_index % 2 == 0 else "value"
    return arrays[f"decoder_decode_step0_output_cache_layer{layer_idx:02d}_{suffix}"]


def output_name(output_index: int) -> str:
    if output_index == 0:
        return "last_hidden_state"
    cache_index = output_index - 1
    layer_idx = cache_index // 2
    suffix = "key" if cache_index % 2 == 0 else "value"
    return f"cache_layer{layer_idx:02d}_{suffix}"


def parse_output_indices(raw_values: List[str], layer_count: int) -> List[int]:
    if not raw_values:
        return [0, 1, 2]
    indices: List[int] = []
    max_index = 1 + layer_count * 2 - 1
    for raw in raw_values:
        for part in raw.split(","):
            value = part.strip()
            if not value:
                continue
            index = int(value)
            if index < 0 or index > max_index:
                die(f"output index out of range: {index}; allowed 0..{max_index}")
            if index not in indices:
                indices.append(index)
    return indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ncnn decoder decode-step outputs with PyTorch dump tensors.")
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA), help="Path to text dump metadata.json.")
    parser.add_argument("--sample", default="en_short", help="Sample id in metadata.json.")
    parser.add_argument("--param", default=str(DEFAULT_PARAM), help="Path to ncnn param file.")
    parser.add_argument("--bin", default=str(DEFAULT_BIN), help="Path to ncnn bin file.")
    parser.add_argument(
        "--output",
        action="append",
        default=[],
        help="Output indices to check. 0 is hidden, 1/2 are layer00 key/value. Can repeat or comma-separate.",
    )
    parser.add_argument("--atol", type=float, default=5e-2, help="Max absolute tolerance.")
    parser.add_argument("--mask-mode", default="4d", choices=["2d", "4d"], help="Attention mask shape expected by ncnn model.")
    parser.add_argument("--num-threads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ncnn, np = import_runtime_deps()

    metadata_path = resolve_project_path(args.metadata)
    metadata = load_json(metadata_path)
    sample = select_sample(metadata, args.sample)
    npz_path = resolve_npz_path(sample, metadata_path)
    if not npz_path.exists():
        die(f"npz not found: {npz_path}")
    param_path = resolve_project_path(args.param)
    bin_path = resolve_project_path(args.bin)
    if not param_path.exists():
        die(f"ncnn param not found: {param_path}")
    if not bin_path.exists():
        die(f"ncnn bin not found: {bin_path}")

    layer_count = int(metadata.get("model", {}).get("num_hidden_layers") or 0)
    if layer_count <= 0:
        die("metadata model.num_hidden_layers is missing or invalid")
    outputs_to_check = parse_output_indices(args.output, layer_count)

    arrays = np.load(npz_path)
    inputs = input_arrays(np, arrays, layer_count, args.mask_mode)

    print(f"[load] param={param_path}")
    print(f"[load] bin={bin_path}")
    print(f"[data] sample={sample.get('sample_id')} npz={npz_path}")

    ok = True
    with ncnn.Net() as net:
        net.opt.use_vulkan_compute = False
        net.opt.num_threads = int(args.num_threads)
        ret = net.load_param(str(param_path))
        if ret != 0:
            die(f"net.load_param failed: {ret}", code=4)
        ret = net.load_model(str(bin_path))
        if ret != 0:
            die(f"net.load_model failed: {ret}", code=4)

        with net.create_extractor() as ex:
            for index, value in enumerate(inputs):
                ex.input(f"in{index}", ncnn.Mat(np.ascontiguousarray(value)).clone())

            for index in outputs_to_check:
                ret, out = ex.extract(f"out{index}")
                if ret != 0:
                    die(f"extract out{index} failed: {ret}", code=5)
                actual = np.array(out)
                expected = expected_output(arrays, index)
                ok = print_stats(f"out{index}.{output_name(index)}", diff_stats(np, actual, expected), args.atol) and ok

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
