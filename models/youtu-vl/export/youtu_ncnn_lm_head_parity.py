#!/usr/bin/env python3
"""Check ncnn lm_head logits against the saved PyTorch decode dump."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from youtu_decoder_wrapper_probe import (
    DEFAULT_METADATA,
    DEFAULT_OUTPUT_DIR,
    die,
    load_json,
    print_stats,
    resolve_npz_path,
    resolve_project_path,
    select_sample,
)
from youtu_ncnn_decoder_parity import diff_stats as numpy_diff_stats


DEFAULT_PARAM = DEFAULT_OUTPUT_DIR / "youtu_lm_head.ncnn.param"
DEFAULT_BIN = DEFAULT_OUTPUT_DIR / "youtu_lm_head.ncnn.bin"


def import_deps():
    try:
        import ncnn
        import numpy as np
    except ModuleNotFoundError as exc:
        die(f"missing Python dependency `{exc.name}`; activate the project venv", code=3)
    return ncnn, np


def run_lm_head(ncnn, np, *, param_path: Path, bin_path: Path, hidden, num_threads: int, no_packing_layout: bool):
    with ncnn.Net() as net:
        net.opt.use_vulkan_compute = False
        net.opt.num_threads = int(num_threads)
        if no_packing_layout:
            net.opt.use_packing_layout = False
        ret = net.load_param(str(param_path))
        if ret != 0:
            die(f"net.load_param failed for {param_path}: {ret}", code=4)
        ret = net.load_model(str(bin_path))
        if ret != 0:
            die(f"net.load_model failed for {bin_path}: {ret}", code=4)
        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(np.ascontiguousarray(hidden.astype("float32", copy=False))).clone())
            ret, out = ex.extract("out0")
            if ret != 0:
                die(f"extract out0 failed for {param_path}: {ret}", code=5)
            return np.array(out, dtype=np.float32)


def topk(np, values, k: int) -> tuple[Any, Any]:
    k = min(int(k), int(values.shape[-1]))
    ids = np.argpartition(-values, kth=k - 1, axis=-1)[..., :k]
    vals = np.take_along_axis(values, ids, axis=-1)
    order = np.argsort(-vals, axis=-1)
    ids = np.take_along_axis(ids, order, axis=-1)
    vals = np.take_along_axis(values, ids, axis=-1)
    return ids, vals


def compare_topk(np, actual, expected, k: int, *, require_order: bool) -> bool:
    actual_ids, actual_values = topk(np, actual, k)
    expected_ids, expected_values = topk(np, expected, k)
    top1_ok = bool(np.array_equal(actual_ids[..., :1], expected_ids[..., :1]))
    if require_order:
        topk_ok = bool(np.array_equal(actual_ids, expected_ids))
        label = f"top{k}_ids"
    else:
        actual_sets = [set(row.tolist()) for row in actual_ids.reshape(-1, actual_ids.shape[-1])]
        expected_sets = [set(row.tolist()) for row in expected_ids.reshape(-1, expected_ids.shape[-1])]
        topk_ok = actual_sets == expected_sets
        label = f"top{k}_id_set"
    print(f"[{'PASS' if top1_ok else 'FAIL'}] logits.top1 actual={actual_ids[..., :1].tolist()} expected={expected_ids[..., :1].tolist()}")
    print(f"[{'PASS' if topk_ok else 'FAIL'}] logits.{label} actual={actual_ids.tolist()} expected={expected_ids.tolist()}")
    if not topk_ok:
        print(f"[info] logits.top{k}_values actual={actual_values.tolist()} expected={expected_values.tolist()}")
    return top1_ok and topk_ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ncnn lm_head logits with PyTorch dump logits.")
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
    parser.add_argument("--sample", default="en_short")
    parser.add_argument("--param", default=str(DEFAULT_PARAM))
    parser.add_argument("--bin", default=str(DEFAULT_BIN))
    parser.add_argument("--hidden-key", default="decoder_decode_step0_last_hidden_state")
    parser.add_argument("--logits-key", default="decode_step_logits")
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--no-packing-layout", action="store_true")
    parser.add_argument("--atol", type=float, default=1e-1)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--require-topk-order", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ncnn, np = import_deps()

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

    arrays = np.load(npz_path)
    hidden = arrays[args.hidden_key].astype("float32", copy=False)
    expected = arrays[args.logits_key][args.step : args.step + 1].astype("float32", copy=False)
    actual = run_lm_head(
        ncnn,
        np,
        param_path=param_path,
        bin_path=bin_path,
        hidden=hidden,
        num_threads=args.num_threads,
        no_packing_layout=args.no_packing_layout,
    ).reshape(expected.shape)

    print(f"[data] sample={sample.get('sample_id')} npz={npz_path}")
    print(f"[ncnn] param={param_path} bin={bin_path}")
    ok = print_stats("lm_head.logits", numpy_diff_stats(np, actual, expected), args.atol)
    ok = compare_topk(np, actual, expected, args.topk, require_order=args.require_topk_order) and ok
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
