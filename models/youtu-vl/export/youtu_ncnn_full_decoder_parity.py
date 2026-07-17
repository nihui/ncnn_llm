#!/usr/bin/env python3
"""Compare the single 40-layer ncnn decoder graph with saved decode-step dumps."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

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
from youtu_ncnn_decoder_stack_parity import print_topk_set, run_single_output_net
from youtu_ncnn_prefill_parity import default_final_norm_paths, default_lm_head_paths, make_rope


FULL_DECODER_STEM = "youtu_decoder_full_manual_kv_sdpa_ncnn"
FULL_DECODER_PREFILL_STEM = "youtu_decoder_full_prefill_manual_kv_sdpa_ncnn"


def import_deps():
    try:
        import ncnn
        import numpy as np
    except ModuleNotFoundError as exc:
        die(f"missing Python dependency `{exc.name}`; activate the project venv", code=3)
    return ncnn, np


def to_ncnn_mat(ncnn, np, value):
    return ncnn.Mat(np.ascontiguousarray(value.astype("float32", copy=False))).clone()


def default_decoder_paths(output_dir: Path, *, mode: str, seq_len: int) -> tuple[Path, Path]:
    if mode == "decode":
        stem = FULL_DECODER_STEM
        return output_dir / f"{stem}.ncnn.patched.param", output_dir / f"{stem}.ncnn.bin"
    elif mode == "prefill":
        return (
            output_dir / f"{FULL_DECODER_PREFILL_STEM}.dynamic.patched.param",
            output_dir / f"{FULL_DECODER_PREFILL_STEM}_s{seq_len}.ncnn.bin",
        )
    else:
        die(f"unsupported mode: {mode}")


def make_decode_mask(np, arrays):
    attention_mask = arrays["decoder_decode_step0_attention_mask"]
    mask = np.zeros(attention_mask.shape, dtype=np.float32)
    mask[attention_mask == 0] = np.finfo(np.float32).min
    return mask


def make_prefill_mask(np, torch, arrays, *, include_dummy_cache_column: bool = True):
    from youtu_decoder_wrapper_probe import prepare_flat_attention_mask

    inputs_embeds = arrays["decoder_prefill_inputs_embeds"].astype(np.float32, copy=False)
    attention_mask_2d = arrays["decoder_prefill_attention_mask"]
    mask = prepare_flat_attention_mask(
        torch,
        torch.from_numpy(attention_mask_2d),
        torch.from_numpy(inputs_embeds),
        torch.from_numpy(arrays["decoder_prefill_cache_position"]),
        "4d",
    ).numpy().astype(np.float32, copy=False)
    if include_dummy_cache_column:
        dummy = np.full(
            (inputs_embeds.shape[0], 1, inputs_embeds.shape[1], 1),
            np.finfo(np.float32).min,
            dtype=np.float32,
        )
        mask = np.concatenate([dummy, mask], axis=-1)
    return np.ascontiguousarray(mask[0])


def add_batch_if_needed(np, actual, expected):
    if actual.shape == expected.shape:
        return actual
    if actual.ndim + 1 == expected.ndim and expected.shape[0] == 1 and actual.shape == expected.shape[1:]:
        return actual.reshape(expected.shape)
    return actual


def crop_prefill_dummy_cache(actual, expected):
    actual_len = actual.shape[2] if actual.ndim == 4 else actual.shape[1] if actual.ndim == 3 else -1
    expected_len = expected.shape[2] if expected.ndim == 4 else expected.shape[1] if expected.ndim == 3 else -1
    if actual_len != expected_len + 1:
        return actual
    if actual.ndim == 4:
        return actual[:, :, 1:, :]
    if actual.ndim == 3:
        return actual[:, 1:, :]
    return actual


def run_full_decoder_net(
    ncnn,
    np,
    torch,
    *,
    mode: str,
    param_path: Path,
    bin_path: Path,
    arrays,
    layer_count: int,
    num_threads: int,
    no_packing_layout: bool,
    omit_prefill_cache_inputs: bool,
):
    if mode == "decode":
        inputs_embeds = arrays["decoder_decode_step0_inputs_embeds"].astype(np.float32, copy=False)
        mask = make_decode_mask(np, arrays)
        cos, sin = make_rope(np, arrays["decoder_decode_step0_cache_position"].astype(np.float32, copy=False))
    elif mode == "prefill":
        inputs_embeds = arrays["decoder_prefill_inputs_embeds"][0].astype(np.float32, copy=False)
        mask = make_prefill_mask(np, torch, arrays, include_dummy_cache_column=not omit_prefill_cache_inputs)
        cos, sin = make_rope(np, arrays["decoder_prefill_cache_position"].astype(np.float32, copy=False))
    else:
        die(f"unsupported mode: {mode}")

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
            ex.input("in0", to_ncnn_mat(ncnn, np, inputs_embeds))
            ex.input("in1", to_ncnn_mat(ncnn, np, mask))
            ex.input("in2", to_ncnn_mat(ncnn, np, cos))
            ex.input("in3", to_ncnn_mat(ncnn, np, sin))

            for layer_idx in range(layer_count):
                if mode == "prefill" and omit_prefill_cache_inputs:
                    continue
                if mode == "decode":
                    key = arrays[f"decoder_decode_step0_input_cache_layer{layer_idx:02d}_key"][0]
                    value = arrays[f"decoder_decode_step0_input_cache_layer{layer_idx:02d}_value"][0]
                else:
                    key = np.zeros((32, 1, 192), dtype=np.float32)
                    value = np.zeros((32, 1, 128), dtype=np.float32)
                ex.input(f"cache_k{layer_idx}", to_ncnn_mat(ncnn, np, key))
                ex.input(f"cache_v{layer_idx}", to_ncnn_mat(ncnn, np, value))

            ret, hidden = ex.extract("out0")
            if ret != 0:
                die(f"extract out0 failed for {param_path}: {ret}", code=5)
            caches = []
            for layer_idx in range(layer_count):
                ret, key = ex.extract(f"out_cache_k{layer_idx}")
                if ret != 0:
                    die(f"extract out_cache_k{layer_idx} failed for {param_path}: {ret}", code=5)
                ret, value = ex.extract(f"out_cache_v{layer_idx}")
                if ret != 0:
                    die(f"extract out_cache_v{layer_idx} failed for {param_path}: {ret}", code=5)
                caches.append((np.array(key, dtype=np.float32), np.array(value, dtype=np.float32)))
            return np.array(hidden, dtype=np.float32), caches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one full ncnn decoder graph and compare with dumps.")
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
    parser.add_argument("--sample", default="en_short")
    parser.add_argument("--mode", default="decode", choices=["decode", "prefill"])
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--param", default="")
    parser.add_argument("--bin", default="")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--no-packing-layout", action="store_true")
    parser.add_argument(
        "--omit-prefill-cache-inputs",
        action="store_true",
        help="Do not feed cache_k/cache_v in prefill mode, matching ncnn_llm native SDPA kv_cache runtime.",
    )
    parser.add_argument("--cache-atol", type=float, default=0.1)
    parser.add_argument("--final-hidden-atol", type=float, default=0.5)
    parser.add_argument("--logits-atol", type=float, default=1.0)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--check-cache", action="store_true")
    parser.add_argument("--check-head", action="store_true")
    parser.add_argument("--final-norm-param", default="")
    parser.add_argument("--final-norm-bin", default="")
    parser.add_argument("--lm-head-param", default="")
    parser.add_argument("--lm-head-bin", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ncnn, np = import_deps()
    try:
        import torch
    except ModuleNotFoundError as exc:
        die(f"missing Python dependency `{exc.name}`; activate the project venv", code=3)

    metadata_path = resolve_project_path(args.metadata)
    metadata: Mapping[str, Any] = load_json(metadata_path)
    sample = select_sample(metadata, args.sample)
    npz_path = resolve_npz_path(sample, metadata_path)
    if not npz_path.exists():
        die(f"npz not found: {npz_path}")
    arrays = np.load(npz_path)
    layer_count = int(metadata.get("model", {}).get("num_hidden_layers") or 0)
    if layer_count <= 0:
        die("metadata model.num_hidden_layers is missing or invalid")

    output_dir = resolve_project_path(args.output_dir)
    seq_len = int(arrays["decoder_prefill_inputs_embeds"].shape[1]) if args.mode == "prefill" else 1
    default_param, default_bin = default_decoder_paths(output_dir, mode=args.mode, seq_len=seq_len)
    param_path = resolve_project_path(args.param) if args.param else default_param
    bin_path = resolve_project_path(args.bin) if args.bin else default_bin
    if not param_path.exists():
        die(f"ncnn decoder param not found: {param_path}")
    if not bin_path.exists():
        die(f"ncnn decoder bin not found: {bin_path}")

    print(f"[data] sample={sample.get('sample_id')} npz={npz_path}")
    print(f"[load] decoder param={param_path}")
    print(f"[load] decoder bin={bin_path}")
    hidden, caches = run_full_decoder_net(
        ncnn,
        np,
        torch,
        mode=args.mode,
        param_path=param_path,
        bin_path=bin_path,
        arrays=arrays,
        layer_count=layer_count,
        num_threads=args.num_threads,
        no_packing_layout=args.no_packing_layout,
        omit_prefill_cache_inputs=args.omit_prefill_cache_inputs,
    )
    print(f"[decoder] hidden={list(hidden.shape)} cache_layers={len(caches)}")

    ok = True
    if args.check_cache:
        worst_name = ""
        worst_value = -1.0
        for layer_idx, (actual_key, actual_value) in enumerate(caches):
            if args.mode == "decode":
                expected_key = arrays[f"decoder_decode_step0_output_cache_layer{layer_idx:02d}_key"][0].astype(
                    np.float32, copy=False
                )
                expected_value = arrays[f"decoder_decode_step0_output_cache_layer{layer_idx:02d}_value"][0].astype(
                    np.float32, copy=False
                )
            else:
                expected_key = arrays[f"decoder_prefill_output_cache_layer{layer_idx:02d}_key"].astype(
                    np.float32, copy=False
                )
                expected_value = arrays[f"decoder_prefill_output_cache_layer{layer_idx:02d}_value"].astype(
                    np.float32, copy=False
                )
            for suffix, actual, expected in (
                ("key", actual_key, expected_key),
                ("value", actual_value, expected_value),
            ):
                if args.mode == "prefill":
                    actual = crop_prefill_dummy_cache(actual, expected)
                actual = add_batch_if_needed(np, actual, expected)
                stats = numpy_diff_stats(np, actual, expected)
                if float(stats["max_abs"]) > worst_value:
                    worst_value = float(stats["max_abs"])
                    worst_name = f"layer{layer_idx:02d}_{suffix}"
                if layer_idx < 2:
                    ok = print_stats(f"cache.layer{layer_idx:02d}.{suffix}", stats, args.cache_atol) and ok
        print(f"[{'PASS' if worst_value <= args.cache_atol else 'FAIL'}] cache_all max_abs={worst_value:.6g} worst={worst_name}")
        ok = (worst_value <= args.cache_atol) and ok

    if args.check_head:
        default_norm_param, default_norm_bin = default_final_norm_paths(output_dir)
        default_lm_param, default_lm_bin = default_lm_head_paths(output_dir)
        final_norm_param = resolve_project_path(args.final_norm_param) if args.final_norm_param else default_norm_param
        final_norm_bin = resolve_project_path(args.final_norm_bin) if args.final_norm_bin else default_norm_bin
        lm_head_param = resolve_project_path(args.lm_head_param) if args.lm_head_param else default_lm_param
        lm_head_bin = resolve_project_path(args.lm_head_bin) if args.lm_head_bin else default_lm_bin
        actual_hidden = add_batch_if_needed(
            np,
            hidden,
            arrays[
                "decoder_decode_step0_last_hidden_state"
                if args.mode == "decode"
                else "decoder_prefill_last_hidden_state"
            ],
        )
        final_norm_input = actual_hidden if args.mode == "decode" else actual_hidden[:, -1:, :]
        final_hidden = run_single_output_net(
            ncnn,
            np,
            param_path=final_norm_param,
            bin_path=final_norm_bin,
            input_value=final_norm_input,
            num_threads=args.num_threads,
            no_packing_layout=args.no_packing_layout,
        )
        if args.mode == "decode":
            expected_final_hidden = arrays["decoder_decode_step0_last_hidden_state"].astype(np.float32, copy=False)
        else:
            expected_final_hidden = arrays["prefill_last_hidden_state"][:, -1:, :].astype(np.float32, copy=False)
            final_hidden = final_hidden.reshape(expected_final_hidden.shape)
        ok = print_stats(
            "head.final_hidden_state",
            numpy_diff_stats(np, final_hidden, expected_final_hidden),
            args.final_hidden_atol,
        ) and ok
        logits = run_single_output_net(
            ncnn,
            np,
            param_path=lm_head_param,
            bin_path=lm_head_bin,
            input_value=final_hidden,
            num_threads=args.num_threads,
            no_packing_layout=args.no_packing_layout,
        )
        if args.mode == "decode":
            logits = logits.reshape(arrays["decode_step_logits"][0:1].shape)
            expected_logits = arrays["decode_step_logits"][0:1].astype(np.float32, copy=False)
            name = "head.logits_step0"
        else:
            logits = logits.reshape(1, -1)[:, : arrays["prefill_logits"].shape[-1]]
            expected_logits = arrays["prefill_logits"].astype(np.float32, copy=False)
            name = "head.prefill_logits"
        ok = print_stats(name, numpy_diff_stats(np, logits, expected_logits), args.logits_atol) and ok
        ok = print_topk_set(np, name, logits, expected_logits, args.topk) and ok

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
