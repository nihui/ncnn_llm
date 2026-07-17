#!/usr/bin/env python3
"""Compare the full ncnn prefill decoder graph with saved PyTorch dumps."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from youtu_decoder_wrapper_probe import (
    DEFAULT_METADATA,
    DEFAULT_OUTPUT_DIR,
    die,
    load_json,
    prepare_flat_attention_mask,
    print_stats,
    resolve_npz_path,
    resolve_project_path,
    select_sample,
)
from youtu_ncnn_decoder_parity import diff_stats as numpy_diff_stats
from youtu_ncnn_decoder_stack_parity import print_topk_set, run_single_output_net


def import_deps():
    try:
        import ncnn
        import numpy as np
        import torch
    except ModuleNotFoundError as exc:
        die(f"missing Python dependency `{exc.name}`; activate the project venv", code=3)
    return ncnn, np, torch


def default_prefill_paths(output_dir: Path) -> tuple[Path, Path]:
    return output_dir / "youtu_decoder_prefill.ncnn.param", output_dir / "youtu_decoder_prefill.ncnn.bin"


def default_manual_prefill_paths(output_dir: Path) -> tuple[Path, Path]:
    return output_dir / "youtu_decoder_prefill_manual.ncnn.patched.param", output_dir / "youtu_decoder_prefill_manual.ncnn.bin"


def default_final_norm_paths(output_dir: Path) -> tuple[Path, Path]:
    return output_dir / "youtu_decoder_final_norm.ncnn.param", output_dir / "youtu_decoder_final_norm.ncnn.bin"


def default_lm_head_paths(output_dir: Path) -> tuple[Path, Path]:
    return output_dir / "youtu_lm_head.ncnn.param", output_dir / "youtu_lm_head.ncnn.bin"


def to_ncnn_input(ncnn, np, value, *, squeeze_batch: bool):
    array = np.ascontiguousarray(value.astype("float32", copy=False))
    if squeeze_batch and array.ndim >= 1 and array.shape[0] == 1:
        array = np.ascontiguousarray(array[0])
    return ncnn.Mat(array).clone()


def add_batch_if_needed(np, actual, expected):
    if actual.shape == expected.shape:
        return actual
    if actual.ndim + 1 == expected.ndim and expected.shape[0] == 1 and actual.shape == expected.shape[1:]:
        return actual.reshape(expected.shape)
    return actual


def run_prefill_net(
    ncnn,
    np,
    *,
    param_path: Path,
    bin_path: Path,
    inputs: list[Any],
    squeeze_flags: list[bool],
    num_threads: int,
    no_packing_layout: bool,
):
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
            for index, value in enumerate(inputs):
                ex.input(f"in{index}", to_ncnn_input(ncnn, np, value, squeeze_batch=squeeze_flags[index]))
            outputs = []
            for index in range(81):
                ret, out = ex.extract(f"out{index}")
                if ret != 0:
                    die(f"extract out{index} failed for {param_path}: {ret}", code=5)
                outputs.append(np.array(out, dtype=np.float32))
            return outputs


def make_rope(np, positions, *, theta: float = 500000.0, dim: int = 64):
    positions = np.asarray(positions, dtype=np.float32).reshape(-1)
    half = dim // 2
    inv_freq = np.array([1.0 / (theta ** (float(i * 2) / float(dim))) for i in range(half)], dtype=np.float32)
    freqs = positions[:, None] * inv_freq[None, :]
    cos_half = np.cos(freqs).astype(np.float32, copy=False)
    sin_half = np.sin(freqs).astype(np.float32, copy=False)
    cos = np.concatenate([cos_half, cos_half], axis=-1).reshape(1, positions.shape[0], dim)
    sin = np.concatenate([sin_half, sin_half], axis=-1).reshape(1, positions.shape[0], dim)
    return cos, sin


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ncnn full prefill decoder graph and compare with dumps.")
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
    parser.add_argument("--sample", default="en_short")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--param", default="")
    parser.add_argument("--bin", default="")
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Run youtu_decoder_prefill_manual, which uses manual RoPE inputs cos/sin and a dummy masked cache slot.",
    )
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--no-packing-layout", action="store_true")
    parser.add_argument("--hidden-atol", type=float, default=1.0)
    parser.add_argument("--cache-atol", type=float, default=0.5)
    parser.add_argument("--check-ncnn-head", action="store_true")
    parser.add_argument("--final-norm-param", default="")
    parser.add_argument("--final-norm-bin", default="")
    parser.add_argument("--lm-head-param", default="")
    parser.add_argument("--lm-head-bin", default="")
    parser.add_argument("--ncnn-final-hidden-atol", type=float, default=0.2)
    parser.add_argument("--ncnn-logits-atol", type=float, default=1.0)
    parser.add_argument("--topk", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ncnn, np, torch = import_deps()

    metadata_path = resolve_project_path(args.metadata)
    metadata: Mapping[str, Any] = load_json(metadata_path)
    sample = select_sample(metadata, args.sample)
    npz_path = resolve_npz_path(sample, metadata_path)
    if not npz_path.exists():
        die(f"npz not found: {npz_path}")

    output_dir = resolve_project_path(args.output_dir)
    default_param, default_bin = default_manual_prefill_paths(output_dir) if args.manual else default_prefill_paths(output_dir)
    param_path = resolve_project_path(args.param) if args.param else default_param
    bin_path = resolve_project_path(args.bin) if args.bin else default_bin
    if not param_path.exists():
        die(f"ncnn prefill param not found: {param_path}")
    if not bin_path.exists():
        die(f"ncnn prefill bin not found: {bin_path}")

    arrays = np.load(npz_path)
    inputs_embeds = arrays["decoder_prefill_inputs_embeds"].astype(np.float32, copy=False)
    attention_mask_2d = arrays["decoder_prefill_attention_mask"]
    cache_position = arrays["decoder_prefill_cache_position"].astype(np.float32, copy=False)

    mask = prepare_flat_attention_mask(
        torch,
        torch.from_numpy(attention_mask_2d),
        torch.from_numpy(inputs_embeds),
        torch.from_numpy(arrays["decoder_prefill_cache_position"]),
        "4d",
    ).numpy().astype(np.float32, copy=False)
    inputs = [inputs_embeds, mask, cache_position]
    squeeze_flags = [True, False, False]
    if args.manual:
        dummy = np.full(
            (inputs_embeds.shape[0], 1, inputs_embeds.shape[1], 1),
            np.finfo(np.float32).min,
            dtype=np.float32,
        )
        mask = np.concatenate([dummy, mask], axis=-1)
        cos, sin = make_rope(np, arrays["decoder_prefill_cache_position"])
        inputs = [inputs_embeds, mask, cos, sin]
        # Keep cos/sin as 3D [1, seq, dim] mats.  Squeezing them to [seq, dim]
        # changes how ncnn broadcasts the RoPE tables inside the traced graph.
        squeeze_flags = [True, True, False, False]

    print(f"[data] sample={sample.get('sample_id')} npz={npz_path}")
    print(f"[load] param={param_path}")
    print(f"[load] bin={bin_path}")
    outputs = run_prefill_net(
        ncnn,
        np,
        param_path=param_path,
        bin_path=bin_path,
        inputs=inputs,
        squeeze_flags=squeeze_flags,
        num_threads=args.num_threads,
        no_packing_layout=args.no_packing_layout,
    )

    ok = True
    expected_hidden = arrays["decoder_prefill_last_hidden_state"].astype(np.float32, copy=False)
    actual_hidden = add_batch_if_needed(np, outputs[0], expected_hidden)
    ok = print_stats(
        "prefill.last_hidden_state",
        numpy_diff_stats(np, actual_hidden, expected_hidden),
        args.hidden_atol,
    ) and ok

    worst_cache = 0.0
    worst_name = ""
    for layer in range(40):
        for suffix, out_index in (("key", 1 + layer * 2), ("value", 2 + layer * 2)):
            name = f"decoder_prefill_output_cache_layer{layer:02d}_{suffix}"
            expected = arrays[name].astype(np.float32, copy=False)
            actual = add_batch_if_needed(np, outputs[out_index], expected)
            stats = numpy_diff_stats(np, actual, expected)
            if float(stats["max_abs"]) > worst_cache:
                worst_cache = float(stats["max_abs"])
                worst_name = name
            if layer < 2:
                ok = print_stats(f"prefill.cache.layer{layer:02d}.{suffix}", stats, args.cache_atol) and ok

    cache_ok = worst_cache <= args.cache_atol
    print(f"[{'PASS' if cache_ok else 'FAIL'}] prefill.cache_all max_abs={worst_cache:.6g} worst={worst_name}")
    ok = cache_ok and ok

    if args.check_ncnn_head:
        default_norm_param, default_norm_bin = default_final_norm_paths(output_dir)
        default_lm_param, default_lm_bin = default_lm_head_paths(output_dir)
        final_norm_param = resolve_project_path(args.final_norm_param) if args.final_norm_param else default_norm_param
        final_norm_bin = resolve_project_path(args.final_norm_bin) if args.final_norm_bin else default_norm_bin
        lm_head_param = resolve_project_path(args.lm_head_param) if args.lm_head_param else default_lm_param
        lm_head_bin = resolve_project_path(args.lm_head_bin) if args.lm_head_bin else default_lm_bin
        if args.manual:
            final_hidden = actual_hidden[:, -1:, :]
        else:
            final_hidden = run_single_output_net(
                ncnn,
                np,
                param_path=final_norm_param,
                bin_path=final_norm_bin,
                input_value=actual_hidden[:, -1:, :],
                num_threads=args.num_threads,
                no_packing_layout=args.no_packing_layout,
            ).reshape(1, 1, -1)
        expected_final_hidden = arrays["prefill_last_hidden_state"][:, -1:, :].astype(np.float32, copy=False)
        ok = print_stats(
            "ncnn_head.prefill_final_hidden",
            numpy_diff_stats(np, final_hidden, expected_final_hidden),
            args.ncnn_final_hidden_atol,
        ) and ok
        logits = run_single_output_net(
            ncnn,
            np,
            param_path=lm_head_param,
            bin_path=lm_head_bin,
            input_value=final_hidden,
            num_threads=args.num_threads,
            no_packing_layout=args.no_packing_layout,
        ).reshape(1, -1)
        logits = logits[:, : arrays["prefill_logits"].shape[-1]]
        expected_logits = arrays["prefill_logits"].astype(np.float32, copy=False)
        ok = print_stats(
            "ncnn_head.prefill_logits",
            numpy_diff_stats(np, logits, expected_logits),
            args.ncnn_logits_atol,
        ) and ok
        ok = print_topk_set(np, "ncnn_head.prefill_logits", logits, expected_logits, args.topk) and ok

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
