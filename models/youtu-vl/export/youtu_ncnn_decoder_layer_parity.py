#!/usr/bin/env python3
"""Compare one ncnn decoder layer with the PyTorch reference path."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from youtu_decoder_layer_probe import (
    expected_from_full_model,
    flatten_sdpa_attention_mask,
    layer_inputs,
    load_model,
    squeeze_batch_cache_for_ncnn,
)
from youtu_ncnn_decoder_parity import diff_stats as numpy_diff_stats
from youtu_decoder_wrapper_probe import (
    DEFAULT_METADATA,
    DEFAULT_OUTPUT_DIR,
    die,
    first_parameter_dtype,
    load_json,
    print_stats,
    resolve_npz_path,
    resolve_project_path,
    select_sample,
)


def import_ncnn_deps():
    try:
        import ncnn
        import numpy as np
        import torch
    except ModuleNotFoundError as exc:
        die(f"missing Python dependency `{exc.name}`; activate the project venv", code=3)
    return ncnn, np, torch


def tensor_to_numpy(tensor):
    return tensor.detach().cpu().float().contiguous().numpy()


def tensor_to_ncnn_mat(ncnn, np, tensor):
    array = tensor_to_numpy(tensor)
    if array.size == 0 and array.ndim == 4:
        # ncnn.Mat(np.empty(...)) loses the zero-length 4D shape in Python
        # bindings and becomes dims=0.  Preserve NCHW as ncnn w,h,d,c.
        return ncnn.Mat(int(array.shape[3]), int(array.shape[2]), int(array.shape[1]), int(array.shape[0]))
    return ncnn.Mat(np.ascontiguousarray(array)).clone()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare one ncnn decoder layer with PyTorch.")
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA), help="Path to text dump metadata.json.")
    parser.add_argument("--model", default="", help="Local HF model dir or model id. Defaults to metadata model.path_or_id.")
    parser.add_argument("--sample", default="en_short", help="Sample id in metadata.json.")
    parser.add_argument("--layer", type=int, default=0, help="Decoder layer index.")
    parser.add_argument("--param", default="", help="Path to patched ncnn param.")
    parser.add_argument("--bin", default="", help="Path to ncnn bin.")
    parser.add_argument("--mask-mode", default="4d", choices=["2d", "4d"])
    parser.add_argument("--float-cache-position", action="store_true")
    parser.add_argument(
        "--zero-cache",
        action="store_true",
        help="Use empty KV cache and cache_position=0. Matches *_zero_cache_manual_kv exports.",
    )
    parser.add_argument(
        "--dummy-cache",
        action="store_true",
        help="Use one masked all-zero KV slot. Matches *_dummy_cache_manual_kv exports.",
    )
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device-map", default="none")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--atol", type=float, default=None, help="Set both hidden/cache max absolute tolerances.")
    parser.add_argument("--hidden-atol", type=float, default=5e-1, help="Hidden-state max absolute tolerance.")
    parser.add_argument("--cache-atol", type=float, default=6e-2, help="KV-cache max absolute tolerance.")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--no-packing-layout", action="store_true", help="Disable ncnn packing layout optimization.")
    parser.add_argument(
        "--sdpa-flat-mask",
        action="store_true",
        help="Feed ncnn in1 as a 2D additive SDPA mask instead of the HF-style 4D mask.",
    )
    parser.add_argument(
        "--sdpa-ncnn-layout",
        action="store_true",
        help="Feed ncnn_llm-style 3D KV cache and compare 3D output cache.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.atol is not None:
        args.hidden_atol = args.atol
        args.cache_atol = args.atol

    ncnn, np, torch = import_ncnn_deps()
    metadata_path = resolve_project_path(args.metadata)
    metadata: Mapping[str, Any] = load_json(metadata_path)
    sample = select_sample(metadata, args.sample)
    npz_path = resolve_npz_path(sample, metadata_path)
    if not npz_path.exists():
        die(f"npz not found: {npz_path}")

    mode = "_dummy_cache" if args.dummy_cache else ("_zero_cache" if args.zero_cache else "")
    suffix = "manual_kv_sdpa_ncnn" if args.sdpa_ncnn_layout else "manual_kv"
    default_stem = f"youtu_decoder_layer{args.layer:02d}_decode_step0{mode}_{suffix}"
    param_arg = args.param or str(DEFAULT_OUTPUT_DIR / f"{default_stem}.ncnn.patched.param")
    bin_arg = args.bin or str(DEFAULT_OUTPUT_DIR / f"{default_stem}.ncnn.bin")
    param_path = resolve_project_path(param_arg)
    bin_path = resolve_project_path(bin_arg)
    if not param_path.exists():
        die(f"ncnn param not found: {param_path}")
    if not bin_path.exists():
        die(f"ncnn bin not found: {bin_path}")

    np_mod, torch_mod, transformers, DynamicCache, model, device, pydensecrf_stubbed = load_model(args, metadata)
    arrays = np_mod.load(npz_path)
    layer_count = int(metadata.get("model", {}).get("num_hidden_layers") or 0)
    model_dtype = first_parameter_dtype(model) or torch_mod.float32

    print(f"[runtime] torch={torch_mod.__version__} transformers={transformers.__version__} pydensecrf_stubbed={pydensecrf_stubbed}")
    print(f"[data] sample={sample.get('sample_id')} layer={args.layer} npz={npz_path}")
    print(f"[load] param={param_path}")
    print(f"[load] bin={bin_path}")

    inputs = layer_inputs(
        torch_mod, DynamicCache, model, arrays, args.layer, layer_count, device=device, dtype=model_dtype, args=args
    )
    expected = expected_from_full_model(
        torch_mod, DynamicCache, model, arrays, args.layer, layer_count, device=device, dtype=model_dtype, args=args
    )

    attention_mask = flatten_sdpa_attention_mask(inputs[1]) if (args.sdpa_flat_mask or args.sdpa_ncnn_layout) else inputs[1]
    past_key, past_value = inputs[3], inputs[4]
    if args.sdpa_ncnn_layout:
        past_key, past_value = squeeze_batch_cache_for_ncnn(past_key, past_value)
        expected = (
            expected[0],
            expected[1].squeeze(0).contiguous(),
            expected[2].squeeze(0).contiguous(),
        )

    # pnnx removes cache_position because this layer wrapper already receives cos/sin.
    ncnn_inputs = [inputs[0], attention_mask, past_key, past_value, inputs[5], inputs[6]]

    ok = True
    with ncnn.Net() as net:
        net.opt.use_vulkan_compute = False
        net.opt.num_threads = int(args.num_threads)
        if args.no_packing_layout:
            net.opt.use_packing_layout = False
        ret = net.load_param(str(param_path))
        if ret != 0:
            die(f"net.load_param failed: {ret}", code=4)
        ret = net.load_model(str(bin_path))
        if ret != 0:
            die(f"net.load_model failed: {ret}", code=4)

        with net.create_extractor() as ex:
            for index, value in enumerate(ncnn_inputs):
                ex.input(f"in{index}", tensor_to_ncnn_mat(ncnn, np, value))

            names = ["hidden_states", "cache_key", "cache_value"]
            for index, name in enumerate(names):
                ret, out = ex.extract(f"out{index}")
                if ret != 0:
                    die(f"extract out{index} failed: {ret}", code=5)
                actual = np.array(out)
                stats = numpy_diff_stats(np, actual, tensor_to_numpy(expected[index]))
                atol = args.hidden_atol if index == 0 else args.cache_atol
                ok = print_stats(f"out{index}.{name}", stats, atol) and ok
                if index in {1, 2} and actual.ndim == 4:
                    past_expected = tensor_to_numpy(inputs[3 if index == 1 else 4])
                    current_expected = tensor_to_numpy(expected[index])[:, :, past_expected.shape[2] :, :]
                    past_actual = actual[:, :, : past_expected.shape[2], :]
                    current_actual = actual[:, :, past_expected.shape[2] :, :]
                    ok = (
                        print_stats(
                            f"out{index}.{name}.past",
                            numpy_diff_stats(np, past_actual, past_expected),
                            args.cache_atol,
                        )
                        and ok
                    )
                    ok = (
                        print_stats(
                            f"out{index}.{name}.current",
                            numpy_diff_stats(np, current_actual, current_expected),
                            args.cache_atol,
                        )
                        and ok
                    )
                    if index == 1 and current_actual.shape[-1] == 192:
                        ok = (
                            print_stats(
                                "out1.cache_key.current.nope128",
                                numpy_diff_stats(np, current_actual[..., :128], current_expected[..., :128]),
                                args.cache_atol,
                            )
                            and ok
                        )
                        ok = (
                            print_stats(
                                "out1.cache_key.current.rope64",
                                numpy_diff_stats(np, current_actual[..., 128:], current_expected[..., 128:]),
                                args.cache_atol,
                            )
                            and ok
                        )
                if index in {1, 2} and actual.ndim == 3:
                    past_expected = tensor_to_numpy(past_key if index == 1 else past_value)
                    current_expected = tensor_to_numpy(expected[index])[:, past_expected.shape[1] :, :]
                    past_actual = actual[:, : past_expected.shape[1], :]
                    current_actual = actual[:, past_expected.shape[1] :, :]
                    ok = (
                        print_stats(
                            f"out{index}.{name}.past",
                            numpy_diff_stats(np, past_actual, past_expected),
                            args.cache_atol,
                        )
                        and ok
                    )
                    ok = (
                        print_stats(
                            f"out{index}.{name}.current",
                            numpy_diff_stats(np, current_actual, current_expected),
                            args.cache_atol,
                        )
                        and ok
                    )
                    if index == 1 and current_actual.shape[-1] == 192:
                        ok = (
                            print_stats(
                                "out1.cache_key.current.nope128",
                                numpy_diff_stats(np, current_actual[..., :128], current_expected[..., :128]),
                                args.cache_atol,
                            )
                            and ok
                        )
                        ok = (
                            print_stats(
                                "out1.cache_key.current.rope64",
                                numpy_diff_stats(np, current_actual[..., 128:], current_expected[..., 128:]),
                                args.cache_atol,
                            )
                            and ok
                        )

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
