#!/usr/bin/env python3
"""Run patched per-layer ncnn decoder exports as a sequential stack."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from youtu_decoder_layer_probe import layer_inputs, load_model
from youtu_ncnn_decoder_parity import diff_stats as numpy_diff_stats
from youtu_ncnn_decoder_layer_parity import tensor_to_numpy
from youtu_decoder_wrapper_probe import (
    DEFAULT_METADATA,
    DEFAULT_OUTPUT_DIR,
    cache_layers_from_npz,
    cache_to_layers,
    die,
    dynamic_cache_from_layers,
    first_parameter_dtype,
    load_json,
    prepare_flat_attention_mask,
    print_stats,
    resolve_npz_path,
    resolve_project_path,
    select_sample,
    tensor_from_np,
)


def import_deps():
    try:
        import ncnn
        import numpy as np
    except ModuleNotFoundError as exc:
        die(f"missing Python dependency `{exc.name}`; activate the project venv", code=3)
    return ncnn, np


def parse_range(raw: str, layer_count: int) -> tuple[int, int]:
    if "-" in raw:
        start_raw, end_raw = raw.split("-", 1)
        start, end = int(start_raw), int(end_raw)
    else:
        start = end = int(raw)
    if start < 0 or end < start or end >= layer_count:
        die(f"invalid layer range {raw!r}; allowed 0..{layer_count - 1}")
    return start, end


def layer_paths(layer_idx: int, output_dir: Path) -> tuple[Path, Path]:
    stem = f"youtu_decoder_layer{layer_idx:02d}_decode_step0_manual_kv"
    return output_dir / f"{stem}.ncnn.patched.param", output_dir / f"{stem}.ncnn.bin"


def default_final_norm_paths(output_dir: Path) -> tuple[Path, Path]:
    return output_dir / "youtu_decoder_final_norm.ncnn.param", output_dir / "youtu_decoder_final_norm.ncnn.bin"


def default_lm_head_paths(output_dir: Path) -> tuple[Path, Path]:
    return output_dir / "youtu_lm_head.ncnn.param", output_dir / "youtu_lm_head.ncnn.bin"


def run_layer(ncnn, np, *, param_path: Path, bin_path: Path, inputs: list[Any], num_threads: int, no_packing_layout: bool):
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
                ex.input(f"in{index}", ncnn.Mat(np.ascontiguousarray(value.astype("float32", copy=False))).clone())
            outputs = []
            for name in ("out0", "out1", "out2"):
                ret, out = ex.extract(name)
                if ret != 0:
                    die(f"extract {name} failed for {param_path}: {ret}", code=5)
                outputs.append(np.array(out, dtype=np.float32))
            return outputs


def run_single_output_net(ncnn, np, *, param_path: Path, bin_path: Path, input_value, num_threads: int, no_packing_layout: bool):
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
            ex.input("in0", ncnn.Mat(np.ascontiguousarray(input_value.astype("float32", copy=False))).clone())
            ret, out = ex.extract("out0")
            if ret != 0:
                die(f"extract out0 failed for {param_path}: {ret}", code=5)
            return np.array(out, dtype=np.float32)


def full_torch_expected(torch, DynamicCache, model, arrays, layer_count: int, *, device, dtype, args):
    inputs_embeds = tensor_from_np(torch, arrays, "decoder_decode_step0_inputs_embeds", device=device, dtype=dtype)
    attention_mask_2d = tensor_from_np(torch, arrays, "decoder_decode_step0_attention_mask", device=device)
    cache_position = tensor_from_np(torch, arrays, "decoder_decode_step0_cache_position", device=device)
    attention_mask = prepare_flat_attention_mask(torch, attention_mask_2d, inputs_embeds, cache_position, args.mask_mode)
    if args.float_cache_position:
        cache_position = cache_position.float()
    input_cache_layers = cache_layers_from_npz(
        torch, arrays, "decoder_decode_step0_input_cache", layer_count, device=device, dtype=dtype
    )
    past = dynamic_cache_from_layers(DynamicCache, input_cache_layers)
    captured: dict[str, Any] = {}

    def hook(_module, _module_inputs, module_output):
        captured["hidden"] = module_output[0].detach()

    handle = model.model.layers[layer_count - 1].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            _ = model.model(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past,
                cache_position=cache_position,
                use_cache=True,
                output_hidden_states=False,
                output_attentions=False,
            )
    finally:
        handle.remove()
    if "hidden" not in captured:
        die(f"layer {layer_count - 1} hook did not capture hidden output")
    return captured["hidden"], cache_to_layers(past)


def final_norm_and_logits(torch, np, model, hidden, *, device, dtype):
    hidden_tensor = torch.from_numpy(np.ascontiguousarray(hidden)).to(device=device, dtype=dtype)
    with torch.inference_mode():
        final_hidden = model.model.norm(hidden_tensor)
        logits = model.lm_head(final_hidden).float()
    return final_hidden.detach().cpu().numpy(), logits[:, -1, :].detach().cpu().numpy()


def topk(np, values, k: int) -> tuple[Any, Any]:
    k = min(int(k), int(values.shape[-1]))
    partition = np.argpartition(-values, kth=k - 1, axis=-1)[..., :k]
    partition_values = np.take_along_axis(values, partition, axis=-1)
    order = np.argsort(-partition_values, axis=-1)
    ids = np.take_along_axis(partition, order, axis=-1)
    top_values = np.take_along_axis(values, ids, axis=-1)
    return ids, top_values


def print_topk(np, name: str, actual, expected, k: int) -> bool:
    actual_ids, actual_values = topk(np, actual, k)
    expected_ids, expected_values = topk(np, expected, k)
    same = bool(np.array_equal(actual_ids, expected_ids))
    status = "PASS" if same else "FAIL"
    print(f"[{status}] {name}.top{k}_ids actual={actual_ids.tolist()} expected={expected_ids.tolist()}")
    if not same:
        print(f"[info] {name}.top{k}_values actual={actual_values.tolist()} expected={expected_values.tolist()}")
    return same


def print_topk_set(np, name: str, actual, expected, k: int) -> bool:
    actual_ids, actual_values = topk(np, actual, k)
    expected_ids, expected_values = topk(np, expected, k)
    top1_ok = bool(np.array_equal(actual_ids[..., :1], expected_ids[..., :1]))
    actual_sets = [set(row.tolist()) for row in actual_ids.reshape(-1, actual_ids.shape[-1])]
    expected_sets = [set(row.tolist()) for row in expected_ids.reshape(-1, expected_ids.shape[-1])]
    set_ok = actual_sets == expected_sets
    print(
        f"[{'PASS' if top1_ok else 'FAIL'}] {name}.top1 "
        f"actual={actual_ids[..., :1].tolist()} expected={expected_ids[..., :1].tolist()}"
    )
    print(
        f"[{'PASS' if set_ok else 'FAIL'}] {name}.top{k}_id_set "
        f"actual={actual_ids.tolist()} expected={expected_ids.tolist()}"
    )
    if not set_ok:
        print(f"[info] {name}.top{k}_values actual={actual_values.tolist()} expected={expected_values.tolist()}")
    return top1_ok and set_ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ncnn decoder layers sequentially and compare with dump tensors.")
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
    parser.add_argument("--model", default="")
    parser.add_argument("--sample", default="en_short")
    parser.add_argument("--layers", default="0-39", help="Sequential layer range, e.g. 0-3 or 0-39.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--mask-mode", default="4d", choices=["4d"])
    parser.add_argument("--float-cache-position", action="store_true")
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "float32"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device-map", default="none")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--no-packing-layout", action="store_true")
    parser.add_argument("--hidden-atol", type=float, default=5.0)
    parser.add_argument("--cache-atol", type=float, default=6e-2)
    parser.add_argument("--final-hidden-atol", type=float, default=5.0)
    parser.add_argument("--logits-atol", type=float, default=5.0)
    parser.add_argument("--ncnn-final-hidden-atol", type=float, default=0.1)
    parser.add_argument("--ncnn-logits-atol", type=float, default=0.4)
    parser.add_argument("--check-cache", action="store_true", help="Compare each updated layer cache with PyTorch dump.")
    parser.add_argument("--check-logits", action="store_true", help="Compare final norm + lm_head logits for full stack.")
    parser.add_argument("--check-ncnn-head", action="store_true", help="Run ncnn final norm + ncnn lm_head after the layer stack.")
    parser.add_argument("--final-norm-param", default="")
    parser.add_argument("--final-norm-bin", default="")
    parser.add_argument("--lm-head-param", default="")
    parser.add_argument("--lm-head-bin", default="")
    parser.add_argument("--topk", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ncnn, np = import_deps()

    metadata_path = resolve_project_path(args.metadata)
    metadata: Mapping[str, Any] = load_json(metadata_path)
    sample = select_sample(metadata, args.sample)
    npz_path = resolve_npz_path(sample, metadata_path)
    if not npz_path.exists():
        die(f"npz not found: {npz_path}")

    layer_count = int(metadata.get("model", {}).get("num_hidden_layers") or 0)
    if layer_count <= 0:
        die("metadata model.num_hidden_layers is missing or invalid")
    start_layer, end_layer = parse_range(args.layers, layer_count)
    if start_layer != 0:
        die("stack parity currently requires layers to start at 0")

    output_dir = resolve_project_path(args.output_dir)
    for layer_idx in range(start_layer, end_layer + 1):
        param_path, bin_path = layer_paths(layer_idx, output_dir)
        if not param_path.exists():
            die(f"patched ncnn param not found: {param_path}")
        if not bin_path.exists():
            die(f"ncnn bin not found: {bin_path}")
    if args.check_ncnn_head:
        default_norm_param, default_norm_bin = default_final_norm_paths(output_dir)
        default_lm_param, default_lm_bin = default_lm_head_paths(output_dir)
        final_norm_param = resolve_project_path(args.final_norm_param) if args.final_norm_param else default_norm_param
        final_norm_bin = resolve_project_path(args.final_norm_bin) if args.final_norm_bin else default_norm_bin
        lm_head_param = resolve_project_path(args.lm_head_param) if args.lm_head_param else default_lm_param
        lm_head_bin = resolve_project_path(args.lm_head_bin) if args.lm_head_bin else default_lm_bin
        for path in (final_norm_param, final_norm_bin, lm_head_param, lm_head_bin):
            if not path.exists():
                die(f"ncnn head file not found: {path}")

    np_mod, torch_mod, transformers, DynamicCache, model, device, pydensecrf_stubbed = load_model(args, metadata)
    arrays = np_mod.load(npz_path)
    model_dtype = first_parameter_dtype(model) or torch_mod.float32
    seed_inputs = layer_inputs(torch_mod, DynamicCache, model, arrays, 0, layer_count, device=device, dtype=model_dtype, args=args)
    expected_hidden, expected_cache_layers = full_torch_expected(
        torch_mod, DynamicCache, model, arrays, layer_count, device=device, dtype=model_dtype, args=args
    )

    hidden = tensor_to_numpy(seed_inputs[0])
    attention_mask = tensor_to_numpy(seed_inputs[1])
    cos = tensor_to_numpy(seed_inputs[5])
    sin = tensor_to_numpy(seed_inputs[6])
    caches: list[tuple[Any, Any]] = []
    for layer_idx in range(layer_count):
        caches.append(
            (
                arrays[f"decoder_decode_step0_input_cache_layer{layer_idx:02d}_key"].astype("float32", copy=False),
                arrays[f"decoder_decode_step0_input_cache_layer{layer_idx:02d}_value"].astype("float32", copy=False),
            )
        )

    print(f"[runtime] torch={torch_mod.__version__} transformers={transformers.__version__} pydensecrf_stubbed={pydensecrf_stubbed}")
    print(f"[data] sample={sample.get('sample_id')} layers={start_layer}-{end_layer} npz={npz_path}")

    ok = True
    for layer_idx in range(start_layer, end_layer + 1):
        param_path, bin_path = layer_paths(layer_idx, output_dir)
        past_key, past_value = caches[layer_idx]
        hidden, out_key, out_value = run_layer(
            ncnn,
            np,
            param_path=param_path,
            bin_path=bin_path,
            inputs=[hidden, attention_mask, past_key, past_value, cos, sin],
            num_threads=args.num_threads,
            no_packing_layout=args.no_packing_layout,
        )
        caches[layer_idx] = (out_key, out_value)
        print(f"[layer] {layer_idx:02d}: hidden={list(hidden.shape)} key={list(out_key.shape)} value={list(out_value.shape)}")

        if args.check_cache:
            expected_key = tensor_to_numpy(expected_cache_layers[layer_idx][0])
            expected_value = tensor_to_numpy(expected_cache_layers[layer_idx][1])
            ok = print_stats(
                f"layer{layer_idx:02d}.cache_key",
                numpy_diff_stats(np, out_key, expected_key),
                args.cache_atol,
            ) and ok
            ok = print_stats(
                f"layer{layer_idx:02d}.cache_value",
                numpy_diff_stats(np, out_value, expected_value),
                args.cache_atol,
            ) and ok

    if end_layer == layer_count - 1:
        ok = print_stats(
            "decoder.layer39_hidden_state",
            numpy_diff_stats(np, hidden, tensor_to_numpy(expected_hidden)),
            args.hidden_atol,
        ) and ok
        if args.check_logits:
            final_hidden, logits = final_norm_and_logits(torch_mod, np, model, hidden, device=device, dtype=model_dtype)
            expected_final_hidden = arrays["decoder_decode_step0_last_hidden_state"].astype("float32", copy=False)
            expected_logits = arrays["decode_step_logits"][0:1].astype("float32", copy=False)
            ok = print_stats(
                "decoder.final_hidden_state",
                numpy_diff_stats(np, final_hidden, expected_final_hidden),
                args.final_hidden_atol,
            ) and ok
            ok = print_stats(
                "decoder.logits_step0",
                numpy_diff_stats(np, logits, expected_logits),
                args.logits_atol,
            ) and ok
            ok = print_topk(np, "decoder.logits_step0", logits, expected_logits, args.topk) and ok
        if args.check_ncnn_head:
            expected_final_hidden = arrays["decoder_decode_step0_last_hidden_state"].astype("float32", copy=False)
            expected_logits = arrays["decode_step_logits"][0:1].astype("float32", copy=False)
            final_hidden = run_single_output_net(
                ncnn,
                np,
                param_path=final_norm_param,
                bin_path=final_norm_bin,
                input_value=hidden,
                num_threads=args.num_threads,
                no_packing_layout=args.no_packing_layout,
            )
            logits = run_single_output_net(
                ncnn,
                np,
                param_path=lm_head_param,
                bin_path=lm_head_bin,
                input_value=final_hidden,
                num_threads=args.num_threads,
                no_packing_layout=args.no_packing_layout,
            ).reshape(expected_logits.shape)
            ok = print_stats(
                "ncnn_head.final_hidden_state",
                numpy_diff_stats(np, final_hidden, expected_final_hidden),
                args.ncnn_final_hidden_atol,
            ) and ok
            ok = print_stats(
                "ncnn_head.logits_step0",
                numpy_diff_stats(np, logits, expected_logits),
                args.ncnn_logits_atol,
            ) and ok
            ok = print_topk_set(np, "ncnn_head.logits_step0", logits, expected_logits, args.topk) and ok
        for layer_idx, (actual_key, actual_value) in enumerate(caches):
            ok = print_stats(
                f"decoder.cache_layer{layer_idx:02d}_key",
                numpy_diff_stats(np, actual_key, tensor_to_numpy(expected_cache_layers[layer_idx][0])),
                args.cache_atol,
            ) and ok
            ok = print_stats(
                f"decoder.cache_layer{layer_idx:02d}_value",
                numpy_diff_stats(np, actual_value, tensor_to_numpy(expected_cache_layers[layer_idx][1])),
                args.cache_atol,
            ) and ok
    else:
        print("[note] final hidden comparison is only available when the range ends at the last decoder layer")

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
