#!/usr/bin/env python3
"""Probe/export one 40-layer ncnn_llm-style decoder graph."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping

from youtu_decoder_layer_probe import (
    flatten_sdpa_attention_mask,
    load_model,
    make_layer_wrapper,
    squeeze_batch_cache_for_ncnn,
)
from youtu_decoder_wrapper_probe import (
    DEFAULT_METADATA,
    DEFAULT_OUTPUT_DIR,
    cache_layers_from_npz,
    cache_to_layers,
    die,
    diff_stats,
    dynamic_cache_from_layers,
    first_parameter_dtype,
    load_json,
    prepare_flat_attention_mask,
    print_cache_summary,
    print_stats,
    resolve_npz_path,
    resolve_project_path,
    run_pnnx,
    select_sample,
    tensor_from_np,
    trace_wrapper,
)


FULL_DECODER_STEM = "youtu_decoder_full_manual_kv_sdpa_ncnn"
FULL_DECODER_PREFILL_STEM = "youtu_decoder_full_prefill_manual_kv_sdpa_ncnn"


def build_full_decoder_wrapper(torch, DynamicCache, model, layer_count: int, *, drop_first_cache_token: bool = False):
    layers = [
        make_layer_wrapper(
            torch,
            DynamicCache,
            layer_idx,
            manual_kv_b_split=True,
            manual_sdpa=True,
            manual_sdpa_ncnn_layout=True,
        )(model.model.layers[layer_idx])
        for layer_idx in range(layer_count)
    ]

    class FullDecoderWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList(layers)

        def forward(self, hidden_states, attention_mask, cos, sin, *flat_cache):
            out_cache = []
            cache_position = hidden_states.new_zeros((hidden_states.shape[1],))
            for layer_idx, layer in enumerate(self.layers):
                past_key = flat_cache[layer_idx * 2]
                past_value = flat_cache[layer_idx * 2 + 1]
                hidden_states, key, value = layer(
                    hidden_states,
                    attention_mask,
                    cache_position,
                    past_key,
                    past_value,
                    cos,
                    sin,
                )
                if drop_first_cache_token:
                    key = key[:, 1:, :]
                    value = value[:, 1:, :]
                out_cache.extend([key, value])
            return (hidden_states, *out_cache)

    return FullDecoderWrapper().eval()


def full_torch_expected_decode(torch, DynamicCache, model, arrays, layer_count: int, *, device, dtype, args):
    inputs_embeds = tensor_from_np(torch, arrays, "decoder_decode_step0_inputs_embeds", device=device, dtype=dtype)
    attention_mask_2d = tensor_from_np(torch, arrays, "decoder_decode_step0_attention_mask", device=device)
    cache_position = tensor_from_np(torch, arrays, "decoder_decode_step0_cache_position", device=device)
    attention_mask = prepare_flat_attention_mask(torch, attention_mask_2d, inputs_embeds, cache_position, "4d")
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


def full_torch_expected_prefill(torch, DynamicCache, model, arrays, layer_count: int, *, device, dtype, args):
    inputs_embeds = tensor_from_np(torch, arrays, "decoder_prefill_inputs_embeds", device=device, dtype=dtype)
    attention_mask_2d = tensor_from_np(torch, arrays, "decoder_prefill_attention_mask", device=device)
    cache_position = tensor_from_np(torch, arrays, "decoder_prefill_cache_position", device=device)
    attention_mask = prepare_flat_attention_mask(torch, attention_mask_2d, inputs_embeds, cache_position, "4d")
    past = DynamicCache()
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


def make_decode_example_inputs(torch, model, arrays, layer_count: int, *, device, dtype, args):
    inputs_embeds = tensor_from_np(torch, arrays, "decoder_decode_step0_inputs_embeds", device=device, dtype=dtype)
    attention_mask_2d = tensor_from_np(torch, arrays, "decoder_decode_step0_attention_mask", device=device)
    cache_position = tensor_from_np(torch, arrays, "decoder_decode_step0_cache_position", device=device)
    attention_mask = prepare_flat_attention_mask(torch, attention_mask_2d, inputs_embeds, cache_position, "4d")
    attention_mask = flatten_sdpa_attention_mask(attention_mask)
    with torch.inference_mode():
        cos, sin = model.model.rotary_emb(inputs_embeds, cache_position.unsqueeze(0))

    input_cache_layers = cache_layers_from_npz(
        torch, arrays, "decoder_decode_step0_input_cache", layer_count, device=device, dtype=dtype
    )
    flat_cache = []
    for key, value in input_cache_layers:
        key, value = squeeze_batch_cache_for_ncnn(key, value)
        flat_cache.extend([key, value])
    return (inputs_embeds, attention_mask, cos, sin, *flat_cache)


def make_prefill_example_inputs(torch, model, arrays, layer_count: int, *, device, dtype, args):
    del args
    inputs_embeds = tensor_from_np(torch, arrays, "decoder_prefill_inputs_embeds", device=device, dtype=dtype)
    attention_mask_2d = tensor_from_np(torch, arrays, "decoder_prefill_attention_mask", device=device)
    cache_position = tensor_from_np(torch, arrays, "decoder_prefill_cache_position", device=device)
    base_mask = prepare_flat_attention_mask(torch, attention_mask_2d, inputs_embeds, cache_position, "4d")
    dummy = torch.full(
        (inputs_embeds.shape[0], 1, inputs_embeds.shape[1], 1),
        torch.finfo(base_mask.dtype).min,
        dtype=base_mask.dtype,
        device=base_mask.device,
    )
    attention_mask = torch.cat([dummy, base_mask], dim=-1)[0].contiguous()
    with torch.inference_mode():
        cos, sin = model.model.rotary_emb(inputs_embeds, cache_position.unsqueeze(0))

    flat_cache = []
    for _ in range(layer_count):
        key = inputs_embeds.new_zeros((32, 1, 192))
        value = inputs_embeds.new_zeros((32, 1, 128))
        flat_cache.extend([key, value])
    return (inputs_embeds, attention_mask, cos, sin, *flat_cache)


def make_example_inputs(torch, model, arrays, layer_count: int, *, device, dtype, args):
    if args.mode == "decode":
        return make_decode_example_inputs(torch, model, arrays, layer_count, device=device, dtype=dtype, args=args)
    if args.mode == "prefill":
        return make_prefill_example_inputs(torch, model, arrays, layer_count, device=device, dtype=dtype, args=args)
    die(f"unsupported mode: {args.mode}")


def full_torch_expected(torch, DynamicCache, model, arrays, layer_count: int, *, device, dtype, args):
    if args.mode == "decode":
        return full_torch_expected_decode(
            torch, DynamicCache, model, arrays, layer_count, device=device, dtype=dtype, args=args
        )
    if args.mode == "prefill":
        return full_torch_expected_prefill(
            torch, DynamicCache, model, arrays, layer_count, device=device, dtype=dtype, args=args
        )
    die(f"unsupported mode: {args.mode}")


def summarize_cache(torch, outputs, expected_cache_layers, *, sample_layers: int, cache_start: int = 0):
    worst_name = ""
    worst_max = -1.0
    samples = []
    for layer_idx, (expected_key, expected_value) in enumerate(expected_cache_layers):
        expected_key = expected_key.squeeze(0).contiguous()
        expected_value = expected_value.squeeze(0).contiguous()
        actual_key = outputs[1 + layer_idx * 2]
        actual_value = outputs[2 + layer_idx * 2]
        if cache_start:
            actual_key = actual_key[:, cache_start:, :]
            actual_value = actual_value[:, cache_start:, :]
        for suffix, actual, expected in (
            ("key", actual_key, expected_key),
            ("value", actual_value, expected_value),
        ):
            stats = diff_stats(torch, actual, expected)
            if float(stats["max_abs"]) > worst_max:
                worst_max = float(stats["max_abs"])
                worst_name = f"layer{layer_idx:02d}_{suffix}"
            if layer_idx < sample_layers:
                samples.append({"name": f"layer{layer_idx:02d}_{suffix}", **stats})
    return {
        "layer_count": len(expected_cache_layers),
        "expected_layer_count": len(expected_cache_layers),
        "max_abs": worst_max,
        "worst": worst_name,
        "samples": samples,
    }


def patch_full_decoder_param(
    python_path: Path,
    raw_param: Path,
    patched_param: Path,
    *,
    mode: str = "decode",
    seq_len: int | None = None,
) -> None:
    cmd = [
        str(python_path),
        str(Path(__file__).resolve().parent / "patch_ncnn_param.py"),
        "--fix-cache-concat-axis-3d",
        "--rename-full-decoder-io",
        "--fuse-sdpa-kv-cache",
        "--merge-kv-cache-input",
    ]
    if mode == "decode":
        cmd.append("--reshape-all-binary-adds")
    elif mode == "prefill":
        cmd.append("--fix-sdpa-output-permute")
        if seq_len is None:
            die("prefill patching requires seq_len for dynamic ncnn dimensions")
        cmd.extend(["--dynamic-seq-len", str(seq_len)])
    else:
        die(f"unsupported patch mode: {mode}")
    cmd.extend([str(raw_param), str(patched_param)])
    print("[patch] " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[1].parent), check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe/export one 40-layer ncnn_llm-style decoder graph.")
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
    parser.add_argument("--model", default="")
    parser.add_argument("--sample", default="en_short")
    parser.add_argument("--mode", default="decode", choices=["decode", "prefill"])
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16"], help="ncnn weight storage precision.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device-map", default="none")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--atol", type=float, default=5.0)
    parser.add_argument("--cache-atol", type=float, default=0.1)
    parser.add_argument("--cache-sample-layers", type=int, default=2)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--torchscript", default="", help="Reuse an existing precision-neutral .pt when running pnnx.")
    parser.add_argument("--run-pnnx", action="store_true")
    parser.add_argument("--patch-param", action="store_true")
    parser.add_argument("--pnnx", default="")
    parser.add_argument("--pnnx-arg", action="append", default=[])
    parser.add_argument("--check-trace", action="store_true")
    parser.set_defaults(dtype="fp32")
    return parser.parse_args()


def output_stem(args, example_inputs) -> str:
    if args.mode == "decode":
        return FULL_DECODER_STEM
    seq_len = int(example_inputs[0].shape[1])
    return f"{FULL_DECODER_PREFILL_STEM}_s{seq_len}"


def main() -> None:
    args = parse_args()
    if args.patch_param and not args.run_pnnx:
        die("--patch-param requires --run-pnnx")

    metadata_path = resolve_project_path(args.metadata)
    metadata: Mapping[str, Any] = load_json(metadata_path)
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

    wrapper = build_full_decoder_wrapper(
        torch,
        DynamicCache,
        model,
        layer_count,
        drop_first_cache_token=False,
    )
    example_inputs = make_example_inputs(torch, model, arrays, layer_count, device=device, dtype=model_dtype, args=args)
    expected_hidden, expected_cache = full_torch_expected(
        torch, DynamicCache, model, arrays, layer_count, device=device, dtype=model_dtype, args=args
    )

    with torch.inference_mode():
        outputs = wrapper(*example_inputs)

    print(f"[runtime] torch={torch.__version__} transformers={transformers.__version__} pydensecrf_stubbed={pydensecrf_stubbed}")
    print(f"[probe] sample={sample.get('sample_id')} layers={layer_count} npz={npz_path}")
    probe_name = f"full_decoder.{args.mode}"
    ok = print_stats(f"{probe_name}.layer39_hidden_state", diff_stats(torch, outputs[0], expected_hidden), args.atol)
    ok = print_cache_summary(
        f"{probe_name}.output_cache",
        summarize_cache(
            torch,
            outputs,
            expected_cache,
            sample_layers=args.cache_sample_layers,
            cache_start=1 if args.mode == "prefill" else 0,
        ),
        args.cache_atol,
    ) and ok
    if not ok:
        raise SystemExit(1)

    output_dir = resolve_project_path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR / ("fp16" if args.precision == "fp16" else "")
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(args, example_inputs)
    torchscript_path = resolve_project_path(args.torchscript) if args.torchscript else DEFAULT_OUTPUT_DIR / f"{stem}.pt"
    if args.trace:
        output_path = output_dir / f"{stem}.pt"
        info = trace_wrapper(torch, wrapper, example_inputs, output_path, args.check_trace)
        print(f"[trace] {info['path']} bytes={info['bytes']} outputs={info['output_count']}")
        torchscript_path = Path(info["path"])
    if args.run_pnnx:
        if not torchscript_path.exists():
            die(f"TorchScript not found: {torchscript_path}; pass --trace or --torchscript")
        pnnx_path = resolve_project_path(args.pnnx) if args.pnnx else None
        if pnnx_path is None or not pnnx_path.exists():
            found = shutil.which("pnnx")
            if not found:
                die("pnnx executable not found; pass --pnnx")
            pnnx_path = Path(found)
        result = run_pnnx(
            torch,
            pnnx_path=pnnx_path,
            torchscript_path=torchscript_path,
            example_inputs=example_inputs,
            output_prefix=output_dir / stem,
            fp16=args.precision == "fp16",
            extra_args=args.pnnx_arg,
        )
        print(f"[pnnx] returncode={result['returncode']} produced={len(result['produced'])}")
        if result["stderr"]:
            print("[pnnx:stderr]")
            print(result["stderr"])
        if not result["passed"]:
            raise SystemExit(result["returncode"] or 1)
        if args.patch_param:
            patched_param = output_dir / f"{stem}.ncnn.patched.param"
            seq_len = None
            if args.mode == "prefill":
                patched_param = output_dir / f"{FULL_DECODER_PREFILL_STEM}.dynamic.patched.param"
                seq_len = int(example_inputs[0].shape[1])
            patch_full_decoder_param(
                Path(__import__("sys").executable),
                output_dir / f"{stem}.ncnn.param",
                patched_param,
                mode=args.mode,
                seq_len=seq_len,
            )


if __name__ == "__main__":
    main()
