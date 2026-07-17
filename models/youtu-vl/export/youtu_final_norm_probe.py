#!/usr/bin/env python3
"""Probe/export the Youtu decoder final RMSNorm."""

from __future__ import annotations

import argparse
from pathlib import Path

from youtu_decoder_layer_probe import load_model
from youtu_decoder_wrapper_probe import (
    DEFAULT_METADATA,
    DEFAULT_OUTPUT_DIR,
    die,
    first_parameter_dtype,
    load_json,
    print_stats,
    resolve_npz_path,
    resolve_pnnx_path,
    resolve_project_path,
    run_pnnx,
    select_sample,
    trace_wrapper,
)
from youtu_ncnn_decoder_parity import diff_stats as numpy_diff_stats
from youtu_ncnn_decoder_layer_parity import tensor_to_numpy
from youtu_ncnn_decoder_stack_parity import full_torch_expected


def make_final_norm_wrapper(torch):
    class _FinalNormWrapper(torch.nn.Module):
        def __init__(self, norm):
            super().__init__()
            self.norm = norm

        def forward(self, hidden_states):
            return self.norm(hidden_states)

    return _FinalNormWrapper


def run_ncnn(ncnn, np, *, param_path: Path, bin_path: Path, hidden, num_threads: int, no_packing_layout: bool):
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


def import_ncnn():
    try:
        import ncnn
        import numpy as np
    except ModuleNotFoundError as exc:
        die(f"missing Python dependency `{exc.name}`; activate the project venv", code=3)
    return ncnn, np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe/export the Youtu decoder final norm.")
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
    parser.add_argument("--model", default="")
    parser.add_argument("--sample", default="en_short")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16"], help="ncnn weight storage precision.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device-map", default="none")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--check-trace", action="store_true")
    parser.add_argument("--run-pnnx", action="store_true")
    parser.add_argument("--pnnx", default="")
    parser.add_argument("--pnnx-arg", action="append", default=[])
    parser.add_argument("--param", default="")
    parser.add_argument("--bin", default="")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--no-packing-layout", action="store_true")
    # Transformers 4.57 replays the captured DynamicCache path with a small
    # attention-order difference; the observed FP32 final-norm maximum is
    # 0.0707 while the downstream greedy token remains unchanged.
    parser.add_argument("--atol", type=float, default=8e-2)
    parser.add_argument("--ncnn-atol", type=float, default=3e-3)
    parser.set_defaults(mask_mode="4d", float_cache_position=False, dtype="fp32")
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

    np_mod, torch, transformers, DynamicCache, model, device, pydensecrf_stubbed = load_model(args, metadata)
    arrays = np_mod.load(npz_path)
    model_dtype = first_parameter_dtype(model) or torch.float32
    layer_count = int(metadata.get("model", {}).get("num_hidden_layers") or 0)
    if layer_count <= 0:
        die("metadata model.num_hidden_layers is missing or invalid")

    print(f"[runtime] torch={torch.__version__} transformers={transformers.__version__} pydensecrf_stubbed={pydensecrf_stubbed}")
    print(f"[probe] sample={sample.get('sample_id')} npz={npz_path}")
    print(f"[probe] device={device} model_dtype={model_dtype} layers={layer_count}")

    layer_hidden, _cache_layers = full_torch_expected(
        torch, DynamicCache, model, arrays, layer_count, device=device, dtype=model_dtype, args=args
    )
    Wrapper = make_final_norm_wrapper(torch)
    wrapper = Wrapper(model.model.norm).eval()
    with torch.inference_mode():
        final_hidden = wrapper(layer_hidden)

    expected_final_hidden = torch.from_numpy(arrays["decoder_decode_step0_last_hidden_state"]).to(
        device=device, dtype=torch.float32
    )
    ok = print_stats("final_norm.hidden_state", numpy_torch_diff(torch, final_hidden.float(), expected_final_hidden), args.atol)
    if not ok:
        raise SystemExit(1)

    output_dir = resolve_project_path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR / ("fp16" if args.precision == "fp16" else "")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = output_dir / "youtu_decoder_final_norm"
    if args.trace:
        info = trace_wrapper(torch, wrapper, (layer_hidden,), output_prefix.with_suffix(".pt"), args.check_trace)
        print(f"[trace] {info['path']} bytes={info['bytes']} outputs={info['output_count']}")
        if args.run_pnnx:
            pnnx_path = resolve_pnnx_path(args.pnnx)
            if pnnx_path is None:
                die("pnnx executable not found; pass --pnnx")
            result = run_pnnx(
                torch,
                pnnx_path=pnnx_path,
                torchscript_path=Path(info["path"]),
                example_inputs=(layer_hidden,),
                output_prefix=output_prefix,
                fp16=args.precision == "fp16",
                extra_args=args.pnnx_arg,
            )
            print(f"[pnnx] returncode={result['returncode']} produced={len(result['produced'])}")
            if result["stderr"]:
                print("[pnnx:stderr]")
                print(result["stderr"])
            if not result["passed"]:
                raise SystemExit(result["returncode"] or 1)

    param_path = resolve_project_path(args.param) if args.param else Path(f"{output_prefix}.ncnn.param")
    bin_path = resolve_project_path(args.bin) if args.bin else Path(f"{output_prefix}.ncnn.bin")
    if param_path.exists() and bin_path.exists():
        ncnn, np = import_ncnn()
        actual = run_ncnn(
            ncnn,
            np,
            param_path=param_path,
            bin_path=bin_path,
            hidden=tensor_to_numpy(layer_hidden),
            num_threads=args.num_threads,
            no_packing_layout=args.no_packing_layout,
        )
        ok = print_stats(
            "ncnn.final_norm.hidden_state",
            numpy_diff_stats(np, actual, tensor_to_numpy(final_hidden)),
            args.ncnn_atol,
        )
        if not ok:
            raise SystemExit(1)
    else:
        print(f"[note] ncnn final norm files not found: {param_path} / {bin_path}")


def numpy_torch_diff(torch, actual, expected):
    diff = (actual - expected).abs()
    return {
        "shape": list(actual.shape),
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "rms": float(torch.sqrt((diff * diff).mean()).item()),
    }


if __name__ == "__main__":
    main()
