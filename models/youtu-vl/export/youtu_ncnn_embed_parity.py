#!/usr/bin/env python3
"""Check ncnn embed_tokens against saved PyTorch input embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

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


DEFAULT_PARAM = DEFAULT_OUTPUT_DIR / "youtu_embed_tokens.ncnn.param"
DEFAULT_BIN = DEFAULT_OUTPUT_DIR / "youtu_embed_tokens.ncnn.bin"


def import_deps():
    try:
        import ncnn
        import numpy as np
    except ModuleNotFoundError as exc:
        die(f"missing Python dependency `{exc.name}`; activate the project venv", code=3)
    return ncnn, np


def run_embed(ncnn, np, *, param_path: Path, bin_path: Path, input_ids, num_threads: int, no_packing_layout: bool):
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
            ids = np.ascontiguousarray(input_ids.astype("int32", copy=False).reshape(-1))
            ex.input("in0", ncnn.Mat(ids).clone())
            ret, out = ex.extract("out0")
            if ret != 0:
                die(f"extract out0 failed for {param_path}: {ret}", code=5)
            return np.array(out, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ncnn embed_tokens with PyTorch dump embeddings.")
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
    parser.add_argument("--sample", default="en_short")
    parser.add_argument("--param", default=str(DEFAULT_PARAM))
    parser.add_argument("--bin", default=str(DEFAULT_BIN))
    parser.add_argument("--input-key", default="decoder_decode_step0_input_ids")
    parser.add_argument("--expected-key", default="decoder_decode_step0_inputs_embeds")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--no-packing-layout", action="store_true")
    parser.add_argument("--atol", type=float, default=1e-6)
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
    input_ids = arrays[args.input_key]
    expected = arrays[args.expected_key].astype("float32", copy=False)
    actual = run_embed(
        ncnn,
        np,
        param_path=param_path,
        bin_path=bin_path,
        input_ids=input_ids,
        num_threads=args.num_threads,
        no_packing_layout=args.no_packing_layout,
    ).reshape(expected.shape)

    print(f"[data] sample={sample.get('sample_id')} npz={npz_path}")
    print(f"[ncnn] param={param_path} bin={bin_path}")
    ok = print_stats("embed_tokens.outputs", numpy_diff_stats(np, actual, expected), args.atol)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
