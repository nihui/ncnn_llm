#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# Rename the decoder graph's input/output blobs to the names the C++ runtime
# expects. pnnx names graph inputs `in0, in1, ...` and outputs `out0, out1, ...`
# in declaration order; we map them to:
#
#   inputs : in0, in1, in2, in3, cache_k0, cache_v0, cache_k1, cache_v1, ...
#   outputs: out0 (hidden), out_cache_k0, out_cache_v0, out_cache_k1, ...
#
# This edits an ncnn .param in place (blob names are plain tokens in the file),
# preserving everything else.
#
# Usage:
#   python tools/rename_decoder_blobs.py --param out_dir/decoder.ncnn.param --layers 28

import argparse
import re


def build_mapping(layers):
    inputs = {"in0": "in0", "in1": "in1", "in2": "in2", "in3": "in3"}
    idx = 4
    for i in range(layers):
        inputs[f"in{idx}"] = f"cache_k{i}"; idx += 1
        inputs[f"in{idx}"] = f"cache_v{i}"; idx += 1

    outputs = {"out0": "out0"}
    oidx = 1
    for i in range(layers):
        outputs[f"out{oidx}"] = f"out_cache_k{i}"; oidx += 1
        outputs[f"out{oidx}"] = f"out_cache_v{i}"; oidx += 1

    mapping = {}
    mapping.update(inputs)
    mapping.update(outputs)
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--param", required=True)
    ap.add_argument("--layers", type=int, required=True)
    args = ap.parse_args()

    mapping = build_mapping(args.layers)
    # Replace whole-token occurrences only (blob names are space-delimited tokens).
    token_re = re.compile(r"(?<!\S)(in\d+|out\d+)(?!\S)")

    with open(args.param, "r", encoding="utf-8") as f:
        lines = f.readlines()

    def repl(m):
        return mapping.get(m.group(1), m.group(1))

    out = [lines[0], lines[1]]  # magic + layer/blob counts header
    for line in lines[2:]:
        out.append(token_re.sub(repl, line))

    with open(args.param, "w", encoding="utf-8") as f:
        f.writelines(out)
    print(f"[ok] renamed {args.layers}-layer decoder blobs in {args.param}")


if __name__ == "__main__":
    main()
