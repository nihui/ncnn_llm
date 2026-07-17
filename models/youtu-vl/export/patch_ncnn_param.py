#!/usr/bin/env python3
"""Patch pnnx-generated ncnn param files for unsupported no-op custom layers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def die(message: str, code: int = 2) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def patch_noop_tensor_to(
    param_path: Path,
    output_path: Path,
    *,
    reshape_residual_adds: bool = False,
    reshape_all_binary_adds: bool = False,
    fix_cache_concat_axis: bool = False,
    fix_cache_concat_axis_3d: bool = False,
    fix_prefill_empty_cache_concat_axis: bool = False,
    rename_full_decoder_io: bool = False,
    rotary_param0: int | None = None,
    softmax_axis: int | None = None,
    residual_seq_len: int = 1,
    fix_sdpa_output_permute: bool = False,
    fuse_sdpa_kv_cache: bool = False,
    merge_kv_cache_input: bool = False,
    dynamic_seq_len: int | None = None,
) -> tuple[int, int, int, int]:
    lines = param_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        die(f"not a valid ncnn param file: {param_path}")

    try:
        layer_count, blob_count = [int(x) for x in lines[1].split()[:2]]
    except ValueError:
        die(f"invalid ncnn param header: {lines[1]!r}")

    rewrites: dict[str, str] = {}
    kept: list[str] = []
    removed = 0

    for line in lines[2:]:
        parts = line.split()
        if len(parts) >= 6 and parts[0] == "Tensor.to" and parts[2] == "1" and parts[3] == "1":
            rewrites[parts[5]] = parts[4]
            removed += 1
            continue
        kept.append(line)

    if removed == 0 and not (
        reshape_residual_adds
        or reshape_all_binary_adds
        or fix_prefill_empty_cache_concat_axis
        or fix_cache_concat_axis
        or fix_cache_concat_axis_3d
        or rename_full_decoder_io
        or rotary_param0 is not None
        or softmax_axis is not None
        or fix_sdpa_output_permute
        or fuse_sdpa_kv_cache
        or merge_kv_cache_input
        or dynamic_seq_len is not None
    ):
        die(f"no Tensor.to layer found in {param_path}", code=1)

    def is_3d_cache_input_blob(blob: str) -> bool:
        if blob.startswith("cache_k") or blob.startswith("cache_v"):
            return True
        if blob.startswith("in") and blob[2:].isdigit():
            return int(blob[2:]) >= 4
        return False

    patched: list[str] = []
    inserted = 0
    fixed_concat = 0
    replaced_addcmul = 0
    removed_blobs: set[str] = set()
    blob_renames: dict[str, str] = {}
    sdpa_output_blobs: set[str] = set()
    kv_concat_sources: dict[str, tuple[str, str]] = {}
    kv_split_sources: dict[str, tuple[str, str, str]] = {}
    if rename_full_decoder_io:
        for layer_idx in range(40):
            blob_renames[f"in{4 + layer_idx * 2}"] = f"cache_k{layer_idx}"
            blob_renames[f"in{5 + layer_idx * 2}"] = f"cache_v{layer_idx}"
            blob_renames[f"out{1 + layer_idx * 2}"] = f"out_cache_k{layer_idx}"
            blob_renames[f"out{2 + layer_idx * 2}"] = f"out_cache_v{layer_idx}"

    def rewrite_param(parts: list[str], key: str, old_value: int, new_value: int) -> bool:
        needle = f"{key}={old_value}"
        replacement = f"{key}={new_value}"
        for index, part in enumerate(parts):
            if part == needle:
                parts[index] = replacement
                return True
        return False

    for line in kept:
        parts = line.split()
        if len(parts) < 4:
            patched.append(line)
            continue

        if parts[0] == "pnnx.Expression" and len(parts) >= 5:
            output_count = int(parts[3])
            output_start = 4
            removed_blobs.update(parts[output_start : output_start + output_count])
            removed += 1
            continue

        if parts[0] == "Split" and len(parts) >= 5 and parts[4] in removed_blobs:
            output_count = int(parts[3])
            output_start = 5
            removed_blobs.update(parts[output_start : output_start + output_count])
            removed += 1
            continue
        input_count = int(parts[2])
        input_start = 4
        input_end = input_start + input_count
        for index in range(input_start, min(input_end, len(parts))):
            while parts[index] in rewrites:
                parts[index] = rewrites[parts[index]]

        if (
            fix_cache_concat_axis
            and len(parts) >= 8
            and parts[0] == "Concat"
            and (parts[1] in {"cat_2", "cat_3"} or "in2" in parts[4:6] or "in3" in parts[4:6])
        ):
            for index, part in enumerate(parts):
                if part == "0=1":
                    parts[index] = "0=2"
                    fixed_concat += 1

        if (
            fix_cache_concat_axis_3d
            and len(parts) >= 8
            and parts[0] == "Concat"
            and any(is_3d_cache_input_blob(blob) for blob in parts[4:6])
        ):
            for index, part in enumerate(parts):
                if part.startswith("0="):
                    parts[index] = "0=1"
                    fixed_concat += 1

        if (
            fuse_sdpa_kv_cache
            and len(parts) >= 8
            and parts[0] == "Concat"
            and int(parts[2]) == 2
            and int(parts[3]) == 1
            and any(is_3d_cache_input_blob(blob) for blob in parts[4:6])
        ):
            cache_blob, current_blob = parts[4], parts[5]
            if not is_3d_cache_input_blob(cache_blob):
                current_blob, cache_blob = cache_blob, current_blob
            output_blob = parts[6]
            kv_concat_sources[output_blob] = (cache_blob, current_blob)
            removed += 1
            continue

        if fuse_sdpa_kv_cache and parts[0] == "Split" and len(parts) >= 7 and parts[4] in kv_concat_sources:
            output_count = int(parts[3])
            outputs = parts[5 : 5 + output_count]
            if output_count == 2:
                out_cache_blob = ""
                sdpa_input_blob = ""
                for blob in outputs:
                    renamed = blob_renames.get(blob, blob)
                    if renamed.startswith("out_cache_") or blob.startswith("out"):
                        out_cache_blob = blob
                    else:
                        sdpa_input_blob = blob
                if out_cache_blob and sdpa_input_blob:
                    cache_blob, current_blob = kv_concat_sources[parts[4]]
                    kv_split_sources[sdpa_input_blob] = (cache_blob, current_blob, out_cache_blob)
                    removed += 1
                    continue

        if fix_prefill_empty_cache_concat_axis and len(parts) >= 8 and parts[0] == "Concat":
            name = parts[1]
            if name.startswith("cat_"):
                try:
                    cat_index = int(name.split("_", 1)[1])
                except ValueError:
                    cat_index = -1
                if cat_index >= 3 and (cat_index - 3) % 4 in {0, 1}:
                    input_count = int(parts[2])
                    output_count = int(parts[3])
                    input_start = 4
                    output_start = input_start + input_count
                    if input_count == 2 and output_count == 1 and len(parts) > output_start:
                        rewrites[parts[output_start]] = parts[input_start + 1]
                        fixed_concat += 1
                        removed += 1
                        continue

        if rotary_param0 is not None and len(parts) >= 7 and parts[0] == "RotaryEmbed":
            replaced = False
            for index, part in enumerate(parts):
                if part.startswith("0="):
                    parts[index] = f"0={rotary_param0}"
                    replaced = True
            if not replaced:
                parts.append(f"0={rotary_param0}")

        if softmax_axis is not None and parts[0] == "Softmax":
            replaced = False
            for index, part in enumerate(parts):
                if part.startswith("0="):
                    parts[index] = f"0={softmax_axis}"
                    replaced = True
            if not replaced:
                parts.append(f"0={softmax_axis}")

        if fix_sdpa_output_permute and parts[0] == "SDPA" and len(parts) >= 6:
            output_count = int(parts[3])
            output_start = 4 + int(parts[2])
            sdpa_output_blobs.update(parts[output_start : output_start + output_count])

        if (
            fuse_sdpa_kv_cache
            and parts[0] == "SDPA"
            and len(parts) >= 9
            and int(parts[2]) == 4
            and int(parts[3]) == 1
            and parts[5] in kv_split_sources
            and parts[6] in kv_split_sources
        ):
            key_cache, key_current, key_out_cache = kv_split_sources[parts[5]]
            value_cache, value_current, value_out_cache = kv_split_sources[parts[6]]
            inputs = [parts[4], key_current, value_current, parts[7], key_cache, value_cache]
            outputs = [parts[8], key_out_cache, value_out_cache]
            params = parts[9:]
            params = [param for param in params if not param.startswith("7=")]
            params.append("7=1")
            parts = [parts[0], parts[1], "6", "3", *inputs, *outputs, *params]
            fixed_concat += 1
            if fix_sdpa_output_permute:
                sdpa_output_blobs.add(outputs[0])

        if (
            fix_sdpa_output_permute
            and parts[0] == "Permute"
            and len(parts) >= 7
            and parts[4] in sdpa_output_blobs
        ):
            replaced = False
            for index, part in enumerate(parts):
                if part.startswith("0="):
                    if part != "0=2":
                        fixed_concat += 1
                    parts[index] = "0=2"
                    replaced = True
            if not replaced:
                parts.append("0=2")
                fixed_concat += 1

        if dynamic_seq_len is not None:
            if parts[0] == "Gemm" and rewrite_param(parts, "7", dynamic_seq_len, 0):
                fixed_concat += 1
            elif parts[0] == "Reshape":
                changed = False
                changed = rewrite_param(parts, "2", dynamic_seq_len, -1) or changed
                changed = rewrite_param(parts, "1", dynamic_seq_len, -1) or changed
                changed = rewrite_param(parts, "11", dynamic_seq_len, -1) or changed
                if changed:
                    fixed_concat += 1

        if len(parts) >= 9 and parts[0] == "aten::addcmul" and parts[2] == "4" and parts[3] == "1":
            mul_blob = f"{parts[1]}_mul"
            patched.append(f"BinaryOp {parts[1]}_mul 2 1 {parts[5]} {parts[6]} {mul_blob} 0=2")
            patched.append(f"BinaryOp {parts[1]}_add 2 1 {parts[4]} {mul_blob} {parts[8]} 0=0")
            inserted += 1
            replaced_addcmul += 1
            continue

        if (reshape_residual_adds or reshape_all_binary_adds) and len(parts) >= 7 and parts[0] == "BinaryOp" and parts[2] == "2" and parts[3] == "1":
            rhs = parts[5]
            should_reshape_add = parts[1] in {"add_2", "add_4", "add_6", "add_8", "add_17", "add_24"}
            should_reshape_add = should_reshape_add or (reshape_all_binary_adds and parts[1].startswith("add_"))
            if should_reshape_add:
                reshaped = f"{rhs}_as3d"
                if residual_seq_len == 1:
                    reshape_params = "0=2560 1=1 2=1"
                else:
                    reshape_params = f"0=2560 1={residual_seq_len}"
                patched.append(f"Reshape youtu_residual_reshape_{inserted} 1 1 {rhs} {reshaped} {reshape_params}")
                parts[5] = reshaped
                inserted += 1
        if blob_renames:
            parts = [blob_renames.get(part, part) for part in parts]
        patched.append(" ".join(parts))

    if merge_kv_cache_input:
        cache_blobs = [name for layer_idx in range(40) for name in (f"cache_k{layer_idx}", f"cache_v{layer_idx}")]
        cache_blob_set = set(cache_blobs)
        merged: list[str] = []
        removed_cache_inputs = 0
        inserted_kv_input = False
        for line in patched:
            parts = line.split()
            is_cache_input = (
                len(parts) == 5
                and parts[0] == "Input"
                and parts[2] == "0"
                and parts[3] == "1"
                and parts[4] in cache_blob_set
            )
            if is_cache_input:
                removed_cache_inputs += 1
                continue
            if not inserted_kv_input and parts and parts[0] not in {"Input", "Split"}:
                merged.append("Input kv_cache 0 80 " + " ".join(cache_blobs))
                inserted_kv_input = True
            merged.append(line)
        if removed_cache_inputs:
            if not inserted_kv_input:
                merged.append("Input kv_cache 0 80 " + " ".join(cache_blobs))
            patched = merged
            removed += removed_cache_inputs - 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join([lines[0], f"{layer_count - removed + inserted} {blob_count - removed + inserted}", *patched]) + "\n"
    )
    return removed, inserted, fixed_concat, replaced_addcmul


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove no-op Tensor.to layers from a pnnx ncnn param file.")
    parser.add_argument("param", help="Input .ncnn.param path.")
    parser.add_argument("output", help="Patched .ncnn.param output path.")
    parser.add_argument(
        "--reshape-residual-adds",
        action="store_true",
        help="Insert 3D reshapes before Youtu decoder layer residual adds.",
    )
    parser.add_argument(
        "--reshape-all-binary-adds",
        action="store_true",
        help="Insert 3D reshapes before every BinaryOp add_* layer. Useful for full decoder graphs.",
    )
    parser.add_argument(
        "--fix-cache-concat-axis",
        action="store_true",
        help="Patch layer0 cache concat axes from ncnn axis 1 to axis 2.",
    )
    parser.add_argument(
        "--fix-cache-concat-axis-3d",
        action="store_true",
        help="Patch ncnn_llm-style 3D KV cache concat axes to sequence axis 1.",
    )
    parser.add_argument(
        "--fix-prefill-empty-cache-concat-axis",
        action="store_true",
        help="Patch full prefill empty-cache concat axes from -1 to ncnn axis 2.",
    )
    parser.add_argument("--rotary-param0", type=int, default=None, help="Override RotaryEmbed 0= parameter.")
    parser.add_argument("--softmax-axis", type=int, default=None, help="Override Softmax 0= axis.")
    parser.add_argument(
        "--residual-seq-len",
        type=int,
        default=1,
        help="Sequence length used by inserted residual-add reshapes. Use 1 for decode and bucket length for prefill.",
    )
    parser.add_argument(
        "--rename-full-decoder-io",
        action="store_true",
        help="Rename full decoder cache input/output blobs to cache_kN/cache_vN and out_cache_kN/out_cache_vN.",
    )
    parser.add_argument(
        "--fix-sdpa-output-permute",
        action="store_true",
        help="Patch native SDPA output permutes from head-major flattening to sequence-major flattening for prefill graphs.",
    )
    parser.add_argument(
        "--fuse-sdpa-kv-cache",
        action="store_true",
        help="Fuse explicit cache Concat/Split nodes into native SDPA kv_cache inputs/outputs.",
    )
    parser.add_argument(
        "--merge-kv-cache-input",
        action="store_true",
        help="Merge cache_k/cache_v Input layers into one ncnn_llm-style multi-output Input named kv_cache.",
    )
    parser.add_argument(
        "--dynamic-seq-len",
        type=int,
        default=None,
        help="Replace this traced prefill sequence length with ncnn dynamic dimensions in Gemm/Reshape layers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    param_path = Path(args.param).resolve()
    output_path = Path(args.output).resolve()
    if not param_path.exists():
        die(f"param file not found: {param_path}")
    removed, inserted, fixed_concat, replaced_addcmul = patch_noop_tensor_to(
        param_path,
        output_path,
        reshape_residual_adds=args.reshape_residual_adds,
        reshape_all_binary_adds=args.reshape_all_binary_adds,
        fix_cache_concat_axis=args.fix_cache_concat_axis,
        fix_cache_concat_axis_3d=args.fix_cache_concat_axis_3d,
        fix_prefill_empty_cache_concat_axis=args.fix_prefill_empty_cache_concat_axis,
        rename_full_decoder_io=args.rename_full_decoder_io,
        rotary_param0=args.rotary_param0,
        softmax_axis=args.softmax_axis,
        residual_seq_len=args.residual_seq_len,
        fix_sdpa_output_permute=args.fix_sdpa_output_permute,
        fuse_sdpa_kv_cache=args.fuse_sdpa_kv_cache,
        merge_kv_cache_input=args.merge_kv_cache_input,
        dynamic_seq_len=args.dynamic_seq_len,
    )
    print(f"[patch] removed Tensor.to layers: {removed}")
    print(f"[patch] inserted helper layers: {inserted}")
    print(f"[patch] fixed cache concat axes: {fixed_concat}")
    print(f"[patch] replaced aten::addcmul layers: {replaced_addcmul}")
    print(f"[patch] wrote: {output_path}")


if __name__ == "__main__":
    main()
