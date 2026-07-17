#!/usr/bin/env python3
"""Export a dynamic-length Youtu-VL SigLIP2 encoder graph to ncnn.

The fixed vision export folds window indices, RoPE tables, and attention masks
into constants. Those constants depend on the image patch grid, so the resulting
ncnn graph only accepts the traced patch count. This exporter keeps the encoder
weights in ncnn and moves all grid-dependent tensors to runtime inputs:

  in0: window-ordered pixel_values, shape [seq_len, 768]
  in1: cos table, shape [seq_len, 1, 72]
  in2: sin table, shape [seq_len, 1, 72]
  in3: window attention mask, shape [1, seq_len, seq_len]
  in4: full attention mask, shape [1, seq_len, seq_len]

The C++ runner computes the dynamic tensors, runs this encoder, reverses the
window ordering, and feeds the result to the standalone merger ncnn graph.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from youtu_vl_reference import DEFAULT_MANIFEST, die, load_json, load_model_and_processor, tensor_to_numpy
from _youtu_vision_export_utils import (
    DEFAULT_INPUT_NPZ,
    DEFAULT_OUTPUT_DIR,
    diff_stats,
    pnnx_dtype_name,
    resolve_path,
    resolve_pnnx_path,
    run_pnnx,
    tensor_shape,
)


DEFAULT_STEM = "youtu_vl_siglip2_encoder_dynamic"
DEFAULT_POST_MERGER_STEM = "youtu_vl_post_merger_dynamic"
MASK_VALUE = -10000.0


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_dynamic_encoder_wrapper(torch):
    def rotate_half(x):
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    class DynamicSiglip2EncoderForNcnn(torch.nn.Module):
        def __init__(self, siglip2):
            super().__init__()
            self.embeddings = siglip2.vision_model.embeddings
            self.layers = siglip2.vision_model.encoder.layers
            self.num_layers = len(self.layers)

        @staticmethod
        def _apply_attention(attn, hidden_states, cos, sin, mask):
            seq_length = hidden_states.shape[0]
            q = attn.q_proj(hidden_states).reshape(seq_length, attn.num_heads, -1)
            k = attn.k_proj(hidden_states).reshape(seq_length, attn.num_heads, -1)
            v = attn.v_proj(hidden_states).reshape(seq_length, attn.num_heads, -1)

            qf = q.float()
            kf = k.float()
            q = ((qf * cos) + (rotate_half(qf) * sin)).type_as(q)
            k = ((kf * cos) + (rotate_half(kf) * sin)).type_as(k)

            q = q.transpose(0, 1)
            k = k.transpose(0, 1)
            v = v.transpose(0, 1)
            attn_weights = torch.matmul(q, k.transpose(1, 2)) / (attn.head_dim ** 0.5)
            attn_weights = attn_weights + mask.to(dtype=attn_weights.dtype, device=attn_weights.device)
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            out = torch.matmul(attn_weights, v)
            return attn.out_proj(out.transpose(0, 1).reshape(seq_length, -1))

        def forward(self, pixel_values, cos, sin, mask_window, mask_full):
            pixel_values = pixel_values.type(self.embeddings.patch_embedding.weight.dtype)
            hidden_states = self.embeddings(pixel_values)
            cos = cos.float()
            sin = sin.float()
            for layer_num, layer in enumerate(self.layers):
                residual = hidden_states
                normed = layer.layer_norm1(hidden_states)
                mask = mask_full if ((layer_num + 1) % 8 == 0 or layer_num == self.num_layers - 1) else mask_window
                hidden_states = residual + self._apply_attention(layer.self_attn, normed, cos, sin, mask)

                residual = hidden_states
                hidden_states = residual + layer.mlp(layer.layer_norm2(hidden_states))
            return hidden_states

    return DynamicSiglip2EncoderForNcnn


def make_post_merger_wrapper(torch):
    class DynamicPostMergerForNcnn(torch.nn.Module):
        def __init__(self, siglip2, merger):
            super().__init__()
            self.post_layernorm = siglip2.vision_model.post_layernorm
            self.merger = merger

        def forward(self, hidden_states):
            hidden_states = self.post_layernorm(hidden_states)
            dummy_spatial_shapes = torch.empty((1, 2), dtype=torch.long, device=hidden_states.device)
            return self.merger(hidden_states, dummy_spatial_shapes)

    return DynamicPostMergerForNcnn


def make_runtime_tensors(torch, siglip2, pixel_values, spatial_shapes, mask_value: float):
    encoder = siglip2.vision_model.encoder
    spatial_shapes = spatial_shapes.long()
    with torch.no_grad():
        rotary = encoder.rot_pos_emb(spatial_shapes)
        window_index, cu_window = encoder.get_window_index(spatial_shapes)
        cu_window = torch.tensor(cu_window, dtype=torch.int64, device=spatial_shapes.device)
        cu_window = torch.unique_consecutive(cu_window)
        cu_full = torch.repeat_interleave(spatial_shapes[:, 0] * spatial_shapes[:, 1], 1).cumsum(dim=0, dtype=torch.int64)
        cu_full = torch.nn.functional.pad(cu_full, (1, 0), value=0)

        seq_len = int(spatial_shapes[:, 0].mul(spatial_shapes[:, 1]).sum().item())
        groups = seq_len // encoder.spatial_merge_unit
        reverse_index = torch.argsort(window_index)

        pixel_values_windowed = pixel_values.reshape(groups, encoder.spatial_merge_unit, -1)
        pixel_values_windowed = pixel_values_windowed[window_index, :, :].reshape(seq_len, -1)

        rotary = rotary.reshape(groups, encoder.spatial_merge_unit, -1)
        rotary = rotary[window_index, :, :].reshape(seq_len, -1)
        emb = torch.cat((rotary, rotary), dim=-1)
        cos = emb.cos().unsqueeze(-2).float()
        sin = emb.sin().unsqueeze(-2).float()

        def mask_from_cu(cu):
            mask = torch.full((1, seq_len, seq_len), float(mask_value), dtype=torch.float32, device=spatial_shapes.device)
            for i in range(1, int(cu.numel())):
                start = int(cu[i - 1].item())
                end = int(cu[i].item())
                mask[..., start:end, start:end] = 0
            return mask

        return pixel_values_windowed, cos, sin, mask_from_cu(cu_window), mask_from_cu(cu_full), reverse_index


def patch_dynamic_param(param: Path, output: Path, traced_seq_len: int) -> None:
    patcher = Path(__file__).resolve().parent / "patch_ncnn_param.py"
    cmd = [
        sys.executable,
        str(patcher),
        "--dynamic-seq-len",
        str(traced_seq_len),
        str(param),
        str(output),
    ]
    print("[patch] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export dynamic-length Youtu-VL SigLIP2 encoder to ncnn.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--model", default="")
    parser.add_argument("--input-npz", default=str(DEFAULT_INPUT_NPZ))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16"], help="ncnn weight storage precision.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device-map", default="none")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--trace", action="store_true", help="Trace both the encoder and post-merger.")
    parser.add_argument("--check-trace", action="store_true")
    parser.add_argument("--run-pnnx", action="store_true", help="Convert both traced graphs with pnnx.")
    parser.add_argument("--pnnx", default="")
    parser.add_argument("--pnnx-arg", action="append", default=[])
    parser.add_argument("--patch-param", action="store_true", help="Patch both ncnn graphs for dynamic lengths.")
    parser.set_defaults(dtype="fp32")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_npz = resolve_path(args.input_npz)
    if not input_npz.exists():
        die(f"input npz not found: {input_npz}")
    output_dir = resolve_path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR / ("fp16" if args.precision == "fp16" else "")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = output_dir / DEFAULT_STEM

    import numpy as np
    import torch

    arrays = np.load(input_npz)
    pixel_values_np = arrays["pixel_values"][0].astype("float32", copy=False)
    spatial_shapes_np = arrays["spatial_shapes"].astype("int64", copy=False)
    traced_seq_len = int(pixel_values_np.shape[0])
    artifacts: dict[str, Any] = {}
    traced_info: dict[str, Any] | None = None
    model_path = ""
    transformers_version = ""

    post_merger_example = None
    if args.trace:
        manifest = load_json(resolve_path(args.manifest))
        np, torch, transformers, model_path, model, _processor, device, _pydensecrf_stubbed = load_model_and_processor(args, manifest)
        transformers_version = transformers.__version__
        pixel_values = torch.from_numpy(pixel_values_np).to(device=device)
        spatial_shapes = torch.from_numpy(spatial_shapes_np).long().to(device=device)
        runtime = make_runtime_tensors(torch, model.siglip2, pixel_values, spatial_shapes, MASK_VALUE)
        pixel_values_windowed, cos, sin, mask_window, mask_full, reverse_index = runtime
        with torch.inference_mode():
            EncoderWrapper = make_dynamic_encoder_wrapper(torch)
            encoder_wrapper = EncoderWrapper(model.siglip2).eval()
            hidden_window = encoder_wrapper(pixel_values_windowed, cos, sin, mask_window, mask_full)
            hidden = hidden_window.reshape(traced_seq_len // 4, 4, -1)[reverse_index, :, :].reshape(traced_seq_len, -1)
            PostMergerWrapper = make_post_merger_wrapper(torch)
            post_merger_wrapper = PostMergerWrapper(model.siglip2, model.merger).eval()
            image_embeds = post_merger_wrapper(hidden)
            post_merger_example = hidden
        # youtu_vl_reference.py writes the public boundary as
        # `image_embeddings`.  Accept the early prototype spelling as a
        # compatibility fallback for dumps made before that schema settled.
        expected_image_embeddings = (
            arrays["image_embeddings"] if "image_embeddings" in arrays.files else arrays["image_embeds"]
        )
        stats = diff_stats(np, tensor_to_numpy(np, torch, image_embeds), expected_image_embeddings)
        print(f"[torch] dynamic encoder + merger image_embeds max_abs_vs_dump={stats['max_abs']}")

        torchscript_path = output_dir / f"{DEFAULT_STEM}.pt"
        print(f"[trace] {torchscript_path}")
        with torch.inference_mode():
            traced = torch.jit.trace(
                encoder_wrapper,
                (pixel_values_windowed, cos, sin, mask_window, mask_full),
                check_trace=args.check_trace,
            )
            traced.save(str(torchscript_path))
        traced_info = {
            "torchscript": str(torchscript_path),
            "bytes": torchscript_path.stat().st_size,
            "example_input_shapes": [tensor_shape(t) for t in (pixel_values_windowed, cos, sin, mask_window, mask_full)],
            "example_input_dtypes": [str(t.dtype) for t in (pixel_values_windowed, cos, sin, mask_window, mask_full)],
            "example_input_pnnx_dtypes": [pnnx_dtype_name(torch, t) for t in (pixel_values_windowed, cos, sin, mask_window, mask_full)],
            "traced_seq_len": traced_seq_len,
        }
        artifacts["trace"] = traced_info

        assert post_merger_example is not None
        post_merger_path = output_dir / f"{DEFAULT_POST_MERGER_STEM}.pt"
        print(f"[trace] {post_merger_path}")
        with torch.inference_mode():
            traced_post = torch.jit.trace(post_merger_wrapper, (post_merger_example,), check_trace=args.check_trace)
            traced_post.save(str(post_merger_path))
        artifacts["post_merger_trace"] = {
            "torchscript": str(post_merger_path),
            "bytes": post_merger_path.stat().st_size,
            "example_input_shapes": [tensor_shape(post_merger_example)],
            "example_input_dtypes": [str(post_merger_example.dtype)],
            "example_input_pnnx_dtypes": [pnnx_dtype_name(torch, post_merger_example)],
        }
    elif args.run_pnnx:
        default_torchscript = output_dir / f"{DEFAULT_STEM}.pt"
        if not default_torchscript.exists():
            die("--run-pnnx requires --trace or an existing default encoder .pt")
        traced_info = {
            "torchscript": str(default_torchscript),
            "traced_seq_len": traced_seq_len,
        }
        artifacts["trace"] = traced_info

    if args.run_pnnx:
        assert traced_info is not None
        pnnx_path = resolve_pnnx_path(args.pnnx)
        if pnnx_path is None:
            die("pnnx executable not found; pass --pnnx")
        example_inputs = (
            torch.from_numpy(pixel_values_np),
            torch.zeros((traced_seq_len, 1, 72), dtype=torch.float32),
            torch.zeros((traced_seq_len, 1, 72), dtype=torch.float32),
            torch.zeros((1, traced_seq_len, traced_seq_len), dtype=torch.float32),
            torch.zeros((1, traced_seq_len, traced_seq_len), dtype=torch.float32),
        )
        pnnx_result = run_pnnx(
            pnnx_path=pnnx_path,
            torchscript_path=Path(traced_info["torchscript"]),
            example_inputs=example_inputs,
            output_prefix=output_prefix,
            extra_args=args.pnnx_arg,
            torch=torch,
            fp16=args.precision == "fp16",
        )
        artifacts["pnnx"] = pnnx_result
        print(f"[pnnx] returncode={pnnx_result['returncode']} produced={len(pnnx_result['produced'])}")
        if pnnx_result["stderr"]:
            print("[pnnx:stderr]")
            print(pnnx_result["stderr"])
        if not pnnx_result["passed"]:
            raise SystemExit(pnnx_result["returncode"] or 1)

    if args.run_pnnx:
        pnnx_path = resolve_pnnx_path(args.pnnx)
        assert pnnx_path is not None
        post_merger_path = output_dir / f"{DEFAULT_POST_MERGER_STEM}.pt"
        if not post_merger_path.exists():
            die("--run-pnnx requires --trace or an existing default post-merger .pt")
        example_input = torch.zeros((traced_seq_len, 1152), dtype=torch.float32)
        pnnx_result = run_pnnx(
            pnnx_path=pnnx_path,
            torchscript_path=post_merger_path,
            example_inputs=(example_input,),
            output_prefix=output_dir / DEFAULT_POST_MERGER_STEM,
            extra_args=args.pnnx_arg,
            torch=torch,
            fp16=args.precision == "fp16",
        )
        artifacts["post_merger_pnnx"] = pnnx_result
        print(f"[pnnx:post_merger] returncode={pnnx_result['returncode']} produced={len(pnnx_result['produced'])}")
        if pnnx_result["stderr"]:
            print("[pnnx:post_merger:stderr]")
            print(pnnx_result["stderr"])
        if not pnnx_result["passed"]:
            raise SystemExit(pnnx_result["returncode"] or 1)

    if args.patch_param:
        raw_param = output_dir / f"{DEFAULT_STEM}.ncnn.param"
        patched_param = output_dir / f"{DEFAULT_STEM}.ncnn.dynamic.patched.param"
        patch_dynamic_param(raw_param, patched_param, traced_seq_len)
        artifacts["patched_param"] = str(patched_param)

        raw_post_merger = output_dir / f"{DEFAULT_POST_MERGER_STEM}.ncnn.param"
        patched_post_merger = output_dir / f"{DEFAULT_POST_MERGER_STEM}.ncnn.dynamic.patched.param"
        temp_post_merger = output_dir / f"{DEFAULT_POST_MERGER_STEM}.ncnn.dynamic.seq.patched.param"
        patch_dynamic_param(raw_post_merger, temp_post_merger, traced_seq_len)
        patch_dynamic_param(temp_post_merger, patched_post_merger, traced_seq_len // 4)
        artifacts["patched_post_merger_param"] = str(patched_post_merger)

    manifest_path = output_dir / f"{DEFAULT_STEM}.manifest.json"
    write_json(
        manifest_path,
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "script": str(Path(__file__).resolve()),
            "input_npz": str(input_npz),
            "model": str(model_path),
            "runtime": {"python": sys.version, "torch": torch.__version__, "transformers": transformers_version},
            "traced_seq_len": traced_seq_len,
            "mask_value": MASK_VALUE,
            "precision": args.precision,
            "artifacts": artifacts,
        },
    )
    print(f"[write] {manifest_path}")


if __name__ == "__main__":
    main()
