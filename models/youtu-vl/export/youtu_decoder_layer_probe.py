#!/usr/bin/env python3
"""Probe and export a single Youtu-VL decoder layer.

The full 40-layer decoder can be traced and converted, but it is too large for
comfortable ncnn debugging.  This script starts with one layer, using the same
decode-step tensors captured by youtu_text_reference.py.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from youtu_decoder_wrapper_probe import (
    DEFAULT_METADATA,
    DEFAULT_OUTPUT_DIR,
    cache_layers_from_npz,
    cache_to_layers,
    diff_stats,
    die,
    dynamic_cache_from_layers,
    first_parameter_device,
    first_parameter_dtype,
    import_runtime_deps,
    install_pydensecrf_import_stub,
    load_json,
    parse_dtype,
    prepare_flat_attention_mask,
    print_cache_summary,
    print_stats,
    resolve_model_arg,
    resolve_npz_path,
    resolve_project_path,
    resolve_torch_device,
    run_pnnx,
    select_sample,
    tensor_from_np,
    trace_wrapper,
)


class LayerDecodeWrapper:
    """Lazily materialized after torch import."""


def make_layer_wrapper(
    torch,
    DynamicCache,
    layer_idx: int,
    *,
    manual_kv_b_split: bool = False,
    manual_sdpa: bool = False,
    manual_sdpa_ncnn_layout: bool = False,
):
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def interleave_rope(x):
        batch_size, head_count, seq_length, head_dim = x.shape
        return x.view(batch_size, head_count, seq_length, head_dim // 2, 2).transpose(4, 3).reshape(
            batch_size, head_count, seq_length, head_dim
        )

    def sdpa_mask_slice(attention_mask, key_length):
        if attention_mask is None:
            return None
        if attention_mask.dim() == 4:
            return attention_mask[:, :, :, :key_length]
        if attention_mask.dim() == 2:
            return attention_mask[:, :key_length]
        return attention_mask

    class _LayerDecodeWrapper(torch.nn.Module):
        def __init__(self, decoder_layer):
            super().__init__()
            self.decoder_layer = decoder_layer
            self.manual_kv_b_split = manual_kv_b_split
            self.manual_sdpa = manual_sdpa
            self.manual_sdpa_ncnn_layout = manual_sdpa_ncnn_layout

            if self.manual_kv_b_split:
                attn = decoder_layer.self_attn
                kv_a_weight = attn.kv_a_proj_with_mqa.weight.detach()
                self.kv_a_latent_proj = torch.nn.Linear(
                    attn.config.hidden_size,
                    attn.kv_lora_rank,
                    bias=attn.kv_a_proj_with_mqa.bias is not None,
                )
                self.kv_a_rope_proj = torch.nn.Linear(
                    attn.config.hidden_size,
                    attn.qk_rope_head_dim,
                    bias=attn.kv_a_proj_with_mqa.bias is not None,
                )
                self.kv_a_latent_proj.weight = torch.nn.Parameter(
                    kv_a_weight[: attn.kv_lora_rank, :].contiguous(),
                    requires_grad=False,
                )
                self.kv_a_rope_proj.weight = torch.nn.Parameter(
                    kv_a_weight[attn.kv_lora_rank :, :].contiguous(),
                    requires_grad=False,
                )
                if attn.kv_a_proj_with_mqa.bias is not None:
                    kv_a_bias = attn.kv_a_proj_with_mqa.bias.detach()
                    self.kv_a_latent_proj.bias = torch.nn.Parameter(
                        kv_a_bias[: attn.kv_lora_rank].contiguous(),
                        requires_grad=False,
                    )
                    self.kv_a_rope_proj.bias = torch.nn.Parameter(
                        kv_a_bias[attn.kv_lora_rank :].contiguous(),
                        requires_grad=False,
                    )

                kv_b_weight = attn.kv_b_proj.weight.detach().view(
                    attn.num_heads,
                    attn.qk_nope_head_dim + attn.v_head_dim,
                    attn.kv_lora_rank,
                )
                key_weight = kv_b_weight[:, : attn.qk_nope_head_dim, :].reshape(
                    attn.num_heads * attn.qk_nope_head_dim,
                    attn.kv_lora_rank,
                )
                value_weight = kv_b_weight[:, attn.qk_nope_head_dim :, :].reshape(
                    attn.num_heads * attn.v_head_dim,
                    attn.kv_lora_rank,
                )

                self.kv_b_key_proj = torch.nn.Linear(attn.kv_lora_rank, attn.num_heads * attn.qk_nope_head_dim, bias=False)
                self.kv_b_value_proj = torch.nn.Linear(attn.kv_lora_rank, attn.num_heads * attn.v_head_dim, bias=False)
                self.kv_b_key_proj.weight = torch.nn.Parameter(key_weight.contiguous(), requires_grad=False)
                self.kv_b_value_proj.weight = torch.nn.Parameter(value_weight.contiguous(), requires_grad=False)

        def forward_manual_kv_b_split(self, hidden_states, attention_mask, cache_position, past_key, past_value, cos, sin):
            del cache_position

            decoder_layer = self.decoder_layer
            attn = decoder_layer.self_attn
            residual = hidden_states

            hidden_states = decoder_layer.input_layernorm(hidden_states)
            batch_size, seq_length = hidden_states.shape[:-1]

            query_shape = (batch_size, seq_length, attn.num_heads, attn.qk_head_dim)
            q_states = attn.q_b_proj(attn.q_a_layernorm(attn.q_a_proj(hidden_states))).view(query_shape).transpose(1, 2)
            q_pass, q_rot = torch.split(q_states, [attn.qk_nope_head_dim, attn.qk_rope_head_dim], dim=-1)

            k_pass = self.kv_a_latent_proj(hidden_states)
            k_rot = self.kv_a_rope_proj(hidden_states)
            kv_latent = attn.kv_a_layernorm(k_pass)

            k_pass = self.kv_b_key_proj(kv_latent).view(
                batch_size, seq_length, attn.num_heads, attn.qk_nope_head_dim
            ).transpose(1, 2)
            value_states = self.kv_b_value_proj(kv_latent)
            value_states = value_states.view(batch_size, seq_length, attn.num_heads, attn.v_head_dim).transpose(1, 2)

            k_rot = k_rot.view(batch_size, 1, seq_length, attn.qk_rope_head_dim)
            if attn.config.rope_interleave:
                q_rot = interleave_rope(q_rot)
                k_rot = interleave_rope(k_rot)
            q_rot = torch.addcmul(q_rot * cos, rotate_half(q_rot), sin)
            k_rot = torch.addcmul(k_rot * cos, rotate_half(k_rot), sin)
            k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

            query_states = torch.cat((q_pass, q_rot), dim=-1)
            key_states = torch.cat((k_pass, k_rot), dim=-1)
            if self.manual_sdpa and self.manual_sdpa_ncnn_layout:
                if batch_size != 1:
                    raise RuntimeError("manual_sdpa_ncnn_layout expects batch_size=1")
                query_states_3d = query_states.squeeze(0)
                current_key_3d = key_states.squeeze(0)
                current_value_3d = value_states.squeeze(0)
                past_key_3d = past_key.squeeze(0) if past_key.dim() == 4 else past_key
                past_value_3d = past_value.squeeze(0) if past_value.dim() == 4 else past_value
                key_states_3d = torch.cat((past_key_3d, current_key_3d), dim=1)
                value_states_3d = torch.cat((past_value_3d, current_value_3d), dim=1)
                sdpa_mask = sdpa_mask_slice(attention_mask, key_states_3d.shape[-2])
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query_states_3d,
                    key_states_3d,
                    value_states_3d,
                    attn_mask=sdpa_mask,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=attn.scaling,
                )
                attn_output = attn_output.transpose(0, 1).contiguous()
                attn_output = attn_output.reshape(batch_size, seq_length, -1)
                attn_output = attn.o_proj(attn_output).view(batch_size, seq_length, -1)

                hidden_states = residual + attn_output
                residual = hidden_states
                hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
                hidden_states = decoder_layer.mlp(hidden_states).view(batch_size, seq_length, -1)
                hidden_states = residual + hidden_states
                return hidden_states, key_states_3d, value_states_3d

            key_states = torch.cat((past_key, key_states), dim=2)
            value_states = torch.cat((past_value, value_states), dim=2)

            if self.manual_sdpa:
                sdpa_mask = sdpa_mask_slice(attention_mask, key_states.shape[-2])
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query_states,
                    key_states,
                    value_states,
                    attn_mask=sdpa_mask,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=attn.scaling,
                )
            else:
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * attn.scaling
                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(batch_size, seq_length, -1)
            attn_output = attn.o_proj(attn_output).view(batch_size, seq_length, -1)

            hidden_states = residual + attn_output
            residual = hidden_states
            hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
            hidden_states = decoder_layer.mlp(hidden_states).view(batch_size, seq_length, -1)
            hidden_states = residual + hidden_states
            return hidden_states, key_states, value_states

        def forward(self, hidden_states, attention_mask, cache_position, past_key, past_value, cos, sin):
            if self.manual_kv_b_split:
                return self.forward_manual_kv_b_split(
                    hidden_states, attention_mask, cache_position, past_key, past_value, cos, sin
                )
            past = dynamic_cache_from_layers(DynamicCache, [(past_key, past_value)])
            outputs = self.decoder_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_value=past,
                output_attentions=False,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=(cos, sin),
            )
            cache_layers = cache_to_layers(past)
            return outputs[0], cache_layers[0][0], cache_layers[0][1]

    return _LayerDecodeWrapper


def flatten_sdpa_attention_mask(attention_mask):
    if attention_mask is None:
        return None
    if attention_mask.dim() == 4:
        if attention_mask.shape[0] != 1 or attention_mask.shape[1] != 1:
            die(f"cannot flatten SDPA mask with shape {tuple(attention_mask.shape)}")
        return attention_mask[0, 0].contiguous()
    if attention_mask.dim() == 2:
        return attention_mask.contiguous()
    die(f"cannot flatten SDPA mask with shape {tuple(attention_mask.shape)}")


def squeeze_batch_cache_for_ncnn(past_key, past_value):
    if past_key.dim() == 4 and past_key.shape[0] == 1:
        past_key = past_key.squeeze(0).contiguous()
    if past_value.dim() == 4 and past_value.shape[0] == 1:
        past_value = past_value.squeeze(0).contiguous()
    return past_key, past_value


def load_model(args, metadata: Mapping[str, Any]):
    np, torch, transformers, AutoModelForCausalLM, DynamicCache = import_runtime_deps()
    pydensecrf_stubbed = install_pydensecrf_import_stub()

    model_path = resolve_model_arg(args, metadata)
    dtype = parse_dtype(torch, args.dtype)
    kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": dtype,
    }
    if args.attn_implementation != "none":
        kwargs["attn_implementation"] = args.attn_implementation
    if args.device_map != "none":
        kwargs["device_map"] = args.device_map

    print(f"[load] model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(str(model_path), **kwargs).eval()
    if args.device_map == "none":
        device = resolve_torch_device(torch, args.device)
        print(f"[load] moving model to {device}")
        model = model.to(device)
    else:
        device = first_parameter_device(model)

    return np, torch, transformers, DynamicCache, model, device, pydensecrf_stubbed


def expected_from_full_model(torch, DynamicCache, model, arrays, layer_idx: int, layer_count: int, *, device, dtype, args):
    inputs_embeds = tensor_from_np(torch, arrays, "decoder_decode_step0_inputs_embeds", device=device, dtype=dtype)
    attention_mask, cache_position, input_cache_layers = make_layer_attention_and_cache(
        torch, arrays, inputs_embeds, layer_count, device=device, dtype=dtype, args=args
    )
    past = dynamic_cache_from_layers(DynamicCache, input_cache_layers)

    captured: Dict[str, Any] = {}

    def hook(_module, _module_inputs, module_output):
        captured["hidden"] = module_output[0].detach()

    handle = model.model.layers[layer_idx].register_forward_hook(hook)
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
        die(f"layer {layer_idx} hook did not capture hidden output")
    output_cache_layers = cache_to_layers(past)
    return captured["hidden"], output_cache_layers[layer_idx][0], output_cache_layers[layer_idx][1]


def layer_inputs(torch, DynamicCache, model, arrays, layer_idx: int, layer_count: int, *, device, dtype, args):
    inputs_embeds = tensor_from_np(torch, arrays, "decoder_decode_step0_inputs_embeds", device=device, dtype=dtype)
    attention_mask, cache_position, input_cache_layers = make_layer_attention_and_cache(
        torch, arrays, inputs_embeds, layer_count, device=device, dtype=dtype, args=args
    )

    if layer_idx == 0:
        hidden_states = inputs_embeds
    else:
        captured: Dict[str, Any] = {}
        past = dynamic_cache_from_layers(DynamicCache, input_cache_layers)

        def pre_hook(_module, module_inputs, module_kwargs):
            hidden = module_kwargs.get("hidden_states") if module_kwargs else None
            if hidden is None and module_inputs:
                hidden = module_inputs[0]
            if hidden is None:
                die(f"layer {layer_idx} pre-hook did not capture hidden input")
            captured["hidden"] = hidden.detach()

        handle = model.model.layers[layer_idx].register_forward_pre_hook(pre_hook, with_kwargs=True)
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
            die(f"layer {layer_idx} pre-hook did not capture hidden input")
        hidden_states = captured["hidden"]

    if getattr(args, "zero_cache", False) or getattr(args, "dummy_cache", False):
        position_ids = torch.zeros((1, hidden_states.shape[1]), device=device, dtype=torch.long)
    else:
        position_ids = tensor_from_np(torch, arrays, "decoder_decode_step0_cache_position", device=device).unsqueeze(0)
    with torch.inference_mode():
        cos, sin = model.model.rotary_emb(hidden_states, position_ids)

    return (
        hidden_states,
        attention_mask,
        cache_position,
        input_cache_layers[layer_idx][0],
        input_cache_layers[layer_idx][1],
        cos,
        sin,
    )


def make_layer_attention_and_cache(torch, arrays, inputs_embeds, layer_count: int, *, device, dtype, args):
    if getattr(args, "dummy_cache", False):
        batch_size, seq_length = inputs_embeds.shape[:2]
        cache_position = torch.zeros((seq_length,), device=device, dtype=torch.float32 if args.float_cache_position else torch.long)
        attention_mask = torch.zeros((batch_size, 1, seq_length, seq_length + 1), device=device, dtype=dtype)
        attention_mask[:, :, :, 0] = torch.finfo(dtype).min
        input_cache_layers = []
        for _layer_idx in range(layer_count):
            key = torch.zeros((batch_size, 32, 1, 192), device=device, dtype=dtype)
            value = torch.zeros((batch_size, 32, 1, 128), device=device, dtype=dtype)
            input_cache_layers.append((key, value))
        return attention_mask, cache_position, input_cache_layers

    if getattr(args, "zero_cache", False):
        batch_size, seq_length = inputs_embeds.shape[:2]
        cache_position = torch.zeros((seq_length,), device=device, dtype=torch.float32 if args.float_cache_position else torch.long)
        attention_mask = torch.zeros((batch_size, 1, seq_length, seq_length), device=device, dtype=dtype)
        input_cache_layers = []
        for _layer_idx in range(layer_count):
            key = torch.empty((batch_size, 32, 0, 192), device=device, dtype=dtype)
            value = torch.empty((batch_size, 32, 0, 128), device=device, dtype=dtype)
            input_cache_layers.append((key, value))
        return attention_mask, cache_position, input_cache_layers

    attention_mask_2d = tensor_from_np(torch, arrays, "decoder_decode_step0_attention_mask", device=device)
    cache_position = tensor_from_np(torch, arrays, "decoder_decode_step0_cache_position", device=device)
    attention_mask = prepare_flat_attention_mask(torch, attention_mask_2d, inputs_embeds, cache_position, args.mask_mode)
    if args.float_cache_position:
        cache_position = cache_position.float()

    input_cache_layers = cache_layers_from_npz(
        torch, arrays, "decoder_decode_step0_input_cache", layer_count, device=device, dtype=dtype
    )
    return attention_mask, cache_position, input_cache_layers


def compare_outputs(torch, outputs, expected, args) -> bool:
    ok = print_stats("layer.hidden_states", diff_stats(torch, outputs[0], expected[0]), args.atol)
    cache_summary = {
        "layer_count": 1,
        "expected_layer_count": 1,
        "max_abs": max(
            diff_stats(torch, outputs[1], expected[1])["max_abs"],
            diff_stats(torch, outputs[2], expected[2])["max_abs"],
        ),
        "worst": "layer_key_or_value",
        "samples": [
            {"name": "layer_key", **diff_stats(torch, outputs[1], expected[1])},
            {"name": "layer_value", **diff_stats(torch, outputs[2], expected[2])},
        ],
    }
    ok = print_cache_summary("layer.output_cache", cache_summary, args.cache_atol) and ok
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe/export one Youtu-VL decoder layer.")
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA), help="Path to text dump metadata.json.")
    parser.add_argument("--model", default="", help="Local HF model dir or model id. Defaults to metadata model.path_or_id.")
    parser.add_argument("--sample", default="en_short", help="Sample id in metadata.json.")
    parser.add_argument("--layer", type=int, default=0, help="Decoder layer index. Only 0 is supported initially.")
    parser.add_argument("--trace", action="store_true", help="Trace layer wrapper to TorchScript after parity passes.")
    parser.add_argument("--run-pnnx", action="store_true", help="Run pnnx after tracing. Requires --trace.")
    parser.add_argument("--pnnx", default="", help="Path to pnnx executable. If omitted, PATH is searched.")
    parser.add_argument("--pnnx-arg", action="append", default=[], help="Extra key=value argument passed to pnnx.")
    parser.add_argument("--pnnx-fp16", action="store_true", help="Ask pnnx to write fp16 ncnn weights.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--check-trace", action="store_true", help="Enable torch.jit.trace check_trace.")
    parser.add_argument("--mask-mode", default="4d", choices=["2d", "4d"])
    parser.add_argument("--float-cache-position", action="store_true")
    parser.add_argument(
        "--zero-cache",
        action="store_true",
        help="Use empty KV cache, cache_position=0, and a single-token causal mask for first-token prefill export.",
    )
    parser.add_argument(
        "--dummy-cache",
        action="store_true",
        help="Use one masked all-zero KV slot to avoid zero-length ncnn tensors for first-token prefill export.",
    )
    parser.add_argument(
        "--manual-kv-b-split",
        action="store_true",
        help="Use an equivalent attention wrapper that splits kv_b_proj weights into key/value projections.",
    )
    parser.add_argument(
        "--manual-sdpa",
        action="store_true",
        help="Use torch scaled_dot_product_attention in the manual KV wrapper to probe ncnn SDPA export.",
    )
    parser.add_argument(
        "--manual-sdpa-flat-mask",
        action="store_true",
        help="Trace the manual SDPA wrapper with a 2D additive mask, matching ncnn_llm decoder mask layout.",
    )
    parser.add_argument(
        "--manual-sdpa-ncnn-layout",
        action="store_true",
        help="Trace manual SDPA with ncnn_llm-style 3D KV cache layout: heads x sequence x dim.",
    )
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device-map", default="none")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--cache-atol", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_pnnx and not args.trace:
        die("--run-pnnx requires --trace")
    if args.manual_sdpa and not args.manual_kv_b_split:
        args.manual_kv_b_split = True
    if args.manual_sdpa_flat_mask or args.manual_sdpa_ncnn_layout:
        args.manual_sdpa = True
        args.manual_kv_b_split = True

    metadata_path = resolve_project_path(args.metadata)
    metadata = load_json(metadata_path)
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
    if args.layer < 0 or args.layer >= layer_count:
        die(f"layer out of range: {args.layer}")

    print(f"[runtime] torch={torch.__version__} transformers={transformers.__version__} pydensecrf_stubbed={pydensecrf_stubbed}")
    print(f"[probe] sample={sample.get('sample_id')} layer={args.layer} npz={npz_path}")
    print(f"[probe] device={device} model_dtype={model_dtype} layers={layer_count}")

    example_inputs = layer_inputs(
        torch, DynamicCache, model, arrays, args.layer, layer_count, device=device, dtype=model_dtype, args=args
    )
    expected = expected_from_full_model(
        torch, DynamicCache, model, arrays, args.layer, layer_count, device=device, dtype=model_dtype, args=args
    )
    if args.manual_sdpa_flat_mask or args.manual_sdpa_ncnn_layout:
        past_key, past_value = example_inputs[3], example_inputs[4]
        if args.manual_sdpa_ncnn_layout:
            past_key, past_value = squeeze_batch_cache_for_ncnn(past_key, past_value)
        example_inputs = (
            example_inputs[0],
            flatten_sdpa_attention_mask(example_inputs[1]),
            example_inputs[2],
            past_key,
            past_value,
            *example_inputs[5:],
        )
    Wrapper = make_layer_wrapper(
        torch,
        DynamicCache,
        args.layer,
        manual_kv_b_split=args.manual_kv_b_split,
        manual_sdpa=args.manual_sdpa,
        manual_sdpa_ncnn_layout=args.manual_sdpa_ncnn_layout,
    )
    wrapper = Wrapper(model.model.layers[args.layer]).eval()
    with torch.inference_mode():
        outputs = wrapper(*example_inputs)

    if args.manual_sdpa_ncnn_layout:
        expected = (expected[0], expected[1].squeeze(0).contiguous(), expected[2].squeeze(0).contiguous())
    ok = compare_outputs(torch, outputs, expected, args)
    if not ok:
        raise SystemExit(1)

    if args.trace:
        output_dir = resolve_project_path(args.output_dir)
        if args.manual_sdpa_ncnn_layout:
            suffix = "_manual_kv_sdpa_ncnn"
        elif args.manual_sdpa_flat_mask:
            suffix = "_manual_kv_sdpa_flatmask"
        else:
            suffix = "_manual_kv_sdpa" if args.manual_sdpa else ("_manual_kv" if args.manual_kv_b_split else "")
        mode = "_dummy_cache" if args.dummy_cache else ("_zero_cache" if args.zero_cache else "")
        name = f"youtu_decoder_layer{args.layer:02d}_decode_step0{mode}{suffix}.pt"
        info = trace_wrapper(torch, wrapper, example_inputs, output_dir / name, args.check_trace)
        print(f"[trace] {info['path']} bytes={info['bytes']} outputs={info['output_count']}")
        if args.run_pnnx:
            pnnx_path = resolve_project_path(args.pnnx) if args.pnnx else None
            if pnnx_path is None or not pnnx_path.exists():
                found = shutil.which("pnnx")
                if not found:
                    die("pnnx executable not found; pass --pnnx")
                pnnx_path = Path(found)
            result = run_pnnx(
                torch,
                pnnx_path=pnnx_path,
                torchscript_path=Path(info["path"]),
                example_inputs=example_inputs,
                output_prefix=output_dir / Path(name).with_suffix("").name,
                fp16=args.pnnx_fp16,
                extra_args=args.pnnx_arg,
            )
            print(f"[pnnx] returncode={result['returncode']} produced={len(result['produced'])}")
            if result["stderr"]:
                print("[pnnx:stderr]")
                print(result["stderr"])
            if not result["passed"]:
                raise SystemExit(result["returncode"] or 1)


if __name__ == "__main__":
    main()
