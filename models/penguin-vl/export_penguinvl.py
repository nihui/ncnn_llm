#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# Export a HuggingFace Penguin-VL checkpoint to ncnn using pnnx.
#
# The model is split into the same sub-graphs the C++ runtime loads:
#   1. patch_embed   Conv2d(3, Hvis, k=patch, s=patch)  -> per-patch projection
#   2. vision_enc    bidirectional Qwen3 encoder + final RMSNorm (2D-RoPE external)
#   3. projector     mlp2x_gelu  (Hvis -> Hllm)
#   4. embed_tokens  LLM token embedding
#   5. decoder       Qwen3 causal decoder with explicit KV-cache in/out
#   6. lm_head       final projection to vocab logits
#
# It also writes model.json describing the deployment.
#
# Requirements (NOT available in a minimal CI box; run this on a torch machine):
#   pip install torch transformers
#   pnnx binary on PATH (built from ncnn/tools/pnnx) or `pip install pnnx`
#
# Usage:
#   python tools/export_penguinvl.py \
#       --model tencent/Penguin-VL-2B \
#       --out   ./penguin-vl-2b-ncnn \
#       --dtype fp16
#
# See docs/EXPORT.md for the full pipeline and the decoder KV-cache contract.

import argparse
import json
import os
import subprocess
import sys

import torch


def run_pnnx(pt_path, input_shapes, out_prefix, dtypes=None):
    """Invoke pnnx on a traced TorchScript file.

    input_shapes: list of shape lists, e.g. [[196, 588]] -> "[196,588]".
    dtypes: optional per-input pnnx type suffix ("f32", "i64", ...); defaults to
    "f32" for every input. Integer inputs (e.g. token ids into an Embedding)
    must be declared i64 or pnnx rejects the traced shape.
    Produces <out_prefix>.ncnn.param / <out_prefix>.ncnn.bin next to pt_path.
    """
    if dtypes is None:
        dtypes = ["f32"] * len(input_shapes)
    shape_arg = ",".join(
        "[" + ",".join(str(d) for d in s) + "]" + t
        for s, t in zip(input_shapes, dtypes))
    cmd = [
        os.environ.get("PNNX", "pnnx"),
        pt_path,
        f"inputshape={shape_arg}",
        f"ncnnparam={out_prefix}.ncnn.param",
        f"ncnnbin={out_prefix}.ncnn.bin",
        f"ncnnpy={out_prefix}_ncnn.py",
        "fp16=0",
    ]
    print("[pnnx]", " ".join(cmd))
    subprocess.check_call(cmd)


def trace_and_export(module, example_inputs, input_shapes, out_prefix, work_dir,
                     dtypes=None):
    module.eval()
    pt_path = os.path.join(work_dir, os.path.basename(out_prefix) + ".pt")
    with torch.no_grad():
        ts = torch.jit.trace(module, example_inputs, check_trace=False)
    ts.save(pt_path)
    run_pnnx(pt_path, input_shapes, out_prefix, dtypes)


# --------------------------------------------------------------------------- #
# Traceable wrappers. Each isolates one stage and applies RoPE using external
# cos/sin so positions are never baked into the graph (the C++ side supplies
# them). They mirror the reference implementation in penguinvl/model/*.
# --------------------------------------------------------------------------- #

class PatchEmbed(torch.nn.Module):
    """Flattened-patch -> embedding. Equivalent to the reference Conv2d applied
    per patch, expressed as a single Linear over the (3*ps*ps) feature vector."""

    def __init__(self, conv, patch_size, num_channels):
        super().__init__()
        w = conv.weight.reshape(conv.out_channels, -1).contiguous()  # (H, 3*ps*ps)
        self.linear = torch.nn.Linear(w.shape[1], w.shape[0], bias=conv.bias is not None)
        with torch.no_grad():
            self.linear.weight.copy_(w)
            if conv.bias is not None:
                self.linear.bias.copy_(conv.bias)

    def forward(self, pixel_values):  # (N, 3*ps*ps) -> (N, H)
        return self.linear(pixel_values)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(x, rep):
    # GQA head expansion via expand+reshape (pnnx->ncnn friendly; avoids the
    # unsupported repeat_interleave op).
    if rep == 1:
        return x
    b, nkv, n, hd = x.shape
    return x[:, :, None, :, :].expand(b, nkv, rep, n, hd).reshape(b, nkv * rep, n, hd)


class RMSNorm(torch.nn.Module):
    """Plain fp32 RMSNorm. We re-implement it (instead of reusing the model's
    Qwen3RMSNorm) so the traced graph contains only the simple pow/mean/rsqrt/mul
    ops that pnnx converts cleanly. The stock module adds dtype-cast (aten::to)
    patterns that crash pnnx/torch-jit's shape pass on this toolchain."""

    def __init__(self, weight, eps):
        super().__init__()
        self.weight = torch.nn.Parameter(weight.detach().float().clone())
        self.eps = eps

    def forward(self, x):
        v = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(v + self.eps) * self.weight


def _clone_linear(real):
    """Copy a real nn.Linear into a fresh plain nn.Linear (weights + bias)."""
    lin = torch.nn.Linear(real.in_features, real.out_features,
                          bias=real.bias is not None)
    with torch.no_grad():
        lin.weight.copy_(real.weight.float())
        if real.bias is not None:
            lin.bias.copy_(real.bias.float())
    return lin


class VisionEncoder(torch.nn.Module):
    """Bidirectional Qwen3 encoder over patch embeddings, 2D-RoPE supplied as
    cos/sin (N, head_dim). Full attention (no mask) for a single image.

    Each real Qwen3 layer is rebuilt here from plain ops with the real weights
    copied in, so the traced graph matches the pnnx-verified self-test exactly.
    Tracing the stock nn.Modules directly crashes pnnx on this torch build."""

    def __init__(self, encoder, head_dim, num_heads, num_kv_heads, eps):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.eps = eps
        self.layers = torch.nn.ModuleList()
        for layer in encoder.layers:
            a = layer.self_attn
            self.layers.append(torch.nn.ModuleDict({
                "input_layernorm": RMSNorm(layer.input_layernorm.weight, eps),
                "post_attention_layernorm": RMSNorm(layer.post_attention_layernorm.weight, eps),
                "q_proj": _clone_linear(a.q_proj),
                "k_proj": _clone_linear(a.k_proj),
                "v_proj": _clone_linear(a.v_proj),
                "o_proj": _clone_linear(a.o_proj),
                "q_norm": RMSNorm(a.q_norm.weight, eps),
                "k_norm": RMSNorm(a.k_norm.weight, eps),
                "gate_proj": _clone_linear(layer.mlp.gate_proj),
                "up_proj": _clone_linear(layer.mlp.up_proj),
                "down_proj": _clone_linear(layer.mlp.down_proj),
            }))
        self.norm = RMSNorm(encoder.norm.weight, eps)

    def forward(self, hidden, cos, sin):  # hidden (N, H), cos/sin (N, head_dim)
        # Keep the token-feature stream 2D (N, H) so pnnx->ncnn emits clean 2D
        # Gemms: a batched 3D input makes the current ncnn Gemm collapse the token
        # dim. Attention adds an explicit batch=1 at dim 0 (which pnnx strips) to
        # get 3D (heads, N, hd) batched matmuls; dim 0 stays the batch and is
        # never permuted, since pnnx cannot transpose the batch axis.
        x = hidden  # (N, H)
        c = cos.view(1, 1, cos.shape[0], self.head_dim)
        s = sin.view(1, 1, sin.shape[0], self.head_dim)
        rep = self.num_heads // self.num_kv_heads
        for m in self.layers:
            residual = x
            h = m["input_layernorm"](x)
            N = h.shape[0]
            q = m["q_norm"](m["q_proj"](h).view(N, self.num_heads, self.head_dim)).unsqueeze(0).transpose(1, 2)
            k = m["k_norm"](m["k_proj"](h).view(N, self.num_kv_heads, self.head_dim)).unsqueeze(0).transpose(1, 2)
            v = m["v_proj"](h).view(N, self.num_kv_heads, self.head_dim).unsqueeze(0).transpose(1, 2)
            q = q * c + rotate_half(q) * s
            k = k * c + rotate_half(k) * s
            k = repeat_kv(k, rep)
            v = repeat_kv(v, rep)
            scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
            probs = torch.softmax(scores, dim=-1)
            o = torch.matmul(probs, v).transpose(1, 2).reshape(N, -1)
            x = residual + m["o_proj"](o)
            residual = x
            h = m["post_attention_layernorm"](x)
            x = residual + m["down_proj"](torch.nn.functional.silu(m["gate_proj"](h)) * m["up_proj"](h))
        x = self.norm(x)
        return x  # (N, H)


class Projector(torch.nn.Module):
    def __init__(self, projector):
        super().__init__()
        self.readout = projector.readout  # nn.Sequential(Linear, GELU, Linear, ...)

    def forward(self, x):  # (N, Hvis) -> (N, Hllm)
        return self.readout(x)


class EmbedTokens(torch.nn.Module):
    def __init__(self, embed):
        super().__init__()
        self.embed = embed

    def forward(self, ids):  # (seq,) int -> (seq, H)
        return self.embed(ids)


class LmHead(torch.nn.Module):
    def __init__(self, lm_head):
        super().__init__()
        self.lm_head = lm_head

    def forward(self, hidden):  # (seq, H) -> (seq, vocab)
        return self.lm_head(hidden)


def build_model_json(args, cfg, vcfg, out_dir):
    doc = {
        "model_type": "penguinvl_qwen3",
        "params": {
            "embed_token_param": "embed.ncnn.param",
            "embed_token_bin": "embed.ncnn.bin",
            "decoder_param": f"{args.decoder_name}.ncnn.param",
            "decoder_bin": f"{args.decoder_name}.ncnn.bin",
            "lm_head_param": "lm_head.ncnn.param",
            "lm_head_bin": "lm_head.ncnn.bin",
        },
        "tokenizer": {
            "type": "bbpe",
            "vocab_file": "vocab.txt",
            "merges_file": "merges.txt",
            "bos": "",
            "eos": "<|im_end|>",
            "additional_special_tokens": [
                "<|im_start|>", "<|im_end|>", "<image>", "<think>", "</think>",
            ],
        },
        "setting": {
            "hidden_size": cfg.hidden_size,
            "attn_cnt": cfg.num_hidden_layers,
            "kv_heads": getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
            "kv_cache": bool(args.kv_cache),
            "system_prompt": "",
            "rope": {
                "type": "RoPE",
                "rope_head_dim": getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads),
                "rope_theta": float(getattr(cfg, "rope_theta", 1000000.0)),
            },
            "vision": {
                "patch_embed_param": "vision_patch_embed.ncnn.param",
                "patch_embed_bin": "vision_patch_embed.ncnn.bin",
                "encoder_param": "vision_encoder.ncnn.param",
                "encoder_bin": "vision_encoder.ncnn.bin",
                "projector_param": "vision_projector.ncnn.param",
                "projector_bin": "vision_projector.ncnn.bin",
                "patch_size": vcfg.patch_size,
                "merge_size": 1,
                "min_tokens": args.min_tokens,
                "max_tokens": args.max_tokens,
                "vision_hidden": vcfg.hidden_size,
                "vision_head_dim": getattr(vcfg, "head_dim", vcfg.hidden_size // vcfg.num_attention_heads),
                "vision_rope_theta": float(getattr(vcfg, "rope_theta", 1000000.0)),
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
                "image_token": "<image>",
            },
        },
    }
    with open(os.path.join(out_dir, "model.json"), "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    print("[ok] wrote model.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--work", default=None, help="scratch dir for .pt/.onnx (default: <out>/_work)")
    ap.add_argument("--vision-encoder", default=None,
                    help="optional local Penguin-Encoder directory; useful for fully offline export")
    ap.add_argument("--vision-tokens", type=int, default=196,
                    help="number of visual patches used to trace the fixed-size vision graphs")
    ap.add_argument("--min-tokens", type=int, default=16,
                    help="runtime image processor minimum visual-token budget")
    ap.add_argument("--max-tokens", type=int, default=196,
                    help="runtime image processor maximum visual-token budget")
    ap.add_argument("--decoder-name", default="decoder_nocache",
                    help="decoder basename written to model.json")
    ap.add_argument("--kv-cache", action="store_true",
                    help="declare a KV-cache decoder in model.json (default: cacheless)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    work = args.work or os.path.join(args.out, "_work")
    os.makedirs(work, exist_ok=True)

    # Import here so the file can be inspected without transformers installed.
    from transformers import AutoConfig, AutoModelForCausalLM
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print(f"[load] {args.model}")
    # Load in the checkpoint's native dtype first, then cast to fp32. Casting
    # bf16->fp32 *during* the safetensors load (torch_dtype=torch.float32) can
    # segfault on some torch builds; converting after load is safe.
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    if args.vision_encoder:
        config.vision_encoder = args.vision_encoder
    model = AutoModelForCausalLM.from_pretrained(
        args.model, config=config, torch_dtype="auto", trust_remote_code=True)
    model = model.float()
    model.eval()

    llm = model.model                 # PenguinVLQwen3Model (VLMMetaModel + Qwen3Model)
    cfg = model.config
    vision = llm.get_vision_encoder()  # PenguinVLVisionEncoderModel
    vcfg = vision.config
    projector = llm.get_vision_projector()

    Hvis = vcfg.hidden_size
    ps = vcfg.patch_size
    # Qwen3 decouples head_dim from hidden_size/num_heads, so read it explicitly.
    vis_head_dim = getattr(vcfg, "head_dim", Hvis // vcfg.num_attention_heads)

    # 1. patch embed
    trace_and_export(
        PatchEmbed(vision.embeddings.patch_embedding, ps, vcfg.num_channels),
        (torch.randn(196, 3 * ps * ps),), [[196, 3 * ps * ps]],
        os.path.join(args.out, "vision_patch_embed"), work)

    # 2. vision encoder
    N = args.vision_tokens
    trace_and_export(
        VisionEncoder(vision.encoder, vis_head_dim, vcfg.num_attention_heads,
                      vcfg.num_key_value_heads, getattr(vcfg, "rms_norm_eps", 1e-6)),
        (torch.randn(N, Hvis), torch.randn(N, vis_head_dim), torch.randn(N, vis_head_dim)),
        [[N, Hvis], [N, vis_head_dim], [N, vis_head_dim]],
        os.path.join(args.out, "vision_encoder"), work)

    # 3. projector
    trace_and_export(
        Projector(projector), (torch.randn(N, Hvis),), [[N, Hvis]],
        os.path.join(args.out, "vision_projector"), work)

    # The token embedding and lm_head are the two largest sub-graphs (each a
    # ~vocab*hidden table). pnnx needs 2-3x their size in RAM to convert, so we
    # drop the rest of the model (decoder layers, vision) first 鈥?otherwise the
    # ~9 GB fp32 model held in this process starves pnnx and it aborts.
    import gc
    vocab_size = cfg.vocab_size
    hidden_size = cfg.hidden_size
    embed_mod = EmbedTokens(llm.embed_tokens)
    lmhead_mod = LmHead(model.lm_head)
    del vision, projector, llm, model
    gc.collect()

    # 4. embed tokens
    trace_and_export(
        embed_mod, (torch.randint(0, vocab_size, (8,)),), [[8]],
        os.path.join(args.out, "embed"), work, dtypes=["i64"])
    del embed_mod
    gc.collect()

    # 6. lm_head
    trace_and_export(
        lmhead_mod, (torch.randn(1, hidden_size),), [[1, hidden_size]],
        os.path.join(args.out, "lm_head"), work)
    del lmhead_mod
    gc.collect()

    # 5. decoder (KV-cache). This uses the ncnn_llm / nihui KV-cache decoder
    #    export convention; see docs/EXPORT.md. The traceable wrapper lives in
    #    tools/export_decoder_kvcache.py and must be exported on a torch machine.
    print("[note] decoder KV-cache export: see docs/EXPORT.md and "
          "tools/export_decoder_kvcache.py (blob names must be in0..in3, "
          "cache_k{i}/cache_v{i}, out_cache_k{i}/out_cache_v{i}, out0).")

    build_model_json(args, cfg, vcfg, args.out)
    print(f"[done] exported sub-graphs to {args.out}")


if __name__ == "__main__":
    main()
