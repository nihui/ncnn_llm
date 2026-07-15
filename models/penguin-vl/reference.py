#!/usr/bin/env python3
"""Deterministic official Penguin-VL PyTorch baseline for the ncnn fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import types
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install_single_image_sdpa(model) -> None:
    """Replace the encoder's FlashAttention-only call with equivalent SDPA.

    The official encoder assumes flash-attn is installed. For a single image,
    its cu_seqlens describe one full bidirectional sequence, so PyTorch SDPA
    with is_causal=False is mathematically equivalent and deterministic.
    """

    layers = model.model.vision_encoder.encoder.layers
    remote = sys.modules[type(layers[0].self_attn).__module__]

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_value=None,
        cache_position=None,
        cu_seqlens=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query, key = remote.apply_multimodal_rotary_pos_emb(
            query, key, *position_embeddings
        )
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=query.shape[1] != key.shape[1],
        )
        output = output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.o_proj(output), None

    for layer in layers:
        layer.self_attn.forward = types.MethodType(forward, layer.self_attn)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--vision-encoder", default=None)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision", default="26ac2ceac1179ec3eda106cbb88e9eb909e68d4e")
    parser.add_argument("--min-tokens", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=196)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image).resolve()
    output_path = Path(args.output).resolve()
    model_path = Path(args.model)
    local_only = model_path.exists()

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    started = time.perf_counter()
    config = AutoConfig.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=local_only
    )
    if args.vision_encoder:
        config.vision_encoder = args.vision_encoder

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        trust_remote_code=True,
        device_map={"": args.device},
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=local_only,
    ).eval()
    processor = AutoProcessor.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=local_only
    )
    processor.image_processor.min_tokens = args.min_tokens
    processor.image_processor.max_tokens = args.max_tokens
    install_single_image_sdpa(model)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": {"image_path": str(image_path)}},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    inputs = {
        key: value.to(args.device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
    if generated.shape[1] > inputs["input_ids"].shape[1]:
        response_ids = generated[:, inputs["input_ids"].shape[1] :]
    else:
        response_ids = generated
    text = processor.batch_decode(response_ids, skip_special_tokens=True)[0].strip()

    record = {
        "schema": 1,
        "model": "tencent/Penguin-VL-2B",
        "revision": args.revision,
        "image": str(image_path),
        "image_sha256": sha256_file(image_path),
        "prompt": args.prompt,
        "min_tokens": args.min_tokens,
        "max_tokens": args.max_tokens,
        "max_new_tokens": args.max_new_tokens,
        "input_ids_shape": list(inputs["input_ids"].shape),
        "pixel_values_shape": list(inputs["pixel_values"].shape),
        "grid_sizes": inputs["grid_sizes"].detach().cpu().tolist(),
        "output_token_ids": response_ids[0].detach().cpu().tolist(),
        "text": text,
        "text_utf8_bytes": len(text.encode("utf-8")),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "wall_seconds": time.perf_counter() - started,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
