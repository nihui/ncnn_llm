#!/usr/bin/env python3
"""Convert the pinned HunyuanOCR 1.0 checkpoint into ncnn runtime graphs.

This file intentionally keeps the PyTorch-only export graph separate from the
C++ runtime. The deployed ``ocr_main`` executable does not import Python or
PyTorch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import HunYuanVLForConditionalGeneration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pnnx", type=Path, required=True)
    return parser.parse_args()


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * value * scale


class VisionAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(1152, 1152)
        self.k_proj = nn.Linear(1152, 1152)
        self.v_proj = nn.Linear(1152, 1152)
        self.o_proj = nn.Linear(1152, 1152)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        length = value.size(1)
        q = self.q_proj(value).reshape(1, length, 16, 72).permute(0, 2, 1, 3)
        k = self.k_proj(value).reshape(1, length, 16, 72).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(1, length, 16, 72).permute(0, 2, 1, 3)
        result = F.scaled_dot_product_attention(q, k, v)
        result = result.permute(0, 2, 1, 3).reshape(1, length, 1152)
        return self.o_proj(result)


class VisionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense_h_to_4h = nn.Linear(1152, 4304)
        self.dense_4h_to_h = nn.Linear(4304, 1152)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.dense_4h_to_h(F.gelu(self.dense_h_to_4h(value)))


class VisionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = VisionAttention()
        self.mlp = VisionMLP()
        self.input_layernorm = nn.LayerNorm(1152, eps=1e-5)
        self.post_attention_layernorm = nn.LayerNorm(1152, eps=1e-5)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value + self.self_attn(self.input_layernorm(value))
        return value + self.mlp(self.post_attention_layernorm(value))


class VisionMerger(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(1152, 2304, kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv2d(2304, 4608, kernel_size=1),
        )
        self.mlp = nn.Linear(4608, 1024)
        self.image_newline = nn.Parameter(torch.zeros(4608))
        self.image_begin = nn.Parameter(torch.zeros(1024))
        self.image_end = nn.Parameter(torch.zeros(1024))
        self.before_rms = RMSNorm(1152)
        self.after_rms = RMSNorm(1024)

    def forward(self, value: torch.Tensor, height: int, width: int) -> torch.Tensor:
        value = self.before_rms(value).permute(0, 2, 1).reshape(1, 1152, height, width)
        value = self.proj(value)
        channels = value.size(1)
        newline = self.image_newline.reshape(1, channels, 1, 1) + value[:, :1, :, :1] * 0.0
        value = torch.cat((value, newline), dim=3)
        value = value.reshape(1, channels, -1).permute(0, 2, 1)
        value = self.mlp(value)
        value = torch.cat(
            (self.image_begin.reshape(1, 1, 1024), value, self.image_end.reshape(1, 1, 1024)),
            dim=1,
        )
        return self.after_rms(value)


class VisionEncoder(nn.Module):
    def __init__(self, position_embedding: torch.Tensor):
        super().__init__()
        self.patch_embedding = nn.Conv2d(3, 1152, kernel_size=16, stride=16)
        self.layers = nn.ModuleList(VisionBlock() for _ in range(27))
        self.perceive = VisionMerger()
        self.position_embedding = nn.Parameter(position_embedding, requires_grad=False)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        value = self.patch_embedding(pixels)
        height, width = value.size(2), value.size(3)
        position = F.interpolate(
            self.position_embedding, size=[height, width], mode="bilinear", align_corners=False
        )
        value = value + position
        value = value.reshape(1, 1152, height * width).permute(0, 2, 1)
        for layer in self.layers:
            value = layer(value)
        return self.perceive(value, height, width)


def rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class DecoderAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(1024, 2048, bias=False)
        self.k_proj = nn.Linear(1024, 1024, bias=False)
        self.v_proj = nn.Linear(1024, 1024, bias=False)
        self.o_proj = nn.Linear(2048, 1024, bias=False)
        self.query_layernorm = RMSNorm(128)
        self.key_layernorm = RMSNorm(128)

    def forward(
        self, value: torch.Tensor, mask: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        length = value.size(1)
        query = self.q_proj(value).reshape(1, length, 16, 128).permute(0, 2, 1, 3)
        key = self.k_proj(value).reshape(1, length, 8, 128).permute(0, 2, 1, 3)
        val = self.v_proj(value).reshape(1, length, 8, 128).permute(0, 2, 1, 3)
        cos = torch.cat((cos, cos), dim=-1)
        sin = torch.cat((sin, sin), dim=-1)
        query = self.query_layernorm(query * cos + rotate_half(query) * sin)
        key = self.key_layernorm(key * cos + rotate_half(key) * sin)
        key = key.repeat_interleave(2, dim=1)
        val = val.repeat_interleave(2, dim=1)
        result = F.scaled_dot_product_attention(query, key, val, attn_mask=mask)
        result = result.permute(0, 2, 1, 3).reshape(1, length, 2048)
        return self.o_proj(result)


class DecoderMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(1024, 3584, bias=False)
        self.up_proj = nn.Linear(1024, 3584, bias=False)
        self.down_proj = nn.Linear(3584, 1024, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(value)) * self.up_proj(value))


class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = DecoderAttention()
        self.mlp = DecoderMLP()
        self.input_layernorm = RMSNorm(1024)
        self.post_attention_layernorm = RMSNorm(1024)

    def forward(
        self, value: torch.Tensor, mask: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        value = value + self.self_attn(self.input_layernorm(value), mask, cos, sin)
        return value + self.mlp(self.post_attention_layernorm(value))


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(DecoderBlock() for _ in range(24))
        self.norm = RMSNorm(1024)

    def forward(
        self, value: torch.Tensor, mask: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        for layer in self.layers:
            value = layer(value, mask, cos, sin)
        return self.norm(value)


class TextEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(120818, 1024)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(token_ids)


class LMHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head = nn.Linear(1024, 120818, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.lm_head(value)


def copy_affine(target: nn.Module, source: nn.Module, *, bias: bool = True) -> None:
    target.weight.copy_(source.weight.float())
    if bias:
        target.bias.copy_(source.bias.float())


@torch.no_grad()
def build_modules(model: nn.Module) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
    vision_source = model.vit
    decoder_source = model.model
    position = vision_source.embeddings.position_embedding.weight[1:].float()
    position = position.reshape(128, 128, 1152).permute(2, 0, 1).unsqueeze(0).contiguous()
    vision = VisionEncoder(position)
    copy_affine(vision.patch_embedding, vision_source.embeddings.patch_embedding)
    for target, source in zip(vision.layers, vision_source.layers, strict=True):
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            copy_affine(getattr(target.self_attn, name), getattr(source.self_attn, name))
        for name in ("dense_h_to_4h", "dense_4h_to_h"):
            copy_affine(getattr(target.mlp, name), getattr(source.mlp, name))
        copy_affine(target.input_layernorm, source.input_layernorm)
        copy_affine(target.post_attention_layernorm, source.post_attention_layernorm)
    copy_affine(vision.perceive.proj[0], vision_source.perceive.proj[0])
    copy_affine(vision.perceive.proj[2], vision_source.perceive.proj[2])
    copy_affine(vision.perceive.mlp, vision_source.perceive.mlp)
    vision.perceive.image_newline.copy_(vision_source.perceive.image_newline.float())
    vision.perceive.image_begin.copy_(vision_source.perceive.image_begin.float())
    vision.perceive.image_end.copy_(vision_source.perceive.image_end.float())
    vision.perceive.before_rms.weight.copy_(vision_source.perceive.before_rms.weight.float())
    vision.perceive.after_rms.weight.copy_(vision_source.perceive.after_rms.weight.float())

    decoder = Decoder()
    for target, source in zip(decoder.layers, decoder_source.layers, strict=True):
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            copy_affine(getattr(target.self_attn, name), getattr(source.self_attn, name), bias=False)
        target.self_attn.query_layernorm.weight.copy_(source.self_attn.query_layernorm.weight.float())
        target.self_attn.key_layernorm.weight.copy_(source.self_attn.key_layernorm.weight.float())
        for name in ("gate_proj", "up_proj", "down_proj"):
            copy_affine(getattr(target.mlp, name), getattr(source.mlp, name), bias=False)
        target.input_layernorm.weight.copy_(source.input_layernorm.weight.float())
        target.post_attention_layernorm.weight.copy_(source.post_attention_layernorm.weight.float())
    decoder.norm.weight.copy_(decoder_source.norm.weight.float())

    embedding = TextEmbed()
    embedding.embed_tokens.weight.copy_(decoder_source.embed_tokens.weight.float())
    head = LMHead()
    head.lm_head.weight.copy_(model.lm_head.weight.float())
    return vision.eval(), embedding.eval(), decoder.eval(), head.eval()


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def pnnx_convert(
    pnnx: Path, traced: Path, output: Path, shape1: str, shape2: str | None = None
) -> None:
    command = [str(pnnx), str(traced), f"inputshape={shape1}"]
    if shape2:
        command.append(f"inputshape2={shape2}")
    command.extend(
        (
            f"ncnnparam={output}.param",
            f"ncnnbin={output}.bin",
            "fp16=0",
            "optlevel=2",
        )
    )
    run(command)


def export_tokenizer(checkpoint: Path, output: Path) -> None:
    tokenizer = json.loads((checkpoint / "tokenizer.json").read_text(encoding="utf-8"))
    config = json.loads((checkpoint / "tokenizer_config.json").read_text(encoding="utf-8"))
    vocab = tokenizer["model"]["vocab"]
    added = config.get("added_tokens_decoder", {})
    size = max([*vocab.values(), *(int(value) for value in added)]) + 1
    ordered = [f"<|unused_{index}|>" for index in range(size)]
    for token, index in vocab.items():
        ordered[index] = token
    for index, value in added.items():
        ordered[int(index)] = value["content"]
    (output / "vocab.txt").write_text("\n".join(ordered) + "\n", encoding="utf-8", newline="\n")
    merges = tokenizer["model"]["merges"]
    lines = [" ".join(value) if isinstance(value, list) else value for value in merges]
    (output / "merges.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    checkpoint, output = args.checkpoint.resolve(), args.output.resolve()
    pnnx_candidate = shutil.which(str(args.pnnx))
    pnnx = Path(pnnx_candidate).resolve() if pnnx_candidate else args.pnnx.resolve()
    if not (checkpoint / "config.json").is_file():
        raise SystemExit(f"checkpoint is incomplete: {checkpoint}")
    if not pnnx.is_file():
        raise SystemExit(f"pnnx executable not found: {pnnx}")
    output.mkdir(parents=True, exist_ok=True)
    work = output / ".torchscript"
    work.mkdir(exist_ok=True)
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    model = HunYuanVLForConditionalGeneration.from_pretrained(
        checkpoint,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).eval()
    vision, embedding, decoder, head = build_modules(model)

    examples = {
        "vision": (vision, (torch.zeros(1, 3, 384, 896),)),
        "text_embed": (embedding, (torch.zeros(1, 8, dtype=torch.int64),)),
        "text_decoder": (
            decoder,
            (
                torch.zeros(1, 8, 1024),
                torch.zeros(1, 1, 8, 8),
                torch.ones(1, 8, 64),
                torch.zeros(1, 8, 64),
            ),
        ),
        "lm_head": (head, (torch.zeros(1, 1, 1024),)),
    }
    for name, (module, inputs) in examples.items():
        torch.jit.trace(module, inputs, check_trace=False).save(str(work / f"{name}.pt"))

    pnnx_convert(
        pnnx,
        work / "vision.pt",
        output / "hunyuan_ocr_vision_encoder.ncnn",
        "[1,3,384,896]f32",
        "[1,3,224,448]f32",
    )
    pnnx_convert(
        pnnx,
        work / "text_embed.pt",
        output / "hunyuan_ocr_text_embed.ncnn",
        "[1,8]i64",
        "[1,64]i64",
    )
    pnnx_convert(
        pnnx,
        work / "text_decoder.pt",
        output / "hunyuan_ocr_text_decoder.ncnn",
        "[1,8,1024]f32,[1,1,8,8]f32,[1,8,64]f32,[1,8,64]f32",
        "[1,16,1024]f32,[1,1,16,16]f32,[1,16,64]f32,[1,16,64]f32",
    )
    pnnx_convert(
        pnnx,
        work / "lm_head.pt",
        output / "hunyuan_ocr_lm_head.ncnn",
        "[1,1,1024]f32",
        "[1,8,1024]f32",
    )
    repo_root = Path(__file__).resolve().parents[2]
    run(
        [
            sys.executable,
            str(repo_root / "export" / "hunyuan_ocr_add_kvcache.py"),
            str(output / "hunyuan_ocr_text_decoder.ncnn.param"),
        ]
    )
    head_bin = output / "hunyuan_ocr_lm_head.ncnn.bin"
    embed_bin = output / "hunyuan_ocr_text_embed.ncnn.bin"
    if sha256(head_bin) != sha256(embed_bin):
        raise RuntimeError("tied lm_head and embedding binaries differ")
    head_bin.unlink()
    export_tokenizer(checkpoint, output)
    shutil.copy2(Path(__file__).with_name("model.json"), output / "model.json")
    shutil.rmtree(work)
    print(f"converted HunyuanOCR runtime: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
