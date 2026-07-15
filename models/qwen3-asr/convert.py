#!/usr/bin/env python3
"""Convert the pinned Qwen3-ASR-0.6B checkpoint to the ncnn runtime layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from qwen_asr.core.transformers_backend import (
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRProcessor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pnnx", type=Path, required=True)
    return parser.parse_args()


class MelFrontend(nn.Module):
    def __init__(self, filters: np.ndarray):
        super().__init__()
        self.register_buffer("window", torch.hann_window(400, periodic=True))
        self.register_buffer("filters", torch.from_numpy(filters.astype(np.float32)).transpose(0, 1))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spectrum = torch.stft(
            waveform,
            n_fft=400,
            hop_length=160,
            win_length=400,
            window=self.window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        power = spectrum.abs().pow(2)
        return torch.matmul(self.filters, power)


class AudioConv(nn.Module):
    def __init__(self, source: nn.Module):
        super().__init__()
        self.conv2d1 = source.conv2d1
        self.conv2d2 = source.conv2d2
        self.conv2d3 = source.conv2d3
        self.conv_out = source.conv_out
        self.register_buffer("position", source.positional_embedding(13).float().unsqueeze(0))

    def forward(self, logmel: torch.Tensor) -> torch.Tensor:
        value = logmel.unsqueeze(0).unsqueeze(0)
        value = F.gelu(self.conv2d1(value))
        value = F.gelu(self.conv2d2(value))
        value = F.gelu(self.conv2d3(value))
        batch, channels, frequency, frames = value.shape
        value = value.permute(0, 3, 1, 2).reshape(batch, frames, channels * frequency)
        return self.conv_out(value) + self.position


class AudioEncoder(nn.Module):
    def __init__(self, source: nn.Module):
        super().__init__()
        self.layers = source.layers
        self.ln_post = source.ln_post
        self.proj1 = source.proj1
        self.proj2 = source.proj2

    @staticmethod
    def _attention(attention: nn.Module, value: torch.Tensor) -> torch.Tensor:
        batch, length, _ = value.shape
        query = attention.q_proj(value).reshape(batch, length, 14, 64).transpose(1, 2)
        key = attention.k_proj(value).reshape(batch, length, 14, 64).transpose(1, 2)
        val = attention.v_proj(value).reshape(batch, length, 14, 64).transpose(1, 2)
        result = F.scaled_dot_product_attention(
            query, key, val, dropout_p=0.0, scale=float(attention.scaling)
        )
        result = result.transpose(1, 2).reshape(batch, length, 896)
        return attention.out_proj(result)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            residual = value
            value = layer.self_attn_layer_norm(value)
            value = residual + self._attention(layer.self_attn, value)
            residual = value
            value = layer.final_layer_norm(value)
            value = layer.fc2(layer.activation_fn(layer.fc1(value)))
            value = residual + value
        value = self.ln_post(value)
        return self.proj2(F.gelu(self.proj1(value)))


def rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class TextDecoder(nn.Module):
    def __init__(self, source: nn.Module):
        super().__init__()
        self.layers = source.layers
        self.norm = source.norm

    @staticmethod
    def _attention(
        attention: nn.Module,
        value: torch.Tensor,
        mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        length = value.size(1)
        query = attention.q_norm(attention.q_proj(value).reshape(1, length, 16, 128)).transpose(1, 2)
        key = attention.k_norm(attention.k_proj(value).reshape(1, length, 8, 128)).transpose(1, 2)
        val = attention.v_proj(value).reshape(1, length, 8, 128).transpose(1, 2)
        cos = torch.cat((cos, cos), dim=-1)
        sin = torch.cat((sin, sin), dim=-1)
        query = query * cos + rotate_half(query) * sin
        key = key * cos + rotate_half(key) * sin
        result = F.scaled_dot_product_attention(
            query,
            key,
            val,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=True,
        )
        result = result.transpose(1, 2).reshape(1, length, 2048)
        return attention.o_proj(result)

    def forward(
        self, value: torch.Tensor, mask: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        for layer in self.layers:
            value = value + self._attention(
                layer.self_attn, layer.input_layernorm(value), mask, cos, sin
            )
            value = value + layer.mlp(layer.post_attention_layernorm(value))
        return self.norm(value)


class TextEmbed(nn.Module):
    def __init__(self, source: nn.Module):
        super().__init__()
        self.embed_tokens = source.embed_tokens

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(token_ids)


class LMHead(nn.Module):
    def __init__(self, source: nn.Module):
        super().__init__()
        self.lm_head = source

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.lm_head(value)


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


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
    vocab = json.loads((checkpoint / "vocab.json").read_text(encoding="utf-8"))
    config = json.loads((checkpoint / "tokenizer_config.json").read_text(encoding="utf-8"))
    added = config.get("added_tokens_decoder", {})
    size = max([*vocab.values(), *(int(index) for index in added)]) + 1
    ordered = [f"<|unused_{index}|>" for index in range(size)]
    for token, index in vocab.items():
        ordered[index] = token
    for index, item in added.items():
        ordered[int(index)] = item["content"]
    (output / "vocab.txt").write_text("\n".join(ordered) + "\n", encoding="utf-8", newline="\n")
    merges = [
        line
        for line in (checkpoint / "merges.txt").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    (output / "merges.txt").write_text("\n".join(merges) + "\n", encoding="utf-8", newline="\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    checkpoint, output = args.checkpoint.resolve(), args.output.resolve()
    pnnx_found = shutil.which(str(args.pnnx))
    pnnx = Path(pnnx_found).resolve() if pnnx_found else args.pnnx.resolve()
    if not (checkpoint / "model.safetensors").is_file():
        raise SystemExit(f"checkpoint is incomplete: {checkpoint}")
    if not pnnx.is_file():
        raise SystemExit(f"pnnx executable not found: {pnnx}")
    output.mkdir(parents=True, exist_ok=True)
    work = output / ".torchscript"
    work.mkdir(exist_ok=True)
    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        checkpoint,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).eval()
    processor = Qwen3ASRProcessor.from_pretrained(
        checkpoint, local_files_only=True, fix_mistral_regex=True
    )
    thinker = model.thinker
    modules: dict[str, tuple[nn.Module, tuple[torch.Tensor, ...]]] = {
        "mel": (
            MelFrontend(processor.feature_extractor.mel_filters).eval(),
            (torch.zeros(16000),),
        ),
        "audio_conv": (AudioConv(thinker.audio_tower).eval(), (torch.zeros(128, 100),)),
        "audio_encoder": (
            AudioEncoder(thinker.audio_tower).eval(),
            (torch.zeros(1, 13, 896),),
        ),
        "text_embed": (TextEmbed(thinker.model).eval(), (torch.zeros(1, 8, dtype=torch.int64),)),
        "decoder": (
            TextDecoder(thinker.model).eval(),
            (
                torch.zeros(1, 8, 1024),
                torch.zeros(1, 1, 8, 8),
                torch.ones(1, 8, 64),
                torch.zeros(1, 8, 64),
            ),
        ),
        "lm_head": (LMHead(thinker.lm_head).eval(), (torch.zeros(1, 1, 1024),)),
    }
    for name, (module, inputs) in modules.items():
        torch.jit.trace(module, inputs, check_trace=False).save(str(work / f"{name}.pt"))

    pnnx_convert(pnnx, work / "mel.pt", output / "qwen3_asr_mel.ncnn", "[16000]f32", "[32000]f32")
    pnnx_convert(
        pnnx, work / "audio_conv.pt", output / "qwen3_asr_audio_conv.ncnn", "[128,100]f32"
    )
    pnnx_convert(
        pnnx,
        work / "audio_encoder.pt",
        output / "qwen3_asr_audio_encoder.ncnn",
        "[1,13,896]f32",
        "[1,26,896]f32",
    )
    pnnx_convert(
        pnnx,
        work / "text_embed.pt",
        output / "qwen3_asr_text_embed.ncnn",
        "[1,8]i64",
        "[1,64]i64",
    )
    pnnx_convert(
        pnnx,
        work / "decoder.pt",
        output / "qwen3_asr_decoder.ncnn",
        "[1,8,1024]f32,[1,1,8,8]f32,[1,8,64]f32,[1,8,64]f32",
        "[1,16,1024]f32,[1,1,16,16]f32,[1,16,64]f32,[1,16,64]f32",
    )
    pnnx_convert(
        pnnx,
        work / "lm_head.pt",
        output / "qwen3_asr_lm_head.ncnn",
        "[1,1,1024]f32",
        "[1,8,1024]f32",
    )
    repo_root = Path(__file__).resolve().parents[2]
    decoder_param = output / "qwen3_asr_decoder.ncnn.param"
    run([sys.executable, str(repo_root / "export" / "hunyuan_ocr_add_kvcache.py"), str(decoder_param)])
    decoder_param.with_suffix(decoder_param.suffix + ".nokv").unlink(missing_ok=True)
    head_bin = output / "qwen3_asr_lm_head.ncnn.bin"
    embed_bin = output / "qwen3_asr_text_embed.ncnn.bin"
    if sha256(head_bin) != sha256(embed_bin):
        raise RuntimeError("tied lm_head and embedding binaries differ")
    head_bin.unlink()
    export_tokenizer(checkpoint, output)
    shutil.copy2(Path(__file__).with_name("model.json"), output / "model.json")
    shutil.rmtree(work)
    print(f"converted Qwen3-ASR runtime: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
