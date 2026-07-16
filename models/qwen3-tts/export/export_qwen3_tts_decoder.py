#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch

from qwen_tts import Qwen3TTSModel


class SpeechDecoderWrapper(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, codes):
        if codes.shape[1] != self.decoder.config.num_quantizers:
            raise ValueError("unexpected quantizer count")

        hidden = self.decoder.quantizer.decode(codes)
        hidden = self.decoder.pre_conv(hidden).transpose(1, 2)

        seq_len = hidden.shape[1]
        mask = torch.full((seq_len, seq_len), torch.finfo(hidden.dtype).min, dtype=hidden.dtype, device=hidden.device)
        mask = torch.triu(mask, diagonal=1).view(1, 1, seq_len, seq_len)
        mask_mapping = {
            "full_attention": mask,
            "sliding_attention": mask,
        }
        hidden = self.decoder.pre_transformer(inputs_embeds=hidden, attention_mask=mask_mapping).last_hidden_state
        hidden = hidden.permute(0, 2, 1)
        for blocks in self.decoder.upsample:
            for block in blocks:
                hidden = block(hidden)
        wav = hidden
        for block in self.decoder.decoder:
            wav = block(wav)
        return wav.clamp(min=-1, max=1)


class SpeechDecoderFromHiddenWrapper(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, hidden):
        seq_len = hidden.shape[1]
        mask = torch.full((seq_len, seq_len), torch.finfo(hidden.dtype).min, dtype=hidden.dtype, device=hidden.device)
        mask = torch.triu(mask, diagonal=1).view(1, 1, seq_len, seq_len)
        mask_mapping = {
            "full_attention": mask,
            "sliding_attention": mask,
        }
        hidden = self.decoder.pre_transformer(inputs_embeds=hidden, attention_mask=mask_mapping).last_hidden_state
        hidden = hidden.permute(0, 2, 1)
        for blocks in self.decoder.upsample:
            for block in blocks:
                hidden = block(hidden)
        wav = hidden
        for block in self.decoder.decoder:
            wav = block(wav)
        return wav.clamp(min=-1, max=1)


class SpeechDecoderPreTransformerWrapper(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, hidden):
        seq_len = hidden.shape[1]
        mask = torch.full((seq_len, seq_len), torch.finfo(hidden.dtype).min, dtype=hidden.dtype, device=hidden.device)
        mask = torch.triu(mask, diagonal=1).view(1, 1, seq_len, seq_len)
        mask_mapping = {
            "full_attention": mask,
            "sliding_attention": mask,
        }
        return self.decoder.pre_transformer(inputs_embeds=hidden, attention_mask=mask_mapping).last_hidden_state


class SpeechDecoderPostTransformerWrapper(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, hidden):
        hidden = hidden.permute(0, 2, 1)
        for blocks in self.decoder.upsample:
            for block in blocks:
                hidden = block(hidden)
        wav = hidden
        for block in self.decoder.decoder:
            wav = block(wav)
        return wav.clamp(min=-1, max=1)


class SpeechDecoderUpsampleOnlyWrapper(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, hidden):
        hidden = hidden.permute(0, 2, 1)
        for blocks in self.decoder.upsample:
            for block in blocks:
                hidden = block(hidden)
        return hidden


class SpeechDecoderWaveOnlyWrapper(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, hidden):
        wav = hidden
        for block in self.decoder.decoder:
            wav = block(wav)
        return wav.clamp(min=-1, max=1)


class SpeechDecoderToHiddenWrapper(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, codes):
        hidden = self.decoder.quantizer.decode(codes)
        hidden = self.decoder.pre_conv(hidden).transpose(1, 2)
        return hidden


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--codes", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", default="full", choices=["full", "from-hidden", "to-hidden", "pre-transformer", "post-transformer", "upsample-only", "wave-only"])
    parser.add_argument("--trace-input", default="")
    parser.add_argument("--trace-seq-len", type=int, default=53)
    parser.add_argument("--trace-wave-len", type=int, default=212)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    tts = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=dtype,
        attn_implementation="eager",
    )
    decoder = tts.model.speech_tokenizer.model.decoder.eval()
    decoder.config._attn_implementation = "eager"
    decoder.pre_transformer.config._attn_implementation = "eager"
    if args.mode == "full":
        wrapper = SpeechDecoderWrapper(decoder).eval()
    elif args.mode == "from-hidden":
        wrapper = SpeechDecoderFromHiddenWrapper(decoder).eval()
    elif args.mode == "pre-transformer":
        wrapper = SpeechDecoderPreTransformerWrapper(decoder).eval()
    elif args.mode == "post-transformer":
        wrapper = SpeechDecoderPostTransformerWrapper(decoder).eval()
    elif args.mode == "upsample-only":
        wrapper = SpeechDecoderUpsampleOnlyWrapper(decoder).eval()
    elif args.mode == "wave-only":
        wrapper = SpeechDecoderWaveOnlyWrapper(decoder).eval()
    else:
        wrapper = SpeechDecoderToHiddenWrapper(decoder).eval()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    codes = torch.from_numpy(__import__("numpy").load(args.codes)).long().transpose(0, 1).unsqueeze(0)
    codes = codes.to(next(wrapper.parameters()).device)
    if args.mode in ("post-transformer", "upsample-only"):
        trace_input = torch.from_numpy(__import__("numpy").fromfile(args.trace_input, dtype="float32").reshape(1, args.trace_seq_len, 1024)).to(next(wrapper.parameters()).device)
    elif args.mode == "wave-only":
        trace_input = torch.from_numpy(__import__("numpy").fromfile(args.trace_input, dtype="float32").reshape(1, 1024, args.trace_wave_len)).to(next(wrapper.parameters()).device)
    elif args.mode in ("from-hidden", "pre-transformer"):
        with torch.inference_mode():
            hidden = decoder.quantizer.decode(codes)
            hidden = decoder.pre_conv(hidden).transpose(1, 2)
        hidden = hidden.clone()
        hidden_np = hidden.detach().cpu().float().numpy()
        __import__("numpy").save(str(out.with_suffix(".input_hidden.npy")), hidden_np)
        hidden_np.astype("float32").tofile(str(out.with_suffix(".input_hidden_f32.bin")))
        if args.mode == "from-hidden" and args.trace_seq_len > hidden.shape[1]:
            padded = torch.zeros((hidden.shape[0], args.trace_seq_len, hidden.shape[2]), dtype=hidden.dtype, device=hidden.device)
            padded[:, : hidden.shape[1], :] = hidden
            trace_input = padded
        else:
            trace_input = hidden
    else:
        trace_input = codes

    traced = torch.jit.trace(wrapper, trace_input, strict=False)
    traced.save(str(out))
    print(f"saved {out}")
    print(f"input_shape={tuple(trace_input.shape)} dtype={trace_input.dtype}")
    with torch.inference_mode():
        wav = wrapper(trace_input)
    print(f"output_shape={tuple(wav.shape)} dtype={wav.dtype}")


if __name__ == "__main__":
    main()
