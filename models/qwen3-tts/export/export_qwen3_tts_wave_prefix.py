#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch

from qwen_tts import Qwen3TTSModel


class WavePrefix(torch.nn.Module):
    def __init__(self, decoder, prefix: int):
        super().__init__()
        self.blocks = torch.nn.ModuleList(list(decoder.decoder[:prefix]))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        if len(self.blocks) == 7:
            x = x.clamp(min=-1, max=1)
        return x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--prefix", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tts = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    decoder = tts.model.speech_tokenizer.model.decoder.eval()
    wrapper = WavePrefix(decoder, args.prefix).eval()

    x = np.fromfile(args.input, dtype=np.float32).reshape(1, 1024, 212)
    xt = torch.from_numpy(x).to(next(wrapper.parameters()).device)

    with torch.inference_mode():
        y = wrapper(xt).detach().cpu().numpy().astype(np.float32)

    stem = f"wave_prefix_{args.prefix}"
    y.tofile(out_dir / f"{stem}_ref_f32.bin")
    np.save(out_dir / f"{stem}_ref.npy", y)

    traced = torch.jit.trace(wrapper, xt.clone(), strict=False)
    traced.save(str(out_dir / f"{stem}.pt"))
    print(f"{stem} input={tuple(xt.shape)} output={tuple(y.shape)}")


if __name__ == "__main__":
    main()
