#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path

import torch

from qwen_tts import Qwen3TTSModel


class TextEmbed(torch.nn.Module):
    def __init__(self, talker):
        super().__init__()
        self.text_embedding = talker.get_text_embeddings()
        self.text_projection = talker.text_projection

    def forward(self, ids):
        return self.text_projection(self.text_embedding(ids))


class CodecEmbed(torch.nn.Module):
    def __init__(self, talker):
        super().__init__()
        self.codec_embedding = talker.get_input_embeddings()

    def forward(self, ids):
        return self.codec_embedding(ids)


def run_pnnx(pnnx: str, pt: Path, shape1: str, shape2: str) -> None:
    cmd = [pnnx, str(pt.name), f"inputshape={shape1}", f"inputshape2={shape2}", "fp16=0"]
    print("running:", " ".join(cmd))
    ret = subprocess.run(cmd, cwd=pt.parent, text=True, capture_output=True)
    if ret.stdout:
        print("\n".join(ret.stdout.splitlines()[-8:]))
    if ret.returncode != 0:
        raise RuntimeError(ret.stderr[-2000:])


def export_one(name: str, module: torch.nn.Module, example, out_dir: Path, pnnx: str, shape1: str, shape2: str) -> None:
    module.eval()
    with torch.inference_mode():
        ref = module(example)
    traced = torch.jit.trace(module, example, strict=False)
    pt = out_dir / f"{name}.pt"
    traced.save(str(pt))
    ref.detach().cpu().float().numpy().astype("float32").tofile(out_dir / f"{name}_ref_f32.bin")
    print(f"{name}: ref_shape={tuple(ref.shape)} pt={pt}")
    run_pnnx(pnnx, pt, shape1, shape2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pnnx", default="/workspace/Tencent/ncnn/build-vulkan/tools/pnnx")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tts = Qwen3TTSModel.from_pretrained(args.model, device_map=args.device, dtype=torch.float32, attn_implementation="eager")
    talker = tts.model.talker

    export_one(
        "talker_text_embed",
        TextEmbed(talker),
        torch.tensor([151672, 151673, 151671, 104198, 3837, 102067, 1773], dtype=torch.int32, device=talker.device),
        out_dir,
        args.pnnx,
        "[7]i32",
        "[97]i32",
    )
    export_one(
        "talker_codec_embed",
        CodecEmbed(talker),
        torch.tensor([2154, 2156, 2055, 2157, 3065, 2148, 2149], dtype=torch.int32, device=talker.device),
        out_dir,
        args.pnnx,
        "[7]i32",
        "[1]i32",
    )


if __name__ == "__main__":
    main()
