#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from qwen_tts import Qwen3TTSModel
from export_qwen3_tts_talker_next import build_nonstream_prefill


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", default="")
    ap.add_argument("--text-file", default="")
    ap.add_argument("--language", required=True)
    ap.add_argument("--speaker", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", default="")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8")
    else:
        text = args.text
    if not text:
        raise ValueError("--text or --text-file is required")

    tts = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    prefill, _ = build_nonstream_prefill(tts, text, args.language, args.speaker)
    arr = prefill.detach().cpu().numpy().astype(np.float32)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(args.out)

    meta = {
        "text": text,
        "language": args.language,
        "speaker": args.speaker,
        "shape": list(arr.shape),
        "numel": int(arr.size),
    }
    if args.meta:
        Path(args.meta).write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
