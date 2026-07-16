#!/usr/bin/env python3
"""Reproducible full Penguin-VL-2B -> pnnx -> ncnn conversion driver."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(command: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.check_call(command, env=env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vision-encoder", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pnnx", required=True)
    parser.add_argument("--vision-tokens", type=int, default=196)
    parser.add_argument("--min-tokens", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=196)
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PNNX"] = str(Path(args.pnnx).resolve())

    run(
        [
            sys.executable,
            str(here / "export_penguinvl.py"),
            "--model",
            args.checkpoint,
            "--vision-encoder",
            args.vision_encoder,
            "--out",
            str(output),
            "--vision-tokens",
            str(args.vision_tokens),
            "--min-tokens",
            str(args.min_tokens),
            "--max-tokens",
            str(args.max_tokens),
            "--decoder-name",
            "decoder_nocache",
        ],
        env,
    )
    run(
        [
            sys.executable,
            str(here / "export_decoder_kvcache.py"),
            "--model",
            args.checkpoint,
            "--vision-encoder",
            args.vision_encoder,
            "--out",
            str(output),
            "--no-cache",
            "--name",
            "decoder_nocache",
        ],
        env,
    )
    run(
        [
            sys.executable,
            str(here / "extract_tokenizer.py"),
            "--model",
            args.checkpoint,
            "--out",
            str(output),
        ],
        env,
    )


if __name__ == "__main__":
    main()
