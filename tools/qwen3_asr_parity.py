#!/usr/bin/env python3
"""Run official Qwen3-ASR and ncnn on one WAV and require exact text parity."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pytorch-model", type=Path, required=True)
    parser.add_argument("--ncnn-model", type=Path, required=True)
    parser.add_argument("--ncnn-executable", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device-map", default="cpu")
    return parser.parse_args()


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def run_pytorch(args: argparse.Namespace) -> tuple[str, str, dict[str, str]]:
    import torch
    from qwen_asr import Qwen3ASRModel

    model = Qwen3ASRModel.from_pretrained(
        str(args.pytorch_model),
        device_map=args.device_map,
        max_new_tokens=args.max_new_tokens,
    )
    result = model.transcribe(str(args.audio), language=None)
    first = result[0] if isinstance(result, list) else result
    return first.language, first.text, {
        "torch": torch.__version__,
        "qwen_asr": importlib.metadata.version("qwen-asr"),
    }


def run_ncnn(args: argparse.Namespace) -> tuple[str, str, str, list[str]]:
    command = [
        str(args.ncnn_executable),
        "--model",
        str(args.ncnn_model),
        "--audio",
        str(args.audio),
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    process = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    language = re.search(r"^language: (.*)$", process.stdout, re.MULTILINE)
    text = re.search(r"^text: (.*)$", process.stdout, re.MULTILINE)
    if not language or not text:
        raise RuntimeError("ncnn output does not contain language/text result lines")
    return language.group(1), text.group(1), process.stdout, command


def main() -> int:
    args = parse_args()
    for path in (args.pytorch_model, args.ncnn_model, args.ncnn_executable, args.audio):
        if not path.exists():
            raise SystemExit(f"not found: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    py_language, py_text, versions = run_pytorch(args)
    nc_language, nc_text, stdout, command = run_ncnn(args)
    (args.output_dir / "pytorch.txt").write_text(py_text + "\n", encoding="utf-8", newline="\n")
    (args.output_dir / "ncnn.txt").write_text(nc_text + "\n", encoding="utf-8", newline="\n")
    (args.output_dir / "ncnn.stdout.txt").write_text(stdout, encoding="utf-8", newline="\n")
    report = {
        "schema_version": 1,
        "parity_pass": py_language == nc_language and py_text == nc_text,
        "comparison": "strict language and UTF-8 final-text equality",
        "pytorch_language": py_language,
        "ncnn_language": nc_language,
        "language_exact": py_language == nc_language,
        "text_exact": py_text == nc_text,
        "audio_sha256": sha256(args.audio),
        "pytorch_config_sha256": sha256(args.pytorch_model / "config.json"),
        "ncnn_config_sha256": sha256(args.ncnn_model / "model.json"),
        "ncnn_command": command,
        "platform": platform.platform(),
        **versions,
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["parity_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
