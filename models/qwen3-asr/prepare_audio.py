#!/usr/bin/env python3
"""Download the official Qwen3-ASR English sample and create the 16 kHz fixture."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path


SOURCE = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"
SOURCE_SHA256 = "f9b4440ac8393e47c14a6240e9739dea09b645bb1592b8f2dd48feb9666cea7f"
OUTPUT_SHA256 = "9c33540cbba2cf82c2b724f733b002466b87f83c06e20fbe3c4eed83c141edc9"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("asr_en_16k_pcm16.wav"))
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    ffmpeg = shutil.which(args.ffmpeg)
    if not ffmpeg:
        raise SystemExit(f"ffmpeg not found: {args.ffmpeg}")
    with tempfile.TemporaryDirectory() as temporary:
        source = Path(temporary) / "asr_en.wav"
        urllib.request.urlretrieve(SOURCE, source)
        if digest(source) != SOURCE_SHA256:
            raise RuntimeError("official source audio hash changed")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(args.output),
            ],
            check=True,
        )
    if digest(args.output) != OUTPUT_SHA256:
        raise RuntimeError("ffmpeg output differs from the pinned fixture")
    print(f"wrote {args.output} (sha256={OUTPUT_SHA256})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
