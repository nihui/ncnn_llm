#!/usr/bin/env python3
"""Compare the deterministic Qwen3-TTS PyTorch and ncnn regression outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import wave


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_i32(path: Path) -> tuple[int, ...]:
    data = path.read_bytes()
    if len(data) % 4:
        raise ValueError(f"unaligned int32 file: {path}")
    return struct.unpack(f"<{len(data) // 4}i", data)


def read_pcm16(path: Path) -> tuple[int, int, tuple[int, ...]]:
    with wave.open(str(path), "rb") as stream:
        if stream.getnchannels() != 1 or stream.getsampwidth() != 2:
            raise ValueError(f"expected mono PCM16 WAV: {path}")
        sample_rate = stream.getframerate()
        frames = stream.getnframes()
        data = stream.readframes(frames)
    return sample_rate, frames, struct.unpack(f"<{frames}h", data)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-codes", type=Path, required=True)
    parser.add_argument("--reference-wav", type=Path, required=True)
    parser.add_argument("--ncnn-codes", type=Path, required=True)
    parser.add_argument("--ncnn-wav", type=Path, required=True)
    parser.add_argument("--max-pcm-lsb", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reference_codes = read_i32(args.reference_codes)
    ncnn_codes = read_i32(args.ncnn_codes)
    reference_rate, reference_frames, reference_pcm = read_pcm16(args.reference_wav)
    ncnn_rate, ncnn_frames, ncnn_pcm = read_pcm16(args.ncnn_wav)
    if len(reference_pcm) != len(ncnn_pcm):
        max_lsb = None
        equal_samples = 0
    else:
        differences = [abs(a - b) for a, b in zip(reference_pcm, ncnn_pcm)]
        max_lsb = max(differences, default=0)
        equal_samples = sum(value == 0 for value in differences)

    report = {
        "schema_version": 1,
        "codes": {
            "count": len(ncnn_codes),
            "exact": reference_codes == ncnn_codes,
            "reference_sha256": sha256(args.reference_codes),
            "ncnn_sha256": sha256(args.ncnn_codes),
        },
        "audio": {
            "sample_rate": ncnn_rate,
            "samples": ncnn_frames,
            "shape_equal": (reference_rate, reference_frames) == (ncnn_rate, ncnn_frames),
            "max_pcm_lsb": max_lsb,
            "max_pcm_lsb_allowed": args.max_pcm_lsb,
            "equal_samples": equal_samples,
            "reference_sha256": sha256(args.reference_wav),
            "ncnn_sha256": sha256(args.ncnn_wav),
        },
    }
    report["passed"] = bool(
        report["codes"]["exact"]
        and report["audio"]["shape_equal"]
        and max_lsb is not None
        and max_lsb <= args.max_pcm_lsb
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
