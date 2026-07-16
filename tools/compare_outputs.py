#!/usr/bin/env python3
"""Strict final-output parity checks for text, token sequences, and PCM WAV."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import wave


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_result(reference: Path, actual: Path) -> dict:
    expected = reference.read_bytes()
    observed = actual.read_bytes()
    return {
        "exact": expected == observed,
        "reference_bytes": len(expected),
        "actual_bytes": len(observed),
    }


def token_result(reference: Path, actual: Path) -> dict:
    expected = json.loads(reference.read_text(encoding="utf-8"))
    observed = json.loads(actual.read_text(encoding="utf-8"))
    if isinstance(expected, dict):
        expected = expected["tokens"]
    if isinstance(observed, dict):
        observed = observed["tokens"]
    mismatch = next((i for i, pair in enumerate(zip(expected, observed)) if pair[0] != pair[1]), None)
    return {
        "exact": expected == observed,
        "reference_tokens": len(expected),
        "actual_tokens": len(observed),
        "first_mismatch": mismatch,
    }


def read_wav(path: Path) -> tuple[dict, bytes]:
    with wave.open(str(path), "rb") as stream:
        metadata = {
            "channels": stream.getnchannels(),
            "sample_width": stream.getsampwidth(),
            "sample_rate": stream.getframerate(),
            "frames": stream.getnframes(),
            "compression": stream.getcomptype(),
        }
        return metadata, stream.readframes(stream.getnframes())


def wav_result(reference: Path, actual: Path) -> dict:
    expected_meta, expected_pcm = read_wav(reference)
    actual_meta, actual_pcm = read_wav(actual)
    mismatch = None
    if expected_pcm != actual_pcm:
        mismatch = next((i for i, pair in enumerate(zip(expected_pcm, actual_pcm)) if pair[0] != pair[1]), None)
    return {
        "exact": expected_meta == actual_meta and expected_pcm == actual_pcm,
        "reference_format": expected_meta,
        "actual_format": actual_meta,
        "reference_pcm_bytes": len(expected_pcm),
        "actual_pcm_bytes": len(actual_pcm),
        "first_pcm_byte_mismatch": mismatch,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("text", "tokens", "wav"))
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    compare = {"text": text_result, "tokens": token_result, "wav": wav_result}[args.mode]
    result = compare(args.reference, args.actual)
    receipt = {
        "schema_version": 1,
        "mode": args.mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "reference": {"path": str(args.reference), "sha256": file_hash(args.reference)},
        "actual": {"path": str(args.actual), "sha256": file_hash(args.actual)},
        "result": result,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    return 0 if result["exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
