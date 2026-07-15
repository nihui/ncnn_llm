#!/usr/bin/env python3
"""Convert the official float32 decoder output to the runtime PCM16 format."""

from __future__ import annotations

import argparse
from pathlib import Path
import wave

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=24000)
    args = parser.parse_args()
    samples = np.fromfile(args.input, dtype=np.float32)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.output), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(args.sample_rate)
        stream.writeframes(pcm.tobytes())
    print(f"wrote {args.output}: samples={samples.size} rate={args.sample_rate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
