#!/usr/bin/env python3
"""Create the fixed square Penguin-VL regression image from the official asset."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    image = Image.open(source).convert("RGB")
    side = min(image.size)
    left = (image.width - side) // 2
    output.parent.mkdir(parents=True, exist_ok=True)
    image.crop((left, 0, left + side, side)).save(output, optimize=True)


if __name__ == "__main__":
    main()
