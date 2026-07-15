#!/usr/bin/env python3
"""Recreate the checked-in OCR fixture from the immutable official image."""

from __future__ import annotations

import argparse
import hashlib
import io
import urllib.request
from pathlib import Path

from PIL import Image


SOURCE = "https://raw.githubusercontent.com/Tencent-Hunyuan/HunyuanOCR/068d890e570a05ac8e1cf8575572b692adf28694/assets/guwan1.png"
SOURCE_SHA256 = "6e40cbd351933f6eecca9a44169bc084d5824cecf7015f2cdb95e6debfb0c833"
CROP = (308, 78, 568, 598)
OUTPUT_SHA256 = "c8faf31d71fea69a2aa35e02436d86b3408b3c810cfdedf8fff760015ef5c4a5"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("guwan-right-column.png"))
    args = parser.parse_args()
    source = urllib.request.urlopen(SOURCE).read()
    if digest(source) != SOURCE_SHA256:
        raise RuntimeError("official source image hash changed")
    image = Image.open(io.BytesIO(source)).convert("RGB").crop(CROP)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    data = buffer.getvalue()
    if digest(data) != OUTPUT_SHA256:
        raise RuntimeError("fixture encoder produced a different PNG")
    args.output.write_bytes(data)
    print(f"wrote {args.output} ({image.width}x{image.height}, sha256={digest(data)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
