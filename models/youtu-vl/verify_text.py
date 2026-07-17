#!/usr/bin/env python3
"""Strictly compare a Youtu-VL runtime text file with the pinned baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=Path(__file__).with_name("testdata") / "reference.json")
    parser.add_argument("--actual", type=Path, required=True)
    args = parser.parse_args()
    expected = json.loads(args.reference.read_text(encoding="utf-8"))["pytorch"]["text"]
    actual = args.actual.read_text(encoding="utf-8")
    if actual != expected:
        raise SystemExit(f"FAIL: actual={actual!r} expected={expected!r}")
    print(f"PASS: exact text match {actual!r}")


if __name__ == "__main__":
    main()
