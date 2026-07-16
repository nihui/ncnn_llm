#!/usr/bin/env python3
"""Compare final ncnn text with the deterministic PyTorch baseline exactly."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--ncnn", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    reference = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    actual = Path(args.ncnn).read_text(encoding="utf-8")
    expected = reference["text"]
    result = {
        "schema": 1,
        "pass": actual == expected,
        "expected": expected,
        "actual": actual,
        "expected_sha256": hashlib.sha256(expected.encode("utf-8")).hexdigest(),
        "actual_sha256": hashlib.sha256(actual.encode("utf-8")).hexdigest(),
        "expected_utf8_bytes": len(expected.encode("utf-8")),
        "actual_utf8_bytes": len(actual.encode("utf-8")),
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
