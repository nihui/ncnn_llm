#!/usr/bin/env python3
"""Create the downloadable asset manifest from a conversion receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import quote


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    receipt = json.loads(args.conversion.read_text(encoding="utf-8"))
    base = args.source.rstrip("/")
    files = []
    for item in receipt["files"]:
        path = item["path"]
        files.append({
            "path": path,
            "size": item["size"],
            "sha256": item["sha256"],
            "urls": [f"{base}/{quote(path, safe='/')}"]
        })
    manifest = {
        "schema_version": 1,
        "model": args.model,
        "source": base,
        "files": files,
    }
    args.output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output}: files={len(files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
