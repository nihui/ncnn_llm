#!/usr/bin/env python3
"""Run a model-specific pnnx converter and write an auditable receipt."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    if spec.get("schema_version") != 1 or not spec.get("command"):
        raise ValueError("conversion spec must use schema_version 1 and contain command[]")
    command = [str(value) for value in spec["command"]]
    started = datetime.now(timezone.utc)
    working_directory = Path(spec.get("working_directory", ".")).resolve()
    process = subprocess.run(command, cwd=working_directory, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    finished = datetime.now(timezone.utc)
    outputs = []
    for name in spec.get("outputs", []):
        path = (working_directory / name).resolve()
        outputs.append({
            "path": name,
            "exists": path.is_file(),
            "size": path.stat().st_size if path.is_file() else None,
            "sha256": digest(path) if path.is_file() else None,
        })
    receipt = {
        "schema_version": 1,
        "model": spec.get("model"),
        "checkpoint": spec.get("checkpoint"),
        "pnnx_version": spec.get("pnnx_version"),
        "command": command,
        "platform": platform.platform(),
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "exit_code": process.returncode,
        "outputs": outputs,
        "log": process.stdout,
    }
    receipt_path = args.receipt or args.spec.with_name("conversion-receipt.json")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(process.stdout, end="")
    if process.returncode != 0 or any(not item["exists"] for item in outputs):
        return process.returncode or 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
