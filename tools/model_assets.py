#!/usr/bin/env python3
"""Download, verify, and reproducibly package external ncnn model assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request
import zipfile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not isinstance(data.get("files"), list):
        raise ValueError("asset manifest must use schema_version 1 and contain files[]")
    return data


def safe_path(root: Path, relative: str) -> Path:
    root = root.resolve()
    target = (root / relative).resolve()
    if root != target and root not in target.parents:
        raise ValueError(f"asset path escapes root: {relative}")
    return target


def verify(manifest: dict, root: Path) -> list[dict]:
    results = []
    for item in manifest["files"]:
        target = safe_path(root, item["path"])
        exists = target.is_file()
        size = target.stat().st_size if exists else None
        digest = sha256(target) if exists else None
        expected_hash = item["sha256"].lower()
        expected_size = item.get("size")
        ok = exists and digest == expected_hash and (
            expected_size is None or size == expected_size
        )
        results.append({
            "path": item["path"], "exists": exists, "size": size,
            "sha256": digest, "ok": ok,
        })
    return results


def download(manifest: dict, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for item in manifest["files"]:
        target = safe_path(root, item["path"])
        if target.is_file() and sha256(target) == item["sha256"].lower():
            print(f"verified {item['path']}")
            continue
        urls = item.get("urls", [])
        if not urls:
            raise RuntimeError(f"no download URL for {item['path']}")
        target.parent.mkdir(parents=True, exist_ok=True)
        error = None
        for url in urls:
            try:
                with tempfile.NamedTemporaryFile(delete=False, dir=target.parent) as tmp:
                    temporary = Path(tmp.name)
                try:
                    print(f"downloading {item['path']} from {url}")
                    urllib.request.urlretrieve(url, temporary)
                    if sha256(temporary) != item["sha256"].lower():
                        raise RuntimeError("SHA-256 mismatch")
                    if item.get("size") is not None and temporary.stat().st_size != item["size"]:
                        raise RuntimeError("size mismatch")
                    os.replace(temporary, target)
                    error = None
                    break
                finally:
                    temporary.unlink(missing_ok=True)
            except Exception as exc:  # try mirrors in declared order
                error = exc
        if error is not None:
            raise RuntimeError(f"failed to download {item['path']}: {error}")


def package(manifest: dict, root: Path, output: Path) -> None:
    failures = [row for row in verify(manifest, root) if not row["ok"]]
    if failures:
        raise RuntimeError(f"refusing to package unverified assets: {failures}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(manifest["files"], key=lambda value: value["path"]):
            source = safe_path(root, item["path"])
            info = zipfile.ZipInfo(item["path"], date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, source.read_bytes())


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("download", "verify", "package"):
        command = subparsers.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--root", type=Path, required=True)
        if name == "package":
            command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    if args.command == "download":
        download(manifest, args.root)
    elif args.command == "verify":
        results = verify(manifest, args.root)
        print(json.dumps({"files": results}, indent=2, ensure_ascii=False))
        return 0 if all(row["ok"] for row in results) else 1
    else:
        package(manifest, args.root, args.output)
        print(json.dumps({"package": str(args.output), "sha256": sha256(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
