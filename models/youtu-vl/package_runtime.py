#!/usr/bin/env python3
"""Assemble the runtime-only Youtu-VL package and write a SHA-256 receipt."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


RUNTIME_FILES = {
    "models/text/youtu_decoder_full_manual_kv_sdpa_ncnn.ncnn.bin": "models/text/youtu_decoder_full_manual_kv_sdpa_ncnn.ncnn.bin",
    "models/text/youtu_decoder_full_manual_kv_sdpa_ncnn.ncnn.patched.param": "models/text/youtu_decoder_full_manual_kv_sdpa_ncnn.ncnn.patched.param",
    "models/text/youtu_decoder_final_norm.ncnn.bin": "models/text/youtu_decoder_final_norm.ncnn.bin",
    "models/text/youtu_decoder_final_norm.ncnn.param": "models/text/youtu_decoder_final_norm.ncnn.param",
    "models/text/youtu_embed_tokens.ncnn.bin": "models/text/youtu_embed_tokens.ncnn.bin",
    "models/text/youtu_embed_tokens.ncnn.param": "models/text/youtu_embed_tokens.ncnn.param",
    "models/text/youtu_lm_head.ncnn.bin": "models/text/youtu_lm_head.ncnn.bin",
    "models/text/youtu_lm_head.ncnn.param": "models/text/youtu_lm_head.ncnn.param",
    "models/vision/youtu_vl_siglip2_encoder_dynamic.ncnn.bin": "models/vision/youtu_vl_siglip2_encoder_dynamic.ncnn.bin",
    "models/vision/youtu_vl_siglip2_encoder_dynamic.ncnn.dynamic.patched.param": "models/vision/youtu_vl_siglip2_encoder_dynamic.ncnn.dynamic.patched.param",
    "models/vision/youtu_vl_post_merger_dynamic.ncnn.bin": "models/vision/youtu_vl_post_merger_dynamic.ncnn.bin",
    "models/vision/youtu_vl_post_merger_dynamic.ncnn.dynamic.patched.param": "models/vision/youtu_vl_post_merger_dynamic.ncnn.dynamic.patched.param",
    "tokenizer/tokenizer.json": "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json": "tokenizer/tokenizer_config.json",
    "tokenizer/special_tokens_map.json": "tokenizer/special_tokens_map.json",
    "tokenizer/chat_template.json": "tokenizer/chat_template.json",
    "testdata/coco_cats.jpg": "testdata/coco_cats.jpg",
    "testdata/vision-boundary/coco_cats_describe.npz": "testdata/vision-boundary/coco_cats_describe.npz",
    "testdata/vision-boundary/metadata.json": "testdata/vision-boundary/metadata.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversion-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--readme", type=Path, default=Path(__file__).with_name("RUNTIME_PACKAGE.md"))
    parser.add_argument("--refresh", action="store_true", help="Refresh an existing package created by this tool.")
    args = parser.parse_args()

    source = args.conversion_output.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.refresh:
        raise SystemExit(f"refusing to merge into non-empty package directory: {output}")
    if args.refresh and output.exists() and not (output / "manifest.json").is_file():
        raise SystemExit(f"refusing to refresh a directory without this tool's manifest: {output}")
    output.mkdir(parents=True, exist_ok=True)

    for source_name, output_name in RUNTIME_FILES.items():
        source_path = source / source_name
        if not source_path.is_file():
            raise SystemExit(f"missing runtime artifact: {source_path}")
        destination = output / output_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)

    shutil.copy2(args.readme.resolve(), output / "README.md")
    model_root = Path(__file__).resolve().parent
    shutil.copy2(model_root / "testdata/reference.json", output / "testdata/reference.json")
    shutil.copy2(model_root / "verify_text.py", output / "verify_text.py")
    entries = []
    for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name != "manifest.json"):
        relative = path.relative_to(output).as_posix()
        entries.append({"path": relative, "size": path.stat().st_size, "sha256": sha256(path)})
    receipt = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": "tencent/Youtu-VL-4B-Instruct",
        "source_revision": "8d30a0e49662a1d628a472b12df264dbcd768753",
        "runtime": "C++/ncnn; no Python or PyTorch required",
        "files": entries,
    }
    (output / "manifest.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[package] {output}")
    print(f"[files] {len(entries)}")
    print(f"[bytes] {sum(item['size'] for item in entries)}")


if __name__ == "__main__":
    main()
