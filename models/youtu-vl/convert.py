#!/usr/bin/env python3
"""Reproduce the Youtu-VL PyTorch baselines and pnnx runtime package."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


def run(command: list[str], *, env: dict[str, str]) -> None:
    print("+", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--pnnx", default="pnnx")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    export = root / "export"
    checkpoint = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    work = Path(args.work).resolve()
    text_dump = work / "text-dumps"
    vision_dump = work / "vision-dumps"
    text_out = output / "models" / "text"
    vision_out = output / "models" / "vision"
    tokenizer_out = output / "tokenizer"
    for directory in (text_dump, vision_dump, text_out, vision_out, tokenizer_out):
        directory.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(export) + os.pathsep + env.get("PYTHONPATH", "")
    common_model = ["--model", str(checkpoint), "--device", args.device,
                    "--device-map", "none", "--attn-implementation", "eager"]

    run([args.python, str(export / "youtu_text_reference.py"),
         "--manifest", str(root / "assets/youtu_text_test/text_manifest.json"),
         "--output-dir", str(text_dump), "--only", "en_short",
         "--dtype", "auto", "--capture-decoder-io",
         "--capture-decode-hidden-states", *common_model], env=env)
    run([args.python, str(export / "youtu_vl_reference.py"),
         "--manifest", str(root / "assets/youtu_vl_test/image_manifest.json"),
         "--output-dir", str(vision_dump), "--only", "coco_cats:describe",
         "--max-new-tokens", "3", "--max-image-patches", "512",
         "--capture-vision-last-hidden", *common_model], env=env)
    run([args.python, str(export / "youtu_vl_vision_embed_probe.py"),
         "--manifest", str(root / "assets/youtu_vl_test/image_manifest.json"),
         "--output-dir", str(output / "testdata/vision-boundary"),
         "--only", "coco_cats:describe", "--max-image-patches", "512",
         "--dtype", "fp32", *common_model], env=env)

    metadata = text_dump / "metadata.json"
    pnnx = str(Path(args.pnnx).resolve())
    pnnx_common = ["--precision", args.precision, "--pnnx", pnnx,
                   "--trace", "--run-pnnx"]
    run([args.python, str(export / "youtu_vl_export.py"),
         "--metadata", str(metadata), "--model", str(checkpoint),
         "--output-dir", str(text_out), "--target", "all",
         "--allow-large-lm-head-pnnx",
         "--device", args.device, "--device-map", "none",
         "--attn-implementation", "eager", *pnnx_common], env=env)
    run([args.python, str(export / "youtu_final_norm_probe.py"),
         "--metadata", str(metadata), "--model", str(checkpoint),
         "--output-dir", str(text_out), "--device", args.device,
         "--device-map", "none", "--attn-implementation", "eager",
         *pnnx_common], env=env)
    decoder_torchscript = text_out / "youtu_decoder_full_manual_kv_sdpa_ncnn.pt"
    run([args.python, str(export / "youtu_full_decoder_probe.py"),
         "--metadata", str(metadata), "--model", str(checkpoint),
         "--output-dir", str(text_out), "--mode", "decode",
         "--device", args.device, "--device-map", "none",
         "--attn-implementation", "eager", "--trace"], env=env)
    run([args.python, str(export / "youtu_full_decoder_pnnx.py"),
         "--torchscript", str(decoder_torchscript),
         "--output-dir", str(text_out), "--pnnx", pnnx,
         "--precision", args.precision], env=env)

    vision_npz = vision_dump / "coco_cats_describe.npz"
    run([args.python, str(export / "youtu_vl_vision_dynamic_export.py"),
         "--manifest", str(root / "assets/youtu_vl_test/image_manifest.json"),
         "--model", str(checkpoint), "--input-npz", str(vision_npz),
         "--output-dir", str(vision_out), "--device", args.device,
         "--device-map", "none", "--attn-implementation", "eager",
         "--patch-param", *pnnx_common], env=env)

    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.json"):
        shutil.copy2(checkpoint / name, tokenizer_out / name)
    shutil.copy2(root / "testdata/coco_cats.jpg", output / "testdata/coco_cats.jpg")
    print(f"Youtu-VL ncnn package: {output}")


if __name__ == "__main__":
    main()
