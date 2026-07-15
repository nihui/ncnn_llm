#!/usr/bin/env python3
"""Run HunyuanOCR with PyTorch and ncnn and compare the final text.

The report always records strict UTF-8 equality. It also reports a conservative
character similarity after normalizing line-break spelling and whitespace,
because converted inference can choose a neighboring token at very small logit
margins. Both raw outputs are retained so the metric is independently auditable.
"""

from __future__ import annotations

import argparse
import difflib
import gc
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pytorch-model", required=True, type=Path)
    parser.add_argument("--ncnn-model", required=True, type=Path)
    parser.add_argument("--ncnn-executable", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prompt", default="提取图中的文字。")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--pytorch-attention",
        choices=("sdpa", "eager"),
        default="sdpa",
        help="PyTorch attention backend; SDPA avoids quadratic eager-attention memory use on large pages.",
    )
    parser.add_argument(
        "--minimum-similarity",
        type=float,
        default=0.90,
        help="Exit successfully when whitespace-insensitive character similarity reaches this value.",
    )
    parser.add_argument(
        "--require-exact",
        action="store_true",
        help="Require strict UTF-8 equality instead of the similarity threshold.",
    )
    return parser.parse_args()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_for_similarity(text: str) -> str:
    text = text.replace("\\r\\n", "\n").replace("\\n", "\n")
    return "".join(text.split())


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def run_pytorch(args: argparse.Namespace) -> tuple[str, str, dict[str, object]]:
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, HunYuanVLForConditionalGeneration

    processor_kwargs: dict[str, object] = {"local_files_only": True}
    if int(transformers.__version__.split(".", 1)[0]) >= 5:
        processor_kwargs["backend"] = "pil"
    else:
        processor_kwargs["use_fast"] = False
    processor = AutoProcessor.from_pretrained(args.pytorch_model, **processor_kwargs)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(args.image)},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image = Image.open(args.image).convert("RGB")
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")

    model = HunYuanVLForConditionalGeneration.from_pretrained(
        args.pytorch_model,
        local_files_only=True,
        attn_implementation=args.pytorch_attention,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    device = next(model.parameters()).device
    inputs = inputs.to(device)

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    input_ids = inputs.get("input_ids", inputs.get("inputs"))
    trimmed = [out[len(src) :] for src, out in zip(input_ids, generated)]
    decoded = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    raw = decoded + "\n"
    versions = {
        "torch": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "transformers": transformers.__version__,
        "pytorch_device": str(device),
        "pytorch_dtype": str(next(model.parameters()).dtype),
        "pytorch_attention": args.pytorch_attention,
    }
    del generated, inputs, model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return decoded.strip(), raw, versions


def extract_ncnn_text(stdout: str) -> str:
    marker = "Generating text:\n"
    start = stdout.find(marker)
    if start < 0:
        raise RuntimeError("ncnn stdout does not contain the generation marker")
    start += len(marker)
    end = stdout.find("\n\nDone.", start)
    if end < 0:
        raise RuntimeError("ncnn stdout does not contain the completion marker")
    return stdout[start:end].strip()


def run_ncnn(args: argparse.Namespace) -> tuple[str, str, list[str]]:
    command = [
        str(args.ncnn_executable),
        "--model",
        str(args.ncnn_model),
        "--image",
        str(args.image),
        "--prompt",
        args.prompt,
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return extract_ncnn_text(completed.stdout), completed.stdout, command


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.minimum_similarity <= 1.0:
        raise SystemExit("--minimum-similarity must be between 0 and 1")
    for path in (
        args.pytorch_model,
        args.ncnn_model,
        args.ncnn_executable,
        args.image,
    ):
        if not path.exists():
            raise SystemExit(f"not found: {path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pytorch_text, pytorch_stdout, versions = run_pytorch(args)
    write_text(args.output_dir / "pytorch.stdout.txt", pytorch_stdout)
    write_text(args.output_dir / "pytorch.txt", pytorch_text + "\n")

    ncnn_text, ncnn_stdout, command = run_ncnn(args)
    write_text(args.output_dir / "ncnn.stdout.txt", ncnn_stdout)
    write_text(args.output_dir / "ncnn.txt", ncnn_text + "\n")

    exact_match = pytorch_text == ncnn_text
    pytorch_normalized = normalize_for_similarity(pytorch_text)
    ncnn_normalized = normalize_for_similarity(ncnn_text)
    similarity = difflib.SequenceMatcher(
        None, pytorch_normalized, ncnn_normalized, autojunk=False
    ).ratio()
    parity_pass = exact_match if args.require_exact else similarity >= args.minimum_similarity
    report = {
        "schema_version": 1,
        "parity_pass": parity_pass,
        "exact_match": exact_match,
        "comparison": "strict UTF-8 equality plus whitespace-insensitive character similarity",
        "whitespace_insensitive_similarity": similarity,
        "minimum_similarity": args.minimum_similarity,
        "require_exact": args.require_exact,
        "pytorch_text_length": len(pytorch_text),
        "ncnn_text_length": len(ncnn_text),
        "prompt": args.prompt,
        "image": str(args.image.resolve()),
        "image_sha256": sha256_file(args.image),
        "max_new_tokens": args.max_new_tokens,
        "pytorch_model": str(args.pytorch_model.resolve()),
        "pytorch_config_sha256": sha256_file(args.pytorch_model / "config.json"),
        "ncnn_model": str(args.ncnn_model.resolve()),
        "ncnn_config_sha256": sha256_file(args.ncnn_model / "model.json"),
        "ncnn_command": command,
        "pytorch_sha256": sha256_text(pytorch_text),
        "ncnn_sha256": sha256_text(ncnn_text),
        "platform": platform.platform(),
        "python": sys.version,
        **versions,
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if parity_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
