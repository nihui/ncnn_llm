#!/usr/bin/env python3
"""Run multi-sample Youtu-VL greedy token parity against the ncnn C++ runner."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _subprocess_utils import run_utf8
from youtu_vl_reference import (
    DEFAULT_MANIFEST,
    build_messages,
    first_parameter_device,
    get_required_inputs,
    iter_samples,
    load_json,
    load_model_and_processor,
    resolve_project_path,
    to_model_device,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNNER = ROOT / "cpp" / "build" / "youtu_ncnn_decode_runner"
DEFAULT_REPORT = ROOT / "assets" / "youtu_vl_test" / "e2e_token_parity.json"
GENERATED_RE = re.compile(r"^\[ids\] generated=(\[.*\])$", re.MULTILINE)
PROMPT_RE = re.compile(r"^\[ids\] prompt=(\[.*\])$", re.MULTILINE)


def die(message: str, code: int = 2) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_runner_ids(output: str, pattern: re.Pattern[str], label: str) -> list[int]:
    match = pattern.search(output)
    if not match:
        raise ValueError(f"runner output missing `[ids] {label}=...`")
    value = ast.literal_eval(match.group(1))
    if not isinstance(value, list) or not all(isinstance(x, int) for x in value):
        raise ValueError(f"runner {label} is not an int list: {value!r}")
    return value


def first_mismatch(actual: list[int], expected: list[int]) -> dict[str, Any] | None:
    for index, (actual_id, expected_id) in enumerate(zip(actual, expected)):
        if actual_id != expected_id:
            return {"index": index, "ncnn": actual_id, "torch": expected_id}
    if len(actual) != len(expected):
        return {
            "index": min(len(actual), len(expected)),
            "ncnn": actual[min(len(actual), len(expected))] if len(actual) > len(expected) else None,
            "torch": expected[min(len(actual), len(expected))] if len(expected) > len(actual) else None,
            "ncnn_length": len(actual),
            "torch_length": len(expected),
        }
    return None


def torch_greedy_generate(torch, model, model_inputs: dict[str, Any], max_new_tokens: int, eos_token_id: int | None) -> list[int]:
    input_device = first_parameter_device(model)
    attention_mask = model_inputs.get("attention_mask")
    forward_inputs = get_required_inputs(
        model_inputs,
        (
            "input_ids",
            "attention_mask",
            "pixel_values",
            "pixel_attention_mask",
            "spatial_shapes",
            "instance_length",
            "coefficients",
            "rope_deltas",
        ),
    )
    generated: list[int] = []

    with torch.inference_mode():
        outputs = model(
            **forward_inputs,
            use_cache=True,
            output_hidden_states=False,
            output_attentions=False,
        )
        next_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        past = outputs.past_key_values

        for _ in range(max_new_tokens):
            token_id = int(next_id.item())
            generated.append(token_id)
            if eos_token_id is not None and token_id == int(eos_token_id):
                break

            if attention_mask is not None:
                one = torch.ones(
                    (attention_mask.shape[0], 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([attention_mask, one], dim=1)
            outputs = model(
                input_ids=next_id.view(1, 1).to(input_device),
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=False,
                output_attentions=False,
            )
            next_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
            past = outputs.past_key_values

    return generated


def run_ncnn(args, image_path: Path, prompt: str) -> dict[str, Any]:
    cmd = [
        str(args.runner),
        "--root",
        str(ROOT),
        "--image",
        str(image_path),
        "--prompt",
        prompt,
        "--max-image-patches",
        str(args.max_image_patches),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--full-decoder-ncnn",
        "--threads",
        str(args.threads),
    ]
    started = time.monotonic()
    result = run_utf8(
        cmd,
        cwd=str(ROOT.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=args.timeout,
        check=False,
    )
    elapsed = time.monotonic() - started
    if args.show_runner_output or result.returncode != 0:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.returncode != 0:
        raise RuntimeError(f"runner failed with exit code {result.returncode}")
    return {
        "prompt_ids": parse_runner_ids(result.stdout, PROMPT_RE, "prompt"),
        "generated_ids": parse_runner_ids(result.stdout, GENERATED_RE, "generated"),
        "seconds": elapsed,
        "stdout": result.stdout if args.keep_runner_output else "",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-sample Youtu-VL PyTorch vs ncnn complete greedy token parity."
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--model", default="")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Run one image_id:prompt_id sample. Repeat to select multiple; default runs all.",
    )
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-image-patches", type=int, default=512)
    parser.add_argument("--dtype", default="fp32", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device-map", default="none")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=900, help="Per-sample ncnn timeout in seconds.")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--show-runner-output", action="store_true")
    parser.add_argument("--keep-runner-output", action="store_true", help="Include full C++ stdout in JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0:
        die("--max-new-tokens must be positive")
    if args.max_image_patches <= 0:
        die("--max-image-patches must be positive")
    args.runner = args.runner.resolve()
    if not args.runner.exists():
        die(f"runner not found: {args.runner}")

    manifest_path = resolve_project_path(str(args.manifest))
    if not manifest_path.exists():
        die(f"manifest not found: {manifest_path}")
    manifest = load_json(manifest_path)
    samples = list(iter_samples(manifest, args.only))
    if not samples:
        die(f"no samples matched --only={args.only}" if args.only else "manifest has no samples")

    (
        _np,
        torch,
        transformers,
        model_path,
        model,
        processor,
        model_device,
        pydensecrf_stubbed,
    ) = load_model_and_processor(args, manifest)
    runtime = {
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "model_device": str(model_device),
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "pydensecrf_stubbed": pydensecrf_stubbed,
    }
    eos_token_id = getattr(processor.tokenizer, "eos_token_id", None)
    image_token_id = getattr(model.config, "image_token_id", None)

    report_samples: list[dict[str, Any]] = []
    test_root = manifest_path.parent
    suite_started = time.monotonic()
    for index, (image, prompt) in enumerate(samples, start=1):
        sample_id = f"{image['id']}:{prompt['id']}"
        image_path = (test_root / image["filename"]).resolve()
        print(f"\n[sample {index}/{len(samples)}] {sample_id}")
        sample_report: dict[str, Any] = {
            "sample_id": sample_id,
            "image_path": str(image_path),
            "prompt": prompt["text"],
        }
        try:
            if not image_path.exists():
                raise FileNotFoundError(f"image not found: {image_path}")
            messages = build_messages(image_path, prompt["text"])
            model_inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                max_image_patches=args.max_image_patches,
            )
            model_inputs = dict(to_model_device(model_inputs, first_parameter_device(model)))
            prompt_ids = [int(x) for x in model_inputs["input_ids"].detach().cpu().reshape(-1).tolist()]

            torch_started = time.monotonic()
            expected = torch_greedy_generate(
                torch,
                model,
                model_inputs,
                args.max_new_tokens,
                eos_token_id,
            )
            torch_seconds = time.monotonic() - torch_started
            ncnn_result = run_ncnn(args, image_path, prompt["text"])

            prompt_mismatch = first_mismatch(ncnn_result["prompt_ids"], prompt_ids)
            generated_mismatch = first_mismatch(ncnn_result["generated_ids"], expected)
            passed = prompt_mismatch is None and generated_mismatch is None
            sample_report.update({
                "status": "pass" if passed else "fail",
                "prompt_ids_match": prompt_mismatch is None,
                "generated_ids_match": generated_mismatch is None,
                "prompt_mismatch": prompt_mismatch,
                "generated_mismatch": generated_mismatch,
                "prompt_token_count": len(prompt_ids),
                "torch_image_token_count": prompt_ids.count(int(image_token_id)) if image_token_id is not None else None,
                "ncnn_image_token_count": ncnn_result["prompt_ids"].count(int(image_token_id)) if image_token_id is not None else None,
                "torch_prompt_ids": prompt_ids,
                "ncnn_prompt_ids": ncnn_result["prompt_ids"],
                "torch_generated_ids": expected,
                "ncnn_generated_ids": ncnn_result["generated_ids"],
                "torch_text": processor.decode(expected, skip_special_tokens=True),
                "ncnn_text": processor.decode(ncnn_result["generated_ids"], skip_special_tokens=True),
                "torch_seconds": torch_seconds,
                "ncnn_seconds": ncnn_result["seconds"],
            })
            if args.keep_runner_output:
                sample_report["runner_stdout"] = ncnn_result["stdout"]
            print(f"[torch] generated={expected} text={sample_report['torch_text']!r}")
            print(f"[ncnn]  generated={ncnn_result['generated_ids']} text={sample_report['ncnn_text']!r}")
            print(f"[{'PASS' if passed else 'FAIL'}] prompt_ids={prompt_mismatch is None} generated_ids={generated_mismatch is None}")
            if generated_mismatch:
                print(f"[mismatch] generated {generated_mismatch}")
        except Exception as exc:
            sample_report.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            print(f"[ERROR] {sample_report['error']}")
            report_samples.append(sample_report)
            if args.fail_fast:
                break
            continue
        report_samples.append(sample_report)
        if args.fail_fast and sample_report["status"] != "pass":
            break

    passed_count = sum(x["status"] == "pass" for x in report_samples)
    failed_count = sum(x["status"] == "fail" for x in report_samples)
    error_count = sum(x["status"] == "error" for x in report_samples)
    all_passed = len(report_samples) == len(samples) and passed_count == len(samples)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path),
        "model": str(model_path),
        "runner": str(args.runner),
        "runtime": runtime,
        "config": {
            "max_new_tokens": args.max_new_tokens,
            "max_image_patches": args.max_image_patches,
            "threads": args.threads,
            "eos_token_id": int(eos_token_id) if eos_token_id is not None else None,
            "image_token_id": int(image_token_id) if image_token_id is not None else None,
        },
        "summary": {
            "selected": len(samples),
            "executed": len(report_samples),
            "passed": passed_count,
            "failed": failed_count,
            "errors": error_count,
            "all_passed": all_passed,
            "seconds": time.monotonic() - suite_started,
        },
        "samples": report_samples,
    }
    output_path = args.output_json if args.output_json.is_absolute() else (ROOT / args.output_json).resolve()
    write_json(output_path, report)
    print(f"\n[summary] pass={passed_count} fail={failed_count} error={error_count} selected={len(samples)}")
    print(f"[write] {output_path}")
    raise SystemExit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
