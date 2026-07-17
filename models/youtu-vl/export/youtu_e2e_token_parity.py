#!/usr/bin/env python3
"""Compare PyTorch/HF greedy text generation against the ncnn C++ runner."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _subprocess_utils import run_utf8
from youtu_text_reference import (
    DEFAULT_MANIFEST,
    build_messages,
    die,
    first_parameter_device,
    import_runtime_deps,
    install_pydensecrf_import_stub,
    load_json,
    parse_dtype,
    prompt_text_from_messages,
    resolve_torch_device,
    to_model_device,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNNER = ROOT / "cpp" / "build" / "youtu_ncnn_decode_runner"
GENERATED_RE = re.compile(r"^\[ids\] generated=(\[.*\])$", re.MULTILINE)
PROMPT_RE = re.compile(r"^\[ids\] prompt=(\[.*\])$", re.MULTILINE)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_output_path(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == ROOT.name:
        return (ROOT.parent / path).resolve()
    return (ROOT / path).resolve()


def resolve_cli_path(path_like: str) -> Path:
    return resolve_output_path(path_like)


def parse_id_list(text: str) -> list[int]:
    ids: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if item:
            ids.append(int(item))
    if not ids:
        die("empty id list")
    return ids


def select_text_sample(manifest: dict[str, Any], sample_id: str) -> dict[str, Any]:
    for sample in manifest.get("samples", []):
        if sample.get("id") == sample_id:
            return sample
    available = ", ".join(str(x.get("id")) for x in manifest.get("samples", []))
    die(f"sample not found: {sample_id}; available: {available}")


def ids_arg(ids: list[int]) -> str:
    return ",".join(str(x) for x in ids)


def parse_runner_ids(output: str, pattern: re.Pattern[str], label: str) -> list[int]:
    match = pattern.search(output)
    if not match:
        die(f"runner output missing `{label}` line")
    try:
        value = ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError) as exc:
        die(f"failed to parse runner {label}: {exc}")
    if not isinstance(value, list) or not all(isinstance(x, int) for x in value):
        die(f"runner {label} is not an int list: {value!r}")
    return value


def first_mismatch(actual: list[int], expected: list[int]) -> str:
    limit = min(len(actual), len(expected))
    for index in range(limit):
        if actual[index] != expected[index]:
            return f"index={index} actual={actual[index]} expected={expected[index]}"
    if len(actual) != len(expected):
        return f"length actual={len(actual)} expected={len(expected)}"
    return ""


def load_model_and_processor(args):
    _, torch, transformers, AutoModelForCausalLM, AutoProcessor = import_runtime_deps()
    pydensecrf_stubbed = install_pydensecrf_import_stub()

    manifest = load_json(resolve_cli_path(args.manifest))
    model_info = manifest.get("model", {})
    model_arg = args.model or model_info.get("local_dir") or model_info.get("hf_model_id")
    if not model_arg:
        die("missing model path/id")
    if str(model_arg).startswith(("tencent/", "TencentCloudADP/")):
        model_ref = str(model_arg)
    else:
        model_path = resolve_cli_path(str(model_arg))
        if not model_path.exists():
            die(f"model path not found: {model_path}")
        model_ref = str(model_path)

    dtype = parse_dtype(torch, args.dtype)
    device = resolve_torch_device(torch, args.device)
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if dtype != "auto":
        kwargs["dtype"] = dtype
    if args.device_map != "none":
        kwargs["device_map"] = args.device_map

    print(f"[load] model={model_ref}")
    print(f"[load] device={device} dtype={args.dtype} transformers={transformers.__version__} pydensecrf_stubbed={pydensecrf_stubbed}")
    processor = AutoProcessor.from_pretrained(model_ref, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_ref, **kwargs)
    model.eval()
    if args.device_map == "none":
        model.to(device)
    return torch, model, processor, manifest


def build_prompt_ids(torch, processor, model, manifest: dict[str, Any], args) -> tuple[list[int], str]:
    if args.ids:
        return parse_id_list(args.ids), ""

    if args.prompt:
        sample = {"id": "cli_prompt", "text": args.prompt}
    else:
        sample = select_text_sample(manifest, args.sample)
    messages = build_messages(sample)
    prompt_text = prompt_text_from_messages(processor, messages)
    inputs = processor(text=prompt_text, return_tensors="pt")
    inputs = to_model_device(inputs, first_parameter_device(model))
    input_ids = inputs["input_ids"].detach().cpu().reshape(-1).tolist()
    return [int(x) for x in input_ids], prompt_text


def torch_greedy_generate(torch, model, prompt_ids: list[int], *, max_new_tokens: int, stop_at_eos: bool, eos_token_id: int | None):
    input_device = first_parameter_device(model)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=input_device)
    attention_mask = torch.ones_like(input_ids)
    generated: list[int] = []

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=False,
            output_attentions=False,
        )
        next_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        past = outputs.past_key_values

        for _step in range(max_new_tokens):
            token_id = int(next_id.item())
            generated.append(token_id)
            if stop_at_eos and eos_token_id is not None and token_id == int(eos_token_id):
                break
            cur_input = next_id.view(1, 1).to(input_device)
            one = torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=input_device)
            attention_mask = torch.cat([attention_mask, one], dim=1)
            outputs = model(
                input_ids=cur_input,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=False,
                output_attentions=False,
            )
            next_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
            past = outputs.past_key_values

    return generated


def run_ncnn_runner(args, prompt_ids: list[int], prompt_text: str) -> tuple[list[int], list[int], str]:
    runner = resolve_cli_path(args.runner)
    if not runner.exists():
        die(f"runner not found: {runner}; build it first with cmake --build youtu_vl_ncnn_port/cpp/build -j4")

    cmd = [
        str(runner),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--full-prefill-ncnn",
        "--full-decoder-ncnn",
    ]
    if args.runner_prompt:
        if not prompt_text:
            die("--runner-prompt requires --prompt or --sample, not --ids")
        cmd.extend(["--prompt", prompt_text, "--no-chat"])
    else:
        cmd.extend(["--ids", ids_arg(prompt_ids)])
    if args.echo_prompt:
        cmd.append("--echo-prompt")

    print("[ncnn] " + " ".join(cmd))
    result = run_utf8(
        cmd,
        cwd=str(ROOT.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=args.timeout,
        check=False,
    )
    print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.returncode != 0:
        die(f"runner failed with exit code {result.returncode}", code=result.returncode)
    return (
        parse_runner_ids(result.stdout, PROMPT_RE, "prompt"),
        parse_runner_ids(result.stdout, GENERATED_RE, "generated"),
        result.stdout,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end greedy token parity: PyTorch vs ncnn C++ runner.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--model", default="")
    parser.add_argument("--sample", default="en_short")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--ids", default="", help="Comma-separated prompt ids. Skips HF tokenization but still uses HF model.")
    parser.add_argument("--runner", default=str(DEFAULT_RUNNER))
    parser.add_argument("--max-new-tokens", type=int, default=3)
    parser.add_argument("--dtype", default="fp32", choices=["auto", "fp32", "float32", "bf16", "bfloat16", "fp16", "float16"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device-map", default="none")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--stop-at-eos", action="store_true")
    parser.add_argument("--runner-prompt", action="store_true", help="Pass prompt text to C++ instead of exact HF prompt ids.")
    parser.add_argument("--echo-prompt", action="store_true")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0:
        die("--max-new-tokens must be positive")

    torch, model, processor, manifest = load_model_and_processor(args)
    prompt_ids, prompt_text = build_prompt_ids(torch, processor, model, manifest, args)
    eos_token_id = getattr(processor.tokenizer, "eos_token_id", None)

    print(f"[prompt] tokens={len(prompt_ids)} ids={prompt_ids}")
    expected = torch_greedy_generate(
        torch,
        model,
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        stop_at_eos=args.stop_at_eos,
        eos_token_id=eos_token_id,
    )
    print(f"[torch] generated={expected}")

    runner_prompt_ids, actual, runner_stdout = run_ncnn_runner(args, prompt_ids, prompt_text)
    prompt_ok = runner_prompt_ids == prompt_ids if not args.runner_prompt else True
    tokens_ok = actual == expected
    mismatch = first_mismatch(actual, expected)
    print(f"[{'PASS' if prompt_ok else 'FAIL'}] prompt_ids actual_len={len(runner_prompt_ids)} expected_len={len(prompt_ids)}")
    print(f"[{'PASS' if tokens_ok else 'FAIL'}] generated_ids actual={actual} expected={expected}")
    if mismatch:
        print(f"[mismatch] {mismatch}")

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample": args.sample,
        "prompt_text": prompt_text,
        "prompt_ids": prompt_ids,
        "runner_prompt_ids": runner_prompt_ids,
        "torch_generated_ids": expected,
        "ncnn_generated_ids": actual,
        "prompt_ids_match": prompt_ok,
        "generated_ids_match": tokens_ok,
        "max_new_tokens": args.max_new_tokens,
        "runner_stdout": runner_stdout,
    }
    if args.output_json:
        output_path = resolve_output_path(args.output_json)
        write_json(output_path, report)
        print(f"[write] {output_path}")

    raise SystemExit(0 if prompt_ok and tokens_ok else 1)


if __name__ == "__main__":
    main()
