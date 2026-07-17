#!/usr/bin/env python3
"""Smoke-test dynamic-seq full decoder prefill through the C++ runner."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from _subprocess_utils import run_utf8

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "cpp" / "build" / "youtu_ncnn_decode_runner"
SMOKE_TIME_RE = re.compile(r"\[prefill\].*\bms=([0-9.]+)")
COMPARE_TIME_RE = re.compile(r"\[time\] prefill\.dynamic_ms=([0-9.]+) token_ms=([0-9.]+) speedup=([0-9.]+)")


def die(message: str, code: int = 2) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def parse_lengths(text: str) -> list[int]:
    values: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            die(f"length must be positive: {value}")
        values.append(value)
    if not values:
        die("no lengths provided")
    return values


def ids_for_length(length: int) -> str:
    if length == 1:
        ids = [128000]
    elif length == 2:
        ids = [128000, 198]
    else:
        ids = [128000] + [12864] * (length - 2) + [198]
    return ",".join(str(x) for x in ids)


def run_command(args: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
    return run_utf8(
        args,
        cwd=str(ROOT.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )


def print_output(output: str) -> None:
    for line in output.splitlines():
        print("    " + line)


def parse_smoke_ms(output: str) -> float | None:
    match = SMOKE_TIME_RE.search(output)
    return float(match.group(1)) if match else None


def parse_compare_time(output: str) -> tuple[float, float, float] | None:
    match = COMPARE_TIME_RE.search(output)
    if not match:
        return None
    return float(match.group(1)), float(match.group(2)), float(match.group(3))


def main() -> None:
    parser = argparse.ArgumentParser(description="Check dynamic prefill across prompt lengths.")
    parser.add_argument("--runner", default=str(RUNNER))
    parser.add_argument("--lengths", default="1,2,3,8,16,33")
    parser.add_argument("--compare-lengths", default="2,3")
    parser.add_argument("--logits-atol", default="2")
    parser.add_argument("--cache-atol", default="2")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    args = parser.parse_args()

    runner = Path(args.runner)
    if not runner.exists():
        die(f"runner not found: {runner}; build it first with cmake --build youtu_vl_ncnn_port/cpp/build -j4")

    ok = True
    smoke_rows: list[tuple[int, float | None, bool]] = []
    compare_rows: list[tuple[int, float | None, float | None, float | None, bool]] = []
    if not args.skip_smoke:
        for length in parse_lengths(args.lengths):
            cmd = [
                str(runner),
                "--ids",
                ids_for_length(length),
                "--full-prefill-ncnn",
                "--full-decoder-ncnn",
                "--max-new-tokens",
                "1",
            ]
            print(f"[smoke] length={length}")
            result = run_command(cmd, timeout=args.timeout)
            print_output(result.stdout)
            length_ok = result.returncode == 0 and "dynamic_seq=1" in result.stdout
            smoke_rows.append((length, parse_smoke_ms(result.stdout), length_ok))
            print(f"[{'PASS' if length_ok else 'FAIL'}] smoke.length{length} returncode={result.returncode}")
            ok = length_ok and ok

    if not args.skip_compare:
        for length in parse_lengths(args.compare_lengths):
            cmd = [
                str(runner),
                "--ids",
                ids_for_length(length),
                "--compare-prefill",
                "--logits-atol",
                args.logits_atol,
                "--cache-atol",
                args.cache_atol,
            ]
            print(f"[compare] length={length}")
            result = run_command(cmd, timeout=args.timeout)
            print_output(result.stdout)
            compare_ok = result.returncode == 0
            times = parse_compare_time(result.stdout)
            if times is None:
                compare_rows.append((length, None, None, None, compare_ok))
            else:
                compare_rows.append((length, times[0], times[1], times[2], compare_ok))
            print(f"[{'PASS' if compare_ok else 'FAIL'}] compare.length{length} returncode={result.returncode}")
            ok = compare_ok and ok

    if smoke_rows:
        print("[summary] smoke")
        for length, ms, row_ok in smoke_rows:
            ms_text = "n/a" if ms is None else f"{ms:.1f}"
            print(f"    length={length} dynamic_ms={ms_text} status={'PASS' if row_ok else 'FAIL'}")

    if compare_rows:
        print("[summary] compare")
        for length, dynamic_ms, token_ms, speedup, row_ok in compare_rows:
            dynamic_text = "n/a" if dynamic_ms is None else f"{dynamic_ms:.1f}"
            token_text = "n/a" if token_ms is None else f"{token_ms:.1f}"
            speedup_text = "n/a" if speedup is None else f"{speedup:.2f}x"
            print(
                f"    length={length} dynamic_ms={dynamic_text} "
                f"token_ms={token_text} speedup={speedup_text} status={'PASS' if row_ok else 'FAIL'}"
            )

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
