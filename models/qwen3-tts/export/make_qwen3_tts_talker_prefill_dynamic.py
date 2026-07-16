#!/usr/bin/env python3
"""Make a Qwen3-TTS talker prefill ncnn param prompt-length dynamic.

The pnnx-exported prefill graph bakes prompt length into Gemm constantM and
attention block Reshape layers. The weights are length-independent, so we can
rewrite only the shape metadata:

  Gemm    7=<seq_len>                 -> 7=0
  Reshape 0=128 1={8,16} 2=<seq_len>  -> 2=-1
  Reshape 0=128 1={8,16} 11=<seq_len> -> 11=-1
  Reshape 0=2048 1=<seq_len>          -> 1=-1

This is intentionally conservative and will fail if the source graph contains
unexpected occurrences of the prompt length.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seq-len", type=int, required=True)
    return parser.parse_args()


def rewrite_line(line: str, seq_len: int) -> tuple[str, bool]:
    parts = line.rstrip("\n").split()
    if not parts:
        return line, False

    changed = False

    if parts[0] == "Gemm":
        for i, tok in enumerate(parts):
            if tok == f"7={seq_len}":
                parts[i] = "7=0"
                changed = True
        return " ".join(parts) + "\n", changed

    if parts[0] == "Reshape":
        kv = {}
        for i, tok in enumerate(parts):
            m = re.fullmatch(r"(\d+)=(-?\d+)", tok)
            if m:
                kv[int(m.group(1))] = (int(m.group(2)), i)

        # q/k/v projection: [seq_len, heads, head_dim] in ncnn layout.
        if kv.get(0, (None,))[0] == 128 and kv.get(1, (None,))[0] in (8, 16) and kv.get(2, (None,))[0] == seq_len:
            parts[kv[2][1]] = "2=-1"
            changed = True
        if kv.get(0, (None,))[0] == 128 and kv.get(1, (None,))[0] in (8, 16) and kv.get(11, (None,))[0] == seq_len:
            parts[kv[11][1]] = "11=-1"
            changed = True

        # output projection: [seq_len, hidden] in ncnn layout.
        if kv.get(0, (None,))[0] == 2048 and kv.get(1, (None,))[0] == seq_len:
            parts[kv[1][1]] = "1=-1"
            changed = True

        return " ".join(parts) + "\n", changed

    return line, False


def main() -> int:
    args = parse_args()
    lines = args.input.read_text().splitlines(keepends=True)

    out_lines = []
    gemm_changed = 0
    reshape_changed = 0
    leftovers = []

    for lineno, line in enumerate(lines, 1):
        new_line, changed = rewrite_line(line, args.seq_len)
        out_lines.append(new_line)

        if changed and line.startswith("Gemm"):
            gemm_changed += 1
        elif changed and line.startswith("Reshape"):
            reshape_changed += 1

        if line.split(" ", 1)[0] in {"Gemm", "Reshape"} and re.search(rf"(^| ){args.seq_len}($| )|={args.seq_len}($| )", new_line):
            leftovers.append((lineno, new_line.rstrip()))

    if leftovers:
        print("unexpected seq_len leftovers in Gemm/Reshape:")
        for lineno, text in leftovers[:20]:
            print(f"  {lineno}: {text}")
        if len(leftovers) > 20:
            print(f"  ... {len(leftovers) - 20} more")
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(out_lines))

    print(f"wrote {args.output}")
    print(f"Gemm constantM rewritten: {gemm_changed}")
    print(f"Reshape seq dims rewritten: {reshape_changed}")
    if gemm_changed != 197 or reshape_changed != 112:
        print("warning: counts differ from the known Qwen3-TTS-0.6B talker prefill graph")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
