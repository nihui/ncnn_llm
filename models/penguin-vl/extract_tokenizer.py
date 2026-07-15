#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# Extract vocab.txt + merges.txt for the C++ byte-level BPE tokenizer.
#
# The Penguin-VL tokenizer is Qwen's byte-level BPE. HuggingFace stores vocab
# tokens already in GPT-2 byte->unicode encoded form, which is exactly what the
# C++ BpeTokenizer expects when use_byte_encoder=true. We therefore:
#   * write id_to_token as one token per line (line number == token id),
#     including added/special tokens at their true ids;
#   * copy the BPE merges (one "a b" pair per line, rank == line order).
#
# Usage:
#   python tools/extract_tokenizer.py --model tencent/Penguin-VL-2B --out ./penguin-vl-2b-ncnn

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Full id -> token table (covers base vocab + added/special tokens).
    vocab = tok.get_vocab()  # token -> id
    max_id = max(vocab.values())
    id_to_token = [""] * (max_id + 1)
    for t, i in vocab.items():
        id_to_token[i] = t

    vocab_path = os.path.join(args.out, "vocab.txt")
    with open(vocab_path, "w", encoding="utf-8") as f:
        for t in id_to_token:
            f.write(t.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t"))
            f.write("\n")
    print(f"[ok] wrote {vocab_path} ({len(id_to_token)} tokens)")

    # Merges: read from the fast tokenizer's serialized state. This works whether
    # the model came from the Hub or a local dir, and handles both the legacy
    # "a b" string form and the newer ["a", "b"] pair form.
    merges = None
    try:
        data = json.loads(tok.backend_tokenizer.to_str())
        merges = data.get("model", {}).get("merges")
    except Exception as e:
        print(f"[warn] could not read merges from backend tokenizer: {e}")

    merges_path = os.path.join(args.out, "merges.txt")
    with open(merges_path, "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        n = 0
        if merges:
            for m in merges:
                if isinstance(m, (list, tuple)):
                    f.write(f"{m[0]} {m[1]}\n")
                else:
                    f.write(f"{m}\n")
                n += 1
    print(f"[ok] wrote {merges_path} ({n} merges)")


if __name__ == "__main__":
    main()
