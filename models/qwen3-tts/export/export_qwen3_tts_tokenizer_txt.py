#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = Path(args.model)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    vocab = json.loads((model / "vocab.json").read_text(encoding="utf-8"))
    merges = (model / "merges.txt").read_text(encoding="utf-8").splitlines()
    if merges and merges[0].startswith("#"):
        merges = merges[1:]
    tok_cfg = json.loads((model / "tokenizer_config.json").read_text(encoding="utf-8"))
    added = {int(k): v["content"] for k, v in tok_cfg["added_tokens_decoder"].items()}

    n_vocab = max(vocab.values()) + 1
    by_id = [""] * n_vocab
    for token, idx in vocab.items():
        by_id[idx] = token

    with out.open("w", encoding="utf-8") as f:
        f.write(f"V {n_vocab}\n")
        for token in by_id:
            f.write(token + "\n")
        f.write(f"M {len(merges)}\n")
        for merge in merges:
            f.write(merge + "\n")
        f.write(f"A {len(added)}\n")
        for idx in sorted(added):
            f.write(f"{idx} {added[idx]}\n")

    print(f"wrote {out}: vocab={n_vocab} merges={len(merges)} added={len(added)}")


if __name__ == "__main__":
    main()
