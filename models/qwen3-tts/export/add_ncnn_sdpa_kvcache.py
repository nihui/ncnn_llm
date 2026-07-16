#!/usr/bin/env python3
import argparse
from pathlib import Path


def noutputs(line: str) -> int:
    f = line.split()
    return int(f[3]) if len(f) >= 4 else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("param")
    ap.add_argument("--backup-suffix", default=".nokv")
    args = ap.parse_args()
    path = Path(args.param)
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    if not lines or lines[0].strip() != "7767517":
        raise SystemExit(f"not an ncnn param: {path}")
    backup = path.with_name(path.name + args.backup_suffix)
    if not backup.exists():
        backup.write_text(raw, encoding="utf-8")

    body = lines[2:]
    insert_idx = None
    for i, line in enumerate(body):
        f = line.split()
        if len(f) >= 5 and f[0] == "Input":
            insert_idx = i
    if insert_idx is None:
        raise SystemExit("could not find Input layer")

    out = []
    sdpa_i = 0
    for line in body:
        f = line.split()
        if f and f[0] == "SDPA":
            nin = int(f[2])
            nout = int(f[3])
            blobs = f[4:4 + nin + nout]
            inputs = blobs[:nin]
            outputs = blobs[nin:]
            params = {}
            for p in f[4 + nin + nout:]:
                if "=" in p:
                    k, v = p.split("=", 1)
                    params[k] = v
            params["5"] = "1"
            params["7"] = "1"
            new_inputs = inputs + [f"cache_k{sdpa_i}", f"cache_v{sdpa_i}"]
            new_outputs = outputs + [f"out_cache_k{sdpa_i}", f"out_cache_v{sdpa_i}"]
            pstr = " ".join(f"{k}={params[k]}" for k in sorted(params, key=lambda x: int(x)))
            out.append("%-24s %-24s %d %d %s %s" % (
                "SDPA", f[1], len(new_inputs), len(new_outputs),
                " ".join(new_inputs + new_outputs), pstr))
            sdpa_i += 1
        else:
            out.append(line)

    cache_blobs = []
    for i in range(sdpa_i):
        cache_blobs += [f"cache_k{i}", f"cache_v{i}"]
    out.insert(insert_idx + 1, "%-24s %-24s 0 %d %s" % ("Input", "kv_cache", len(cache_blobs), " ".join(cache_blobs)))
    layer_lines = [line for line in out if line.strip()]
    new_lines = [lines[0], f"{len(layer_lines)} {sum(noutputs(line) for line in layer_lines)}"] + out
    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    print(f"patched {path}: sdpa={sdpa_i}, layers={len(layer_lines)}, blobs={sum(noutputs(line) for line in layer_lines)}")


if __name__ == "__main__":
    main()
