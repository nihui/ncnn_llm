#!/usr/bin/env python3
import argparse
from pathlib import Path


MAGIC = "7767517"


def read_lines(path: Path):
    return path.read_text().splitlines()


def write_text(path: Path, text: str):
    path.write_text(text if text.endswith("\n") else text + "\n")


def split_body_param(src: Path, dst: Path):
    lines = read_lines(src)
    if lines[0].strip() != MAGIC:
        raise RuntimeError(f"unexpected ncnn magic in {src}")

    header = lines[1].split()
    layers, blobs = int(header[0]), int(header[1])
    body_lines = lines[:-1]
    if not lines[-1].startswith("Gemm "):
        raise RuntimeError(f"last layer is not Gemm in {src}: {lines[-1]}")

    # Removing the final lm_head Gemm removes only the out1 blob.  Keep the
    # preceding Split so out0 remains the last-hidden output expected by runtime.
    body_lines[1] = f"{layers - 1} {blobs - 1}"
    write_text(dst, "\n".join(body_lines))


def parse_head_weight_bytes(param: Path) -> int:
    last = read_lines(param)[-1]
    if not last.startswith("Gemm "):
        raise RuntimeError(f"last layer is not Gemm in {param}: {last}")
    values = {}
    for tok in last.split()[5:]:
        if "=" in tok:
            k, v = tok.split("=", 1)
            values[int(k)] = int(v)
    n = values[8]
    k = values[9]
    return 4 + n * k * 4


def write_head_param(dst: Path):
    write_text(
        dst,
        "\n".join(
            [
                MAGIC,
                "2 2",
                "Input in0 0 1 in0",
                "Gemm head 1 1 in0 out0 10=-1 2=0 3=1 4=0 5=1 6=1 7=1 8=2048 9=1024",
            ]
        ),
    )


def copy_tail(src: Path, dst: Path, nbytes: int):
    size = src.stat().st_size
    if size <= nbytes:
        raise RuntimeError(f"{src} too small for tail {nbytes}")
    with src.open("rb") as f:
        f.seek(size - nbytes)
        data = f.read(nbytes)
    dst.write_bytes(data)


def copy_prefix(src: Path, dst: Path, nbytes: int):
    with src.open("rb") as f:
        data = f.read(src.stat().st_size - nbytes)
    dst.write_bytes(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    src = Path(args.src_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    prefill_param = src / "codepred_prefill_s2_step00.ncnn.param"
    prefill_bin = src / "codepred_prefill_s2_step00.ncnn.bin"
    decode01_param = src / "codepred_decode_s1_step01.ncnn.param"
    decode01_bin = src / "codepred_decode_s1_step01.ncnn.bin"

    head_bytes = parse_head_weight_bytes(decode01_param)
    split_body_param(prefill_param, out / "codepred_prefill_body_s2.ncnn.param")
    split_body_param(decode01_param, out / "codepred_decode_body_s1.ncnn.param")
    copy_prefix(prefill_bin, out / "codepred_body_shared.ncnn.bin", head_bytes)

    write_head_param(out / "codepred_head.ncnn.param")
    copy_tail(prefill_bin, out / "codepred_head_step00.ncnn.bin", head_bytes)
    for step in range(1, 15):
        copy_tail(src / f"codepred_decode_s1_step{step:02d}.ncnn.bin", out / f"codepred_head_step{step:02d}.ncnn.bin", head_bytes)

    total = sum(p.stat().st_size for p in out.glob("*.ncnn.bin"))
    print(f"wrote {out}")
    print(f"head_bytes={head_bytes}")
    print(f"shared_bin_bytes={(out / 'codepred_body_shared.ncnn.bin').stat().st_size}")
    print(f"total_bin_bytes={total}")


if __name__ == "__main__":
    main()
