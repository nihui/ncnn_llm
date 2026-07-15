#!/usr/bin/env python3
"""Reproducibly convert Qwen3-TTS 0.6B CustomVoice into an ncnn package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


HERE = Path(__file__).resolve().parent
EXPORT = HERE / "export"


def run(*args: object, cwd: Path | None = None) -> None:
    command = [str(arg) for arg in args]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pnnx(pnnx_exe: Path, pt: Path, *options: str) -> tuple[Path, Path]:
    run(pnnx_exe, pt.name, *options, "fp16=0", cwd=pt.parent)
    param = pt.with_suffix(".ncnn.param")
    binary = pt.with_suffix(".ncnn.bin")
    if not param.is_file() or not binary.is_file():
        raise RuntimeError(f"pnnx did not produce {param.name} and {binary.name}")
    return param, binary


def copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def validate_native_ncnn_param(path: Path) -> None:
    unsupported = []
    for line in path.read_text(encoding="utf-8").splitlines()[2:]:
        layer_type = line.split(maxsplit=1)[0] if line.strip() else ""
        if layer_type.startswith("pnnx.") or layer_type.startswith("Tensor."):
            unsupported.append(layer_type)
    if unsupported:
        raise RuntimeError(f"{path.name} contains unsupported runtime layers: {sorted(set(unsupported))}")


def write_receipt(output: Path, checkpoint: Path, args: argparse.Namespace) -> None:
    files = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        if path.name == "conversion.json":
            continue
        files.append({
            "path": path.relative_to(output).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256(path),
        })
    receipt = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "pnnx": str(args.pnnx),
        "device": args.device,
        "test_case": {
            "frames": args.frames,
            "text": args.text,
            "language": args.language,
            "speaker": args.speaker,
        },
        "files": files,
    }
    (output / "conversion.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, ensure_ascii=False))


def locate_speech_tokenizer_weights(checkpoint: Path) -> Path:
    candidates = [
        checkpoint / "speech_tokenizer" / "model.safetensors",
        checkpoint / "speech_tokenizer_v2_25hz" / "model.safetensors",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("speech tokenizer model.safetensors was not found")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--pnnx", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, default=25)
    parser.add_argument("--text", default="Hello, welcome to Qwen text to speech.")
    parser.add_argument("--language", default="English")
    parser.add_argument("--speaker", default="Ryan")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse complete stage outputs after an interrupted long conversion.",
    )
    parser.add_argument(
        "--receipt-only",
        action="store_true",
        help="Refresh conversion.json for an already complete package and exit.",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    work = args.work.resolve()
    if not args.pnnx.is_file():
        raise FileNotFoundError(args.pnnx)
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(checkpoint / "config.json")
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    if args.receipt_only:
        if not (output / "model.json").is_file():
            raise FileNotFoundError(output / "model.json")
        write_receipt(output, checkpoint, args)
        return 0

    reference = work / "reference"
    prefix = f"talker_loop_{args.frames}f"
    codes_npy = reference / f"{prefix}_codes_ref.npy"
    codes_bin = reference / f"{prefix}_codes_ref_i32.bin"
    reference_files = (
        codes_npy,
        codes_bin,
        reference / f"{prefix}_wav_ref_f32.bin",
        reference / f"{prefix}_meta.json",
    )
    if not args.resume or not all(path.is_file() for path in reference_files):
        run(
            sys.executable,
            EXPORT / "export_qwen3_tts_long_ref.py",
            "--model", checkpoint,
            "--out-dir", reference,
            "--frames", args.frames,
            "--text", args.text,
            "--language", args.language,
            "--speaker", args.speaker,
            "--device", args.device,
            "--prefix", prefix,
        )
    else:
        print("[resume] reference", flush=True)
    first_frame = work / "first_frame_codes_i32.bin"
    first_frame.write_bytes(codes_bin.read_bytes()[: 16 * 4])

    talker = work / "talker"
    talker_outputs = (
        output / "talker" / "talker_prefill_dynamic_kv.ncnn.param",
        output / "talker" / "talker_decode_s1_kv.ncnn.param",
        output / "talker" / "talker.ncnn.bin",
        output / "talker" / "tts_pad_embed_f32.bin",
    )
    if not args.resume or not all(path.is_file() for path in talker_outputs):
        run(
            sys.executable,
            EXPORT / "export_qwen3_tts_talker_simple_rope.py",
            "--model", checkpoint,
            "--out-dir", talker,
            "--text", args.text,
            "--language", args.language,
            "--speaker", args.speaker,
            "--device", args.device,
            "--first-frame-codes", first_frame,
            "--trace-decode",
        )
        talker_meta = json.loads((talker / "talker_simple_rope_meta.json").read_text())
        seq_len = int(talker_meta["input_shape"][1])
        prefill_param, prefill_bin = pnnx(
            args.pnnx,
            talker / "talker_simple_rope_s22.pt",
            f"inputshape=[1,{seq_len},1024]f32,[1,1,{seq_len},{seq_len}]f32,[1,{seq_len},128]f32,[1,{seq_len},128]f32",
        )
        run(sys.executable, EXPORT / "add_ncnn_sdpa_kvcache.py", prefill_param)
        dynamic_prefill = talker / "talker_prefill_dynamic_kv.ncnn.param"
        run(
            sys.executable,
            EXPORT / "make_qwen3_tts_talker_prefill_dynamic.py",
            prefill_param,
            dynamic_prefill,
            "--seq-len", seq_len,
        )
        decode_param, decode_bin = pnnx(
            args.pnnx,
            talker / "talker_simple_rope_decode_s1.pt",
            "inputshape=[1,1,1024]f32,[1,1,1,1]f32,[1,1,128]f32,[1,1,128]f32",
        )
        run(sys.executable, EXPORT / "add_ncnn_sdpa_kvcache.py", decode_param)
        if sha256(prefill_bin) != sha256(decode_bin):
            raise RuntimeError("talker prefill/decode weights are not identical")
        copy(dynamic_prefill, talker_outputs[0])
        copy(decode_param, talker_outputs[1])
        copy(prefill_bin, talker_outputs[2])
        copy(talker / "tts_pad_embed_f32.bin", talker_outputs[3])
    else:
        print("[resume] talker", flush=True)

    codepred = work / "code_predictor"
    first_code = int.from_bytes(first_frame.read_bytes()[:4], "little", signed=True)
    body_out = output / "code_predictor" / "body"
    body_outputs = [body_out / f"code_predictor_body_s{seq:02d}.ncnn.param" for seq in range(2, 17)]
    body_outputs.append(body_out / "code_predictor_body_shared.ncnn.bin")
    if not args.resume or not all(path.is_file() for path in body_outputs):
        run(
            sys.executable,
            EXPORT / "export_qwen3_tts_code_predictor_body.py",
            "--model", checkpoint,
            "--talker-hidden", talker / "talker_simple_hidden_ref_f32.bin",
            "--first-code", first_code,
            "--out-dir", codepred,
            "--device", args.device,
            "--all-lengths",
        )
        shared_hash = None
        shared_bin = None
        for seq in range(2, 17):
            pt = codepred / "code_predictor_body_by_len" / f"code_predictor_body_s{seq:02d}.pt"
            param, binary = pnnx(
                args.pnnx,
                pt,
                f"inputshape=[1,{seq},1024]f32,[1,1,{seq},{seq}]f32,[1,{seq}]f32,[{seq}]i64",
            )
            validate_native_ncnn_param(param)
            digest = sha256(binary)
            if shared_hash is None:
                shared_hash, shared_bin = digest, binary
            elif digest != shared_hash:
                raise RuntimeError(f"code predictor weights differ at sequence length {seq}")
            copy(param, body_out / f"code_predictor_body_s{seq:02d}.ncnn.param")
        assert shared_bin is not None
        copy(shared_bin, body_out / "code_predictor_body_shared.ncnn.bin")
        for source in sorted((codepred / "code_predictor_weights").glob("*.f32")):
            copy(source, output / "code_predictor" / "weights" / source.name)
    else:
        print("[resume] code predictor", flush=True)

    decoder = work / "decoder"
    decoder.mkdir(parents=True, exist_ok=True)
    decoder_pt = decoder / "speech_decoder_from_hidden_s325.pt"
    run(
        sys.executable,
        EXPORT / "export_qwen3_tts_decoder.py",
        "--model", checkpoint,
        "--codes", codes_npy,
        "--out", decoder_pt,
        "--mode", "from-hidden",
        "--trace-seq-len", 325,
        "--device", args.device,
        "--dtype", "float32",
    )
    decoder_param, decoder_bin = pnnx(args.pnnx, decoder_pt, "inputshape=[1,325,1024]f32")
    copy(decoder_param, output / "decoder" / "speech_decoder_from_hidden_s325.ncnn.param")
    copy(decoder_bin, output / "decoder" / "speech_decoder_from_hidden_s325.ncnn.bin")

    run(
        sys.executable,
        EXPORT / "export_qwen3_tts_front_weights.py",
        "--safetensors", locate_speech_tokenizer_weights(checkpoint),
        "--out-dir", output / "front_weights",
    )
    frontend = work / "frontend"
    run(
        sys.executable,
        EXPORT / "export_qwen3_tts_frontend_nets.py",
        "--model", checkpoint,
        "--out-dir", frontend,
        "--pnnx", args.pnnx,
        "--device", args.device,
    )
    for name in (
        "talker_text_embed.ncnn.param", "talker_text_embed.ncnn.bin",
        "talker_codec_embed.ncnn.param", "talker_codec_embed.ncnn.bin",
    ):
        copy(frontend / name, output / "talker" / name)
    run(
        sys.executable,
        EXPORT / "export_qwen3_tts_tokenizer_txt.py",
        "--model", checkpoint,
        "--out", output / "talker" / "tokenizer.txt",
    )

    model_json = {
        "schema_version": 1,
        "model_type": "qwen3_tts_12hz_0.6b_customvoice",
        "front_weights_dir": "front_weights",
        "decoder_param": "decoder/speech_decoder_from_hidden_s325.ncnn.param",
        "decoder_bin": "decoder/speech_decoder_from_hidden_s325.ncnn.bin",
        "talker_prefill_param": "talker/talker_prefill_dynamic_kv.ncnn.param",
        "talker_decode_param": "talker/talker_decode_s1_kv.ncnn.param",
        "talker_bin": "talker/talker.ncnn.bin",
        "codepred_body_dir": "code_predictor/body",
        "codepred_weights_dir": "code_predictor/weights",
        "tts_pad_embed": "talker/tts_pad_embed_f32.bin",
        "tokenizer": "talker/tokenizer.txt",
        "text_embed_param": "talker/talker_text_embed.ncnn.param",
        "text_embed_bin": "talker/talker_text_embed.ncnn.bin",
        "codec_embed_param": "talker/talker_codec_embed.ncnn.param",
        "codec_embed_bin": "talker/talker_codec_embed.ncnn.bin",
        "decoder_chunk_frames": 325,
        "decoder_context_frames": 25,
        "sample_rate": 24000,
        "samples_per_frame": 1920,
        "fp16": False,
    }
    (output / "model.json").write_text(
        json.dumps(model_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_receipt(output, checkpoint, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
