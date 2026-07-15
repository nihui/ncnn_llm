# Qwen3-TTS 0.6B CustomVoice

This directory contains the reproducible pnnx conversion, C++/ncnn runtime
integration, and strict regression fixture for Tencent/ncnn#6791.  Runtime
inference does not import Python, PyTorch, or Transformers.

## Pinned checkpoint

- Model: `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`
- Revision: `85e237c12c027371202489a0ec509ded67b5e4b5`
- License: Apache-2.0
- Per-file sizes and SHA-256 values: `checkpoint.json`
- Converted package: `https://huggingface.co/March-77/Qwen3-TTS-0.6B-ncnn`
- Download manifest: `assets.json`

The converter produces a self-contained package with the talker prefill/decode
graphs, 15 fixed-shape codebook predictor graphs, audio decoder, C++ frontend
weights, tokenizer, `model.json`, and `conversion.json` receipt.

## Conversion

```bash
python models/qwen3-tts/convert.py \
  --checkpoint /path/to/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --output /path/to/qwen3-tts-ncnn \
  --work /path/to/qwen3-tts-work \
  --pnnx /path/to/pnnx \
  --device cuda:0 --frames 25
```

Use `--resume` after an interrupted conversion.  The converter supplies
explicit input shapes for every pnnx graph and rejects packages containing
residual `pnnx.*` or `Tensor.*` runtime layers.

## Build and run

```bash
cmake -S . -B build -DNCNN_SOURCE_DIR=/path/to/ncnn
cmake --build build --target qwen3_tts_main -j

build/qwen3_tts_main \
  --model /path/to/qwen3-tts-ncnn/model.json \
  --frames 2 \
  --text-file models/qwen3-tts/testdata/prompt.txt \
  --speaker Ryan --language English \
  --out output.wav --codes output-codes-i32.bin --threads 8
```

The same command was executed with native WSL/Linux and native Windows builds.
Both platforms produced byte-identical ncnn WAV and code files.

## Strict parity fixture

The committed fixture uses two 12 Hz frames, the English prompt in
`testdata/prompt.txt`, speaker `Ryan`, and greedy generation.  All 32 generated
audio codes match PyTorch exactly.  The final mono PCM16 waveform has 3,840
samples at 24 kHz; all samples are within one integer PCM LSB of the PyTorch
decoder result, and 3,732 samples are identical.  The one-LSB allowance only
accounts for the final float-to-PCM16 rounding boundary.

```bash
python models/qwen3-tts/compare.py \
  --reference-codes models/qwen3-tts/testdata/pytorch-exact-2f-codes-i32.bin \
  --reference-wav models/qwen3-tts/testdata/pytorch-exact-2f.wav \
  --ncnn-codes models/qwen3-tts/testdata/ncnn-wsl-exact-2f-codes-i32.bin \
  --ncnn-wav models/qwen3-tts/testdata/ncnn-wsl-exact-2f.wav
```

The 25-frame stress run is retained in the evidence logs.  It is not presented
as strict parity: autoregressive floating-point differences first change a
greedy branch at frame 6.  The two-frame fixture is the pinned exact regression
case required for review, while longer synthesis remains fully functional.

## Evidence

`evidence/` contains the official PyTorch log, pnnx conversion logs, native WSL
and Windows runtime logs, and machine-readable comparison reports.  Model
weights are intentionally hosted separately and verified through
`conversion.json`; they are not committed to git.
