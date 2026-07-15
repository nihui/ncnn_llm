# Penguin-VL-2B ncnn port

This port implements the official `tencent/Penguin-VL-2B` checkpoint at the
immutable revision in `checkpoint.json`. It reuses ncnn_llm's byte-level BPE,
Qwen3 decoding, sampling, RoPE and CPU/Vulkan backends, while adding the Penguin
image processor, 28-layer bidirectional vision encoder, projector, 2D-RoPE and
visual-token insertion path.

The deployed `penguin_vl_main` executable depends on ncnn and the C++ standard
library only. Python, PyTorch, Transformers and pnnx are conversion/reference
dependencies and are not required at runtime.

## Fixed regression case

- Source image: `assets/inputs/newspaper.png` from the official
  `tencent-ailab/Penguin-VL` repository.
- Fixture recipe: `make_fixture.py` takes a deterministic top-aligned center
  square crop.
- Fixture SHA-256: `822e2c49b8f11461e1d7f3bba30a651877946d614374b8371ff7e224dcb1cd3f`.
- Prompt: `What month is printed at the top? Answer with the month only.`
- Processor budget: min 16, max 196 visual tokens; grid `1 x 14 x 14`.
- Greedy PyTorch result: `July`.
- Output token IDs: `28427 151645`.
- Text SHA-256: `e9cf7a1ca4a760843ff30e1fbb125e6a066c63e883c58523bdbe04c9daf5dbae`.

`reference.py` uses official model code. If flash-attn is unavailable, it
replaces only the single-image bidirectional attention call with mathematically
equivalent PyTorch SDPA; model weights, preprocessing, 2D-RoPE, projector,
prompt and decoder remain official.

## Convert

Create an environment with PyTorch, pnnx and `requirements.txt`, download the
pinned checkpoint, then run:

```bash
python tools/run_conversion.py \
  --spec models/penguin-vl/conversion.json
```

The underlying fully explicit command is:

```bash
python models/penguin-vl/convert.py \
  --checkpoint /models/Penguin-VL-2B \
  --vision-encoder /models/Penguin-Encoder \
  --output /models/Penguin-VL-2B-ncnn \
  --pnnx /opt/pnnx
```

The conversion emits six pnnx/ncnn graphs: patch embedding, vision encoder,
projector, token embedding, Qwen3 decoder and tied language-model head. The
regression package fixes the vision graph at 196 patches, matching the committed
fixture and processor budget.

## Build

Linux/WSL:

```bash
cmake -S . -B build -G Ninja -DNCNN_SOURCE_DIR=/src/ncnn
cmake --build build --target penguin_vl_main penguin_vl_tests
ctest --test-dir build -R penguin_vl_tests --output-on-failure
```

Windows (PowerShell with Ninja/GCC or MSVC):

```powershell
cmake -S . -B build-win -G Ninja -DNCNN_SOURCE_DIR=C:/src/ncnn
cmake --build build-win --target penguin_vl_main penguin_vl_tests
ctest --test-dir build-win -R penguin_vl_tests --output-on-failure
```

## Run and compare

```bash
./build/penguin_vl_main \
  --model /models/Penguin-VL-2B-ncnn \
  --image models/penguin-vl/testdata/newspaper_top_square.png \
  --prompt "What month is printed at the top? Answer with the month only." \
  --max-new-tokens 16 \
  --output ncnn.txt

python models/penguin-vl/compare.py \
  --reference models/penguin-vl/testdata/pytorch.json \
  --ncnn ncnn.txt \
  --output parity.json
```

Both Linux/WSL and native Windows produce `July` byte-for-byte. An additional
date-reading diagnostic is kept in the validation directory: the official PIL
path produced `July 4, 1776`, while ncnn bicubic downsampling produced `1764`.
That diagnostic is deliberately not presented as a pass; it documents the
remaining pixel-sensitive OCR edge case transparently. Its structured baseline
is `testdata/pytorch-date-diagnostic.json`; `testdata/pytorch.json` is the sole
canonical passing baseline.

The Linux and Windows evidence logs, exact comparison receipts, asset hashes
and package URL are recorded under `docs/validation/penguin-vl/` and
`assets.json`.
