# HunyuanOCR 1.0

This port uses the shared `ncnn_llm` tokenizer, greedy sampler, decoder/KV-cache,
CPU/Vulkan backends, model loader and the `ocr_main` executable. The deployed
program depends on ncnn and the C++ runtime only; PyTorch, Transformers and pnnx
are used for conversion and the reference run.

## Pinned inputs

- Official checkpoint: `tencent/HunyuanOCR`
- Revision: `45ca99a02acc15fc45a86e5fd5ca2ab57c1cc84d` (HunyuanOCR 1.0)
- Tested converted package: `assets.json`
- Test image source: official HunyuanOCR `assets/guwan1.png` at Git revision
  `068d890e570a05ac8e1cf8575572b692adf28694`
- Deterministic crop: `(left=308, top=78, right=568, bottom=598)`

The fixture is redistributable under the official repository's license and can
be recreated and hash-checked with:

```bash
python models/hunyuanocr/prepare_fixture.py \
  --output models/hunyuanocr/tests/guwan-right-column.png
```

## Convert with pnnx

Download the pinned checkpoint with Hugging Face or its ModelScope mirror, then
run the converter. The converter splits the checkpoint into vision encoder,
embedding, 24-layer text decoder, and tied LM-head graphs; exports each graph
through pnnx; and deterministically adds ncnn SDPA KV-cache inputs/outputs.

```bash
python models/hunyuanocr/convert.py \
  --checkpoint /path/to/HunyuanOCR-1.0 \
  --output /path/to/hunyuan_ocr \
  --pnnx /path/to/pnnx
```

To capture a conversion receipt, edit only the three local paths in
`conversion.json` and run:

```bash
python tools/run_conversion.py \
  --spec models/hunyuanocr/conversion.json \
  --receipt build/hunyuanocr-conversion-receipt.json
```

The tested package can instead be downloaded and verified without trusting the
transport or mirror:

```bash
python tools/model_assets.py download \
  --manifest models/hunyuanocr/assets.json --root /path/to/hunyuan_ocr
python tools/model_assets.py verify \
  --manifest models/hunyuanocr/assets.json --root /path/to/hunyuan_ocr
```

## Build

The same CMake project was built and run on native Windows and Ubuntu under WSL.

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DNCNN_SOURCE_DIR=/path/to/ncnn -DNCNN_LLM_BUILD_TESTS=ON
cmake --build build --target ocr_main ncnn_llm_tests
ctest --test-dir build --output-on-failure
```

PowerShell uses the same command shape:

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DNCNN_SOURCE_DIR=C:/path/to/ncnn -DNCNN_LLM_BUILD_TESTS=ON
cmake --build build --target ocr_main ncnn_llm_tests
ctest --test-dir build --output-on-failure
```

## Exact parity

Reference and ncnn were run with prompt `OCR`, greedy decoding, and 32 maximum
new tokens. The final output on all three paths was:

```text
清江浦河庫道
```

Linux/WSL end-to-end PyTorch + ncnn:

```bash
python tools/hunyuan_ocr_parity.py \
  --pytorch-model /path/to/HunyuanOCR-1.0 \
  --ncnn-model /path/to/hunyuan_ocr \
  --ncnn-executable build/ocr_main \
  --image models/hunyuanocr/tests/guwan-right-column.png \
  --prompt OCR --max-new-tokens 32 --pytorch-attention eager \
  --output-dir build/hunyuanocr-parity
```

Native Windows ncnn against the same saved PyTorch baseline:

```powershell
build/ocr_main.exe --model C:/path/to/hunyuan_ocr `
  --image models/hunyuanocr/tests/guwan-right-column.png `
  --prompt OCR --max-new-tokens 32
python tools/compare_outputs.py text `
  --reference models/hunyuanocr/tests/pytorch.txt `
  --actual models/hunyuanocr/tests/windows-ncnn.txt
```

`tests/linux-comparison.json` and `tests/windows-comparison.json` both record
strict equality: 19 bytes versus 19 bytes with identical SHA-256
`171c58169bf5fb3a5c6f7df69bd972da294ab8d494e15cc46167f9a7fb44738a`.
The raw runtime logs are retained beside those receipts.

## Measured environment

| Path | Toolchain/runtime | Result |
|---|---|---|
| Windows 11 native | CMake 4.4, Ninja 1.13.2, GCC 16.1, ncnn `13b6d531` | build + CTest 2/2 + real OCR exact |
| Ubuntu 24.04 WSL | CMake 3.28.3, Ninja 1.11.1, GCC 13.3, ncnn `13b6d531` | build + CTest 2/2 + real OCR exact |
| PyTorch baseline | torch 2.11.0+cu130, Transformers 4.57.1, CUDA BF16 eager | exact text baseline |
