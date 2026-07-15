# Qwen3-ASR-0.6B

This port reuses the shared `ncnn_llm` tokenizer, text embedding, decoder,
KV-cache, greedy sampler, CPU/Vulkan backends and model loader. Model-specific
code covers WAV ingestion, Whisper-compatible log-mel preprocessing, the audio
convolution stack, audio transformer and multimodal prompt splice.

## Pinned checkpoint and input

- Official checkpoint: `Qwen/Qwen3-ASR-0.6B`
- Immutable revision: `5eb144179a02acc5e5ba31e748d22b0cf3e303b0`
- Mainland-China mirror: ModelScope `Qwen/Qwen3-ASR-0.6B`
- Official sample: `https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav`

Recreate the checked-in 16 kHz fixture with FFmpeg 8.1 or a bit-exact
compatible build:

```bash
python models/qwen3-asr/prepare_audio.py \
  --output models/qwen3-asr/tests/asr_en_16k_pcm16.wav
```

## Convert and download

The one-command converter exports six pnnx graphs: mel, audio convolution,
audio transformer, text embedding, 28-layer decoder with ncnn KV cache, and
tied LM head.

```bash
python models/qwen3-asr/convert.py \
  --checkpoint /path/to/Qwen3-ASR-0.6B \
  --output /path/to/qwen3_asr_0.6b \
  --pnnx /path/to/pnnx
```

For a conversion receipt, adjust the three local paths in `conversion.json`
and run `python tools/run_conversion.py --spec models/qwen3-asr/conversion.json`.

The exact model package used for the results below is independently available
from the ncnn model-zoo mirror. Every file is pinned by byte size and SHA-256:

```bash
python tools/model_assets.py download \
  --manifest models/qwen3-asr/assets.json --root /path/to/qwen3_asr_0.6b
python tools/model_assets.py verify \
  --manifest models/qwen3-asr/assets.json --root /path/to/qwen3_asr_0.6b
```

## Build and run

Linux/WSL:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DNCNN_SOURCE_DIR=/path/to/ncnn -DNCNN_LLM_BUILD_TESTS=ON
cmake --build build --target asr_main ncnn_llm_tests
ctest --test-dir build --output-on-failure
build/asr_main --model /path/to/qwen3_asr_0.6b \
  --audio models/qwen3-asr/tests/asr_en_16k_pcm16.wav --max-new-tokens 256
```

Native Windows PowerShell uses the same CMake project:

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DNCNN_SOURCE_DIR=C:/path/to/ncnn -DNCNN_LLM_BUILD_TESTS=ON
cmake --build build --target asr_main ncnn_llm_tests
ctest --test-dir build --output-on-failure
build/asr_main.exe --model C:/path/to/qwen3_asr_0.6b `
  --audio models/qwen3-asr/tests/asr_en_16k_pcm16.wav --max-new-tokens 256
```

## Strict PyTorch/ncnn parity

```bash
python tools/qwen3_asr_parity.py \
  --pytorch-model /path/to/Qwen3-ASR-0.6B \
  --ncnn-model /path/to/qwen3_asr_0.6b \
  --ncnn-executable build/asr_main \
  --audio models/qwen3-asr/tests/asr_en_16k_pcm16.wav \
  --output-dir build/qwen3-asr-parity --device-map cpu
```

All three paths returned language `English` and this exact final text:

```text
Hmm. Oh yeah, yeah. He wasn't even that big when I started listening to him, but and his solo music didn't do overly well, but he did very well when he started writing for other people.
```

The final text is 186 UTF-8 bytes on each path with identical SHA-256
`817cafe879a57eb4b7f2e786eb766ae14ec5e0b3f5cef483ff4ac25bd54b3dc6`.

## Measured environment

| Path | Toolchain/runtime | Result |
|---|---|---|
| Windows 11 native | CMake 4.4, Ninja 1.13.2, GCC 16.1, ncnn `13b6d531` | build + CTest + real ASR exact |
| Ubuntu 24.04 WSL | CMake 3.28.3, Ninja 1.11.1, GCC 13.3, ncnn `13b6d531` | build + CTest + real ASR exact |
| PyTorch baseline | torch 2.13.0+cpu, Transformers 4.57.6, qwen-asr 0.0.6 | language/text exact |
