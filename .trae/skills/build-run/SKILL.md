---
name: "build-run"
description: "Build and run ncnn_llm project using xmake. Invoke when user asks to compile, build, or run the project or any example targets."
---

# Build and Run ncnn_llm

This skill provides instructions for building and running the ncnn_llm project.

## Project Overview

ncnn_llm is a C++ project that provides Large Language Model (LLM) support for the ncnn framework. It uses **xmake** as the build system.

## Build Commands

### Configure Project

```powershell
xmake config -y -m release
```

For debug mode:

```powershell
xmake config -y -m debug
```

### Build Project

Build all targets:

```powershell
xmake build -y
```

Or simply:

```powershell
xmake -y
```

Build specific target:

```powershell
xmake build -y <target_name>
```

## Available Targets

The project provides the following executable targets:

| Target | Description |
|--------|-------------|
| `minicpm4_main` | MiniCPM4-0.5B model inference |
| `qwen3_main` | Qwen3 (0.6B) model inference |
| `qwen2.5_vl_main` | Qwen2.5-VL multimodal model (requires OpenCV) |
| `nllb_main` | NLLB translation model |
| `bytelevelbpe_main` | Byte-level BPE tokenizer example |
| `qwen3_openai_api` | OpenAI-style HTTP API server |
| `llm_ncnn_run` | Unified CLI/OpenAI server with auto-download and MCP support |
| `benchllm` | Benchmark tool |

## Run Commands

Run a specific target:

```powershell
xmake run <target_name>
```

### Examples

Run MiniCPM4:

```powershell
xmake run minicpm4_main
```

Run Qwen3:

```powershell
xmake run qwen3_main
```

Run llm_ncnn_run in CLI mode:

```powershell
xmake run llm_ncnn_run --mode cli --model qwen3_0.6b
```

Run llm_ncnn_run in OpenAI API mode:

```powershell
xmake run llm_ncnn_run --mode openai --port 8080 --model qwen3_0.6b
```

## Install to Distribution Directory

```powershell
xmake install -o dist -y
```

## Dependencies

The project requires the following dependencies (automatically managed by xmake):

- ncnn (master branch, with simple_vulkan enabled)
- nlohmann_json
- cpp-httplib
- libcurl (for llm_ncnn_run)
- opencv (optional, for vision models like qwen2.5_vl_main)

## Model Files

Models should be placed in the `assets/` directory. Download from:

https://mirrors.sdu.edu.cn/ncnn_modelzoo/

## Troubleshooting

### HTTPS Certificate Issues

If model download fails due to TLS certificate errors:

```powershell
$env:NCNN_LLM_CA_FILE = "C:\path\to\ca-certificates.crt"
xmake run llm_ncnn_run --mode openai --model qwen2.5_vl_3b
```

### Clean Build

```powershell
xmake clean
xmake build -y
```

### Reconfigure

```powershell
xmake f -c
xmake config -y -m release
xmake build -y
```
