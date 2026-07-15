<p align="center">
  <img src="logo.png" alt="ncnn_llm" width="220">
</p>

<h1 align="center">ncnn_llm</h1>

<p align="center">
  <b>基于 ncnn 的 LLM、VLM、OCR、翻译和嵌入模型推理运行时。</b>
</p>

<p align="center">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue"></a>
  <img alt="Build" src="https://img.shields.io/badge/build-CMake%20%7C%20xmake-4c8eda">
  <img alt="Backend" src="https://img.shields.io/badge/backend-ncnn-orange">
  <img alt="Platform" src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Android-lightgrey">
</p>

<p align="center">
  <a href="readme.md">English</a>
  ·
  <a href="#快速开始">快速开始</a>
  ·
  <a href="#支持模型">支持模型</a>
  ·
  <a href="#模型库">模型库</a>
</p>

---

`ncnn_llm` 提供一个轻量级 C++ 运行时，用于在 [ncnn](https://github.com/Tencent/ncnn) 上运行语言模型和嵌入模型。项目关注可落地的本地推理场景，包括边缘设备、桌面 CPU 和支持 Vulkan 的 GPU。

本项目源自 **nihui** 对 ncnn `kvcache` 的实验性工作，并在此基础上扩展出可复用的示例、模型加载、分词器、视觉预处理、OCR 推理和嵌入模型 API。

## 特性

- 统一的聊天和视觉语言模型 CLI 示例
- 支持 KV cache 的自回归解码，可运行在 CPU 或 Vulkan 后端
- 支持 Qwen / MiniCPM 风格的 LLM
- 支持 Qwen VL 图文输入
- 提供 GLM-OCR 图像文字识别示例
- 提供 NLLB 翻译示例
- 提供文本嵌入和多模态嵌入 API
- 支持 BPE 和 Unigram 分词器
- 使用 xmake 构建，示例程序保持小而独立

## 支持模型

| 类别 | 模型 | 状态 | 说明 |
| --- | --- | --- | --- |
| LLM | YoutuLLM | 已支持 | 聊天 / 文本生成 |
| LLM | MiniCPM4 | 已支持 | 聊天 / 文本生成 |
| LLM | Qwen3 | 已支持 | 聊天 / 文本生成 |
| VLM | Qwen3.5 | 已支持 | 图像 + 文本输入 |
| VLM | Qwen2.5-VL | 已支持 | 图像 + 文本输入 |
| OCR | GLM-OCR | 已支持 | OCR |
| OCR | HunyuanOCR | 已支持 | OCR |
| ASR | Qwen3 ASR | 已支持 | ASR |
| 翻译 | NLLB | 已支持 | 翻译 |
| 嵌入 | Jina-Embeddings-v5-Text-Nano | 已支持 | 768 维文本嵌入 |
| 嵌入 | Jina-CLIP-v2 | 已支持 | 1024 维文本 + 图像嵌入 |

## 快速开始

### 1. 依赖

- `xmake`
- 从 `master` 分支构建的 ncnn

### 2. 克隆仓库

```bash
git clone https://github.com/nihui/ncnn_llm.git
cd ncnn_llm
```

### 3. 构建

```bash
xmake build
```

也可以使用可复现的 CMake preset；它会获取固定版本的依赖并运行跨平台
单元测试与一致性工具测试：

```bash
cmake --preset ci
cmake --build --preset ci
ctest --preset ci
```

只构建单个 target：

```bash
xmake build llm_ncnn_run
```

### 4. 下载模型

从下面的镜像下载已转换的 ncnn 模型目录：

https://mirrors.sdu.edu.cn/ncnn_modelzoo/

将模型目录放到 `assets/` 下，例如：

```text
assets/
└── qwen3_0.6b/
    ├── model.json
    ├── *.ncnn.param
    ├── *.ncnn.bin
    └── tokenizer files
```

## CLI 聊天

`llm_ncnn_run` 是主要的交互式示例，支持文本模型和视觉语言模型。

```bash
xmake run llm_ncnn_run --model ./assets/qwen3_0.6b
```

指定运行参数：

```bash
xmake run llm_ncnn_run --model ./assets/qwen3_0.6b --threads 4
xmake run llm_ncnn_run --model ./assets/qwen3_0.6b --vulkan --vulkan-device 0
```

视觉语言输入：

```bash
xmake run llm_ncnn_run --model ./assets/qwen2.5_vl_3b --image ./assets/test.jpg
```

### CLI 选项

| 选项 | 说明 |
| --- | --- |
| `--model` | 模型目录 |
| `--threads` | CPU 线程数 |
| `--vulkan` | 启用 Vulkan 计算 |
| `--vulkan-device` | Vulkan 设备编号 |
| `--image` | VLM 输入图像路径 |
| `--builtin-tools` | 启用内置演示工具 |

示例会话：

```text
llm_ncnn_run (cli). Type 'exit' or 'quit' to end the conversation.
User: Hello
Assistant: Hello! How can I help you today?
```

## OCR

GLM-OCR 和 HunyuanOCR 使用各自的图像 prefill 路径，并复用共享文本解码运行时。

```bash
xmake build ocr_main
xmake run ocr_main --model ./assets/glm_ocr --image ./test_ocr.png --prompt "Read the text in the image."
xmake run ocr_main --model ./assets/hunyuan_ocr --image ./test_ocr.png --prompt "提取图中的文字。"
```

可以使用 `--max-new-tokens <count>` 限制解码长度，便于进行可复现的一致性和延迟测试。

HunyuanOCR 的 ncnn 转换模型可从
`https://mirrors.sdu.edu.cn/ncnn_modelzoo/hunyuan_ocr/` 下载。

同一个 OCR 程序也可以使用 CMake 构建：

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/path/to/ncnn \
  -DNCNN_LLM_FETCH_DEPS=ON
cmake --build build --target hunyuan_ocr_ncnn
```

`tools/hunyuan_ocr_parity.py` 会用同一张图片分别运行原版 PyTorch
模型和 ncnn 程序，并保存两边的原始输出。JSON 报告会记录严格 UTF-8
相等结果、忽略空白的字符相似度、输入输出哈希、模型配置哈希、平台、
运行时版本、设备、数据类型和注意力后端。默认通过阈值为 0.90；需要
逐字节一致时可加 `--require-exact`。

输出示例：

```text
Generating text:
Hello World 123
```

## 嵌入模型

`ncnn_embedding` 为文本嵌入和 CLIP 风格的图文嵌入提供统一 API。

### 文本嵌入

```bash
xmake build embedding_main
xmake run embedding_main --model ./assets/jina-embeddings-v5-text-nano
```

### CLIP 多模态嵌入

```bash
xmake build clip_main
xmake run clip_main --model ./assets/jina_clip_v2 --image ./assets/ganyu.jpg
```

### C++ API

```cpp
#include "ncnn_embedding.h"

ncnn_embedding embed("./assets/jina_clip_v2", false, 4);

std::vector<float> text_vec = embed.encode_text("Hello world");

if (embed.supports_image()) {
    std::vector<float> image_vec = embed.encode_image_file("./image.jpg");
    float score = cosine_similarity(text_vec, image_vec);
}
```

## 其他示例

| Target | 用途 |
| --- | --- |
| `llm_ncnn_run` | 统一聊天 / VLM CLI |
| `ocr_main` | GLM-OCR 推理 |
| `embedding_main` | 文本嵌入推理 |
| `clip_main` | CLIP 图文嵌入推理 |
| `nllb_main` | NLLB 翻译示例 |
| `unigram_main` | Unigram 分词器示例 |
| `benchllm` | LLM 性能测试 |
| `test_llm` | 单元测试 |

构建并运行测试：

```bash
xmake build test_llm
xmake run test_llm
```

运行 benchmark：

```bash
xmake build benchllm
xmake run benchllm [loop_count] [num_threads] [powersave] [gpu_device] [cooling_down] [seqlen]
```

## 模型库

已转换的 ncnn 模型权重可从下面的镜像下载：

https://mirrors.sdu.edu.cn/ncnn_modelzoo/

每个模型目录应包含 `model.json`、ncnn param/bin 文件和分词器文件。可以将模型目录放在 `assets/` 下，也可以通过 `--model` 直接传入路径。

## 配置

每个模型目录都由 `model.json` 描述。不同模型族的字段会有差异，一个典型文本模型配置如下：

```json
{
  "model_type": "llm",
  "params": {
    "embed_param": "embed.ncnn.param",
    "embed_bin": "embed.ncnn.bin",
    "decoder_param": "decoder.ncnn.param",
    "decoder_bin": "decoder.ncnn.bin",
    "lm_head_param": "lm_head.ncnn.param",
    "lm_head_bin": "lm_head.ncnn.bin"
  },
  "tokenizer": {
    "type": "bbpe",
    "vocab_file": "vocab.txt",
    "merges_file": "merges.txt"
  },
  "setting": {
    "attn_cnt": 32,
    "hidden_size": 1024,
    "rope": {
      "type": "RoPE",
      "rope_head_dim": 64,
      "rope_theta": 1000000.0
    }
  }
}
```

嵌入模型和 OCR 模型使用各自的 `model_type` 和参数区段。具体写法可以参考 `assets/` 下的模型文件。

## 项目结构

```text
ncnn_llm/
├── assets/                 # 本地模型目录和演示资源
├── benchmark/              # Benchmark 入口
├── examples/               # CLI 和功能示例
│   ├── llm_ncnn_run/       # 统一聊天 / VLM 运行器
│   ├── ocr_main.cpp        # OCR 示例
│   ├── embedding_main.cpp  # 文本嵌入示例
│   ├── clip_main.cpp       # CLIP 示例
│   └── nllb_main.cpp       # 翻译示例
├── export/                 # 导出脚本
├── src/                    # 核心运行时
│   ├── ncnn_llm_gpt.*      # LLM / VLM 运行时
│   ├── ncnn_llm_ocr.*      # OCR 图像 prefill + 共享解码
│   ├── ncnn_embedding.*    # 嵌入模型运行时
│   ├── ncnn_text_runtime.* # 共享文本解码辅助函数
│   └── utils/              # 分词器、图像、RoPE、prompt 工具
├── tests/                  # 单元测试
└── xmake.lua               # 构建配置
```

## 路线图

- 保持 decoder 和 KV cache 运行时在不同模型族之间共享
- 扩展更多模型架构和分词器支持
- 提升 Vulkan 和 CPU 推理性能
- 增加 INT8 量化支持
- 更完整地文档化模型导出流程

随着运行时演进，旧导出脚本可能会过时。建议优先参考最新模型示例和 `model.json` 文件。

## 社区

欢迎提交 issue、修复、转换模型和测试结果。

- QQ 群：`767178345`

## 许可证

Apache License 2.0。详见 [LICENSE](LICENSE)。
