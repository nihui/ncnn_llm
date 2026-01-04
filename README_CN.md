# ncnn_llm

英文版: [readme.md](readme.md)

**ncnn_llm** 为 [ncnn](https://github.com/Tencent/ncnn) 框架提供大语言模型 (LLM) 支持。

ncnn 是一个为移动端和嵌入式设备深度优化的高性能神经网络推理框架。通过将 LLM 集成到 ncnn 中，本项目使得在资源受限的环境（边缘设备、手机、IoT）上运行复杂的自然语言处理任务成为可能。

---

## 🚀 项目起源

本项目源自 **nihui** 为 ncnn 实现的 `kvcache` 功能，这为在 ncnn 上运行 LLM 打开了大门。受开源精神鼓舞，本仓库将该功能整理并扩展为一个独立项目。

目标是提供完整的流水线，方便开发者在 ncnn 上使用 LLM，并推动生态建设。

> **⚠️ 重要提示：**
> ncnn 对 `kvcache` 的支持目前仍处于 **实验阶段**。你 **必须** 从 `master` 分支编译 ncnn，以确保具备运行本项目所需的最新特性。

---

## 📊 模型支持矩阵

项目仍在积极开发中。以下是当前模型兼容性状态。

### ✅ 完全支持

*这些模型可以使用已实现的分词器和推理流程顺利运行。*

* **MiniCPM4-0.5B**
* **Qwen3** (0.6B)
* **Qwen2.5-VL**
* **NLLB** (No Language Left Behind)
* **Youtu-LLM-2B**

### ⚠️ 可运行但存在问题

*这些模型可以加载并运行，但可能存在 bug 或性能欠佳。*

* **Hunyuan 0.5B**

### 🚧 理论支持（开发中）

*这些模型理论上可以工作，但在当前版本中仍失败或未验证。*

* Qwen3-VL-2B-Instruct
* TinyLlama-1.1B-Chat-v1.0
* Qwen2.5-0.5B
* Llama-3.2-1B-Instruct
* DeepSeek-R1-Distill-Qwen-1.5b

### 🔜 即将支持

* Hunyuan OCR
* PaddleOCR-VL

---

## 🛠️ 构建与使用

本项目使用 `xmake` 构建。

### 1. 克隆仓库

```bash
git clone https://github.com/futz12/ncnn_llm.git
cd ncnn_llm

```

### 2. 构建

```bash
xmake build

```

### 3. 运行（示例：MiniCPM4）

运行前请确保已下载模型权重（见下文）。

```bash
xmake run minicpm4_main

```

### 示例输出

```text
 * Executing task: xmake run minicpm4_main 

Chat with MiniCPM4-0.5B! Type 'exit' or 'quit' to end the conversation.
User: Hello
Assistant: 
Hello, I am your intelligent assistant. I can help you check the weather, news, music, translation, etc. Is there anything you need help with?
User: Do you know what OpenCV is?
Assistant: OpenCV (Open Source Computer Vision Library) is an open-source computer vision and machine learning software library. It contains many algorithms and tools for image and video processing...

```

---

## 🚀 llm_ncnn_run（CLI / OpenAI API + 自动下载 + MCP）

`llm_ncnn_run` 是一个统一示例，支持：
- 两种模式：CLI 对话（`--mode cli`）和 OpenAI 风格的 HTTP 服务（`--mode openai`）
- 内置工具（random/add）+ 外部 MCP 工具
- 自动从 https://mirrors.sdu.edu.cn/ncnn_modelzoo/ 下载模型（解析 `model.json`）

### 构建

```bash
xmake build llm_ncnn_run
```

### 运行（CLI 模式）

```bash
xmake run llm_ncnn_run --mode cli --model qwen2.5_vl_3b
```

说明：
- 如果 `--model` 是纯名称（不含路径分隔符），会下载到 `./assets/<name>`。
- 也可以传入显式路径：`--model ./assets/qwen3_0.6b`。

### 运行（OpenAI API 模式）

```bash
xmake run llm_ncnn_run --mode openai --port 8080 --model qwen3_0.6b
```

接口：
- `http://localhost:8080/`（网页聊天）
- `http://localhost:8080/v1/chat/completions`（OpenAI 风格 API）

### MCP（stdio 工具）

CLI 模式：

```bash
xmake run llm_ncnn_run --mode cli --mcp-server "./my_mcp_server --flag"
```

OpenAI 模式：

```bash
xmake run llm_ncnn_run --mode openai --port 8080 --mcp-server "./my_mcp_server --flag"
```

常用 MCP 参数：
- `--mcp-transport lsp|jsonl`
- `--mcp-debug`
- `--mcp-timeout-ms <n>`

### HTTPS 证书问题

如果下载因 TLS 证书错误失败，请设置 CA 路径：

```bash
NCNN_LLM_CA_FILE=/etc/ssl/certs/ca-certificates.crt \
xmake run llm_ncnn_run --mode openai --model qwen2.5_vl_3b
```

---

## 📥 模型库

你可以从以下镜像下载已转换的 ncnn 模型权重：

🔗 **[ncnn 模型库镜像](https://mirrors.sdu.edu.cn/ncnn_modelzoo/)**

---

## 🔮 路线图

我们将持续改进 ncnn_llm，未来计划包括：

* **上游优化：** 向 ncnn 上游提交优化补丁，提升核心 LLM 支持。
* **支持扩展：** 增加更多模型架构与分词器支持。
* **性能：** 优化推理速度并降低内存占用。
* **INT8 量化：** 实现 INT8 量化支持。
* **文档：** 完善导出流水线文档并增加更多 C++ 使用示例。

*注：虽然我们提供完整的导出流水线，但旧的流程可能会随库演进而过时。请参考最新示例代码进行调整。*

---

## 🤝 社区与联系

欢迎大家关注并参与本项目，共同推动 ncnn 在大语言模型领域的发展！

* **QQ群：** `767178345`

---

## 📝 许可证

Apache 2.0
