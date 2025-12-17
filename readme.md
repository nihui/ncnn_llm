# ncnn llm
ncnn llm 旨在为ncnn框架提供大语言模型（LLM）的支持。ncnn 是一个高性能的神经网络前向计算框架，专为移动设备和嵌入式设备设计。通过集成大语言模型，ncnn llm 使得在资源受限的环境中运行复杂的自然语言处理任务成为可能。

ncnn llm is designed to provide support for large language models (LLM) within the ncnn framework. ncnn is a high-performance neural network inference framework optimized for mobile and embedded devices. By integrating large language models, ncnn llm enables the execution of complex natural language processing tasks in resource-constrained environments.

## 项目起源

本项目起源于nihui为ncnn添加了kvcache功能，这使得在ncnn上运行大语言模型成为可能。本人本着为社区贡献的精神，决定将这一功能进行整理和扩展，形成一个独立的项目，以便更多的开发者能够方便地使用和贡献。

The project originated from nihui's addition of the kvcache feature to ncnn, which made it possible to run large language models on ncnn. Motivated by a spirit of community contribution, I decided to organize and expand this functionality into an independent project, making it easier for more developers to use and contribute.

**ncnn对kvcache的支持处于实验性阶段，请编译master分支以获得最新功能。**

**ncnn's support for kvcache is in an experimental stage; please compile the master branch to obtain the latest features.**

## 目前状态

目前，ncnn llm 仍处于早期开发阶段，实现了基本的tokenizer和nllb模型的支持。

Currently, ncnn llm is still in the early stages of development, with basic support for tokenizers and the nllb model implemented.

本项目尽可能提供了详尽的文档和示例代码和完整的导出pipeline，帮助用户快速上手。但是不可避免的，随着库的更新老的导出pipeline可能会失效，用户可以参考示例代码进行调整或者提出issue寻求帮助。

The project provides detailed documentation, example code, and a complete export pipeline to help users get started quickly. However, as the library evolves, some older export pipelines may become obsolete. Users can refer to the example code for adjustments or raise issues for assistance.

## 未来计划

未来计划包括但不限于：

- 为上游提供相关优化补丁，提升ncnn对大语言模型的支持（直接提交上游，而不会出现在本项目中）
- 支持更多的模型和tokenizer
- 优化性能，提升推理速度和降低内存占用
- 增加更多的示例和文档，帮助用户更好地理解和使用本项目

Future plans include but are not limited to:
- Providing relevant optimization patches to upstream to enhance ncnn's support for large language models (directly submitted upstream and not appearing in this project)
- Supporting more models and tokenizers
- Optimizing performance to improve inference speed and reduce memory usage
- Adding more examples and documentation to help users better understand and use the project

欢迎大家关注和参与本项目，共同推动ncnn在大语言模型领域的发展！

TODO LIST:
- [x] MiniCPM4-0.5B
- [x] QWen3 0.6B
- [ ] INT8 量化
- [ ] 完善的推理过程

## 模型获取方法

模型可以从以下链接获取：
[ncnn modelzoo](https://mirrors.sdu.edu.cn/ncnn_modelzoo/)

## 编译和使用

```
git clone https://github.com/futz12/ncnn_llm.git
cd ncnn_llm
xmake build
xmake run minicpm4_main
```

## 示例：Qwen3 OpenAI API Server（支持 MCP stdio 工具）

该示例提供一个 OpenAI 风格的 HTTP API：`POST /v1/chat/completions`，并内置一个简单网页前端：`http://localhost:8080/`。

```
xmake build qwen3_openai_api
xmake run qwen3_openai_api
```

启动后访问：

- `http://localhost:8080/`（网页聊天）
- `http://localhost:8080/v1/chat/completions`（OpenAI 风格接口）

### 接入你自己的 MCP server（stdin/stdout）

该 demo 支持通过 stdio 启动 MCP server，并把 MCP 的 `tools/list` 注入到模型可用工具中；当模型产生 `<tool_call>` 时会调用 `tools/call`，并把工具结果作为 `tool_response` 注入继续生成。

> 注意：不同 MCP server 的 framing 可能不同。若你的 server 是“一行一个 JSON”（JSONL），需要加 `--mcp-transport jsonl`。

示例（JSONL）：

```
xmake run qwen3_openai_api --mcp-transport jsonl --mcp-server "$HOME/path/to/your-mcp-server --flag"
```

常用参数：

- `--mcp-debug`：打印 MCP 收发的 JSON（排查卡住/协议不匹配时很有用）
- `--mcp-timeout-ms <n>`：等待 MCP 响应的超时
- `--port <n>`：HTTP 监听端口（默认 8080）

### 图片工具的返回（base64 / file）

对于图片类工具（例如 `sd_txt2img`），demo **不会把 base64 图片塞回模型 prompt**（避免占用上下文/拖慢推理），而是通过 HTTP 响应的 `artifacts` 字段把图片交给前端渲染。

- 默认：`base64`（前端会直接用 `data:image/png;base64,...` 渲染）
- 可选：`file` 或 `both`（会写到 `examples/web/generated/` 并通过 `/generated/<name>.png` 访问）

你可以在请求 JSON 里控制：

```json
{
  "mcp_image_delivery": "base64"
}
```

返回示例（截断）：

```json
{
  "artifacts": [
    { "kind": "image", "mime_type": "image/png", "data_base64": "..." }
  ]
}
```

### 用 curl 调用接口

```
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"qwen3-0.6b",
    "stream": false,
    "messages":[
      {"role":"system","content":"You are a helpful assistant."},
      {"role":"user","content":"Hello!"}
    ]
  }'
```

![MCP绘图](assets/img/mcpimggen.png)


## 效果测试

minicpm4

```
 *  正在执行任务: xmake run minicpm4_main 

Chat with MiniCPM4-0.5B! Type 'exit' or 'quit' to end the conversation.
User: 你好
Assistant: 
你好，我是你的智能助手。我可以帮助你查询天气、新闻、音乐、翻译等。请问你有什么需要帮助的吗？
User: 测试
Assistant:  你好，我是你的智能助手。你好，请问有什么我可以帮助你的吗？
User: 你知道什么是opencv吗？
Assistant:  opencv，全称OpenCV，是一个开源的计算机视觉和机器学习软件库，它包含了许多用于图像和视频处理的算法和工具。它可以帮助 你处理和理解图像和视频数据，从而实现各种计算机视觉任务，如目标检测、图像分类、人脸识别等。你是否对某个具体的任务或者算法感兴趣 ？
```
