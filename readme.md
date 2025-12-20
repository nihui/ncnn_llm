# ncnn_llm

**ncnn_llm** provides Large Language Model (LLM) support for the [ncnn](https://github.com/Tencent/ncnn) framework.

ncnn is a high-performance neural network inference framework specifically optimized for mobile and embedded devices. By integrating LLMs into ncnn, this project enables the execution of complex natural language processing tasks in resource-constrained environments (edge devices, mobile phones, IoT).

---

## 🚀 Project Origin

This project originated from **nihui's** implementation of the `kvcache` feature for ncnn, which opened the door for running LLMs on the framework. Motivated by the spirit of open-source contribution, this repository organizes and expands upon that functionality into an independent project.

The goal is to provide a complete pipeline, making it easier for developers to use LLMs on ncnn and contribute to the ecosystem.

> **⚠️ Important Note:**
> ncnn's support for `kvcache` is currently in an **experimental stage**. You **must** compile ncnn from the `master` branch to ensure you have the latest features required for this project to run.

---

## 📊 Model Support Matrix

The project is currently in active development. Below is the current compatibility status of various models.

### ✅ Perfectly Supported

*These models run smoothly with the implemented tokenizer and inference pipeline.*

* **MiniCPM4-0.5B**
* **Qwen3** (0.6B)
* **Qwen2.5-VL**
* **NLLB** (No Language Left Behind)

### ⚠️ Running with Issues

*These models can be loaded and run, but may experience bugs or suboptimal performance.*

* **Hunyuan 0.5B**

### 🚧 Theoretical Support (Work in Progress)

*These models should theoretically work but are currently failing or unverified in the current build.*

* Qwen3-VL-2B-Instruct
* TinyLlama-1.1B-Chat-v1.0
* Qwen2.5-0.5B
* Llama-3.2-1B-Instruct
* DeepSeek-R1-Distill-Qwen-1.5b

### 🔜 Coming Soon

* Hunyuan OCR
* PaddleOCR-VL

---

## 🛠️ Build and Usage

This project uses `xmake` for building.

### 1. Clone the Repository

```bash
git clone https://github.com/futz12/ncnn_llm.git
cd ncnn_llm

```

### 2. Build

```bash
xmake build

```

### 3. Run (Example: MiniCPM4)

Ensure you have downloaded the model weights (see below) before running.

```bash
xmake run minicpm4_main

```

### Example Output

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

## 📥 Model Zoo

You can download the converted ncnn-compatible model weights from the following mirror:

🔗 **[ncnn Model Zoo Mirror](https://mirrors.sdu.edu.cn/ncnn_modelzoo/)**

---

## 🔮 Roadmap

We are committed to improving ncnn_llm. Our future plans include:

* **Upstream Optimization:** Submitting optimization patches directly to the upstream ncnn repository to improve core LLM support.
* **Expanded Support:** Adding support for more model architectures and tokenizers.
* **Performance:** Optimizing inference speed and reducing memory footprint.
* **INT8 Quantization:** Implementing INT8 quantization support.
* **Documentation:** Improving the export pipeline docs and adding more C++ usage examples.

*Note: While we provide a complete export pipeline, older pipelines may become obsolete as the library evolves. Please refer to the latest example code for adjustments.*

---

## 🤝 Community & Contact

We welcome everyone to pay attention to and participate in this project to jointly promote the development of ncnn in the field of Large Language Models!

* **QQ Group:** `767178345`

---

## 📝 License

Apache 2.0
