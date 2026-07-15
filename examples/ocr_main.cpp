#include <iostream>
#include <string>
#include <vector>
#include "ncnn_llm_ocr.h"
#include "utf8_args.h"

int main(int argc, char** argv) {
    enable_utf8_console();
    std::vector<std::string> args = get_utf8_args(argc, argv);

    std::string model_path = "assets/glm_ocr";
    std::string image_path;
    std::string prompt;
    bool prompt_set = false;
    int max_new_tokens = 1024;

    for (size_t i = 1; i < args.size(); i++) {
        const std::string& arg = args[i];
        if (arg == "--model" && i + 1 < args.size()) {
            model_path = args[++i];
        } else if (arg == "--image" && i + 1 < args.size()) {
            image_path = args[++i];
        } else if (arg == "--prompt" && i + 1 < args.size()) {
            prompt = args[++i];
            prompt_set = true;
        } else if (arg == "--max-new-tokens" && i + 1 < args.size()) {
            try {
                max_new_tokens = std::stoi(args[++i]);
            } catch (...) {
                fprintf(stderr, "Invalid --max-new-tokens value\n");
                return 1;
            }
        }
    }

    if (image_path.empty() || max_new_tokens <= 0) {
        fprintf(stderr, "Usage: %s --image <image_path> [--model <model_path>] [--prompt <prompt>] [--max-new-tokens <count>]\n", argv[0]);
        return 1;
    }

    printf("Loading OCR model from %s\n", model_path.c_str());

    ncnn_llm_ocr ocr(model_path, false, 4);
    if (!ocr.ok()) {
        fprintf(stderr, "Failed to load OCR model\n");
        return 1;
    }

    // Default prompt depends on the model when the user did not pass one.
    if (!prompt_set) {
        if (ocr.model_type() == "hunyuan_ocr") {
            prompt = "\xE6\xA3\x80\xE6\xB5\x8B\xE5\xB9\xB6\xE8\xAF\x86\xE5\x88\xAB\xE5\x9B\xBE\xE7\x89\x87"
                     "\xE4\xB8\xAD\xE7\x9A\x84\xE6\x96\x87\xE5\xAD\x97\xEF\xBC\x8C\xE5\xB0\x86\xE6\x96\x87"
                     "\xE6\x9C\xAC\xE5\x9D\x90\xE6\xA0\x87\xE6\xA0\xBC\xE5\xBC\x8F\xE5\x8C\x96\xE8\xBE\x93"
                     "\xE5\x87\xBA\xE3\x80\x82";  // 检测并识别图片中的文字，将文本坐标格式化输出。
        } else {
            prompt = "Read the text in the image.";
        }
    }

    printf("Loading image: %s\n", image_path.c_str());
    ncnn::Mat bgr = load_image_to_ncnn_mat(image_path);
    if (ncnn_mat_empty(bgr)) {
        fprintf(stderr, "Failed to load image: %s\n", image_path.c_str());
        return 1;
    }

    printf("Running OCR prefill with prompt: %s\n", prompt.c_str());
    auto ctx = ocr.prefill(prompt, bgr);

    printf("Generating text:\n");

    GenerateConfig cfg;
    cfg.max_new_tokens = max_new_tokens;
    cfg.temperature = 0.0f;
    cfg.top_p = 0.00001f;
    cfg.top_k = 1;
    cfg.repetition_penalty = (ocr.model_type() == "hunyuan_ocr") ? 1.03f : 1.1f;
    cfg.do_sample = 0;

    ctx = ocr.generate(ctx, cfg, [](const std::string& token) {
        printf("%s", token.c_str());
        fflush(stdout);
    });

    printf("\n\nDone.\n");
    return 0;
}
