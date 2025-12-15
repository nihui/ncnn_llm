// Copyright 2025 Tencent
// SPDX-License-Identifier: Apache-2.0

#include "qwen2.5_vl_3b.h"
#include "utils/prompt.h"
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "usage: %s [image-path]\n", argv[0]);
        return -1;
    }

    const char* image_path = argv[1];

    cv::Mat bgr = cv::imread(image_path, 1);
    if (bgr.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", image_path);
        return -1;
    }

    qwen2_5_vl_3b model("./assets/qwen2.5_vl_3b/qwen2.5-vl_vision_embed_patch.ncnn.param",
                        "./assets/qwen2.5_vl_3b/qwen2.5-vl_vision_embed_patch.ncnn.bin",
                        "./assets/qwen2.5_vl_3b/qwen2.5-vl_vision_encoder.ncnn.param",
                        "./assets/qwen2.5_vl_3b/qwen2.5-vl_vision_encoder.ncnn.bin",
                        "./assets/qwen2.5_vl_3b/qwen2.5-vl_embed_token.ncnn.param",
                        "./assets/qwen2.5_vl_3b/qwen2.5-vl_embed_token.ncnn.bin",
                        "./assets/qwen2.5_vl_3b/qwen2.5-vl_proj_out.ncnn.param",
                        "./assets/qwen2.5_vl_3b/qwen2.5-vl_decoder.ncnn.param",
                        "./assets/qwen2.5_vl_3b/qwen2.5-vl_decoder.ncnn.bin",
                        "./assets/qwen2.5_vl_3b/vocab.txt",
                        "./assets/qwen2.5_vl_3b/merges.txt",
                       /*use_vulkan=*/false);

    std::cout << "Chat with Qwen2.5-VL-3B! Type 'exit' or 'quit' to end the conversation.\n";

    std::string prompt = apply_chat_template({
        {"system", "You are a helpful assistant."},
    }, {}, false, false);

    // std::cout << "prompt = " << prompt << std::endl;

    auto ctx = model.prefill(prompt);

    {
        std::string user_input = "分析图片内容";

        std::string user_message = apply_chat_template({
            {"user", "<|vision_start|><|image_pad|><|vision_end|>" + user_input}
        }, {}, true);

        // std::cout << "user_message = " << user_message << std::endl;

        ctx = model.prefill(user_message, bgr, ctx);

        GenerateConfig cfg;
        cfg.beam_size = 1;
        cfg.top_k = 50;
        cfg.top_p = 1.0;
        cfg.temperature = 1.0;
        cfg.repetition_penalty = 1.05;
        cfg.do_sample = false;
        cfg.max_new_tokens = 8192;

        ctx = model.generate(ctx, cfg, [](const std::string& token){
            std::cout << token << std::flush;
        });
        std::cout << std::endl;
    }

    while (true) {
        std::string input;
        std::cout << "User: ";
        std::getline(std::cin, input);
        if (input == "exit" || input == "quit") {
            break;
        }
        std::string user_message = apply_chat_template({
            {"user", input}
        }, {}, true);

        ctx = model.prefill(user_message, ctx);

        std::cout << "Assistant: ";

        GenerateConfig cfg;
        cfg.beam_size = 1;
        cfg.top_k = 50;
        cfg.top_p = 1.0;
        cfg.temperature = 1.0;
        cfg.repetition_penalty = 1.05;
        cfg.do_sample = false;
        cfg.max_new_tokens = 8192;

        ctx = model.generate(ctx, cfg, [](const std::string& token){
            std::cout << token << std::flush;
        });
        std::cout << std::endl;
    }

    return 0;
}
