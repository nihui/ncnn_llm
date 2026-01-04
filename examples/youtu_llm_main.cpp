// Copyright 2025 Tencent
// SPDX-License-Identifier: Apache-2.0

#include "ncnn_llm_gpt.h"
#include "utils/prompt.h"
#include <cstdio>
#include <iostream>


int main() {

    ncnn_llm_gpt model("./assets/youtu_llm_2b",
                       /*use_vulkan=*/false);

    std::cout << "Chat with Youtu-LLM-2B! Type 'exit' or 'quit' to end the conversation.\n";

    // first time
    std::string input;
    std::cout << "User: ";
    std::getline(std::cin, input);
    if (input == "exit" || input == "quit") {
        return 0;
    }

    std::string user_message = "<|begin_of_text|><|User|>" + input + "<|Assistant|><think>\n\n</think>\n\n";

    auto ctx = model.prefill(user_message);

    std::cout << "Assistant: ";
    GenerateConfig cfg;
    cfg.beam_size = 1;
    cfg.top_k = 20;
    cfg.top_p = 0.95f;
    cfg.temperature = 1.0f;
    cfg.repetition_penalty = 1.05f;
    cfg.do_sample = true;

    ctx = model.generate(ctx, cfg, [](const std::string& token){
        std::cout << token << std::flush;
    });
    std::cout << "\n";

    while (true) {
        std::string input;
        std::cout << "User: ";
        std::getline(std::cin, input);
        if (input == "exit" || input == "quit") {
            break;
        }

        std::string user_message = "<|begin_of_text|><|User|>" + input + "<|Assistant|><think>\n\n</think>\n\n";

        ctx = model.prefill(user_message, ctx);

        std::cout << "Assistant: ";
        GenerateConfig cfg;
        cfg.beam_size = 1;
        cfg.top_k = 20;
        cfg.top_p = 0.95f;
        cfg.temperature = 1.0f;
        cfg.repetition_penalty = 1.05f;
        cfg.do_sample = true;

        ctx = model.generate(ctx, cfg, [](const std::string& token){
            std::cout << token << std::flush;
        });
        std::cout << "\n";
    }

    return 0;
}
