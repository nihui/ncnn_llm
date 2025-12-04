#include "qwen3_0.6b.h"
#include "utils/prompt.h"
#include <cstdio>
#include <iostream>


int main() {
    qwen3_0_6b model(
                        "./assets/qwen3_0.6b/qwen3_embed_token.ncnn.param",
                        "./assets/qwen3_0.6b/qwen3_embed_token.ncnn.bin",
                        "./assets/qwen3_0.6b/qwen3_proj_out.ncnn.param",
                        "./assets/qwen3_0.6b/qwen3_decoder.ncnn.param",
                        "./assets/qwen3_0.6b/qwen3_decoder.ncnn.bin",
                        "./assets/qwen3_0.6b/vocab.txt",
                        "./assets/qwen3_0.6b/merges.txt",
                       /*use_vulkan=*/false);

    std::cout << "Chat with Qwen3-0.6B! Type 'exit' or 'quit' to end the conversation.\n";

    std::string prompt = apply_chat_template({
        {"system", "You are a helpful assistant."},
    }, {}, false, false);

    auto ctx = model.prefill(prompt);

    while (true) {
        std::string input;
        std::cout << "User: ";
        std::getline(std::cin, input);
        if (input == "exit" || input == "quit") {
            break;
        }
        std::string user_message = apply_chat_template({
            {"user", input}
        }, {}, true, false);
        ctx = model.prefill(user_message, ctx);
        
        std::cout << "Assistant: ";
        GenerateConfig cfg;
        cfg.beam_size = 1;
        cfg.top_k = 40;
        cfg.top_p = 0.9;
        cfg.temperature = 0.7;
        cfg.do_sample = false;

        ctx = model.generate(ctx, cfg, [](const std::string& token){
            std::cout << token << std::flush;
        });
        std::cout << "\n";
    }

    return 0;
}