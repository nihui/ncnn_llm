#include "minicpm4_0.5b.h"
#include <cstdio>
#include <iostream>


int main() {
    
    minicpm4_0_5b model("./assets/minicpm4_0.5b/minicpm4_embed_token.ncnn.param",
                        "./assets/minicpm4_0.5b/minicpm4_embed_token.ncnn.bin",
                        "./assets/minicpm4_0.5b/minicpm4_proj_out.ncnn.param",
                        "./assets/minicpm4_0.5b/minicpm4_decoder.ncnn.param",
                        "./assets/minicpm4_0.5b/minicpm4_decoder.ncnn.bin",
                        "./assets/minicpm4_0.5b/vocab.txt",
                        "./assets/minicpm4_0.5b/merges.txt",
                       /*use_vulkan=*/false);

    std::cout << "Chat with MiniCPM4-0.5B! Type 'exit' or 'quit' to end the conversation.\n";

    std::string prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";

    auto ctx = model.prefill(prompt);

    while (true) {
        std::string input;
        std::cout << "User: ";
        std::getline(std::cin, input);
        if (input == "exit" || input == "quit") {
            break;
        }
        std::string user_message = "<|im_start|>user\n" + input + "<|im_end|>\n<|im_start|>assistant\n";
        ctx = model.prefill(user_message, ctx);
        
        std::cout << "Assistant: ";
        model.decode(ctx, [](const std::string& token) {
            std::string token_str = token;
            // replace /t/n with actual newline
            size_t pos = 0;
            while ((pos = token_str.find("\\n", pos)) != std::string::npos) {
                token_str.replace(pos, 2, "\n");
                pos += 1;
            }
            pos = 0;
            while ((pos = token_str.find("\\t", pos)) != std::string::npos) {
                token_str.replace(pos, 2, "\t");
                pos += 1;
            }

            // replace ▁ with space
            pos = 0;
            while ((pos = token_str.find("▁", pos)) != std::string::npos) {
                token_str.replace(pos, 3, " ");
                pos += 1;
            }

            std::cout << token_str << std::flush;
        });
        std::cout << std::endl;
    }

    return 0;
}