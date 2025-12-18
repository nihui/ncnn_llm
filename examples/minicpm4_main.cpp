#include "model.h"
#include "utils/prompt.h"
#include <cstdio>
#include <iostream>


int main() {
    
    ncnn_llm_gpt model("./assets/minicpm4_0.5b",
                       /*use_vulkan=*/false);

    std::cout << "Chat with MiniCPM4-0.5B! Type 'exit' or 'quit' to end the conversation.\n";

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
        }, {}, true);

        ctx = model.prefill(user_message, ctx);
        
        std::cout << "Assistant: ";
        GenerateConfig cfg;
        cfg.beam_size = 2;
        cfg.top_k = 40;
        cfg.top_p = 0.9;
        cfg.temperature = 0.7;
        cfg.do_sample = false;

        ctx = model.generate(ctx, cfg, [](const std::string& token){
            std::cout << token << std::flush;
        });
        std::cout << std::endl;
    }

    return 0;
}