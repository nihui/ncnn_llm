#include "cli_runner.h"

#include "json_utils.h"
#include "tools.h"

#include <iostream>

int run_cli(const Options& opt,
            ncnn_llm_gpt& model,
            const std::vector<json>& builtin_tools,
            const std::unordered_map<std::string, std::function<json(const json&)>>& builtin_router
#if NCNN_LLM_WITH_OPENCV
            , const cv::Mat& image
#endif
            ) {
    std::cout << "llm_ncnn_run (cli). Type 'exit' or 'quit' to end the conversation.\n";

    std::string system_prompt = "You are a helpful assistant.";
    std::string prompt = apply_chat_template({{"system", system_prompt}}, {}, false, false);
    auto ctx = model.prefill(prompt);

    if (!builtin_tools.empty()) {
        ctx = model.define_tools(ctx, builtin_tools, system_prompt);
    }

#if NCNN_LLM_WITH_OPENCV
    bool has_image = !image.empty();
    bool first_turn = true;
#else
    bool has_image = false;
    bool first_turn = false;
#endif

    while (true) {
        std::string input;
        std::cout << "User: ";
        if (!std::getline(std::cin, input)) break;
        if (input == "exit" || input == "quit") break;

#if NCNN_LLM_WITH_OPENCV
        if (first_turn && has_image) {
            std::string user_message = apply_chat_template({
                {"user", "<|vision_start|><|image_pad|><|vision_end|>" + input}
            }, {}, true, false);
            ctx = model.prefill(user_message, image, ctx);
            first_turn = false;
        } else {
            std::string user_message = apply_chat_template({
                {"user", input}
            }, {}, true, false);
            ctx = model.prefill(user_message, ctx);
        }
#else
        std::string user_message = apply_chat_template({
            {"user", input}
        }, {}, true, true);
        ctx = model.prefill(user_message, ctx);
#endif

        std::cout << "Assistant: ";
        GenerateConfig cfg;
        cfg.beam_size = 2;
        cfg.top_k = 40;
        cfg.top_p = 0.9f;
        cfg.temperature = 0.7f;
        cfg.do_sample = false;

        cfg.tool_callback = [&](const json& call) {
            json result;
            try {
                std::string fname = call.at("name").get<std::string>();
                json args = call.value("arguments", json::object());
                bool handled = false;

                if (!builtin_tools.empty()) {
                    if (auto it = builtin_router.find(fname); it != builtin_router.end()) {
                        result = it->second(args);
                        handled = true;
                    }
                }

                if (!handled) {
                    result = json{{"error", "unknown function"}, {"name", fname}};
                }
            } catch (const std::exception& e) {
                result = json{{"error", e.what()}};
            }

            // Output tool call and result
            std::cout << "\n[Tool Call]: " << call.dump() << "\n";
            std::cout << "[Tool Result]: " << result.dump() << "\n";
            std::cout << "Assistant: " << std::flush;

            return json{
                {"result", result},
                {"call", call}
            };
        };

        ctx = model.generate(ctx, cfg, [](const std::string& token) {
            std::cout << token << std::flush;
        });
        std::cout << "\n";
    }

    return 0;
}
