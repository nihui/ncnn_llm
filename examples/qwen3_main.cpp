#include "qwen3_0.6b.h"
#include "utils/prompt.h"
#include <cstdio>
#include <iostream>
#include <unordered_map>
#include <functional>

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

    // 首次对话上下文（system）
    std::string prompt = apply_chat_template({
        {"system", "You are a helpful assistant."},
    }, {}, false, false);

    auto ctx = model.prefill(prompt);

    // 定义工具：自动生成 JSON schema
    auto random_tool = qwen3_0_6b::make_function_tool<int, int, int>(
        "random",
        "Generate a random number between two integers.",
        {"floor", "ceiling"}
    );
    auto add_tool = qwen3_0_6b::make_function_tool<int, int, int>(
        "add",
        "Add two integers.",
        {"a", "b"}
    );

    // 仅调用一次，将 tools 注入上下文
    ctx = model.define_tools(ctx, {random_tool, add_tool});

    // -------- 外部可定义的函数路由表 --------
    std::unordered_map<std::string, std::function<nlohmann::json(const nlohmann::json&)>> tool_router;
    tool_router["random"] = [](const nlohmann::json& args) {
        int lo = args.value("floor", 0);
        int hi = args.value("ceiling", 1);
        if (lo > hi) std::swap(lo, hi);
        int val = lo + (rand() % (hi - lo + 1));
        return nlohmann::json{{"value", val}};
    };
    tool_router["add"] = [](const nlohmann::json& args) {
        int a = args.value("a", 0);
        int b = args.value("b", 0);
        return nlohmann::json{{"value", a + b}};
    };

    while (true) {
        std::string input;
        std::cout << "User: ";
        std::getline(std::cin, input);
        if (input == "exit" || input == "quit") {
            break;
        }

        // 用户消息不必再重复带 tools
        std::string user_message = apply_chat_template({
            {"user", input}
        }, {}, true, false);
        ctx = model.prefill(user_message, ctx);
        
        std::cout << "Assistant: ";
        GenerateConfig cfg;
        cfg.beam_size = 2;          // 测试 beam 下的 tool call
        cfg.top_k = 40;
        cfg.top_p = 0.9;
        cfg.temperature = 0.7;
        cfg.do_sample = false;
        cfg.debug = true;           // 打开调试输出

        // 将外部函数路由到 tool_callback
        cfg.tool_callback = [&](const nlohmann::json& call) {
            nlohmann::json result;
            try {
                auto fname = call.at("name").get<std::string>();
                auto args  = call.value("arguments", nlohmann::json::object());
                if (auto it = tool_router.find(fname); it != tool_router.end()) {
                    result = it->second(args);
                } else {
                    result = nlohmann::json{{"error", "unknown function"}, {"name", fname}};
                }
            } catch (const std::exception& e) {
                result = nlohmann::json{{"error", e.what()}};
            }
            return nlohmann::json{
                {"result", result},
                {"call", call}
            };
        };

        ctx = model.generate(ctx, cfg, [](const std::string& token){
            std::cout << token << std::flush;
        });
        std::cout << "\n";
    }

    return 0;
}