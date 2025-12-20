#include "cli_runner.h"

#include "json_utils.h"
#include "tools.h"

#include <iostream>

int run_cli(const Options& opt,
            ncnn_llm_gpt& model,
            const std::vector<json>& builtin_tools,
            const std::unordered_map<std::string, std::function<json(const json&)>>& builtin_router,
            const McpState& mcp,
            std::mutex& mcp_mutex) {
    std::cout << "llm_ncnn_run (cli). Type 'exit' or 'quit' to end the conversation.\n";

    std::string system_prompt = "You are a helpful assistant.";
    std::string prompt = apply_chat_template({{"system", system_prompt}}, {}, false, false);
    auto ctx = model.prefill(prompt);

    std::vector<json> tools = builtin_tools;
    if (!mcp.openai_tools.empty()) {
        tools = merge_tools_by_name(tools, mcp.openai_tools);
    }

    if (!tools.empty()) {
        ctx = model.define_tools(ctx, tools, system_prompt);
    }

    while (true) {
        std::string input;
        std::cout << "User: ";
        if (!std::getline(std::cin, input)) break;
        if (input == "exit" || input == "quit") break;

        std::string user_message = apply_chat_template({
            {"user", input}
        }, {}, true, false);
        ctx = model.prefill(user_message, ctx);

        std::cout << "Assistant: ";
        GenerateConfig cfg;
        cfg.beam_size = 2;
        cfg.top_k = 40;
        cfg.top_p = 0.9f;
        cfg.temperature = 0.7f;
        cfg.do_sample = false;
        cfg.debug = true;

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

                if (!handled && mcp.client && mcp.tool_names.find(fname) != mcp.tool_names.end()) {
                    std::string err;
                    {
                        std::lock_guard<std::mutex> lock(mcp_mutex);
                        result = mcp.client->call_tool(fname, args, &err);
                    }
                    if (!err.empty() || result.is_null()) {
                        result = json{{"error", "mcp tools/call failed"}, {"detail", err}};
                    }
                    result = strip_image_payloads(result);
                    result = truncate_large_strings(result, opt.mcp_max_string_bytes_in_prompt);
                    handled = true;
                }

                if (!handled) {
                    result = json{{"error", "unknown function"}, {"name", fname}};
                }
            } catch (const std::exception& e) {
                result = json{{"error", e.what()}};
            }
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
