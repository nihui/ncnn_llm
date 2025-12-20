#pragma once

#include <cstddef>
#include <string>

enum class RunMode {
    Cli,
    OpenAI
};

struct Options {
    RunMode mode = RunMode::Cli;
    std::string model_path = "./assets/qwen3_0.6b";
    bool use_vulkan = false;
    bool enable_builtin_tools = true;
    int port = 8080;
    std::string mcp_server_cmdline;
    bool mcp_merge_tools = true;
    int mcp_timeout_ms = 15000;
    bool mcp_debug = false;
    std::string mcp_transport = "lsp"; // lsp|jsonl
    size_t mcp_max_string_bytes_in_prompt = 4096;
};

Options parse_options(int argc, char** argv);
