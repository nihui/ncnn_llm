#include "cli_runner.h"
#include "llm_ncnn_run/model_downloader.h"
#include "mcp.h"
#include "options.h"
#include "openai_server.h"
#include "tools.h"
#include "util.h"

#include "ncnn_llm_gpt.h"

#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>

int main(int argc, char** argv) {
    Options opt = parse_options(argc, argv);
    if (opt.mcp_server_cmdline.empty()) {
        if (const char* env = std::getenv("NCNN_LLM_MCP_SERVER")) {
            opt.mcp_server_cmdline = env;
        }
    }
    if (!opt.mcp_debug) {
        if (const char* env = std::getenv("NCNN_LLM_MCP_DEBUG")) {
            opt.mcp_debug = (std::string(env) == "1" || std::string(env) == "true" || std::string(env) == "TRUE");
        }
    }
    if (const char* env = std::getenv("NCNN_LLM_MCP_TRANSPORT")) {
        std::string v = env;
        if (v == "lsp" || v == "jsonl") opt.mcp_transport = v;
    }
    if (const char* env = std::getenv("NCNN_LLM_MCP_TIMEOUT_MS")) {
        if (auto v = parse_int(env)) opt.mcp_timeout_ms = *v;
    }
    if (const char* env = std::getenv("NCNN_LLM_MCP_MAX_STRING_BYTES")) {
        if (auto v = parse_int(env)) opt.mcp_max_string_bytes_in_prompt = (size_t)*v;
    }

    std::string dl_err;
    if (!ensure_model_present(opt.model_path, &dl_err)) {
        std::cerr << "Model download failed: " << dl_err << "\n";
        return 1;
    }

    McpState mcp = init_mcp(opt);

    ncnn_llm_gpt model(opt.model_path, opt.use_vulkan);
    std::vector<json> builtin_tools = opt.enable_builtin_tools ? make_builtin_tools() : std::vector<json>();
    auto builtin_router = make_builtin_router();
    std::mutex mcp_mutex;

    if (opt.mode == RunMode::Cli) {
        return run_cli(opt, model, builtin_tools, builtin_router, mcp, mcp_mutex);
    }
    return run_openai_server(opt, model, builtin_tools, builtin_router, mcp, mcp_mutex);
}
