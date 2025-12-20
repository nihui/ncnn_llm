#include "options.h"

#include "util.h"

#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>

namespace {

void print_usage(const char* argv0) {
    std::cout
        << "Usage: " << (argv0 ? argv0 : "llm_ncnn_run") << " [options]\n"
        << "\n"
        << "Options:\n"
        << "  --mode <cli|openai>        Run mode (default: cli)\n"
        << "  --model <path>             Model path (default: ./assets/qwen3_0.6b)\n"
        << "  --use-vulkan               Enable Vulkan backend\n"
        << "  --no-builtin-tools         Disable built-in tools (random/add)\n"
        << "  --port <n>                 Listen port for openai mode (default: 8080)\n"
        << "  --mcp-server <cmdline>     Launch an MCP server over stdio\n"
        << "  --mcp-transport <mode>     MCP stdio framing: lsp|jsonl (default: lsp)\n"
        << "  --no-mcp-merge-tools       Do not merge MCP tools into request tools\n"
        << "  --mcp-timeout-ms <n>       MCP request timeout in ms (default: 15000)\n"
        << "  --mcp-max-string-bytes <n> Truncate huge tool strings in prompt (default: 4096)\n"
        << "  --mcp-debug                Enable verbose MCP logs\n"
        << "  --help                     Show this help\n"
        << "\n"
        << "Examples:\n"
        << "  " << (argv0 ? argv0 : "llm_ncnn_run") << " --mode cli\n"
        << "  " << (argv0 ? argv0 : "llm_ncnn_run")
        << " --mode openai --port 8080 --mcp-server \"./my_mcp_server --flag\"\n";
}

std::optional<RunMode> parse_mode(const std::string& s) {
    if (s == "cli") return RunMode::Cli;
    if (s == "openai" || s == "server") return RunMode::OpenAI;
    return std::nullopt;
}

} // namespace

Options parse_options(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--") {
            break;
        }
        if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (a == "--mode") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --mode\n";
                std::exit(2);
            }
            auto mode = parse_mode(argv[++i]);
            if (!mode) {
                std::cerr << "Invalid --mode value (expected cli|openai)\n";
                std::exit(2);
            }
            opt.mode = *mode;
        } else if (a == "--model") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --model\n";
                std::exit(2);
            }
            opt.model_path = argv[++i];
        } else if (a == "--use-vulkan") {
            opt.use_vulkan = true;
        } else if (a == "--no-builtin-tools") {
            opt.enable_builtin_tools = false;
        } else if (a == "--port") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --port\n";
                std::exit(2);
            }
            auto v = parse_int(argv[++i]);
            if (!v) {
                std::cerr << "Invalid --port value\n";
                std::exit(2);
            }
            opt.port = *v;
        } else if (a == "--mcp-server") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --mcp-server\n";
                std::exit(2);
            }
            opt.mcp_server_cmdline = argv[++i];
        } else if (a == "--mcp-transport") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --mcp-transport\n";
                std::exit(2);
            }
            opt.mcp_transport = argv[++i];
            if (opt.mcp_transport != "lsp" && opt.mcp_transport != "jsonl") {
                std::cerr << "Invalid --mcp-transport value (expected lsp|jsonl)\n";
                std::exit(2);
            }
        } else if (a == "--no-mcp-merge-tools") {
            opt.mcp_merge_tools = false;
        } else if (a == "--mcp-timeout-ms") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --mcp-timeout-ms\n";
                std::exit(2);
            }
            auto v = parse_int(argv[++i]);
            if (!v) {
                std::cerr << "Invalid --mcp-timeout-ms value\n";
                std::exit(2);
            }
            opt.mcp_timeout_ms = *v;
        } else if (a == "--mcp-max-string-bytes") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --mcp-max-string-bytes\n";
                std::exit(2);
            }
            auto v = parse_int(argv[++i]);
            if (!v) {
                std::cerr << "Invalid --mcp-max-string-bytes value\n";
                std::exit(2);
            }
            opt.mcp_max_string_bytes_in_prompt = (size_t)*v;
        } else if (a == "--mcp-debug") {
            opt.mcp_debug = true;
        } else {
            std::cerr << "Unknown option: " << a << "\n";
            print_usage(argv[0]);
            std::exit(2);
        }
    }
    return opt;
}
