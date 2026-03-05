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
        << "  --model <path>             Model path (default: ./assets/qwen3_0.6b)\n"
        << "  --image <path>             Image path for VL models (optional)\n"
        << "  --use-vulkan               Enable Vulkan backend\n"
        << "  --no-builtin-tools         Disable built-in tools (random/add)\n"
        << "  --help                     Show this help\n"
        << "\n"
        << "Examples:\n"
        << "  " << (argv0 ? argv0 : "llm_ncnn_run") << "\n"
        << "  " << (argv0 ? argv0 : "llm_ncnn_run") << " --model ./assets/qwen3_0.6b\n"
        << "  " << (argv0 ? argv0 : "llm_ncnn_run") << " --model ./assets/qwen2.5_vl_3b --image ./assets/test.jpg\n";
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
        } else if (a == "--model") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --model\n";
                std::exit(2);
            }
            opt.model_path = argv[++i];
        } else if (a == "--image") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --image\n";
                std::exit(2);
            }
            opt.image_path = argv[++i];
        } else if (a == "--use-vulkan") {
            opt.use_vulkan = true;
        } else if (a == "--no-builtin-tools") {
            opt.enable_builtin_tools = false;
        } else {
            std::cerr << "Unknown option: " << a << "\n";
            print_usage(argv[0]);
            std::exit(2);
        }
    }
    return opt;
}
