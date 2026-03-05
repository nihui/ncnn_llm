#pragma once

#include <string>

struct Options {
    std::string model_path = "./assets/qwen3_0.6b";
    std::string image_path;
    bool use_vulkan = false;
    bool enable_builtin_tools = true;
};

Options parse_options(int argc, char** argv);
