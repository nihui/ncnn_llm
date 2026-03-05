#include "cli_runner.h"
#include "options.h"
#include "tools.h"

#include "ncnn_llm_gpt.h"

#include <filesystem>
#include <iostream>
#include <string>

#if NCNN_LLM_WITH_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif

namespace {

std::string normalize_model_path(std::string path) {
    std::filesystem::path p(path);
    if (p.is_absolute()) return path;
    if (!p.has_parent_path()) {
        return (std::filesystem::path("./assets") / p).string();
    }
    return path;
}

} // namespace

int main(int argc, char** argv) {
    Options opt = parse_options(argc, argv);
    opt.model_path = normalize_model_path(opt.model_path);

    if (!std::filesystem::exists(opt.model_path)) {
        std::cerr << "Model path does not exist: " << opt.model_path << "\n";
        return 1;
    }

    ncnn_llm_gpt model(opt.model_path, opt.use_vulkan);
    std::vector<json> builtin_tools = opt.enable_builtin_tools ? make_builtin_tools() : std::vector<json>();
    auto builtin_router = make_builtin_router();

#if NCNN_LLM_WITH_OPENCV
    cv::Mat image;
    if (!opt.image_path.empty()) {
        image = cv::imread(opt.image_path, cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Failed to load image: " << opt.image_path << "\n";
            return 1;
        }
        std::cerr << "Image loaded: " << opt.image_path << "\n";
    }
    return run_cli(opt, model, builtin_tools, builtin_router, image);
#else
    if (!opt.image_path.empty()) {
        std::cerr << "Warning: --image option is not supported without OpenCV\n";
    }
    return run_cli(opt, model, builtin_tools, builtin_router);
#endif
}
