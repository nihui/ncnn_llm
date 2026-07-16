// SPDX-License-Identifier: Apache-2.0
//
// Penguin-VL-ncnn command line runner.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "penguin_vl.h"

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#pragma comment(lib, "shell32")
#endif

namespace {

#ifdef _WIN32
// On Windows the ANSI argv is encoded in the active code page (e.g. GBK/936),
// which corrupts non-ASCII prompts before they reach the (UTF-8) tokenizer.
// Rebuild argv from the wide command line and re-encode every argument as UTF-8.
std::vector<std::string> utf8_args_storage;
std::vector<char*> utf8_argv_ptrs;
void make_utf8_argv(int& argc, char**& argv) {
    int wargc = 0;
    LPWSTR* wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
    if (!wargv) return;
    SetConsoleOutputCP(CP_UTF8);
    utf8_args_storage.reserve(wargc);
    for (int i = 0; i < wargc; ++i) {
        int len = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, nullptr, 0, nullptr, nullptr);
        std::string s(len > 0 ? len : 0, '\0');
        if (len > 0) {
            WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, s.data(), len, nullptr, nullptr);
            s.pop_back();  // discard the terminating NUL written by Win32
        }
        utf8_args_storage.push_back(std::move(s));
    }
    LocalFree(wargv);
    utf8_argv_ptrs.reserve(utf8_args_storage.size() + 1);
    for (auto& s : utf8_args_storage) utf8_argv_ptrs.push_back(s.data());
    utf8_argv_ptrs.push_back(nullptr);
    argc = wargc;
    argv = utf8_argv_ptrs.data();
}
#endif


void print_usage(const char* argv0) {
    std::printf(
        "Penguin-VL-ncnn runner\n"
        "Usage: %s --model <dir> [--image <path>] --prompt <text> [options]\n\n"
        "Options:\n"
        "  --model <dir>          model directory containing model.json (required)\n"
        "  --prompt <text>        user prompt (required)\n"
        "  --image <path>         image path for vision-language input\n"
        "  --threads <n>          CPU threads (default 4)\n"
        "  --max-new-tokens <n>   max tokens to generate (default 512)\n"
        "  --sample               enable sampling (default: greedy / deterministic)\n"
        "  --temperature <f>      sampling temperature (default 1.0)\n"
        "  --top-k <n>            top-k sampling (default 0 = off)\n"
        "  --top-p <f>            top-p sampling (default 1.0 = off)\n"
        "  --vulkan              use the ncnn Vulkan backend\n"
        "  --output <path>       write only the final response text to a file\n"
        "  --think                keep the assistant thinking turn open\n"
        "  -h, --help             show this help\n",
        argv0);
}

std::string next_arg(int& i, int argc, char** argv, const char* name) {
    if (i + 1 >= argc) {
        std::fprintf(stderr, "error: %s requires a value\n", name);
        std::exit(2);
    }
    return argv[++i];
}

}  // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    make_utf8_argv(argc, argv);
#endif
    std::string model_dir, image_path, prompt, output_path;
    int threads = 4;
    bool use_vulkan = false;
    pvl::GenerateConfig cfg;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model") model_dir = next_arg(i, argc, argv, "--model");
        else if (a == "--image") image_path = next_arg(i, argc, argv, "--image");
        else if (a == "--prompt") prompt = next_arg(i, argc, argv, "--prompt");
        else if (a == "--threads") threads = std::stoi(next_arg(i, argc, argv, "--threads"));
        else if (a == "--max-new-tokens") cfg.max_new_tokens = std::stoi(next_arg(i, argc, argv, "--max-new-tokens"));
        else if (a == "--sample") cfg.do_sample = true;
        else if (a == "--temperature") cfg.temperature = std::stof(next_arg(i, argc, argv, "--temperature"));
        else if (a == "--top-k") cfg.top_k = std::stoi(next_arg(i, argc, argv, "--top-k"));
        else if (a == "--top-p") cfg.top_p = std::stof(next_arg(i, argc, argv, "--top-p"));
        else if (a == "--vulkan") use_vulkan = true;
        else if (a == "--output") output_path = next_arg(i, argc, argv, "--output");
        else if (a == "--think") cfg.add_think = true;
        else if (a == "-h" || a == "--help") { print_usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "unknown argument: %s\n", a.c_str()); print_usage(argv[0]); return 2; }
    }

    if (model_dir.empty() || prompt.empty()) {
        print_usage(argv[0]);
        return 2;
    }

    try {
        pvl::PenguinVL model(model_dir, threads, use_vulkan);
        std::printf("Assistant: ");
        std::fflush(stdout);
        const std::string response = model.chat(prompt, image_path, cfg, [](const std::string& piece) {
            std::printf("%s", piece.c_str());
            std::fflush(stdout);
        });
        std::printf("\n");
        if (!output_path.empty()) {
            std::ofstream output(output_path, std::ios::binary);
            if (!output) throw std::runtime_error("failed to open output file: " + output_path);
            output.write(response.data(), static_cast<std::streamsize>(response.size()));
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "\nerror: %s\n", e.what());
        return 1;
    }
    return 0;
}
