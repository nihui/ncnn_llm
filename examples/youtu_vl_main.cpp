// SPDX-License-Identifier: Apache-2.0
#include "youtu_app.h"

#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>

namespace {

std::vector<std::string> utf8_args_storage;
std::vector<char*> utf8_argv_ptrs;

void make_utf8_argv(int& argc, char**& argv) {
    int wargc = 0;
    LPWSTR* wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
    if (!wargv) return;
    SetConsoleOutputCP(CP_UTF8);
    utf8_args_storage.reserve(wargc);
    for (int i = 0; i < wargc; ++i) {
        const int len = WideCharToMultiByte(
            CP_UTF8, 0, wargv[i], -1, nullptr, 0, nullptr, nullptr);
        std::string value(len > 0 ? len : 0, '\0');
        if (len > 0) {
            WideCharToMultiByte(
                CP_UTF8, 0, wargv[i], -1, value.data(), len, nullptr, nullptr);
            value.pop_back();
        }
        utf8_args_storage.push_back(std::move(value));
    }
    LocalFree(wargv);
    utf8_argv_ptrs.reserve(utf8_args_storage.size() + 1);
    for (auto& value : utf8_args_storage) utf8_argv_ptrs.push_back(value.data());
    utf8_argv_ptrs.push_back(nullptr);
    argc = wargc;
    argv = utf8_argv_ptrs.data();
}

}  // namespace
#endif

int main(int argc, char** argv) {
#ifdef _WIN32
    make_utf8_argv(argc, argv);
#endif
    return youtu::run_app(argc, argv);
}
