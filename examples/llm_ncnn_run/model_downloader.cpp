#include "model_downloader.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

std::string normalize_base_url(std::string base) {
    if (!base.empty() && base.back() != '/') base.push_back('/');
    return base;
}

bool has_model_files(const fs::path& dir) {
    if (!fs::exists(dir) || !fs::is_directory(dir)) return false;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        const auto name = entry.path().filename().string();
        if (name == "model.json") return true;
        if (entry.path().extension() == ".param") return true;
    }
    return false;
}

std::string quote_arg(const std::string& s) {
#ifdef _WIN32
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else out += c;
    }
    out += "\"";
    return out;
#else
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    out += "'";
    return out;
#endif
}

bool command_exists(const std::string& cmd) {
#ifdef _WIN32
    std::string check = "where " + cmd + " >nul 2>&1";
#else
    std::string check = "command -v " + cmd + " >/dev/null 2>&1";
#endif
    return std::system(check.c_str()) == 0;
}

bool run_command_capture(const std::string& cmd, std::string& out) {
#ifdef _WIN32
    FILE* pipe = _popen(cmd.c_str(), "r");
#else
    FILE* pipe = popen(cmd.c_str(), "r");
#endif
    if (!pipe) return false;

    char buffer[4096];
    while (fgets(buffer, sizeof(buffer), pipe)) {
        out.append(buffer);
    }

#ifdef _WIN32
    int rc = _pclose(pipe);
#else
    int rc = pclose(pipe);
#endif
    return rc == 0;
}

bool fetch_url(const std::string& url, std::string& out, std::string* err) {
    if (command_exists("curl")) {
        std::string cmd = "curl -f -L -s " + quote_arg(url);
        if (run_command_capture(cmd, out)) return true;
        if (err) *err = "curl failed to fetch " + url;
        return false;
    }
    if (command_exists("wget")) {
        std::string cmd = "wget -q -O - " + quote_arg(url);
        if (run_command_capture(cmd, out)) return true;
        if (err) *err = "wget failed to fetch " + url;
        return false;
    }
    if (err) *err = "curl/wget not found; please install one to auto-download models";
    return false;
}

bool download_file(const std::string& url, const fs::path& out_path, std::string* err) {
    const std::string out_str = out_path.string();
    if (command_exists("curl")) {
        std::string cmd = "curl -f -L --retry 3 --retry-delay 1 -o " + quote_arg(out_str) + " " + quote_arg(url);
        if (std::system(cmd.c_str()) == 0) return true;
        if (err) *err = "curl failed to download " + url;
        return false;
    }
    if (command_exists("wget")) {
        std::string cmd = "wget -O " + quote_arg(out_str) + " " + quote_arg(url);
        if (std::system(cmd.c_str()) == 0) return true;
        if (err) *err = "wget failed to download " + url;
        return false;
    }
    if (err) *err = "curl/wget not found; please install one to auto-download models";
    return false;
}

std::vector<std::string> parse_listing_files(const std::string& html) {
    std::vector<std::string> files;
    const std::string needle = "href=\"./";
    size_t pos = 0;
    while ((pos = html.find(needle, pos)) != std::string::npos) {
        pos += needle.size();
        size_t end = html.find('"', pos);
        if (end == std::string::npos) break;
        std::string name = html.substr(pos, end - pos);
        pos = end + 1;
        if (name.empty()) continue;
        if (name.back() == '/') continue;
        files.push_back(std::move(name));
    }
    std::sort(files.begin(), files.end());
    files.erase(std::unique(files.begin(), files.end()), files.end());
    return files;
}

std::string infer_model_name(const fs::path& model_dir) {
    fs::path p = model_dir;
    if (p.filename().empty()) p = p.parent_path();
    return p.filename().string();
}

bool download_model_dir(const fs::path& model_dir, const std::string& base_url, std::string* err) {
    const std::string model_name = infer_model_name(model_dir);
    if (model_name.empty()) {
        if (err) *err = "invalid model path";
        return false;
    }

    const std::string base = normalize_base_url(base_url);
    const std::string list_url = base + model_name + "/";

    std::string listing;
    if (!fetch_url(list_url, listing, err)) {
        return false;
    }

    auto files = parse_listing_files(listing);
    if (files.empty()) {
        if (err) *err = "no files found at " + list_url;
        return false;
    }

    std::error_code ec;
    fs::create_directories(model_dir, ec);
    if (ec) {
        if (err) *err = "failed to create model dir: " + model_dir.string();
        return false;
    }

    for (const auto& f : files) {
        fs::path out_path = model_dir / f;
        if (fs::exists(out_path)) continue;
        std::cout << "Downloading " << f << "...\n";
        const std::string file_url = base + model_name + "/" + f;
        if (!download_file(file_url, out_path, err)) {
            return false;
        }
    }

    return true;
}

} // namespace

bool ensure_model_present(const std::string& model_dir, const std::string& base_url, std::string* err) {
    fs::path dir(model_dir);
    if (has_model_files(dir)) return true;
    if (!download_model_dir(dir, base_url, err)) return false;
    return has_model_files(dir);
}

bool ensure_model_present(const std::string& model_dir, std::string* err) {
    return ensure_model_present(model_dir, "https://mirrors.sdu.edu.cn/ncnn_modelzoo/", err);
}
