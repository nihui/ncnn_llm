#include "model_downloader.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <curl/curl.h>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

namespace {

std::string normalize_base_url(std::string base) {
    if (!base.empty() && base.back() != '/') base.push_back('/');
    return base;
}

bool is_file_reference(const std::string& s) {
    const char* exts[] = {".param", ".bin", ".txt", ".json"};
    for (const char* ext : exts) {
        const size_t len = std::strlen(ext);
        if (s.size() >= len && s.compare(s.size() - len, len, ext) == 0) return true;
    }
    return false;
}

void collect_file_refs(const nlohmann::json& v, std::vector<std::string>& out) {
    if (v.is_string()) {
        const std::string s = v.get<std::string>();
        if (is_file_reference(s)) out.push_back(s);
        return;
    }
    if (v.is_array()) {
        for (const auto& el : v) collect_file_refs(el, out);
        return;
    }
    if (v.is_object()) {
        for (auto it = v.begin(); it != v.end(); ++it) {
            collect_file_refs(it.value(), out);
        }
    }
}

class CurlGlobal {
public:
    CurlGlobal() { curl_global_init(CURL_GLOBAL_DEFAULT); }
    ~CurlGlobal() { curl_global_cleanup(); }
};

struct ProgressState {
    std::string name;
    std::chrono::steady_clock::time_point last_tick = std::chrono::steady_clock::now();
    double last_pct = -1.0;
    curl_off_t last_bytes = 0;
    double last_speed = 0.0;
};

std::string format_bytes(double v) {
    const char* suffix[] = {"B", "KB", "MB", "GB", "TB"};
    size_t idx = 0;
    while (v >= 1024.0 && idx + 1 < (sizeof(suffix) / sizeof(suffix[0]))) {
        v /= 1024.0;
        ++idx;
    }
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.1f%s", v, suffix[idx]);
    return buf;
}

const char* find_ca_bundle_path() {
    static std::string path;
    if (!path.empty()) return path.c_str();

    const char* envs[] = {"NCNN_LLM_CA_FILE", "CURL_CA_BUNDLE", "SSL_CERT_FILE"};
    for (const char* env : envs) {
        if (const char* v = std::getenv(env)) {
            if (*v && fs::exists(v)) {
                path = v;
                return path.c_str();
            }
        }
    }

    if (const auto* vi = curl_version_info(CURLVERSION_NOW)) {
        if (vi->cainfo && fs::exists(vi->cainfo)) {
            path = vi->cainfo;
            return path.c_str();
        }
    }

#ifdef _WIN32
    return nullptr;
#else
    const char* candidates[] = {
        "/etc/ssl/certs/ca-certificates.crt",
        "/etc/ssl/cert.pem"
    };
    for (const char* c : candidates) {
        if (fs::exists(c)) {
            path = c;
            return path.c_str();
        }
    }
    return nullptr;
#endif
}

const char* find_ca_bundle_dir() {
    static std::string path;
    if (!path.empty()) return path.c_str();

    const char* envs[] = {"NCNN_LLM_CA_DIR", "CURL_CA_PATH", "SSL_CERT_DIR"};
    for (const char* env : envs) {
        if (const char* v = std::getenv(env)) {
            if (*v && fs::exists(v)) {
                path = v;
                return path.c_str();
            }
        }
    }

    if (const auto* vi = curl_version_info(CURLVERSION_NOW)) {
        if (vi->capath && fs::exists(vi->capath)) {
            path = vi->capath;
            return path.c_str();
        }
    }

#ifdef _WIN32
    return nullptr;
#else
    const char* candidates[] = {
        "/etc/ssl/certs"
    };
    for (const char* c : candidates) {
        if (fs::exists(c)) {
            path = c;
            return path.c_str();
        }
    }
    return nullptr;
#endif
}

size_t write_file_cb(void* ptr, size_t size, size_t nmemb, void* userdata) {
    std::ofstream* out = static_cast<std::ofstream*>(userdata);
    if (!out || !out->good()) return 0;
    out->write(static_cast<const char*>(ptr), static_cast<std::streamsize>(size * nmemb));
    return out->good() ? size * nmemb : 0;
}

size_t discard_cb(void*, size_t size, size_t nmemb, void*) {
    return size * nmemb;
}

struct RangeInfo {
    curl_off_t total = -1;
};

size_t header_cb(char* buffer, size_t size, size_t nmemb, void* userdata) {
    const size_t n = size * nmemb;
    if (!userdata || n == 0) return n;
    auto* info = static_cast<RangeInfo*>(userdata);
    std::string line(buffer, n);
    std::string lower = line;
    for (char& c : lower) c = (char)std::tolower((unsigned char)c);
    const std::string prefix = "content-range:";
    if (lower.rfind(prefix, 0) == 0) {
        auto slash = line.find('/');
        if (slash != std::string::npos) {
            std::string tail = line.substr(slash + 1);
            char* end = nullptr;
            long long total = std::strtoll(tail.c_str(), &end, 10);
            if (total > 0) info->total = (curl_off_t)total;
        }
    }
    return n;
}

int progress_cb(void* clientp, curl_off_t dltotal, curl_off_t dlnow, curl_off_t, curl_off_t) {
    auto* state = static_cast<ProgressState*>(clientp);
    if (!state) return 0;

    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - state->last_tick).count();
    if (elapsed < 200 && dltotal > 0) return 0;
    state->last_tick = now;

    if (elapsed > 0) {
        curl_off_t delta = dlnow - state->last_bytes;
        if (delta >= 0) {
            state->last_speed = (double)delta * 1000.0 / (double)elapsed;
        }
        state->last_bytes = dlnow;
    }

    const std::string speed_str = format_bytes(state->last_speed) + "/s";
    if (dltotal > 0) {
        double pct = (double)dlnow * 100.0 / (double)dltotal;
        if (pct < state->last_pct + 0.1 && elapsed < 500) return 0;
        state->last_pct = pct;
        std::fprintf(stderr, "\rDownloading %s: %5.1f%% (%s/%s) %s",
                     state->name.c_str(),
                     pct,
                     format_bytes((double)dlnow).c_str(),
                     format_bytes((double)dltotal).c_str(),
                     speed_str.c_str());
    } else {
        std::fprintf(stderr, "\rDownloading %s: %s %s",
                     state->name.c_str(),
                     format_bytes((double)dlnow).c_str(),
                     speed_str.c_str());
    }
    std::fflush(stderr);
    return 0;
}

void apply_curl_common(CURL* curl) {
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "ncnn_llm/llm_ncnn_run");
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 15L);

    const char* ca = find_ca_bundle_path();
    const char* ca_dir = find_ca_bundle_dir();
    if (ca && *ca) {
        curl_easy_setopt(curl, CURLOPT_CAINFO, ca);
    }
    if (ca_dir && *ca_dir) {
        curl_easy_setopt(curl, CURLOPT_CAPATH, ca_dir);
    }
}

bool get_remote_size(const std::string& url, curl_off_t* out_size, std::string* err) {
    if (!out_size) return false;
    static CurlGlobal curl_global;

    CURL* curl = curl_easy_init();
    if (!curl) return false;
    apply_curl_common(curl);
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 15L);
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 1L);
    CURLcode rc = curl_easy_perform(curl);
    if (rc == CURLE_OK) {
        curl_off_t len = -1;
        curl_easy_getinfo(curl, CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &len);
        curl_easy_cleanup(curl);
        if (len > 0) {
            *out_size = len;
            return true;
        }
    }
    curl_easy_cleanup(curl);

    CURL* range = curl_easy_init();
    if (!range) return false;
    RangeInfo info;
    apply_curl_common(range);
    curl_easy_setopt(range, CURLOPT_URL, url.c_str());
    curl_easy_setopt(range, CURLOPT_RANGE, "0-0");
    curl_easy_setopt(range, CURLOPT_WRITEFUNCTION, discard_cb);
    curl_easy_setopt(range, CURLOPT_HEADERFUNCTION, header_cb);
    curl_easy_setopt(range, CURLOPT_HEADERDATA, &info);
    curl_easy_setopt(range, CURLOPT_TIMEOUT, 15L);
    curl_easy_setopt(range, CURLOPT_NOPROGRESS, 1L);
    rc = curl_easy_perform(range);
    curl_easy_cleanup(range);
    if (rc == CURLE_OK && info.total > 0) {
        *out_size = info.total;
        return true;
    }
    if (err && rc != CURLE_OK) {
        *err = std::string("libcurl failed to query size ") + url + ": " + curl_easy_strerror(rc);
    }
    return false;
}

bool download_file(const std::string& url, const fs::path& out_path, std::string* err) {
    static CurlGlobal curl_global;
    std::ofstream ofs(out_path, std::ios::binary);
    if (!ofs) {
        if (err) *err = "failed to open output file: " + out_path.string();
        return false;
    }

    CURL* curl = curl_easy_init();
    if (!curl) {
        if (err) *err = "failed to initialize libcurl";
        return false;
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_file_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &ofs);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 0L);
    apply_curl_common(curl);

    ProgressState progress;
    progress.name = out_path.filename().string();
    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, progress_cb);
    curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &progress);
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);

    const char* ca = find_ca_bundle_path();
    const char* ca_dir = find_ca_bundle_dir();

    CURLcode rc = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    ofs.close();

    if (rc != CURLE_OK) {
        std::error_code ec;
        fs::remove(out_path, ec);
        std::fprintf(stderr, "\n");
        if (err) {
            std::string msg = std::string("libcurl failed to download ") + url + ": " + curl_easy_strerror(rc);
            if (ca && *ca) msg += " (CAINFO=" + std::string(ca) + ")";
            if (ca_dir && *ca_dir) msg += " (CAPATH=" + std::string(ca_dir) + ")";
            *err = msg;
        }
        return false;
    }

    std::fprintf(stderr, "\n");
    return true;
}

bool ensure_file(const fs::path& out_path, const std::string& url, std::string* err) {
    std::error_code ec;
    if (fs::exists(out_path)) {
        auto size = fs::file_size(out_path, ec);
        if (!ec && size > 0) {
            curl_off_t remote = -1;
            std::string size_err;
            if (get_remote_size(url, &remote, &size_err) && remote > 0) {
                if ((uintmax_t)remote == size) return true;
                std::cout << "Re-downloading " << out_path.filename().string() << " (size mismatch)\n";
            } else {
                return true;
            }
        }
        fs::remove(out_path, ec);
    }
    return download_file(url, out_path, err);
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

    std::error_code ec;
    fs::create_directories(model_dir, ec);
    if (ec) {
        if (err) *err = "failed to create model dir: " + model_dir.string();
        return false;
    }

    fs::path model_json_path = model_dir / "model.json";
    const std::string model_json_url = base + model_name + "/model.json";
    if (!ensure_file(model_json_path, model_json_url, err)) return false;

    nlohmann::json meta;
    for (int attempt = 0; attempt < 2; ++attempt) {
        std::ifstream ifs(model_json_path);
        if (!ifs) {
            if (err) *err = "failed to read " + model_json_path.string();
            return false;
        }
        try {
            ifs >> meta;
            break;
        } catch (...) {
            if (attempt == 1) {
                if (err) *err = "failed to parse " + model_json_path.string();
                return false;
            }
            std::error_code rm_ec;
            fs::remove(model_json_path, rm_ec);
            if (!download_file(model_json_url, model_json_path, err)) return false;
        }
    }

    std::vector<std::string> files;
    collect_file_refs(meta, files);
    files.push_back("model.json");
    std::sort(files.begin(), files.end());
    files.erase(std::unique(files.begin(), files.end()), files.end());

    for (const auto& f : files) {
        if (f.find("..") != std::string::npos) {
            if (err) *err = "invalid file reference in model.json: " + f;
            return false;
        }
        fs::path out_path = model_dir / f;
        const std::string file_url = base + model_name + "/" + f;
        if (!ensure_file(out_path, file_url, err)) return false;
    }

    return true;
}

} // namespace

bool ensure_model_present(const std::string& model_dir, const std::string& base_url, std::string* err) {
    fs::path dir(model_dir);
    if (!download_model_dir(dir, base_url, err)) return false;
    return true;
}

bool ensure_model_present(const std::string& model_dir, std::string* err) {
    return ensure_model_present(model_dir, "https://mirrors.sdu.edu.cn/ncnn_modelzoo/", err);
}
