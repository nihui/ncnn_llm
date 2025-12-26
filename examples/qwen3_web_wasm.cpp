#include "ncnn_llm_gpt.h"
#include "utils/prompt.h"

#include <emscripten/bind.h>

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <string>

using nlohmann::json;

namespace {

// Replace malformed UTF-8 sequences with '?', so JS strings stay valid.
std::string sanitize_utf8(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    auto is_cont = [&](unsigned char c) { return (c & 0xC0) == 0x80; };
    for (size_t i = 0; i < s.size();) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        if (c < 0x80) { out.push_back(static_cast<char>(c)); ++i; continue; }
        if ((c >> 5) == 0x6 && i + 1 < s.size() && is_cont(static_cast<unsigned char>(s[i + 1]))) {
            out.append(s, i, 2); i += 2; continue;
        }
        if ((c >> 4) == 0xE && i + 2 < s.size() &&
            is_cont(static_cast<unsigned char>(s[i + 1])) &&
            is_cont(static_cast<unsigned char>(s[i + 2]))) {
            out.append(s, i, 3); i += 3; continue;
        }
        if ((c >> 3) == 0x1E && i + 3 < s.size() &&
            is_cont(static_cast<unsigned char>(s[i + 1])) &&
            is_cont(static_cast<unsigned char>(s[i + 2])) &&
            is_cont(static_cast<unsigned char>(s[i + 3]))) {
            out.append(s, i, 4); i += 4; continue;
        }
        out.push_back('?');
        ++i;
    }
    return out;
}

bool read_int(const json& j, const char* key, int* out) {
    if (!j.contains(key) || !j[key].is_number_integer()) return false;
    *out = j[key].get<int>();
    return true;
}

bool read_float(const json& j, const char* key, float* out) {
    if (!j.contains(key) || !j[key].is_number()) return false;
    *out = j[key].get<float>();
    return true;
}

bool read_bool(const json& j, const char* key, bool* out) {
    if (!j.contains(key) || !j[key].is_boolean()) return false;
    *out = j[key].get<bool>();
    return true;
}

GenerateConfig parse_config(const std::string& options_json, bool* enable_thinking, std::string* err) {
    GenerateConfig cfg;
    if (options_json.empty()) return cfg;

    try {
        json j = json::parse(options_json);
        int iv = 0;
        float fv = 0.0f;
        bool bv = false;

        if (read_int(j, "max_tokens", &iv) || read_int(j, "max_new_tokens", &iv)) {
            cfg.max_new_tokens = iv;
        }
        if (read_float(j, "temperature", &fv)) cfg.temperature = fv;
        if (read_float(j, "top_p", &fv)) cfg.top_p = fv;
        if (read_int(j, "top_k", &iv)) cfg.top_k = iv;
        if (read_float(j, "repetition_penalty", &fv)) cfg.repetition_penalty = fv;
        if (read_int(j, "beam_size", &iv)) cfg.beam_size = iv;
        if (read_bool(j, "debug", &bv)) cfg.debug = bv;
        if (read_bool(j, "do_sample", &bv)) cfg.do_sample = bv ? 1 : 0;
        if (!j.contains("do_sample") && cfg.temperature <= 0.0f) {
            cfg.do_sample = 0;
        }
        if (enable_thinking && read_bool(j, "enable_thinking", &bv)) {
            *enable_thinking = bv;
        }
    } catch (const std::exception& e) {
        if (err) *err = e.what();
    }

    return cfg;
}

} // namespace

class Qwen3WebWasm {
public:
    explicit Qwen3WebWasm(const std::string& model_dir)
        : model_(model_dir, false) {
        logf("[wasm] init model dir: %s\n", model_dir.c_str());
        reset("You are a helpful assistant.", false);
    }

    void reset(const std::string& system_prompt, bool enable_thinking) {
        system_prompt_ = system_prompt.empty() ? "You are a helpful assistant." : system_prompt;
        enable_thinking_ = enable_thinking;
        logf("[wasm] reset: system_len=%zu enable_thinking=%d\n",
             system_prompt_.size(), enable_thinking_ ? 1 : 0);
        std::string prompt = apply_chat_template({{"system", system_prompt_}}, {}, false, enable_thinking_);
        logf("[wasm] reset: prompt_len=%zu\n", prompt.size());
        ctx_ = model_.prefill(prompt);
    }

    std::string generate(const std::string& user, const std::string& options_json) {
        last_error_.clear();
        if (!ctx_) {
            logf("[wasm] ctx missing, auto-reset\n");
            reset(system_prompt_, enable_thinking_);
        }

        bool use_thinking = enable_thinking_;
        GenerateConfig cfg = parse_config(options_json, &use_thinking, &last_error_);
        if (!last_error_.empty()) {
            logf("[wasm] options parse warning: %s\n", last_error_.c_str());
        }

        std::string user_prompt = apply_chat_template({{"user", user}}, {}, true, use_thinking);
        logf("[wasm] generate: user_len=%zu prompt_len=%zu\n", user.size(), user_prompt.size());
        ctx_ = model_.prefill(user_prompt, ctx_);

        std::string output;
        auto start = std::chrono::steady_clock::now();
        ctx_ = model_.generate(ctx_, cfg, [&](const std::string& token) {
            output += sanitize_utf8(token);
        });
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        logf("[wasm] generate: output_len=%zu time_ms=%lld\n",
             output.size(), static_cast<long long>(ms));
        return output;
    }

    std::string last_error() const { return last_error_; }
    std::string system_prompt() const { return system_prompt_; }
    bool enable_thinking() const { return enable_thinking_; }
    void set_enable_thinking(bool enable) { enable_thinking_ = enable; }
    void set_verbose(bool enable) { verbose_ = enable; }
    bool verbose() const { return verbose_; }

private:
    void logf(const char* fmt, ...) {
        if (!verbose_) return;
        va_list args;
        va_start(args, fmt);
        std::vfprintf(stderr, fmt, args);
        std::fflush(stderr);
        va_end(args);
    }

    ncnn_llm_gpt model_;
    std::shared_ptr<ncnn_llm_gpt_ctx> ctx_;
    std::string system_prompt_;
    bool enable_thinking_ = false;
    std::string last_error_;
    bool verbose_ = true;
};

EMSCRIPTEN_BINDINGS(qwen3_web_wasm) {
    emscripten::class_<Qwen3WebWasm>("Qwen3WebWasm")
        .constructor<std::string>()
        .function("reset", &Qwen3WebWasm::reset)
        .function("generate", &Qwen3WebWasm::generate)
        .function("last_error", &Qwen3WebWasm::last_error)
        .function("system_prompt", &Qwen3WebWasm::system_prompt)
        .function("enable_thinking", &Qwen3WebWasm::enable_thinking)
        .function("set_enable_thinking", &Qwen3WebWasm::set_enable_thinking)
        .function("set_verbose", &Qwen3WebWasm::set_verbose)
        .function("verbose", &Qwen3WebWasm::verbose);
}

int main() {
    return 0;
}
