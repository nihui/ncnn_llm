#pragma once

#include <functional>
#include <locale>
#include <memory>
#include <vector>
#include <string>

struct GenerateConfig {
    int max_new_tokens = 4096;
    float temperature = 0.3f;
    float top_p = 0.8f;
    int top_k = 50;
    float repetition_penalty = 1.1f;
    int beam_size = 1;
    int do_sample = 1;
};

struct qwen3_0_6b_ctx;

class qwen3_0_6b {
public:
    qwen3_0_6b(std::string embed_param,
              std::string embed_bin,
              std::string proj_out_param,
              std::string decoder_param,
              std::string decoder_bin,
              std::string vocab_file,
              std::string merges_file,
              bool use_vulkan);

    ~qwen3_0_6b();

    std::shared_ptr<qwen3_0_6b_ctx> prefill(const std::string& input_text);

    std::shared_ptr<qwen3_0_6b_ctx> prefill(const std::string& input_text,
                                         const std::shared_ptr<qwen3_0_6b_ctx> ctx);

    std::shared_ptr<qwen3_0_6b_ctx> generate(const std::shared_ptr<qwen3_0_6b_ctx>& ctx, const GenerateConfig& cfg, std::function<void(const std::string&)> callback);

    bool decode(std::shared_ptr<qwen3_0_6b_ctx> ctx,
                std::function<void(const std::string&)> callback);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};