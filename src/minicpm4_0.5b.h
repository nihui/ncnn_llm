#pragma once

#include <functional>
#include <locale>
#include <memory>
#include <vector>
#include <string>

struct Message {
    enum class Role {
        System,
        User,
        Assistant
    };

    Role role;
    std::string content;
};

struct GenerateConfig {
    int max_new_tokens = 4096;
    float temperature = 0.3f;
    float top_p = 0.8f;
    int top_k = 50;
    float repetition_penalty = 1.0f;
    int beam_size = 1;
    int do_sample = 1;
};

std::string make_chat_prompt(const std::vector<Message>& messages, bool add_generation_prompt);

struct minicpm4_0_5b_ctx;

class minicpm4_0_5b {
public:
    minicpm4_0_5b(std::string embed_param,
              std::string embed_bin,
              std::string proj_out_param,
              std::string decoder_param,
              std::string decoder_bin,
              std::string vocab_file,
              std::string merges_file,
              bool use_vulkan);

    ~minicpm4_0_5b();

    std::shared_ptr<minicpm4_0_5b_ctx> prefill(const std::string& input_text);

    std::shared_ptr<minicpm4_0_5b_ctx> prefill(const std::string& input_text,
                                         const std::shared_ptr<minicpm4_0_5b_ctx> ctx);

    bool decode(std::shared_ptr<minicpm4_0_5b_ctx> ctx,
                std::function<void(const std::string&)> callback);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};