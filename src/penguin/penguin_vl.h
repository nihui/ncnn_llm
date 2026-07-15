// SPDX-License-Identifier: Apache-2.0
//
// Top-level Penguin-VL runtime on ncnn: tokenizer + Qwen3 KV-cache decoder +
// Penguin vision encoder. Text decoding uses standard 1D Qwen3 RoPE; images are
// injected as visual tokens at the image-placeholder positions.
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <mat.h>
#include <net.h>

#include "bpe_tokenizer.h"
#include "penguin_config.h"
#include "penguin_vision.h"

namespace pvl {

struct GenerateConfig {
    int max_new_tokens = 512;
    bool do_sample = false;    // greedy by default for deterministic, PyTorch-matching output
    float temperature = 1.0f;
    int top_k = 0;
    float top_p = 1.0f;
    bool add_think = false;    // whether to keep the model's thinking turn open
};

class PenguinVL {
public:
    explicit PenguinVL(const std::string& model_dir, int num_threads = 4,
                       bool use_vulkan = false);

    // One-shot chat with an optional image. `on_token` is called with each
    // decoded piece of text as it is generated. Returns the full response.
    std::string chat(const std::string& user_text, const std::string& image_path,
                     const GenerateConfig& cfg = {},
                     const std::function<void(const std::string&)>& on_token = nullptr);

    // Build the templated prompt for a single-turn user message.
    std::string build_prompt(const std::string& user_text, bool has_image,
                             bool add_think) const;

    const PenguinConfig& config() const { return cfg_; }

private:
    // Turn a templated prompt (with a single image placeholder) + optional visual
    // tokens into the input token embeddings, replacing placeholders in order.
    ncnn::Mat build_input_embeds(const std::string& prompt,
                                 const ncnn::Mat& image_embeds,
                                 std::vector<int>& token_ids_out) const;

    ncnn::Mat embed_tokens(const std::vector<int>& ids) const;
    ncnn::Mat embed_token(int id) const;
    ncnn::Mat lm_head(const ncnn::Mat& hidden) const;

    std::string model_dir_;
    int num_threads_ = 4;
    bool use_vulkan_ = false;

    PenguinConfig cfg_;
    std::shared_ptr<BpeTokenizer> tok_;
    std::shared_ptr<ncnn::Net> embed_net_;
    std::shared_ptr<ncnn::Net> decoder_net_;
    std::shared_ptr<ncnn::Net> lm_head_net_;
    std::unique_ptr<PenguinVision> vision_;

    int bos_id_ = -1;
    int eos_id_ = -1;
    int image_token_id_ = -1;
};

}  // namespace pvl
