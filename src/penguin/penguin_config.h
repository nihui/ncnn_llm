// SPDX-License-Identifier: Apache-2.0
//
// Parsed representation of a Penguin-VL-ncnn `model.json`.
#pragma once

#include <string>
#include <vector>

namespace pvl {

// Vision (Penguin-Encoder) settings. All values come from model.json so the
// runtime never hardcodes model-specific dimensions.
struct VisionConfig {
    bool enabled = false;

    // ncnn sub-graphs.
    std::string patch_embed_param;   // Conv2d(3, H, k=patch, s=patch) as a linear patch projection
    std::string patch_embed_bin;
    std::string encoder_param;       // stack of bidirectional Qwen3 blocks + final RMSNorm
    std::string encoder_bin;
    std::string projector_param;     // mlp2x_gelu: Linear -> GELU -> Linear
    std::string projector_bin;

    // geometry / preprocessing
    int patch_size = 14;
    int merge_size = 1;              // image merge_size (Penguin default 1); video uses 2
    int min_tokens = 16;            // 4 * 4
    int max_tokens = 16384;
    int vision_hidden = 1024;        // encoder hidden size (== projector input dim)
    int vision_head_dim = 128;       // per-head dim used by the 2D-RoPE (Qwen3 decoupled head_dim)
    float vision_rope_theta = 1000000.0f;

    // normalization (Penguin-VL preprocessor: mean=std=0.5, i.e. [-1, 1] scaling)
    float image_mean[3] = {0.5f, 0.5f, 0.5f};
    float image_std[3] = {0.5f, 0.5f, 0.5f};

    // Placeholder token that marks visual positions in the prompt.
    std::string image_token = "<image>";
};

struct PenguinConfig {
    std::string model_type = "penguinvl_qwen3";

    // LLM (Qwen3) ncnn sub-graphs.
    std::string embed_param;
    std::string embed_bin;
    std::string decoder_param;
    std::string decoder_bin;
    std::string lm_head_param;
    std::string lm_head_bin;

    // Tokenizer.
    std::string tokenizer_type = "bbpe";   // Qwen uses byte-level BPE
    std::string vocab_file;
    std::string merges_file;
    std::string bos_token;
    std::string eos_token = "<|im_end|>";
    std::vector<std::string> additional_special_tokens;

    // LLM decoder structure.
    int hidden_size = 2048;
    int num_layers = 28;             // number of transformer blocks == kv-cache count
    int rope_head_dim = 128;
    int kv_heads = 8;                // decoder key/value heads (GQA); kv-cache h axis
    bool kv_cache = true;            // false => cacheless decoder (reprocess full seq)
    float rope_theta = 1000000.0f;

    // Chat template pieces (kept explicit so the C++ side matches the HF template).
    // Penguin-VL's chat template injects no default system message, so this is
    // empty by default; set it in model.json only if your deployment adds one.
    std::string system_prompt = "";

    VisionConfig vision;

    // Load and validate a model directory that contains model.json.
    static PenguinConfig load(const std::string& model_dir);
};

}  // namespace pvl
