// SPDX-License-Identifier: Apache-2.0
//
// Penguin-Encoder vision pipeline driver: patch embedding -> bidirectional
// Qwen3 encoder (with 2D-RoPE applied inside the exported graph) -> mlp2x_gelu
// projector, producing LLM-space visual tokens.
#pragma once

#include <memory>

#include <mat.h>
#include <net.h>

#include "penguin_config.h"

namespace pvl {

// Generate the Penguin 2D multimodal RoPE cos/sin cache.
//
// Layout matches transformers' apply_multimodal_rotary_pos_emb with
// rope_section = [head_dim/2, head_dim/2]:
//   row[i]              = f(h_coord * inv_freq[i])   for i in [0, head_dim/2)
//   row[head_dim/2 + i] = f(w_coord * inv_freq[i])   for i in [0, head_dim/2)
// with inv_freq[i] = theta^(-2i/head_dim), and patch order row-major over
// (grid_h, grid_w) (merge_size == 1).
//
// Output mats have shape (w = head_dim, h = grid_h*grid_w).
void generate_penguin_vision_rope(int grid_h, int grid_w, int head_dim,
                                  float theta, ncnn::Mat& cos_cache,
                                  ncnn::Mat& sin_cache);

class PenguinVision {
public:
    PenguinVision(const std::string& model_dir, const VisionConfig& vc,
                  int num_threads, bool use_vulkan);

    // Run the full vision pipeline on preprocessed patches.
    // pixel_values: (w = 3*patch*patch, h = num_patches)
    // returns visual tokens: (w = llm_hidden, h = num_patches) for merge_size==1.
    ncnn::Mat encode(const ncnn::Mat& pixel_values, int grid_h, int grid_w) const;

    const VisionConfig& config() const { return vc_; }

private:
    VisionConfig vc_;
    std::shared_ptr<ncnn::Net> patch_embed_;
    std::shared_ptr<ncnn::Net> encoder_;
    std::shared_ptr<ncnn::Net> projector_;
};

}  // namespace pvl
