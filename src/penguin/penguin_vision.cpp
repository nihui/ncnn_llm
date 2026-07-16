// SPDX-License-Identifier: Apache-2.0
#include "penguin_vision.h"

#include <cmath>
#include <stdexcept>

namespace pvl {

void generate_penguin_vision_rope(int grid_h, int grid_w, int head_dim,
                                  float theta, ncnn::Mat& cos_cache,
                                  ncnn::Mat& sin_cache) {
    if (head_dim % 2 != 0) throw std::runtime_error("vision rope: head_dim must be even");
    const int half = head_dim / 2;
    const int seq_len = grid_h * grid_w;

    std::vector<float> inv_freq(half);
    for (int i = 0; i < half; ++i)
        inv_freq[i] = 1.0f / std::pow(theta, (float)(2 * i) / (float)head_dim);

    cos_cache.create(head_dim, seq_len);
    sin_cache.create(head_dim, seq_len);

    for (int gh = 0; gh < grid_h; ++gh) {
        for (int gw = 0; gw < grid_w; ++gw) {
            const int idx = gh * grid_w + gw;  // row-major, merge_size == 1
            float* c = cos_cache.row(idx);
            float* s = sin_cache.row(idx);
            for (int i = 0; i < half; ++i) {
                const float ah = (float)gh * inv_freq[i];
                const float aw = (float)gw * inv_freq[i];
                c[i] = std::cos(ah);
                s[i] = std::sin(ah);
                c[half + i] = std::cos(aw);
                s[half + i] = std::sin(aw);
            }
        }
    }
}

static void load_net(ncnn::Net& net, const std::string& param, const std::string& bin,
                     int num_threads, bool use_vulkan) {
    if (num_threads > 0) net.opt.num_threads = num_threads;
    net.opt.use_vulkan_compute = use_vulkan;
    if (net.load_param(param.c_str()) != 0)
        throw std::runtime_error("failed to load param: " + param);
    if (net.load_model(bin.c_str()) != 0)
        throw std::runtime_error("failed to load bin: " + bin);
}

PenguinVision::PenguinVision(const std::string& model_dir, const VisionConfig& vc,
                             int num_threads, bool use_vulkan)
    : vc_(vc) {
    patch_embed_ = std::make_shared<ncnn::Net>();
    encoder_ = std::make_shared<ncnn::Net>();
    projector_ = std::make_shared<ncnn::Net>();
    load_net(*patch_embed_, model_dir + "/" + vc.patch_embed_param,
             model_dir + "/" + vc.patch_embed_bin, num_threads, use_vulkan);
    load_net(*encoder_, model_dir + "/" + vc.encoder_param,
             model_dir + "/" + vc.encoder_bin, num_threads, use_vulkan);
    load_net(*projector_, model_dir + "/" + vc.projector_param,
             model_dir + "/" + vc.projector_bin, num_threads, use_vulkan);
}

ncnn::Mat PenguinVision::encode(const ncnn::Mat& pixel_values, int grid_h, int grid_w) const {
    if (vc_.merge_size != 1)
        throw std::runtime_error("PenguinVision::encode: only merge_size==1 supported");

    // 1. patch embedding: (3*ps*ps, N) -> (Hvis, N)
    ncnn::Mat hidden;
    {
        ncnn::Extractor ex = patch_embed_->create_extractor();
        ex.input("in0", pixel_values);
        ex.extract("out0", hidden);
    }

    // 2. 2D-RoPE cache for the encoder's bidirectional attention.
    ncnn::Mat cos_cache, sin_cache;
    generate_penguin_vision_rope(grid_h, grid_w, vc_.vision_head_dim,
                                 vc_.vision_rope_theta, cos_cache, sin_cache);

    // 3. bidirectional encoder (full attention over all patches, no mask needed
    //    for a single image): (Hvis, N) + cos + sin -> (Hvis, N)
    ncnn::Mat enc_out;
    {
        ncnn::Extractor ex = encoder_->create_extractor();
        ex.input("in0", hidden);
        ex.input("in1", cos_cache);
        ex.input("in2", sin_cache);
        ex.extract("out0", enc_out);
    }

    // 4. projector mlp2x_gelu: (Hvis, N) -> (llm_hidden, N)
    ncnn::Mat proj_out;
    {
        ncnn::Extractor ex = projector_->create_extractor();
        ex.input("in0", enc_out);
        ex.extract("out0", proj_out);
    }

    return proj_out;
}

}  // namespace pvl
