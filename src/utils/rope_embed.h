#pragma once

#include <ncnn/mat.h>
#include <vector>

struct RopeScalingParams {
    float alpha;
    float beta_fast;
    float beta_slow;
    float factor;
    float mscale;
    float mscale_all_dim;
};

void generate_ntk_rope_embed_cache(
    int seqlen, 
    int embed_dim, 
    int position_id, 
    ncnn::Mat& cos_cache, 
    ncnn::Mat& sin_cache, 
    float rope_theta, 
    const RopeScalingParams& scaling_params
);

void generate_hunyuan_rope_embed_cache(
    int seqlen, 
    int embed_dim, 
    int position_id, 
    ncnn::Mat& cos_cache, 
    ncnn::Mat& sin_cache, 
    float rope_theta, 
    const RopeScalingParams& scaling_params
);

void generate_yarn_rope_embed_cache(
    int seqlen, 
    int embed_dim, 
    int position_id, 
    ncnn::Mat& cos_cache, 
    ncnn::Mat& sin_cache, 
    float rope_theta, 
    const RopeScalingParams& scaling_params
);

void generate_rope_embed_cache(int seqlen, int embed_dim, int position_id, ncnn::Mat& cos_cache, ncnn::Mat& sin_cache, float rope_theta = 100000);

void generate_rope_embed_cache_LongRoPE(int seqlen,
                                      int embed_dim,
                                      int position_id,
                                      ncnn::Mat& cos_cache,
                                      ncnn::Mat& sin_cache,
                                      float rope_theta,
                                      const float* SHORT_FACTOR,
                                      const float* LONG_FACTOR,
                                      int ORIGINAL_MAX_POSITION_EMBEDDINGS = 32768);

void inject_image_embeds(std::vector<int>& token_ids, ncnn::Mat& token_embed, int& image_pad_index, int image_pad_id, const ncnn::Mat& image_embeds);

void generate_rope_embed_cache_vision_mrope(int seqlen, 
                                          int embed_dim, 
                                          int position_id, 
                                          int image_pad_index, 
                                          int image_embeds_size, 
                                          int num_patches_w, 
                                          int spatial_merge_size,
                                          const std::vector<int>& mrope_section,
                                          ncnn::Mat& cos_cache, 
                                          ncnn::Mat& sin_cache, 
                                          float rope_theta = 100000);