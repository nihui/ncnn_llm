#pragma once

#include <ncnn/mat.h>

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
void inject_image_embeds(std::vector<int>& token_ids, ncnn::Mat& token_embed, int& image_pad_index, const ncnn::Mat& image_embeds);
void generate_rope_embed_cache_vision_mrope(int seqlen, int embed_dim, int position_id, int image_pad_index, int image_embeds_size, int num_patches_w, ncnn::Mat& cos_cache, ncnn::Mat& sin_cache, float rope_theta = 100000);