#pragma once

#include <ncnn/mat.h>

void generate_rope_embed_cache(int seqlen, int embed_dim, int position_id, ncnn::Mat& cos_cache, ncnn::Mat& sin_cache);
void generate_rope_embed_cache_LongRoPE(int seqlen, int embed_dim, int position_id, ncnn::Mat& cos_cache, ncnn::Mat& sin_cache);