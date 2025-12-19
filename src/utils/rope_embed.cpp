#include "rope_embed.h"
#include <cmath>
#include <vector>
#include <cstring>
#include <cstdio>

void generate_hunyuan_rope_embed_cache(
    int seqlen, 
    int embed_dim, 
    int position_id, 
    ncnn::Mat& cos_cache, 
    ncnn::Mat& sin_cache, 
    float rope_theta, 
    const RopeScalingParams& scaling_params
) {
    // HunYuanDenseV1 Logic:
    // base = theta * alpha ^ (dim / (dim - 2))
    
    float dim = (float)embed_dim;
    float alpha = scaling_params.alpha; // 1000.0 from your config

    // Calculate the modified base theta
    // Matches Python: base = ... * alpha ** (head_dim / (head_dim - 2))
    float base_correction = powf(alpha, dim / (dim - 2.0f));
    float hunyuan_theta = rope_theta * base_correction;

    // Precompute inverse frequencies
    std::vector<float> inv_freq(embed_dim / 2);
    for (int i = 0; i < embed_dim / 2; i++) {
        // Matches Python: 1.0 / (base ** (i / head_dim)) 
        // Note: in Python arange(0, dim, 2) implies indices are multiplied by 2
        float exponent = (float)(i * 2) / dim;
        inv_freq[i] = 1.0f / powf(hunyuan_theta, exponent);
    }

    // Allocate ncnn Mats
    cos_cache.create(embed_dim / 2, seqlen);
    sin_cache.create(embed_dim / 2, seqlen);

    // Fill Cache
    for (int i = 0; i < seqlen; i++) {
        float* cos_ptr = cos_cache.row(i);
        float* sin_ptr = sin_cache.row(i);

        for (int j = 0; j < embed_dim / 2; j++) {
            const int pos = position_id + i;
            const float t = pos * inv_freq[j];
            
            // HunYuanDenseV1 sets attention_scaling = 1.0, so no mscale multiplication needed
            *cos_ptr++ = cosf(t);
            *sin_ptr++ = sinf(t);
        }
    }
}

static float yarn_ramp(float low, float high, float val) {
    if (val < low) return 0.0f;
    if (val > high) return 1.0f;
    return (val - low) / (high - low);
}

void generate_yarn_rope_embed_cache(
    int seqlen, 
    int embed_dim, 
    int position_id, 
    ncnn::Mat& cos_cache, 
    ncnn::Mat& sin_cache, 
    float rope_theta, 
    const RopeScalingParams& scaling_params
) {
    cos_cache.create(embed_dim / 2, seqlen);
    sin_cache.create(embed_dim / 2, seqlen);

    std::vector<float> inv_freq(embed_dim / 2);
    
    float scale = scaling_params.alpha; 
    float beta_fast = scaling_params.beta_fast;
    float beta_slow = scaling_params.beta_slow;

    for (int i = 0; i < embed_dim / 2; i++) {
        float freq = 1.0f / powf(rope_theta, (float)(i * 2) / (float)embed_dim);
        
        float r = (float)(i * 2) / (float)embed_dim;
        
        float ramp = yarn_ramp(beta_slow, beta_fast, (float)embed_dim / (i * 2 + 1)); 
        float base_correction = powf(scale, (float)embed_dim / ((float)embed_dim - 2.0f));
        float ntk_theta = rope_theta * base_correction;
        inv_freq[i] = 1.0f / powf(ntk_theta, (float)(i * 2) / (float)embed_dim);
    }

    for (int i = 0; i < seqlen; i++) {
        float* cos_ptr = cos_cache.row(i);
        float* sin_ptr = sin_cache.row(i);

        for (int j = 0; j < embed_dim / 2; j++) {
            const int pos = position_id + i;
            const float t = pos * inv_freq[j];
            
            float mscale = scaling_params.mscale;
            if (fabs(mscale - 1.0f) > 1e-6) {
                *cos_ptr++ = cosf(t) * mscale;
                *sin_ptr++ = sinf(t) * mscale;
            } else {
                *cos_ptr++ = cosf(t);
                *sin_ptr++ = sinf(t);
            }
        }
    }
}

void generate_ntk_rope_embed_cache(
    int seqlen, 
    int embed_dim, 
    int position_id, 
    ncnn::Mat& cos_cache, 
    ncnn::Mat& sin_cache, 
    float rope_theta, 
    const RopeScalingParams& scaling_params
) {
    float dim = (float)embed_dim;
    float alpha = scaling_params.alpha;
    
    float base_correction = powf(alpha, dim / (dim - 2.0f));
    float ntk_theta = rope_theta * base_correction;

    std::vector<float> inv_freq(embed_dim / 2);
    for (int i = 0; i < embed_dim / 2; i++)
    {
        inv_freq[i] = 1.f / powf(ntk_theta, (float)(i * 2) / dim);
    }

    cos_cache.create(embed_dim / 2, seqlen);
    sin_cache.create(embed_dim / 2, seqlen);

    for (int i = 0; i < seqlen; i++)
    {
        float* cos_ptr = cos_cache.row(i);
        float* sin_ptr = sin_cache.row(i);

        for (int j = 0; j < embed_dim / 2; j++)
        {
            const int pos = position_id + i;
            const float t = pos * inv_freq[j];
            
            *cos_ptr++ = cosf(t);
            *sin_ptr++ = sinf(t);
        }
    }
}

void generate_rope_embed_cache(int seqlen, int embed_dim, int position_id, ncnn::Mat& cos_cache, ncnn::Mat& sin_cache, float rope_theta)
{
    std::vector<float> inv_freq(embed_dim / 2);
    for (int i = 0; i < embed_dim / 2; i++)
    {
        inv_freq[i] = 1.f / powf(rope_theta, (float)(i * 2) / embed_dim);
    }

    cos_cache.create(embed_dim / 2, seqlen);
    sin_cache.create(embed_dim / 2, seqlen);

    for (int i = 0; i < seqlen; i++)
    {
        float* cos_ptr = cos_cache.row(i);
        float* sin_ptr = sin_cache.row(i);

        for (int j = 0; j < embed_dim / 2; j++)
        {
            const int pos = position_id + i;
            const float t = pos * inv_freq[j];
            const float cos_val = cosf(t);
            const float sin_val = sinf(t);
            *cos_ptr++ = cos_val;
            *sin_ptr++ = sin_val;
        }
    }
}

static inline float compute_scaling_factor(int max_position_embeddings, int ORIGINAL_MAX_POSITION_EMBEDDINGS = 32768) {
    float scale = static_cast<float>(max_position_embeddings) / static_cast<float>(ORIGINAL_MAX_POSITION_EMBEDDINGS);
    return std::sqrt(1.0f + std::log(scale) / std::log(static_cast<float>(ORIGINAL_MAX_POSITION_EMBEDDINGS)));
}

void generate_rope_embed_cache_LongRoPE(int seqlen,
                                      int embed_dim,
                                      int position_id,
                                      ncnn::Mat& cos_cache,
                                      ncnn::Mat& sin_cache,
                                      float rope_theta,
                                      const float* SHORT_FACTOR,
                                      const float* LONG_FACTOR,
                                      int ORIGINAL_MAX_POSITION_EMBEDDINGS)
{
    if (embed_dim % 2 != 0 || seqlen <= 0) {
        cos_cache.release();
        sin_cache.release();
        return;
    }

    const int half_dim = embed_dim / 2;

    cos_cache.create(half_dim, seqlen);
    sin_cache.create(half_dim, seqlen);

    if (cos_cache.empty() || sin_cache.empty()) {
        return;
    }

    float* cos_ptr = cos_cache.channel(0);
    float* sin_ptr = sin_cache.channel(0);

    std::vector<float> inv_freq(half_dim);
    for (int j = 0; j < half_dim; ++j) {
        float exponent = (2.0f * j) / static_cast<float>(embed_dim);
        inv_freq[j] = 1.0f / std::pow(rope_theta, exponent);
    }

    const float* ext_factor = (seqlen > ORIGINAL_MAX_POSITION_EMBEDDINGS) ? LONG_FACTOR : SHORT_FACTOR;
    const float scaling_factor = compute_scaling_factor(ORIGINAL_MAX_POSITION_EMBEDDINGS);

    for (int i = 0; i < seqlen; ++i) {
        int t = position_id + i;
        float* row_cos = cos_ptr + i * half_dim;
        float* row_sin = sin_ptr + i * half_dim;

        for (int j = 0; j < half_dim; ++j) {
            float freq = (static_cast<float>(t) * inv_freq[j]) / ext_factor[j];
            row_cos[j] = std::cos(freq) * scaling_factor;
            row_sin[j] = std::sin(freq) * scaling_factor;
        }
    }
}

void inject_image_embeds(std::vector<int>& token_ids, ncnn::Mat& token_embed, int& image_pad_index, int image_pad_id, const ncnn::Mat& image_embeds)
{
    image_pad_index = -1;
    if (image_embeds.empty())
    {
        return;
    }

    std::vector<int> token_ids_injected(token_ids.size() - 1 + image_embeds.h);
    ncnn::Mat token_embed_injected(token_embed.w, token_embed.h - 1 + image_embeds.h);

    for (int i = 0; i < (int)token_ids.size(); i++)
    {
        if (token_ids[i] == image_pad_id)
        {
            image_pad_index = i;

            // inject token ids
            memcpy(token_ids_injected.data(), token_ids.data(), i * sizeof(int));
            memset(token_ids_injected.data() + i, image_pad_id, image_embeds.h * sizeof(int));
            memcpy(token_ids_injected.data() + i + image_embeds.h, token_ids.data() + i + 1, (token_ids.size() - 1 - i) * sizeof(int));

            // inject token embed
            memcpy(token_embed_injected.row(0), token_embed.row(0), i * token_embed.w * sizeof(float));
            memcpy(token_embed_injected.row(i), image_embeds.row(0), image_embeds.h * token_embed.w * sizeof(float));
            memcpy(token_embed_injected.row(i + image_embeds.h), token_embed.row(i + 1), (token_ids.size() - 1 - i) * token_embed.w * sizeof(float));

            break;
        }
    }

    token_ids = token_ids_injected;
    token_embed = token_embed_injected;
}

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
                                          float rope_theta)
{
    if (mrope_section.size() < 3) return;

    std::vector<float> inv_freq(embed_dim / 2);
    for (int i = 0; i < embed_dim / 2; i++)
    {
        inv_freq[i] = 1.f / powf(rope_theta, (float)(i * 2) / embed_dim);
    }

    cos_cache.create(embed_dim / 2, seqlen);
    sin_cache.create(embed_dim / 2, seqlen);

    for (int i = 0; i < seqlen; i++)
    {
        float* cos_ptr = cos_cache.row(i);
        float* sin_ptr = sin_cache.row(i);

        for (int j = 0; j < embed_dim / 2; j++)
        {
            int pos = position_id;
            if (i < image_pad_index)
            {
                pos += i;
            }
            else if (i >= image_pad_index + image_embeds_size)
            {
                pos += i - image_embeds_size + (num_patches_w / spatial_merge_size);
            }
            else
            {
                // inside image tokens range
                if (j < mrope_section[0])
                {
                    // temporal
                    pos += image_pad_index;
                }
                else if (j < mrope_section[0] + mrope_section[1])
                {
                    // height
                    int hid = (i - image_pad_index) / (num_patches_w / spatial_merge_size);
                    pos += image_pad_index + hid;
                }
                else
                {
                    // width
                    int wid = (i - image_pad_index) % (num_patches_w / spatial_merge_size);
                    pos += image_pad_index + wid;
                }
            }

            const float t = pos * inv_freq[j];
            const float cos_val = cosf(t);
            const float sin_val = sinf(t);
            *cos_ptr++ = cos_val;
            *sin_ptr++ = sin_val;
        }
    }
}