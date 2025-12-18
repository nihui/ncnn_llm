#include "rope_embed.h"
#include <cmath>
#include <vector>

void generate_rope_embed_cache(int seqlen, int embed_dim, int position_id, ncnn::Mat& cos_cache, ncnn::Mat& sin_cache, float rope_theta)
{
    const float attention_factor = 1.f;

    // prepare inv_freq
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

    // ncnn::Mat (w=half_dim, h=seqlen, c=1)
    cos_cache.create(half_dim, seqlen);
    sin_cache.create(half_dim, seqlen);

    if (cos_cache.empty() || sin_cache.empty()) {
        return;
    }

    float* cos_ptr = cos_cache.channel(0);
    float* sin_ptr = sin_cache.channel(0);

    // idx_j = 2*j => exponent = (2*j)/embed_dim
    std::vector<float> inv_freq(half_dim);
    for (int j = 0; j < half_dim; ++j) {
        float exponent = (2.0f * j) / static_cast<float>(embed_dim);
        inv_freq[j] = 1.0f / std::pow(rope_theta, exponent);
    }

    const float* ext_factor = (seqlen > ORIGINAL_MAX_POSITION_EMBEDDINGS) ? LONG_FACTOR : SHORT_FACTOR;
    const float scaling_factor = compute_scaling_factor(ORIGINAL_MAX_POSITION_EMBEDDINGS);

    // freqs[i, j] = ( (t_i) * inv_freq[j] ) / ext_factor[j]
    // t_i = position_id + i
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

void inject_image_embeds(std::vector<int>& token_ids, ncnn::Mat& token_embed, int& image_pad_index, const ncnn::Mat& image_embeds)
{
    image_pad_index = -1;
    if (image_embeds.empty())
    {
        return;
    }

    // <|image_pad|>
    fprintf(stderr, "token_embed %d x %d\n", token_embed.w, token_embed.h);
    fprintf(stderr, "image_embeds %d x %d\n", image_embeds.w, image_embeds.h);

    std::vector<int> token_ids_injected(token_ids.size() - 1 + image_embeds.h);
    ncnn::Mat token_embed_injected(token_embed.w, token_embed.h - 1 + image_embeds.h);

    for (int i = 0; i < (int)token_ids.size(); i++)
    {
        // FIXME hardcode
        // find <|image_pad|>
        if (token_ids[i] == 151655)
        {
            image_pad_index = i;

            // inject token ids
            memcpy(token_ids_injected.data(), token_ids.data(), i * sizeof(int));
            memset(token_ids_injected.data() + i, 151655, image_embeds.h * sizeof(int));
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

void generate_rope_embed_cache_vision_mrope(int seqlen, int embed_dim, int position_id, int image_pad_index, int image_embeds_size, int num_patches_w, ncnn::Mat& cos_cache, ncnn::Mat& sin_cache, float rope_theta)
{
    const int merge_size = 2;

    const int mrope[3] = {16,24,24};

    // assert mrope[0] + mrope[1] + mrope[2] == embed_dim / 2

    // const float rope_theta = 100000;
    // const float rope_theta = 1000000;
    const float attention_factor = 1.f;

    // prepare inv_freq
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
                pos += i - image_embeds_size + (num_patches_w / merge_size);
            }
            else
            {
                // we are inside image tokens range
                if (j < mrope[0])
                {
                    // temporal
                    pos += image_pad_index;
                }
                else if (j < mrope[0] + mrope[1])
                {
                    // height
                    int hid = (i - image_pad_index) / (num_patches_w / merge_size);
                    pos += image_pad_index + hid;
                }
                else // if (j < mrope[0] + mrope[1] + mrope[2])
                {
                    // width
                    int wid = (i - image_pad_index) % (num_patches_w / merge_size);
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
