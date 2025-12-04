#include "rope_embed.h"
#include <cmath>
#include <vector>

void generate_rope_embed_cache(int seqlen, int embed_dim, int position_id, ncnn::Mat& cos_cache, ncnn::Mat& sin_cache)
{
    const float rope_theta = 100000;
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

// LongRoPE
static const int ORIGINAL_MAX_POSITION_EMBEDDINGS = 32768;
static const float ROPE_BASE = 10000.0f;

static const float SHORT_FACTOR[32] = {
1.0004360675811768, 1.0668443441390991, 1.1631425619125366, 1.3025742769241333, 1.5040205717086792, 1.7941505908966064, 2.2101221084594727, 2.802666664123535, 3.6389970779418945, 4.804192543029785, 6.39855432510376, 8.527148246765137, 11.277542114257812, 14.684998512268066, 18.69317054748535, 23.13019371032715, 27.72362518310547, 32.1606559753418, 36.168827056884766, 39.57627868652344, 42.32667541503906, 44.45526885986328, 46.04962921142578, 47.21482849121094, 48.05115509033203, 48.64370346069336, 49.05967712402344, 49.34980392456055, 49.551246643066406, 49.69068145751953, 49.78697967529297, 49.85338592529297
};

static const float LONG_FACTOR[32] = {
1.0004360675811768, 1.0668443441390991, 1.1631425619125366, 1.3025742769241333, 1.5040205717086792, 1.7941505908966064, 2.2101221084594727, 2.802666664123535, 3.6389970779418945, 4.804192543029785, 6.39855432510376, 8.527148246765137, 11.277542114257812, 14.684998512268066, 18.69317054748535, 23.13019371032715, 27.72362518310547, 32.1606559753418, 36.168827056884766, 39.57627868652344, 42.32667541503906, 44.45526885986328, 46.04962921142578, 47.21482849121094, 48.05115509033203, 48.64370346069336, 49.05967712402344, 49.34980392456055, 49.551246643066406, 49.69068145751953, 49.78697967529297, 49.85338592529297
};

static inline float compute_scaling_factor(int max_position_embeddings) {
    float scale = static_cast<float>(max_position_embeddings) / static_cast<float>(ORIGINAL_MAX_POSITION_EMBEDDINGS);
    return std::sqrt(1.0f + std::log(scale) / std::log(static_cast<float>(ORIGINAL_MAX_POSITION_EMBEDDINGS)));
}

void generate_rope_embed_cache_LongRoPE(int seqlen,
                                      int embed_dim,
                                      int position_id,
                                      ncnn::Mat& cos_cache,
                                      ncnn::Mat& sin_cache)
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
        inv_freq[j] = 1.0f / std::pow(ROPE_BASE, exponent);
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