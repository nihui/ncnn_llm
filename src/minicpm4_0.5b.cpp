#include "minicpm4_0.5b.h"

#include <cstdio>
#include <memory>
#include <ncnn/mat.h>
#include <ncnn/net.h>
#include <utility>
#include <vector>
#include <random>

#include "utils/tokenizer/bpe_tokenizer.h"

static std::mt19937 rng(std::random_device{}());

const static int attn_cnt = 24;

struct minicpm4_0_5b_ctx {
    std::vector<std::pair<ncnn::Mat, ncnn::Mat>> kv_cache;

    int cur_token = 0;
};

static void softmax_vec(std::vector<float>& logits, float temperature) {
    float max_logit = *std::max_element(logits.begin(), logits.end());
    float sum = 0.f;
    for (float& x : logits) {
        x = std::exp((x - max_logit) / temperature);
        sum += x;
    }
    for (float& x : logits) x /= sum;
}

static void apply_top_k(std::vector<float>& probs, int k) {
    if (k <= 0 || k >= (int)probs.size()) return;
    std::vector<float> tmp = probs;
    std::nth_element(tmp.begin(), tmp.end() - k, tmp.end());
    float threshold = tmp[tmp.size() - k];
    for (float& p : probs) if (p < threshold) p = 0.f;
}

static void apply_top_p(std::vector<float>& probs, float p) {
    if (p >= 1.0f) return;
    std::vector<std::pair<float,int>> v;
    v.reserve(probs.size());
    for (int i = 0; i < (int)probs.size(); ++i) {
        v.emplace_back(probs[i], i);
    }
    std::sort(v.begin(), v.end(), std::greater<>());

    float cum = 0.f;
    float last_prob = 0.f;
    size_t cutoff = v.size();
    for (size_t i = 0; i < v.size(); ++i) {
        cum += v[i].first;
        last_prob = v[i].first;
        if (cum >= p) {
            cutoff = i + 1;
            break;
        }
    }
    // 把在 cumulative p 以外的全部置 0
    std::vector<char> keep(probs.size(), 0);
    for (size_t i = 0; i < cutoff; ++i) {
        keep[v[i].second] = 1;
    }
    for (int i = 0; i < (int)probs.size(); ++i) {
        if (!keep[i]) probs[i] = 0.f;
    }
}

static int sample_from_probs(const std::vector<float>& probs) {
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return dist(rng);
}

// ===============================================================
//                     Beam 状态结构
// ===============================================================
struct Beam {
    std::shared_ptr<minicpm4_0_5b_ctx> ctx;
    float score = 0.f;
    bool finished = false;
    std::vector<int> tokens;
};

// 深拷贝 KV Cache
static std::shared_ptr<minicpm4_0_5b_ctx>
clone_ctx(const std::shared_ptr<minicpm4_0_5b_ctx>& src) {
    auto dst = std::make_shared<minicpm4_0_5b_ctx>();
    dst->cur_token = src->cur_token;
    dst->kv_cache.resize(src->kv_cache.size());
    for (size_t i = 0; i < src->kv_cache.size(); ++i) {
        dst->kv_cache[i].first = src->kv_cache[i].first.clone();
        dst->kv_cache[i].second = src->kv_cache[i].second.clone();
    }
    return dst;
}

std::string make_chat_prompt(const std::vector<Message>& messages, bool add_generation_prompt)
{
    std::string prompt;
    for (const auto& msg : messages) {
        switch (msg.role) {
            case Message::Role::System:
                prompt += "<|im_start|>system\n" + msg.content + "<|im_end|>\n";
                break;
            case Message::Role::User:
                prompt += "<|im_start|>user\n" + msg.content + "<|im_end|>\n";
                break;
            case Message::Role::Assistant:
                prompt += "<|im_start|>assistant\n" + msg.content + "<|im_end|>\n";
                break;
        }
    }
    if (add_generation_prompt) {
        prompt += "<|im_start|>assistant\n";
    }
    return prompt;
}


class minicpm4_0_5b::Impl {
public:
    ncnn::Net embed_net;
    ncnn::Net proj_out_net;
    ncnn::Net decoder_net;

    BpeTokenizer bpe;

    int im_end_id = -1;

    Impl(std::string embed_param,
         std::string embed_bin,
         std::string proj_out_param,
         std::string decoder_param,
         std::string decoder_bin,
         std::string vocab_file,
         std::string merges_file,
         bool use_vulkan)
        : bpe(BpeTokenizer::LoadFromFiles(
              vocab_file,
              merges_file,
              SpecialTokensConfig{
                  .bos_token = "<s>",
                  .eos_token = "</s>",
                  .unk_token = "<unk>",
                  .sep_token = "<SEP>",
                  .cls_token = "<CLS>",
                  .mask_token = "<mask>",
              })) {
        if (use_vulkan) {
            embed_net.opt.use_vulkan_compute = true;
            proj_out_net.opt.use_vulkan_compute = true;
            decoder_net.opt.use_vulkan_compute = true;
        }
        embed_net.load_param(embed_param.c_str());
        embed_net.load_model(embed_bin.c_str());
        proj_out_net.load_param(proj_out_param.c_str());
        proj_out_net.load_model(embed_bin.c_str());
        decoder_net.load_param(decoder_param.c_str());
        decoder_net.load_model(decoder_bin.c_str());

        bpe.AddAdditionalSpecialToken("<|im_end|>");
        bpe.AddAdditionalSpecialToken("<|im_start|>");
        bpe.AddAdditionalSpecialToken("<|tool_call|>");
        bpe.AddAdditionalSpecialToken("<|execute_start|>");
        bpe.AddAdditionalSpecialToken("<|execute_end|>");
        bpe.AddAdditionalSpecialToken("<|fim_prefix|>");
        bpe.AddAdditionalSpecialToken("<|fim_middle|>");
        bpe.AddAdditionalSpecialToken("<|fim_suffix|>");

        auto it = bpe.token_to_id().find("<|im_end|>");
        if (it != bpe.token_to_id().end()) {
            im_end_id = it->second;
        }
    }
};

// MiniCPM LongRoPE 超参数常量
static const int ORIGINAL_MAX_POSITION_EMBEDDINGS = 32768;
static const float ROPE_BASE = 10000.0f;

// 来自配置的 short_factor 与 long_factor（长度应为 head_dim/2，MiniCPM4-0.5B head_dim=64 => 32）
static const float SHORT_FACTOR[32] = {
1.0004360675811768, 1.0668443441390991, 1.1631425619125366, 1.3025742769241333, 1.5040205717086792, 1.7941505908966064, 2.2101221084594727, 2.802666664123535, 3.6389970779418945, 4.804192543029785, 6.39855432510376, 8.527148246765137, 11.277542114257812, 14.684998512268066, 18.69317054748535, 23.13019371032715, 27.72362518310547, 32.1606559753418, 36.168827056884766, 39.57627868652344, 42.32667541503906, 44.45526885986328, 46.04962921142578, 47.21482849121094, 48.05115509033203, 48.64370346069336, 49.05967712402344, 49.34980392456055, 49.551246643066406, 49.69068145751953, 49.78697967529297, 49.85338592529297
};

static const float LONG_FACTOR[32] = {
1.0004360675811768, 1.0668443441390991, 1.1631425619125366, 1.3025742769241333, 1.5040205717086792, 1.7941505908966064, 2.2101221084594727, 2.802666664123535, 3.6389970779418945, 4.804192543029785, 6.39855432510376, 8.527148246765137, 11.277542114257812, 14.684998512268066, 18.69317054748535, 23.13019371032715, 27.72362518310547, 32.1606559753418, 36.168827056884766, 39.57627868652344, 42.32667541503906, 44.45526885986328, 46.04962921142578, 47.21482849121094, 48.05115509033203, 48.64370346069336, 49.05967712402344, 49.34980392456055, 49.551246643066406, 49.69068145751953, 49.78697967529297, 49.85338592529297
};

// 可选：如果 max_position_embeddings 与 original 不同，需要动态计算 scaling_factor
// 当前配置中二者相等 => scaling_factor = 1
static inline float compute_scaling_factor(int max_position_embeddings) {
    float scale = static_cast<float>(max_position_embeddings) / static_cast<float>(ORIGINAL_MAX_POSITION_EMBEDDINGS);
    // 当 scale == 1 时，log(scale)=0 => scaling_factor=1
    return std::sqrt(1.0f + std::log(scale) / std::log(static_cast<float>(ORIGINAL_MAX_POSITION_EMBEDDINGS)));
}

/**
 * 生成 RoPE 的 cos / sin 缓存（仅前半部分）。
 *
 * @param seqlen      要生成的序列长度（位置数）
 * @param embed_dim   单头维度（head_dim），需为偶数
 * @param position_id 起始位置偏移（用于增量生成 KV cache 时的偏移）
 * @param cos_cache   输出：形状 [seqlen, embed_dim/2] 的余弦缓存（ncnn: w=embed_dim/2, h=seqlen）
 * @param sin_cache   输出：形状 [seqlen, embed_dim/2] 的正弦缓存
 */
static void generate_rope_embed_cache(int seqlen,
                                      int embed_dim,
                                      int position_id,
                                      ncnn::Mat& cos_cache,
                                      ncnn::Mat& sin_cache)
{
    // 安全检查
    if (embed_dim % 2 != 0 || seqlen <= 0) {
        // 简单处理，可根据需要改为抛异常或返回错误码
        cos_cache.release();
        sin_cache.release();
        return;
    }

    const int half_dim = embed_dim / 2;

    // 分配 ncnn::Mat (w=half_dim, h=seqlen, c=1)
    cos_cache.create(half_dim, seqlen);
    sin_cache.create(half_dim, seqlen);

    if (cos_cache.empty() || sin_cache.empty()) {
        return;
    }

    float* cos_ptr = cos_cache.channel(0);
    float* sin_ptr = sin_cache.channel(0);

    // 计算 inv_freq: 公式与 Python 中 torch.arange(0, dim, 2)/dim 对应
    // idx_j = 2*j => exponent = (2*j)/embed_dim
    std::vector<float> inv_freq(half_dim);
    for (int j = 0; j < half_dim; ++j) {
        float exponent = (2.0f * j) / static_cast<float>(embed_dim);
        inv_freq[j] = 1.0f / std::pow(ROPE_BASE, exponent);
    }

    // 根据 seqlen 是否超过 original_max_position_embeddings 选择 factor
    const float* ext_factor = (seqlen > ORIGINAL_MAX_POSITION_EMBEDDINGS) ? LONG_FACTOR : SHORT_FACTOR;

    // 当前配置中 max_position_embeddings == ORIGINAL => scaling_factor = 1
    // 如果你的实际使用中不同，可把 max_position_embeddings 作为参数传入并计算
    const float scaling_factor = compute_scaling_factor(ORIGINAL_MAX_POSITION_EMBEDDINGS);

    // 主循环：生成 cos / sin（仅半维度）
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


minicpm4_0_5b::minicpm4_0_5b(std::string embed_param,
                                 std::string embed_bin,
                                 std::string proj_out_param,
                                 std::string decoder_param,
                                 std::string decoder_bin,
                                 std::string vocab_file,
                                 std::string merges_file,
                                 bool use_vulkan)
    : impl_(std::make_unique<Impl>(std::move(embed_param),
                                  std::move(embed_bin),
                                  std::move(proj_out_param),
                                  std::move(decoder_param),
                                  std::move(decoder_bin),
                                  std::move(vocab_file),
                                  std::move(merges_file),
                                  use_vulkan)) {
    
}

minicpm4_0_5b::~minicpm4_0_5b() = default;

std::shared_ptr<minicpm4_0_5b_ctx> minicpm4_0_5b::prefill(const std::string& input_text) {
    auto token_ids = impl_->bpe.encode(input_text, true, false);
    int last_token_id = token_ids.back();
    token_ids.pop_back();

    ncnn::Mat cos_cache;
    ncnn::Mat sin_cache;
    generate_rope_embed_cache(token_ids.size(), 64, 0, cos_cache, sin_cache);

    ncnn::Mat input_ids_mat = ncnn::Mat((int)token_ids.size(), 1, (void*)token_ids.data()).clone();
    ncnn::Mat token_embed;
    {
        ncnn::Extractor ex = impl_->embed_net.create_extractor();
        ex.input("in0", input_ids_mat);
        ex.extract("out0", token_embed);
    }

    /*
    ncnn::Mat token_embed(1024, src-seqlen);
    ncnn::Mat mask(cur-seqlen, src-seqlen);
    ncnn::Mat cos_cache(32, cur-seqlen);
    ncnn::Mat sin_cache(32, cur-seqlen);
    */

    ncnn::Mat mask((int)token_ids.size(), (int)token_ids.size());
    mask.fill(0.0f);
    for (int i = 0; i < (int)token_ids.size(); i++)
    {
        float* row = mask.row(i);
        for (int j = i + 1; j < (int)token_ids.size(); j++) {
            row[j] = -1e38f;
        }
    }

    std::vector<std::pair<ncnn::Mat, ncnn::Mat>> kv_cache;
    ncnn::Mat decode_out;
    {
        ncnn::Extractor ex = impl_->decoder_net.create_extractor();
        ex.input("in0", token_embed);
        ex.input("in1", mask);
        ex.input("in2", cos_cache);
        ex.input("in3", sin_cache);

        for (int i = 0; i < attn_cnt; i++) {
            char name_k_out[32], name_v_out[32];
            std::snprintf(name_k_out, sizeof(name_k_out), "out_cache_k%d", i);
            std::snprintf(name_v_out, sizeof(name_v_out), "out_cache_v%d", i);
            ncnn::Mat k_cache, v_cache;
            ex.extract(name_k_out, k_cache);
            ex.extract(name_v_out, v_cache);

            kv_cache.emplace_back(std::move(k_cache), std::move(v_cache));
        }
    }

    // full process last token
    ncnn::Mat last_token_mat = ncnn::Mat(1, 1, (void*)&last_token_id).clone();
    ncnn::Mat last_token_embed;
    {
        ncnn::Extractor ex = impl_->embed_net.create_extractor();
        ex.input("in0", last_token_mat);
        ex.extract("out0", last_token_embed);
    }
    ncnn::Mat last_cos_cache;
    ncnn::Mat last_sin_cache;
    generate_rope_embed_cache(1, 64, (int)token_ids.size(), last_cos_cache, last_sin_cache);
    ncnn::Mat last_mask((int)token_ids.size() + 1, 1);
    last_mask.fill(0.0f);

    {
        ncnn::Extractor ex = impl_->decoder_net.create_extractor();
        ex.input("in0", last_token_embed);
        ex.input("in1", last_mask);
        ex.input("in2", last_cos_cache);
        ex.input("in3", last_sin_cache);

        for (int i = 0; i < attn_cnt; i++) {
            char name_k_in[16], name_v_in[16];
            std::snprintf(name_k_in, sizeof(name_k_in), "cache_k%d", i);
            std::snprintf(name_v_in, sizeof(name_v_in), "cache_v%d", i);
            ex.input(name_k_in, kv_cache[i].first);
            ex.input(name_v_in, kv_cache[i].second);
        }

        for (int i = 0; i < attn_cnt; i++) {
            char name_k_out[32], name_v_out[32];
            std::snprintf(name_k_out, sizeof(name_k_out), "out_cache_k%d", i);
            std::snprintf(name_v_out, sizeof(name_v_out), "out_cache_v%d", i);
            ncnn::Mat k_cache, v_cache;
            ex.extract(name_k_out, k_cache);
            ex.extract(name_v_out, v_cache);
            kv_cache[i] = std::make_pair(std::move(k_cache), std::move(v_cache));
        }

        ex.extract("out0", decode_out);
    }

    ncnn::Mat logits;
    {
        ncnn::Extractor ex = impl_->proj_out_net.create_extractor();
        ex.input("in0", decode_out);
        ex.extract("out0", logits);
    }

    int next_token_id = 0;
    {
        const float* p = logits;
        int max_idx = 0;
        float max_val = p[0];
        for (int i = 1; i < logits.w; ++i) {
            if (p[i] > max_val) {
                max_val = p[i];
                max_idx = i;
            }
        }
        next_token_id = max_idx;
    }

    auto ctx = std::make_shared<minicpm4_0_5b_ctx>();
    ctx->kv_cache = std::move(kv_cache);
    ctx->cur_token = next_token_id;

    return ctx;
}

std::shared_ptr<minicpm4_0_5b_ctx> minicpm4_0_5b::prefill(const std::string& input_text,
                                                 const std::shared_ptr<minicpm4_0_5b_ctx> ctx) {
    std::shared_ptr<minicpm4_0_5b_ctx> new_ctx = clone_ctx(ctx);

    auto token_ids = impl_->bpe.encode(input_text, false, false);
    int last_token_id = token_ids.back();
    token_ids.pop_back();

    ncnn::Mat cos_cache;
    ncnn::Mat sin_cache;
    generate_rope_embed_cache(token_ids.size(), 64, new_ctx->kv_cache[0].first.h, cos_cache, sin_cache);
    ncnn::Mat input_ids_mat = ncnn::Mat((int)token_ids.size(), 1, (void*)token_ids.data()).clone();
    ncnn::Mat token_embed;
    {
        ncnn::Extractor ex = impl_->embed_net.create_extractor();
        ex.input("in0", input_ids_mat);
        ex.extract("out0", token_embed);
    }

    ncnn::Mat mask((int)token_ids.size() + new_ctx->kv_cache[0].first.h, (int)token_ids.size());
    mask.fill(0.0f);
    for (int i = 0; i < (int)token_ids.size(); i++)
    {
        float* row = mask.row(i);
        for (int j = new_ctx->kv_cache[0].first.h + i + 1; j < (int)token_ids.size() + new_ctx->kv_cache[0].first.h; j++) {
            row[j] = -1e38f;
        }
    }
    ncnn::Mat decode_out;
    {
        ncnn::Extractor ex = impl_->decoder_net.create_extractor();
        ex.input("in0", token_embed);
        ex.input("in1", mask);
        ex.input("in2", cos_cache);
        ex.input("in3", sin_cache);

        for (int i = 0; i < attn_cnt; i++) {
            char name_k_out[32], name_v_out[32];
            std::snprintf(name_k_out, sizeof(name_k_out), "cache_k%d", i);
            std::snprintf(name_v_out, sizeof(name_v_out), "cache_v%d", i);
            ex.input(name_k_out, new_ctx->kv_cache[i].first);
            ex.input(name_v_out, new_ctx->kv_cache[i].second);
        }

        for (int i = 0; i < attn_cnt; i++) {
            char name_k_out[32], name_v_out[32];
            std::snprintf(name_k_out, sizeof(name_k_out), "out_cache_k%d", i);
            std::snprintf(name_v_out, sizeof(name_v_out), "out_cache_v%d", i);
            ncnn::Mat k_cache, v_cache;
            ex.extract(name_k_out, k_cache);
            ex.extract(name_v_out, v_cache);
            new_ctx->kv_cache[i] = std::make_pair(std::move(k_cache), std::move(v_cache));
        }
    }

    // full process last token
    ncnn::Mat last_token_mat = ncnn::Mat(1, 1, (void*)&last_token_id).clone();
    ncnn::Mat last_token_embed;
    {
        ncnn::Extractor ex = impl_->embed_net.create_extractor();
        ex.input("in0", last_token_mat);
        ex.extract("out0", last_token_embed);
    }
    ncnn::Mat last_cos_cache;
    ncnn::Mat last_sin_cache;

    generate_rope_embed_cache(1, 64, new_ctx->kv_cache[0].first.h, last_cos_cache, last_sin_cache);
    ncnn::Mat last_mask(new_ctx->kv_cache[0].first.h + 1, 1);
    last_mask.fill(0.0f);

    {
        ncnn::Extractor ex = impl_->decoder_net.create_extractor();
        ex.input("in0", last_token_embed);
        ex.input("in1", last_mask);
        ex.input("in2", last_cos_cache);
        ex.input("in3", last_sin_cache);

        for (int i = 0; i < attn_cnt; i++) {
            char name_k_in[16], name_v_in[16];
            std::snprintf(name_k_in, sizeof(name_k_in), "cache_k%d", i);
            std::snprintf(name_v_in, sizeof(name_v_in), "cache_v%d", i);
            ex.input(name_k_in, new_ctx->kv_cache[i].first);
            ex.input(name_v_in, new_ctx->kv_cache[i].second);
        }

        for (int i = 0; i < attn_cnt; i++) {
            char name_k_out[32], name_v_out[32];
            std::snprintf(name_k_out, sizeof(name_k_out), "out_cache_k%d", i);
            std::snprintf(name_v_out, sizeof(name_v_out), "out_cache_v%d", i);
            ncnn::Mat k_cache, v_cache;
            ex.extract(name_k_out, k_cache);
            ex.extract(name_v_out, v_cache);
            new_ctx->kv_cache[i] = std::make_pair(std::move(k_cache), std::move(v_cache));
        }
        ex.extract("out0", decode_out);
    }

    ncnn::Mat logits;
    {
        ncnn::Extractor ex = impl_->proj_out_net.create_extractor();
        ex.input("in0", decode_out);
        ex.extract("out0", logits);
    }
    int next_token_id = 0;
    {
        const float* p = logits;
        int max_idx = 0;
        float max_val = p[0];
        for (int i = 1; i < logits.w; ++i) {
            if (p[i] > max_val) {
                max_val = p[i];
                max_idx = i;
            }
        }
        next_token_id = max_idx;
    }
    new_ctx->cur_token = next_token_id;
    return new_ctx;
}

bool minicpm4_0_5b::decode(std::shared_ptr<minicpm4_0_5b_ctx> ctx,
                            std::function<void(const std::string&)> callback) {

    while (ctx->cur_token != impl_->im_end_id && ctx->cur_token != impl_->bpe.special_ids().eos_id) {
        callback(impl_->bpe.id_to_token()[ctx->cur_token]);

        ncnn::Mat cur_token_mat = ncnn::Mat(1, 1, (void*)&ctx->cur_token).clone();
        ncnn::Mat cur_token_embed;
        {
            ncnn::Extractor ex = impl_->embed_net.create_extractor();
            ex.input("in0", cur_token_mat);
            ex.extract("out0", cur_token_embed);
        }
        ncnn::Mat cos_cache;
        ncnn::Mat sin_cache;
        generate_rope_embed_cache(1, 64, ctx->kv_cache[0].first.h, cos_cache, sin_cache);
        ncnn::Mat mask(ctx->kv_cache[0].first.h + 1, 1);
        mask.fill(0.0f);

        ncnn::Mat decode_out;
        {
            ncnn::Extractor ex = impl_->decoder_net.create_extractor();
            ex.input("in0", cur_token_embed);
            ex.input("in1", mask);
            ex.input("in2", cos_cache);
            ex.input("in3", sin_cache);

            for (int i = 0; i < attn_cnt; i++) {
                char name_k_in[16], name_v_in[16];
                std::snprintf(name_k_in, sizeof(name_k_in), "cache_k%d", i);
                std::snprintf(name_v_in, sizeof(name_v_in), "cache_v%d", i);
                ex.input(name_k_in, ctx->kv_cache[i].first);
                ex.input(name_v_in, ctx->kv_cache[i].second);
            }

            for (int i = 0; i < attn_cnt; i++) {
                char name_k_out[32], name_v_out[32];
                std::snprintf(name_k_out, sizeof(name_k_out), "out_cache_k%d", i);
                std::snprintf(name_v_out, sizeof(name_v_out), "out_cache_v%d", i);
                ncnn::Mat k_cache, v_cache;
                ex.extract(name_k_out, k_cache);
                ex.extract(name_v_out, v_cache);
                ctx->kv_cache[i] = std::make_pair(std::move(k_cache), std::move(v_cache));
            }

            ex.extract("out0", decode_out);
        }

        ncnn::Mat logits;
        {
            ncnn::Extractor ex = impl_->proj_out_net.create_extractor();
            ex.input("in0", decode_out);
            ex.extract("out0", logits);
        }

        int next_token_id = 0;
        {
            const float* p = logits;
            int max_idx = 0;
            float max_val = p[0];
            for (int i = 1; i < impl_->bpe.vocab_size(); ++i) {
                if (p[i] > max_val) {
                    max_val = p[i];
                    max_idx = i;
                }
            }
            next_token_id = max_idx;
        }
        ctx->cur_token = next_token_id;
    }

    return true;
}


std::shared_ptr<minicpm4_0_5b_ctx> minicpm4_0_5b::generate(
    const std::shared_ptr<minicpm4_0_5b_ctx>& ctx_in,
    const GenerateConfig& cfg,
    std::function<void(const std::string&)> callback)
{
    const int vocab_size = impl_->bpe.vocab_size();
    const int eos     = impl_->bpe.special_ids().eos_id;
    const int im_end  = impl_->im_end_id;

    // ---------- Do Sample or Greedy ----------
    if (cfg.do_sample == 1 || cfg.beam_size <= 1) {
        auto ctx = clone_ctx(ctx_in);
        std::vector<int> history;
        history.push_back(ctx->cur_token);

        for (int step = 0; step < cfg.max_new_tokens; ++step) {
            // Stop
            if (ctx->cur_token == eos || ctx->cur_token == im_end) {
                break;
            }

            callback(impl_->bpe.id_to_token()[ctx->cur_token]);

            ncnn::Mat cur_token_mat = ncnn::Mat(1, 1, (void*)&ctx->cur_token).clone();
            ncnn::Mat cur_embed;
            {
                ncnn::Extractor ex = impl_->embed_net.create_extractor();
                ex.input("in0", cur_token_mat);
                ex.extract("out0", cur_embed);
            }

            ncnn::Mat cos_cache, sin_cache;
            generate_rope_embed_cache(1, 64, ctx->kv_cache[0].first.h, cos_cache, sin_cache);

            ncnn::Mat mask(ctx->kv_cache[0].first.h + 1, 1);
            mask.fill(0.f);

            ncnn::Mat decode_out;
            {
                ncnn::Extractor ex = impl_->decoder_net.create_extractor();
                ex.input("in0", cur_embed);
                ex.input("in1", mask);
                ex.input("in2", cos_cache);
                ex.input("in3", sin_cache);

                for (int i = 0; i < attn_cnt; ++i) {
                    char kname[16], vname[16];
                    std::snprintf(kname, sizeof(kname), "cache_k%d", i);
                    std::snprintf(vname, sizeof(vname), "cache_v%d", i);
                    ex.input(kname, ctx->kv_cache[i].first);
                    ex.input(vname, ctx->kv_cache[i].second);
                }

                for (int i = 0; i < attn_cnt; ++i) {
                    char kname[32], vname[32];
                    std::snprintf(kname, sizeof(kname), "out_cache_k%d", i);
                    std::snprintf(vname, sizeof(vname), "out_cache_v%d", i);
                    ncnn::Mat k_cache, v_cache;
                    ex.extract(kname, k_cache);
                    ex.extract(vname, v_cache);
                    ctx->kv_cache[i] = { k_cache, v_cache };
                }

                ex.extract("out0", decode_out);
            }

            ncnn::Mat logits_mat;
            {
                ncnn::Extractor ex = impl_->proj_out_net.create_extractor();
                ex.input("in0", decode_out);
                ex.extract("out0", logits_mat);
            }

            std::vector<float> logits(vocab_size);
            memcpy(logits.data(), logits_mat.data, sizeof(float) * vocab_size);

            // repetition_penalty
            for (int t : history) {
                logits[t] /= cfg.repetition_penalty;
            }

            softmax_vec(logits, cfg.temperature);
            if (cfg.top_k > 0)       apply_top_k(logits, cfg.top_k);
            if (cfg.top_p < 1.0f)    apply_top_p(logits, cfg.top_p);

            int next_id;
            if (cfg.do_sample == 1) {
                next_id = sample_from_probs(logits);
            } else {
                next_id = std::max_element(logits.begin(), logits.end()) - logits.begin();
            }

            ctx->cur_token = next_id;
            history.push_back(next_id);
        }

        return ctx;  // Update
    }

    // ---------- Beam Search ----------

    auto base_ctx = clone_ctx(ctx_in);
    std::vector<Beam> beams;
    beams.reserve(cfg.beam_size);

    Beam b0;
    b0.ctx = base_ctx;
    b0.tokens.push_back(base_ctx->cur_token);
    beams.push_back(std::move(b0));

    for (int step = 0; step < cfg.max_new_tokens; ++step) {
        std::vector<Beam> candidates;

        for (auto& beam : beams) {
            auto& bctx = *beam.ctx;
            if (beam.finished || bctx.cur_token == eos || bctx.cur_token == im_end) {
                beam.finished = true;
                candidates.push_back(beam);
                continue;
            }

            ncnn::Mat cur_token_mat = ncnn::Mat(1, 1, (void*)&bctx.cur_token).clone();
            ncnn::Mat cur_embed;
            {
                ncnn::Extractor ex = impl_->embed_net.create_extractor();
                ex.input("in0", cur_token_mat);
                ex.extract("out0", cur_embed);
            }

            ncnn::Mat cos_cache, sin_cache;
            generate_rope_embed_cache(1, 64, bctx.kv_cache[0].first.h, cos_cache, sin_cache);

            ncnn::Mat mask(bctx.kv_cache[0].first.h + 1, 1);
            mask.fill(0.f);

            ncnn::Mat decode_out;
            {
                ncnn::Extractor ex = impl_->decoder_net.create_extractor();
                ex.input("in0", cur_embed);
                ex.input("in1", mask);
                ex.input("in2", cos_cache);
                ex.input("in3", sin_cache);

                for (int i = 0; i < attn_cnt; ++i) {
                    char kname[16], vname[16];
                    std::snprintf(kname, sizeof(kname), "cache_k%d", i);
                    std::snprintf(vname, sizeof(vname), "cache_v%d", i);
                    ex.input(kname, bctx.kv_cache[i].first);
                    ex.input(vname, bctx.kv_cache[i].second);
                }

                for (int i = 0; i < attn_cnt; ++i) {
                    char kname[32], vname[32];
                    std::snprintf(kname, sizeof(kname), "out_cache_k%d", i);
                    std::snprintf(vname, sizeof(vname), "out_cache_v%d", i);
                    ncnn::Mat k_cache, v_cache;
                    ex.extract(kname, k_cache);
                    ex.extract(vname, v_cache);
                    bctx.kv_cache[i] = { k_cache, v_cache };
                }

                ex.extract("out0", decode_out);
            }

            ncnn::Mat logits_mat;
            {
                ncnn::Extractor ex = impl_->proj_out_net.create_extractor();
                ex.input("in0", decode_out);
                ex.extract("out0", logits_mat);
            }

            std::vector<float> logits(vocab_size);
            memcpy(logits.data(), logits_mat.data, sizeof(float) * vocab_size);

            // repetition_penalty
            for (int t : beam.tokens) {
                logits[t] /= cfg.repetition_penalty;
            }

            // softmax
            softmax_vec(logits, cfg.temperature);

            int K = std::min(cfg.beam_size, vocab_size);
            std::vector<std::pair<float,int>> top;
            top.reserve(vocab_size);
            for (int i = 0; i < vocab_size; ++i) {
                top.emplace_back(logits[i], i);
            }
            std::partial_sort(top.begin(), top.begin() + K, top.end(),
                              [](auto& a, auto& b){ return a.first > b.first; });

            for (int i = 0; i < K; ++i) {
                int tok = top[i].second;
                float p  = top[i].first;
                Beam nb;
                nb.ctx = clone_ctx(beam.ctx);
                nb.ctx->cur_token = tok;
                nb.tokens = beam.tokens;
                nb.tokens.push_back(tok);
                nb.score  = beam.score + std::log(p + 1e-9f);
                nb.finished = (tok == eos || tok == im_end);
                candidates.push_back(std::move(nb));
            }
        }

        std::sort(candidates.begin(), candidates.end(),
                  [](const Beam& a, const Beam& b) {
                      return a.score > b.score;
                  });
        if ((int)candidates.size() > cfg.beam_size)
            candidates.resize(cfg.beam_size);

        beams = std::move(candidates);

        auto& best = beams[0];
        int tok = best.ctx->cur_token;
        if (tok == eos || tok == im_end || best.finished) {
            break;
        }
        callback(impl_->bpe.id_to_token()[tok]);

        bool all_finished = true;
        for (auto& b : beams) {
            if (!b.finished) { all_finished = false; break; }
        }
        if (all_finished) break;
    }

    auto best_it = std::max_element(
        beams.begin(), beams.end(),
        [](const Beam& a, const Beam& b) { return a.score < b.score; });
    return best_it->ctx;
}