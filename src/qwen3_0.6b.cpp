#include "qwen3_0.6b.h"

#include <cstdio>
#include <memory>
#include <ncnn/mat.h>
#include <ncnn/net.h>
#include <string>
#include <utility>
#include <vector>
#include <random>

#include "utils/tokenizer/bpe_tokenizer.h"
#include "utils/rope_embed.h"

#include "nlohmann/json.hpp"

static std::mt19937 rng(std::random_device{}());

const static int attn_cnt = 28;

struct qwen3_0_6b_ctx {
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
    std::shared_ptr<qwen3_0_6b_ctx> ctx;
    float score = 0.f;
    bool finished = false;
    std::vector<int> tokens;
};

// 深拷贝 KV Cache
static std::shared_ptr<qwen3_0_6b_ctx>
clone_ctx(const std::shared_ptr<qwen3_0_6b_ctx>& src) {
    auto dst = std::make_shared<qwen3_0_6b_ctx>();
    dst->cur_token = src->cur_token;
    dst->kv_cache.resize(src->kv_cache.size());
    for (size_t i = 0; i < src->kv_cache.size(); ++i) {
        dst->kv_cache[i].first = src->kv_cache[i].first;
        dst->kv_cache[i].second = src->kv_cache[i].second;
    }
    return dst;
}

class qwen3_0_6b::Impl {
public:
    ncnn::Net embed_net;
    ncnn::Net proj_out_net;
    ncnn::Net decoder_net;

    BpeTokenizer bpe;

    int im_end_id = -1;
    int tool_call_id = -1;
    int tool_call_end_id = -1;

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
              SpecialTokensConfig{},
            false,
            true,
            true
            )) {
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

        bpe.AddAdditionalSpecialToken("<|endoftext|>");
        bpe.AddAdditionalSpecialToken("<|im_start|>");
        bpe.AddAdditionalSpecialToken("<|im_end|>");
        bpe.AddAdditionalSpecialToken("<|object_ref_start|>");
        bpe.AddAdditionalSpecialToken("<|object_ref_end|>");
        bpe.AddAdditionalSpecialToken("<|box_start|>");
        bpe.AddAdditionalSpecialToken("<|box_end|>");
        bpe.AddAdditionalSpecialToken("<|quad_start|>");
        bpe.AddAdditionalSpecialToken("<|quad_end|>");
        bpe.AddAdditionalSpecialToken("<|vision_start|>");
        bpe.AddAdditionalSpecialToken("<|vision_end|>");
        bpe.AddAdditionalSpecialToken("<|vision_pad|>");
        bpe.AddAdditionalSpecialToken("<|image_pad|>");
        bpe.AddAdditionalSpecialToken("<|video_pad|>");
        bpe.AddAdditionalSpecialToken("<tool_call>");
        bpe.AddAdditionalSpecialToken("</tool_call>");
        bpe.AddAdditionalSpecialToken("<|fim_prefix|>");
        bpe.AddAdditionalSpecialToken("<|fim_middle|>");
        bpe.AddAdditionalSpecialToken("<|fim_suffix|>");
        bpe.AddAdditionalSpecialToken("<|fim_pad|>");
        bpe.AddAdditionalSpecialToken("<|repo_name|>");
        bpe.AddAdditionalSpecialToken("<|file_sep|>");
        bpe.AddAdditionalSpecialToken("<tool_response>");
        bpe.AddAdditionalSpecialToken("</tool_response>");
        bpe.AddAdditionalSpecialToken("<think>");
        bpe.AddAdditionalSpecialToken("</think>");

        
        auto it = bpe.token_to_id().find("<|im_end|>");
        if (it != bpe.token_to_id().end()) {
            im_end_id = it->second;
        }

        it = bpe.token_to_id().find("<tool_call>");
        if (it != bpe.token_to_id().end()) {
            tool_call_id = it->second;
        }

        it = bpe.token_to_id().find("</tool_call>");
        if (it != bpe.token_to_id().end()) {
            tool_call_end_id = it->second;
        }
    }
};

qwen3_0_6b::qwen3_0_6b(std::string embed_param,
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

qwen3_0_6b::~qwen3_0_6b() = default;

std::shared_ptr<qwen3_0_6b_ctx> qwen3_0_6b::prefill(const std::string& input_text) {
    auto token_ids = impl_->bpe.encode(input_text, false, false);
    int last_token_id = token_ids.back();
    token_ids.pop_back();

    ncnn::Mat cos_cache;
    ncnn::Mat sin_cache;
    generate_rope_embed_cache(token_ids.size(), 128, 0, cos_cache, sin_cache);

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
    ncnn::Mat cos_cache(64, cur-seqlen);
    ncnn::Mat sin_cache(64, cur-seqlen);
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
    generate_rope_embed_cache(1, 128, (int)token_ids.size(), last_cos_cache, last_sin_cache);
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

    auto ctx = std::make_shared<qwen3_0_6b_ctx>();
    ctx->kv_cache = std::move(kv_cache);
    ctx->cur_token = next_token_id;

    return ctx;
}

std::shared_ptr<qwen3_0_6b_ctx> qwen3_0_6b::prefill(const std::string& input_text,
                                                 const std::shared_ptr<qwen3_0_6b_ctx> ctx) {
    std::shared_ptr<qwen3_0_6b_ctx> new_ctx = clone_ctx(ctx);

    auto token_ids = impl_->bpe.encode(input_text, false, false);
    int last_token_id = token_ids.back();
    token_ids.pop_back();

    ncnn::Mat cos_cache;
    ncnn::Mat sin_cache;
    generate_rope_embed_cache(token_ids.size(), 128, new_ctx->kv_cache[0].first.h, cos_cache, sin_cache);
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

    generate_rope_embed_cache(1, 128, new_ctx->kv_cache[0].first.h, last_cos_cache, last_sin_cache);
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

bool qwen3_0_6b::decode(std::shared_ptr<qwen3_0_6b_ctx> ctx,
                            std::function<void(const std::string&)> callback) {

    while (ctx->cur_token != impl_->im_end_id && ctx->cur_token != impl_->bpe.special_ids().eos_id) {
        callback(impl_->bpe.decode({ctx->cur_token}, false));

        ncnn::Mat cur_token_mat = ncnn::Mat(1, 1, (void*)&ctx->cur_token).clone();
        ncnn::Mat cur_token_embed;
        {
            ncnn::Extractor ex = impl_->embed_net.create_extractor();
            ex.input("in0", cur_token_mat);
            ex.extract("out0", cur_token_embed);
        }
        ncnn::Mat cos_cache;
        ncnn::Mat sin_cache;
        generate_rope_embed_cache(1, 128, ctx->kv_cache[0].first.h, cos_cache, sin_cache);
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


std::shared_ptr<qwen3_0_6b_ctx> qwen3_0_6b::generate(
    const std::shared_ptr<qwen3_0_6b_ctx>& ctx_in,
    const GenerateConfig& cfg,
    std::function<void(const std::string&)> callback)
{
    const int vocab_size = impl_->bpe.vocab_size();
    const int eos     = impl_->bpe.special_ids().eos_id;
    const int im_end  = impl_->im_end_id;

    bool flag_in_tool_call = false;
    std::string tool_call_content;

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

            if (ctx->cur_token == impl_->tool_call_id) {
                flag_in_tool_call = true;
                fprintf(stderr,"\n[Debug] Entering tool call mode.\n");
            } else if (ctx->cur_token == impl_->tool_call_end_id) {
                flag_in_tool_call = false;
                
                nlohmann::json tool_call_json;
                try {
                    tool_call_json = nlohmann::json::parse(tool_call_content);
                } catch (const std::exception& e) {
                    fprintf(stderr, "\n[Error] Failed to parse tool call JSON: %s\n", e.what());
                    tool_call_json = nlohmann::json::object();
                }
                // print json with indentation
                fprintf(stderr, "\n[Tool Call] %s\n", tool_call_json.dump(4).c_str());

                std::string tool_response_pre = "<|im_end|>\n<|im_start|>user\n<tool_response>\n\n";
                std::string tool_response_post = "\n\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n\n";

                ctx = prefill(tool_response_pre + "{\"result\": 114514}" + tool_response_post, ctx);

                fprintf(stderr,"\n[Debug] Exiting tool call mode.\n");

                continue;
            } else if (flag_in_tool_call) {
                fprintf(stderr, "%s", impl_->bpe.decode({ctx->cur_token}, false).c_str());
                tool_call_content += impl_->bpe.decode({ctx->cur_token}, false);
            }
            else 
                callback(impl_->bpe.decode({ctx->cur_token}, false));

            ncnn::Mat cur_token_mat = ncnn::Mat(1, 1, (void*)&ctx->cur_token).clone();
            ncnn::Mat cur_embed;
            {
                ncnn::Extractor ex = impl_->embed_net.create_extractor();
                ex.input("in0", cur_token_mat);
                ex.extract("out0", cur_embed);
            }

            ncnn::Mat cos_cache, sin_cache;
            generate_rope_embed_cache(1, 128, ctx->kv_cache[0].first.h, cos_cache, sin_cache);

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

    callback(impl_->bpe.decode({ctx_in->cur_token}, false));

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
            generate_rope_embed_cache(1, 128, bctx.kv_cache[0].first.h, cos_cache, sin_cache);

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
        callback(impl_->bpe.decode({tok}, false));

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