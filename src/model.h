// Copyright 2025 Tencent
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cassert>
#include <cstdio>
#include <exception>
#include <functional>
#include <locale>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>
#include <fstream>
#include <cmath>
#include <random>
#include <unordered_set>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include "utils/tokenizer/bpe_tokenizer.h"
#include "utils/rope_embed.h"
#include "utils/prompt.h"

#include <ncnn/mat.h>
#include <ncnn/net.h>
#include <nlohmann/json.hpp>
using nlohmann::json;

static std::mt19937 rng(std::random_device{}());

struct GenerateConfig {
    int max_new_tokens = 4096;
    float temperature = 0.3f;
    float top_p = 0.8f;
    int top_k = 50;
    float repetition_penalty = 1.1f;
    int beam_size = 1;
    int do_sample = 1;

    std::function<nlohmann::json(const nlohmann::json&)> tool_callback = nullptr;

    bool debug = false;
};


struct ncnn_llm_gpt_ctx
{
    std::vector<std::pair<ncnn::Mat, ncnn::Mat>> kv_cache;
    int cur_token = 0;
};


// ==== Sampling utilities ====

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

// ==== Beam state structure ====

struct Beam {
    std::shared_ptr<ncnn_llm_gpt_ctx> ctx;
    float score = 0.f;
    bool finished = false;
    bool in_tool_call = false;
    std::string tool_buffer;
    std::unordered_set<int> tokens;
    
    // Added fields for correct emission tracking
    int prev_token = -1;
    bool prev_in_tool_call = false;
};


static std::shared_ptr<ncnn_llm_gpt_ctx>
clone_ctx(const std::shared_ptr<ncnn_llm_gpt_ctx>& src) {
    auto dst = std::make_shared<ncnn_llm_gpt_ctx>();
    dst->cur_token = src->cur_token;
    dst->kv_cache.resize(src->kv_cache.size());
    for (size_t i = 0; i < src->kv_cache.size(); ++i) {
        dst->kv_cache[i].first = src->kv_cache[i].first;
        dst->kv_cache[i].second = src->kv_cache[i].second;
    }
    return dst;
}

class ncnn_llm_gpt {
public:
    ncnn_llm_gpt(
            const std::string& model_path,
            bool use_vulkan = false
    ) 
    {
        try {
            json config;
            {
                std::ifstream ifs(model_path + "/model.json");
                ifs >> config;
            }
            
            // load base model
            {
                decoder_net = std::make_shared<ncnn::Net>();
                embed_net = std::make_shared<ncnn::Net>();
                proj_out_net =  std::make_shared<ncnn::Net>();

                if (use_vulkan) {
                    decoder_net->opt.use_vulkan_compute = true;
                    embed_net->opt.use_vulkan_compute = true;
                    proj_out_net->opt.use_vulkan_compute = true;
                }

                std::string decoder_param = model_path + "/" + config["params"]["decoder_param"].get<std::string>();
                std::string decoder_bin = model_path + "/" + config["params"]["decoder_bin"].get<std::string>();
                std::string embed_param = model_path + "/" + config["params"]["embed_token_param"].get<std::string>();
                std::string embed_bin = model_path + "/" + config["params"]["embed_token_bin"].get<std::string>();
                std::string proj_out_param = model_path + "/" + config["params"]["proj_out_param"].get<std::string>();

                decoder_net->load_param(decoder_param.c_str());
                decoder_net->load_model(decoder_bin.c_str());
                embed_net->load_param(embed_param.c_str());
                embed_net->load_model(embed_bin.c_str());
                proj_out_net->load_param(proj_out_param.c_str());
                proj_out_net->load_model(embed_bin.c_str());
            }

            // load tokenizer
            {
                std::string type = "bpe";
                if (config["tokenizer"].contains("type")) {
                    type = config["tokenizer"]["type"].get<std::string>();
                }
                std::string vocab_file = model_path + "/" + config["tokenizer"]["vocab_file"].get<std::string>();
                std::string merges_file = model_path + "/" + config["tokenizer"]["merges_file"].get<std::string>();

                bpe = std::make_shared<BpeTokenizer>(BpeTokenizer::LoadFromFiles(
                    vocab_file,
                    merges_file,
                    SpecialTokensConfig{},
                    false,
                    true,
                    type == "bbpe" ? true : false
                ));

                // add special tokens
                std::vector<std::string> additional_special_tokens = config["tokenizer"]["additional_special_tokens"].get<std::vector<std::string>>();
                for (const auto& token : additional_special_tokens) {
                    bpe->AddAdditionalSpecialToken(token);
                }

                auto eos_token = config["tokenizer"]["eos"].get<std::string>();
                if (eos_token != "") 
                    eos = bpe->token_to_id().at(eos_token);
                else
                    eos = -1;

                auto bos_token = config["tokenizer"]["bos"].get<std::string>();
                if (bos_token != "")
                    bos = bpe->token_to_id().at(bos_token);
                else
                    bos = -1;

            }
            
            // model settings
            {
                // attn cnt
                if (config["setting"].contains("attn_cnt")) {
                    attn_cnt = config["setting"]["attn_cnt"].get<int>();
                }

                // rope
                if (config["setting"].contains("rope")) {
                    auto rope_cfg = config["setting"]["rope"];
                    if (rope_cfg.contains("rope_head_dim")) {
                        rope_head_dim = rope_cfg["rope_head_dim"].get<int>();
                    }
                    if (rope_cfg["type"] == "LongRoPE") {
                        rope_type = RoPE_Type::LongRoPE;

                        short_factor = rope_cfg["short_factor"].get<std::vector<float>>();
                        long_factor = rope_cfg["long_factor"].get<std::vector<float>>();
                        original_max_position_embeddings = rope_cfg["original_max_position_embeddings"].get<int>();
                    } else if (rope_cfg["type"] == "RoPE") {
                        rope_type = RoPE_Type::RoPE;
                    }
                    rope_theta = rope_cfg["rope_theta"].get<float>();
                }

                // tool_call
                if (config["setting"].contains("functions")) {
                    auto func_cfg = config["setting"]["functions"];
                    std::string type = func_cfg["type"].get<std::string>();
                    if (type == "tool_call") {
                        if (func_cfg.contains("tool_call_id")) {
                        tool_call_id = bpe->token_to_id().at(func_cfg["tool_call_id"].get<std::string>());
                        }
                        if (func_cfg.contains("tool_call_end_id")) {
                            tool_call_end_id = bpe->token_to_id().at(func_cfg["tool_call_end_id"].get<std::string>());
                        }
                    }
                }
            }
        
        } catch (std::exception &e)
        {
            throw std::runtime_error(std::string("ncnn_llm_gpt load model failed: ") + e.what());
        }
    };
    
    std::shared_ptr<ncnn_llm_gpt_ctx> prefill(const std::string& input_text) const
    {
        auto token_ids = bpe->encode(input_text, false, false);
        if (bos >= 0)
        {
            token_ids.insert(token_ids.begin(), bos);    
        }

        int last_token_id = token_ids.back();
        token_ids.pop_back();

        ncnn::Mat cos_cache;
        ncnn::Mat sin_cache;

        if (rope_type == RoPE_Type::LongRoPE) {
            generate_rope_embed_cache_LongRoPE(token_ids.size(), rope_head_dim, 0, cos_cache, sin_cache, rope_theta, short_factor.data(), long_factor.data(), original_max_position_embeddings);
        } else {
            generate_rope_embed_cache(token_ids.size(), rope_head_dim, 0, cos_cache, sin_cache, rope_theta);
        }

        ncnn::Mat input_ids_mat = ncnn::Mat((int)token_ids.size(), 1, (void*)token_ids.data()).clone();
        ncnn::Mat token_embed;
        {
            ncnn::Extractor ex = embed_net->create_extractor();
            ex.input("in0", input_ids_mat);
            ex.extract("out0", token_embed);
        }

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
            ncnn::Extractor ex = decoder_net->create_extractor();
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

        ncnn::Mat last_token_mat = ncnn::Mat(1, 1, (void*)&last_token_id).clone();
        ncnn::Mat last_token_embed;
        {
            ncnn::Extractor ex = embed_net->create_extractor();
            ex.input("in0", last_token_mat);
            ex.extract("out0", last_token_embed);
        }
        ncnn::Mat last_cos_cache;
        ncnn::Mat last_sin_cache;
        if (rope_type == RoPE_Type::LongRoPE) {
            generate_rope_embed_cache_LongRoPE(1, rope_head_dim, (int)token_ids.size(), last_cos_cache, last_sin_cache, rope_theta, short_factor.data(), long_factor.data(), original_max_position_embeddings);
        } else {
            generate_rope_embed_cache(1, rope_head_dim, (int)token_ids.size(), last_cos_cache, last_sin_cache, rope_theta);
        }

        ncnn::Mat last_mask((int)token_ids.size() + 1, 1);
        last_mask.fill(0.0f);

        {
            ncnn::Extractor ex = decoder_net->create_extractor();
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
            ncnn::Extractor ex = proj_out_net->create_extractor();
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

        auto ctx = std::make_shared<ncnn_llm_gpt_ctx>();
        ctx->kv_cache = std::move(kv_cache);
        ctx->cur_token = next_token_id;

        return ctx;
    }

    std::shared_ptr<ncnn_llm_gpt_ctx> prefill(const std::string& input_text, const cv::Mat& bgr, const std::shared_ptr<ncnn_llm_gpt_ctx> ctx) const;
    
    std::shared_ptr<ncnn_llm_gpt_ctx> prefill(const std::string& input_text, const std::shared_ptr<ncnn_llm_gpt_ctx> ctx) const
    {
        std::shared_ptr<ncnn_llm_gpt_ctx> new_ctx = clone_ctx(ctx);

        auto token_ids = bpe->encode(input_text, false, false);
        int last_token_id = token_ids.back();
        token_ids.pop_back();

        ncnn::Mat cos_cache;
        ncnn::Mat sin_cache;

        if (rope_type == RoPE_Type::LongRoPE) {
            generate_rope_embed_cache_LongRoPE(token_ids.size(), rope_head_dim, new_ctx->kv_cache[0].first.h, cos_cache, sin_cache, rope_theta, short_factor.data(), long_factor.data(), original_max_position_embeddings);
        } else {
            generate_rope_embed_cache(token_ids.size(), rope_head_dim, new_ctx->kv_cache[0].first.h, cos_cache, sin_cache, rope_theta);
        }
        
        ncnn::Mat input_ids_mat = ncnn::Mat((int)token_ids.size(), 1, (void*)token_ids.data()).clone();
        ncnn::Mat token_embed;
        {
            ncnn::Extractor ex = embed_net->create_extractor();
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
            ncnn::Extractor ex = decoder_net->create_extractor();
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

        ncnn::Mat last_token_mat = ncnn::Mat(1, 1, (void*)&last_token_id).clone();
        ncnn::Mat last_token_embed;
        {
            ncnn::Extractor ex = embed_net->create_extractor();
            ex.input("in0", last_token_mat);
            ex.extract("out0", last_token_embed);
        }
        ncnn::Mat last_cos_cache;
        ncnn::Mat last_sin_cache;

        if (rope_type == RoPE_Type::LongRoPE) {
            generate_rope_embed_cache_LongRoPE(1, rope_head_dim, new_ctx->kv_cache[0].first.h, last_cos_cache, last_sin_cache, rope_theta, short_factor.data(), long_factor.data(), original_max_position_embeddings);
        } else {
            generate_rope_embed_cache(1, rope_head_dim, new_ctx->kv_cache[0].first.h, last_cos_cache, last_sin_cache, rope_theta);
        }
        
        ncnn::Mat last_mask(new_ctx->kv_cache[0].first.h + 1, 1);
        last_mask.fill(0.0f);

        {
            ncnn::Extractor ex = decoder_net->create_extractor();
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
            ncnn::Extractor ex = proj_out_net->create_extractor();
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

    std::shared_ptr<ncnn_llm_gpt_ctx> generate(const std::shared_ptr<ncnn_llm_gpt_ctx>& ctx_in, const GenerateConfig& cfg, std::function<void(const std::string&)> callback) const
    {
        const int vocab_size = bpe->vocab_size();

        auto handle_tool = [&](const std::string& tool_call_text,
                            std::shared_ptr<ncnn_llm_gpt_ctx>& ctx_ref) {
            if (cfg.debug) {
                fprintf(stderr, "\n[Debug] Raw tool_call text: %s\n", tool_call_text.c_str());
            }

            nlohmann::json tool_call_json;
            try {
                tool_call_json = nlohmann::json::parse(tool_call_text);
            } catch (const std::exception& e) {
                fprintf(stderr, "\n[Error] Failed to parse tool call JSON: %s\n", e.what());
                tool_call_json = nlohmann::json::object();
            }

            if (cfg.debug) {
                fprintf(stderr, "[Debug] Parsed tool_call JSON:\n%s\n", tool_call_json.dump(2).c_str());
            }

            nlohmann::json tool_resp;
            if (cfg.tool_callback) {
                tool_resp = cfg.tool_callback(tool_call_json);
            } else {
                tool_resp = nlohmann::json{{"tool_call", tool_call_json}};
            }

            if (cfg.debug) {
                fprintf(stderr, "[Debug] Tool callback response:\n%s\n", tool_resp.dump(2).c_str());
            }

            std::string tool_response_pre = "<|im_end|>\n<|im_start|>user\n<tool_response>\n\n";
            std::string tool_response_post = "\n\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n\n";

            ctx_ref = prefill(tool_response_pre + tool_resp.dump() + tool_response_post, ctx_ref);

            if (cfg.debug) {
                fprintf(stderr, "[Debug] Tool response injected, continue decoding.\n");
            }
        };

        // ---------- Do Sample or Greedy ----------
        if (cfg.do_sample == 1 || cfg.beam_size <= 1) {
            auto ctx = clone_ctx(ctx_in);
            std::unordered_set<int> history;
            history.insert(ctx->cur_token);

            bool flag_in_tool_call = false;
            std::string tool_call_content;

            for (int step = 0; step < cfg.max_new_tokens; ++step) {
                if (ctx->cur_token == eos) {
                    break;
                }

                if (ctx->cur_token == tool_call_id) {
                    flag_in_tool_call = true;
                    if (cfg.debug) fprintf(stderr, "[Debug] Enter tool_call mode (greedy).\n");
                } else if (ctx->cur_token == tool_call_end_id) {
                    flag_in_tool_call = false;
                    if (cfg.debug) fprintf(stderr, "[Debug] Exit tool_call mode, handling call.\n");
                    handle_tool(tool_call_content, ctx);
                    tool_call_content.clear();
                    history.clear();
                    history.insert(ctx->cur_token);
                    continue;
                } else if (flag_in_tool_call) {
                    tool_call_content += bpe->decode({ctx->cur_token}, false);
                } else {
                    callback(bpe->decode({ctx->cur_token}, false));
                }

                ncnn::Mat cur_token_mat = ncnn::Mat(1, 1, (void*)&ctx->cur_token).clone();
                ncnn::Mat cur_embed;
                {
                    ncnn::Extractor ex = embed_net->create_extractor();
                    ex.input("in0", cur_token_mat);
                    ex.extract("out0", cur_embed);
                }

                ncnn::Mat cos_cache, sin_cache;
                if (rope_type == RoPE_Type::LongRoPE) {
                    generate_rope_embed_cache_LongRoPE(1, rope_head_dim, ctx->kv_cache[0].first.h, cos_cache, sin_cache, rope_theta, short_factor.data(), long_factor.data(), original_max_position_embeddings);
                } else {
                    generate_rope_embed_cache(1, rope_head_dim, ctx->kv_cache[0].first.h, cos_cache, sin_cache, rope_theta);
                }

                ncnn::Mat mask(ctx->kv_cache[0].first.h + 1, 1);
                mask.fill(0.f);

                ncnn::Mat decode_out;
                {
                    ncnn::Extractor ex = decoder_net->create_extractor();
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
                    ncnn::Extractor ex = proj_out_net->create_extractor();
                    ex.input("in0", decode_out);
                    ex.extract("out0", logits_mat);
                }

                std::vector<float> logits(vocab_size);
                memcpy(logits.data(), logits_mat.data, sizeof(float) * vocab_size);

                for (int t : history) {
                    if (logits[t] < 0)
                        logits[t] *= cfg.repetition_penalty;
                    else
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
                history.insert(next_id);
            }

            return ctx;
        }

        // ---------- Beam Search ----------

        auto base_ctx = clone_ctx(ctx_in);
        std::vector<Beam> beams;
        beams.reserve(cfg.beam_size);

        Beam b0;
        b0.ctx = base_ctx;
        b0.tokens.insert(base_ctx->cur_token);
        // For the first beam, prev_token is technically the last prompt token, 
        // but base_ctx->cur_token holds the FIRST generated token (from prefill).
        // If we want to emit it, we should ensure logic handles it. 
        // We initialize prev_token to -1 here because b0 is input to loop.
        // The candidates generated FROM b0 will have b0.ctx->cur_token as prev_token.
        b0.prev_token = -1; 
        beams.push_back(std::move(b0));

        // Detect for init token
        if (beams[0].ctx->cur_token == tool_call_id) {
            beams[0].in_tool_call = true;
            if (cfg.debug) fprintf(stderr, "[Debug] Initial token is <tool_call>, enter tool_call mode.\n");
        }

        auto maybe_emit_prev = [&](const Beam& best){
            int t = best.prev_token;
            // Emit only if valid, not special control tokens, and NOT inside a tool call
            if (t != -1 
                && !best.prev_in_tool_call
                && t != eos
                && t != tool_call_id
                && t != tool_call_end_id) {
                callback(bpe->decode({t}, false));
            }
        };

        for (int step = 0; step < cfg.max_new_tokens; ++step) {
            std::vector<Beam> candidates;
            candidates.reserve(cfg.beam_size * 2);

            Beam tool_completed;
            bool has_tool_completed = false;

            for (auto& beam : beams) {
                auto& bctx = *beam.ctx;
                if (beam.finished || bctx.cur_token == eos) {
                    beam.finished = true;
                    // For finished beams, we just keep them alive for sorting
                    // We don't advance them.
                    candidates.push_back(beam);
                    continue;
                }

                ncnn::Mat cur_token_mat = ncnn::Mat(1, 1, (void*)&bctx.cur_token).clone();
                ncnn::Mat cur_embed;
                {
                    ncnn::Extractor ex = embed_net->create_extractor();
                    ex.input("in0", cur_token_mat);
                    ex.extract("out0", cur_embed);
                }

                ncnn::Mat cos_cache, sin_cache;
                
                if (rope_type == RoPE_Type::LongRoPE) {
                    generate_rope_embed_cache_LongRoPE(1, rope_head_dim, bctx.kv_cache[0].first.h, cos_cache, sin_cache, rope_theta, short_factor.data(), long_factor.data(), original_max_position_embeddings);
                } else {
                    generate_rope_embed_cache(1, rope_head_dim, bctx.kv_cache[0].first.h, cos_cache, sin_cache, rope_theta);
                }


                ncnn::Mat mask(bctx.kv_cache[0].first.h + 1, 1);
                mask.fill(0.f);

                ncnn::Mat decode_out;
                {
                    ncnn::Extractor ex = decoder_net->create_extractor();
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
                    ncnn::Extractor ex = proj_out_net->create_extractor();
                    ex.input("in0", decode_out);
                    ex.extract("out0", logits_mat);
                }

                std::vector<float> logits(vocab_size);
                memcpy(logits.data(), logits_mat.data, sizeof(float) * vocab_size);

                for (int t : beam.tokens) {
                    if (logits[t] < 0)
                        logits[t] *= cfg.repetition_penalty;
                    else
                        logits[t] /= cfg.repetition_penalty;
                }

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
                    
                    // SAVE PREVIOUS TOKEN & STATE
                    nb.prev_token = beam.ctx->cur_token;
                    nb.prev_in_tool_call = beam.in_tool_call;

                    nb.tokens = beam.tokens;
                    nb.tokens.insert(tok);
                    nb.score  = beam.score + std::log(p + 1e-9f);
                    nb.finished = (tok == eos);
                    nb.in_tool_call = beam.in_tool_call;
                    nb.tool_buffer = beam.tool_buffer;

                    if (tok == tool_call_id) {
                        nb.in_tool_call = true;
                        if (cfg.debug) fprintf(stderr, "[Debug] Beam enter tool_call mode.\n");
                    } else if (tok == tool_call_end_id && nb.in_tool_call) {
                        nb.in_tool_call = false;
                        if (cfg.debug) fprintf(stderr, "[Debug] Beam exit tool_call, handling call.\n");
                        handle_tool(nb.tool_buffer, nb.ctx);
                        nb.tool_buffer.clear();
                        nb.tokens.clear();
                        nb.tokens.insert(nb.ctx->cur_token);
                        nb.finished = (nb.ctx->cur_token == eos);
                        tool_completed = nb;
                        has_tool_completed = true;
                    } else if (nb.in_tool_call) {
                        nb.tool_buffer += bpe->decode({tok}, false);
                    }

                    candidates.push_back(std::move(nb));
                }
            }

            if (has_tool_completed) {
                if (cfg.debug) fprintf(stderr, "[Debug] Tool_call beam completed; promote to sole beam.\n");
                beams.clear();
                beams.push_back(tool_completed);
                auto& b = beams[0];
                if (!b.in_tool_call &&
                    b.ctx->cur_token != eos &&
                    b.ctx->cur_token != tool_call_id && b.ctx->cur_token != tool_call_end_id) {
                    callback(bpe->decode({b.ctx->cur_token}, false));
                }
                if (b.ctx->cur_token == eos || b.finished) break;
                continue;
            }

            // Select top beams
            std::sort(candidates.begin(), candidates.end(),
                    [](const Beam& a, const Beam& b) {
                        return a.score > b.score;
                    });

            int best_tool_idx = -1;
            for (int i = 0; i < (int)candidates.size(); ++i) {
                if (candidates[i].in_tool_call || !candidates[i].tool_buffer.empty()) {
                    best_tool_idx = i;
                    break;
                }
            }

            std::vector<Beam> next_beams;
            next_beams.reserve(cfg.beam_size);
            for (int i = 0; i < (int)candidates.size() && (int)next_beams.size() < cfg.beam_size; ++i) {
                next_beams.push_back(candidates[i]);
            }

            if (best_tool_idx >= 0 && best_tool_idx >= (int)next_beams.size()) {
                if ((int)next_beams.size() < cfg.beam_size) {
                    next_beams.push_back(candidates[best_tool_idx]);
                } else {
                    next_beams.back() = candidates[best_tool_idx];
                }
            }

            int promote_idx = -1;
            for (int i = 0; i < (int)next_beams.size(); ++i) {
                if (next_beams[i].in_tool_call || !next_beams[i].tool_buffer.empty()) {
                    promote_idx = i;
                    break;
                }
            }
            if (promote_idx > 0) {
                std::swap(next_beams[0], next_beams[promote_idx]);
                if (cfg.debug) fprintf(stderr, "[Debug] Promote tool_call beam from rank %d to rank 0.\n", promote_idx);
            }

            beams = std::move(next_beams);

            auto& best = beams[0];

            // Emit the PREVIOUS token of the best beam (the one that led to current state)
            // We do this BEFORE checking for EOS/break to ensure the token preceding EOS is emitted.
            maybe_emit_prev(best);

            if (best.ctx->cur_token == eos || best.finished) {
                break;
            }
            
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

        // ---------- 工具定义辅助模板 ----------
    template<typename T>
    static constexpr const char* json_type_name() {
        if constexpr (std::is_same_v<T, int> || std::is_same_v<T, long> || std::is_same_v<T, long long>) return "integer";
        else if constexpr (std::is_same_v<T, bool>) return "boolean";
        else if constexpr (std::is_floating_point_v<T>) return "number";
        else return "string";
    }

    // 根据 C++ 函数形参类型与名称自动生成 JSON schema。
    template<typename Ret, typename... Args>
    static nlohmann::json make_function_tool(
        const std::string& name,
        const std::string& description,
        const std::array<std::string, sizeof...(Args)>& arg_names)
    {
        // 运行时防御（MSVC 对参数 constexpr 要求较严格）
        assert(arg_names.size() == sizeof...(Args));
        nlohmann::json properties = nlohmann::json::object();
        size_t idx = 0;
        // 展开参数类型与名称
        ((
            properties[arg_names[idx]] = nlohmann::json{
                {"type", json_type_name<Args>()},
                {"description", ""}
            },
            ++idx
        ), ...);

        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", description},
                {"parameters", {
                    {"type", "object"},
                    {"properties", properties},
                    {"required", arg_names}
                }}
            }}
        };
    }

    std::shared_ptr<ncnn_llm_gpt_ctx> define_tools(
        const std::shared_ptr<ncnn_llm_gpt_ctx>& ctx,
        const std::vector<nlohmann::json>& tools,
        const std::string& system_prompt = "You are a helpful assistant.")
    {
        if (tool_call_id < 0 || tool_call_end_id < 0) {
            fprintf(stderr, "Model does not support tool calling.\n");
            return ctx;
        }

        this->tools = tools;
        std::string tool_prompt = apply_chat_template({
            {"system", system_prompt}
        }, tools, false, false);

        if (ctx) return prefill(tool_prompt, ctx);
        return prefill(tool_prompt);
    }

private:
    static int get_visiual_features(const cv::Mat& bgr, ncnn::Mat& image_embeds, int& num_patches_w, int& num_patches_h);

    // base model
    std::shared_ptr<ncnn::Net> decoder_net;
    std::shared_ptr<ncnn::Net> embed_net;
    std::shared_ptr<ncnn::Net> proj_out_net;
    // vision encoder
    std::shared_ptr<ncnn::Net> vision_embed_patch;
    std::shared_ptr<ncnn::Net> vision_encoder;
    // tokenizer
    std::shared_ptr<BpeTokenizer> bpe;

protected:

    std::string model_type;

    int bos = 0;
    int eos = 0;

    int tool_call_id = -1;
    int tool_call_end_id = -1;

    int attn_cnt = 32;

    // RoPE
    int rope_head_dim = 64;

    enum RoPE_Type {
        RoPE = 0,
        LongRoPE = 1
    } rope_type;
    float rope_theta = 100000.0f;

    // LongRoPE
    std::vector<float> short_factor;
    std::vector<float> long_factor;
    int original_max_position_embeddings = 0;

    std::vector<nlohmann::json> tools;

};
