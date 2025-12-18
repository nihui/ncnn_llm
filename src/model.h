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
#include <opencv2/imgproc/imgproc.hpp>

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
    int position_id = 0;
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
    dst->position_id = src->position_id;
    
    dst->kv_cache.resize(src->kv_cache.size());
    for (size_t i = 0; i < src->kv_cache.size(); ++i) {
        dst->kv_cache[i].first = src->kv_cache[i].first;
        dst->kv_cache[i].second = src->kv_cache[i].second;
    }
    return dst;
}

class ncnn_llm_gpt {
private:
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

    // Vision
    int image_pad_id = -1;
    int patch_size = 14;
    int patch_dim = 1280;
    int max_num_patches = 49152;
    int spatial_merge_size = 2;

    // VisionRoPE
    enum VisionRoPE_Type {
        mRoPE = 0
    } vision_rope_type;

    std::vector<int> mrope_section;

    std::vector<nlohmann::json> tools;


public:
    ncnn_llm_gpt(
            const std::string& model_path,
            bool use_vulkan = false
    ) {
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
                std::string proj_out_bin = model_path + "/" + config["params"]["proj_out_bin"].get<std::string>();

                decoder_net->load_param(decoder_param.c_str());
                decoder_net->load_model(decoder_bin.c_str());
                embed_net->load_param(embed_param.c_str());
                embed_net->load_model(embed_bin.c_str());
                proj_out_net->load_param(proj_out_param.c_str());
                proj_out_net->load_model(proj_out_bin.c_str());
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

            // vision
            {
                std::string vision_type = "close";
                auto vision_cfg = config["setting"]["vision"];
                if (config["setting"].contains("vision")) {
                    auto vision_cfg = config["setting"]["vision"];
                    vision_type = vision_cfg["type"].get<std::string>();
                }
                if (vision_type != "close") {
                    std::string vision_embed_patch_param = model_path + "/" + vision_cfg["vision_embed_patch_param"].get<std::string>();
                    std::string vision_embed_patch_bin = model_path + "/" + vision_cfg["vision_embed_patch_bin"].get<std::string>();
                    std::string vision_encoder_param = model_path + "/" + vision_cfg["vision_encoder_param"].get<std::string>();
                    std::string vision_encoder_bin = model_path + "/" + vision_cfg["vision_encoder_bin"].get<std::string>();

                    vision_embed_patch = std::make_shared<ncnn::Net>();
                    vision_encoder = std::make_shared<ncnn::Net>();

                    if (use_vulkan) {
                        vision_embed_patch->opt.use_vulkan_compute = true;
                        vision_encoder->opt.use_vulkan_compute = true;
                    }
                    vision_embed_patch->load_param(vision_embed_patch_param.c_str());
                    vision_embed_patch->load_model(vision_embed_patch_bin.c_str());

                    vision_encoder->load_param(vision_encoder_param.c_str());
                    vision_encoder->load_model(vision_encoder_bin.c_str());

                    std::string image_pad_token = "<|image_pad|>";
                    auto it = bpe->token_to_id().find(image_pad_token);
                    if (it != bpe->token_to_id().end()) {
                        image_pad_id = it->second;
                    }

                    patch_size = vision_cfg["patch_size"].get<int>();
                    patch_dim = vision_cfg["patch_dim"].get<int>();
                    max_num_patches = vision_cfg["max_num_patches"].get<int>();
                    spatial_merge_size = vision_cfg["spatial_merge_size"].get<int>();

                    // rope for vision
                    auto rope_cfg = vision_cfg["rope"];

                    if (rope_cfg["type"] == "mRoPE") {
                        vision_rope_type = VisionRoPE_Type::mRoPE;
                        mrope_section = rope_cfg["mrope_section"].get<std::vector<int>>();
                    }
                }
            }
        } catch (std::exception &e)
        {
            throw std::runtime_error(std::string("ncnn_llm_gpt load model failed: ") + e.what());
        }
    }
    
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
        ctx->position_id = (int)token_ids.size() + 1;

        return ctx;
    }

    std::shared_ptr<ncnn_llm_gpt_ctx> prefill(const std::string& input_text, const cv::Mat& bgr, const std::shared_ptr<ncnn_llm_gpt_ctx> ctx) const
    {
        std::shared_ptr<ncnn_llm_gpt_ctx> new_ctx = clone_ctx(ctx);

        ncnn::Mat image_embeds;
        int num_patches_w = 0;
        int num_patches_h = 0;
        get_visiual_features(bgr, image_embeds, num_patches_w, num_patches_h);

        const int image_embeds_size = image_embeds.h;

        auto token_ids = bpe->encode(input_text, false, false);
        int last_token_id = token_ids.back();
        token_ids.pop_back();

        ncnn::Mat input_ids_mat = ncnn::Mat((int)token_ids.size(), 1, (void*)token_ids.data()).clone();
        ncnn::Mat token_embed;
        {
            ncnn::Extractor ex = embed_net->create_extractor();
            ex.input("in0", input_ids_mat);
            ex.extract("out0", token_embed);
        }

        // inject image_embeds
        int image_pad_index = -1;
        inject_image_embeds(token_ids, token_embed, image_pad_index, image_embeds);

        ncnn::Mat cos_cache;
        ncnn::Mat sin_cache;
        if (image_embeds.empty())
        {
            generate_rope_embed_cache(token_ids.size(), rope_head_dim, new_ctx->position_id, cos_cache, sin_cache, rope_theta);
            new_ctx->position_id += token_ids.size();
        }
        else
        {
            generate_rope_embed_cache_vision_mrope(token_ids.size(), rope_head_dim, new_ctx->position_id, image_pad_index, image_embeds_size, num_patches_w, cos_cache, sin_cache, rope_theta);
            const int merge_size = 2;
            new_ctx->position_id += token_ids.size() - image_embeds_size + (num_patches_w / merge_size);
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

        generate_rope_embed_cache(1, rope_head_dim, new_ctx->position_id, last_cos_cache, last_sin_cache, rope_theta);
        new_ctx->position_id += 1;

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
    
    std::shared_ptr<ncnn_llm_gpt_ctx> prefill(const std::string& input_text, const std::shared_ptr<ncnn_llm_gpt_ctx> ctx) const
    {
        std::shared_ptr<ncnn_llm_gpt_ctx> new_ctx = clone_ctx(ctx);

        auto token_ids = bpe->encode(input_text, false, false);
        int last_token_id = token_ids.back();
        token_ids.pop_back();

        ncnn::Mat cos_cache;
        ncnn::Mat sin_cache;

        int current_pos = new_ctx->position_id;

        if (rope_type == RoPE_Type::LongRoPE) {
            generate_rope_embed_cache_LongRoPE(token_ids.size(), rope_head_dim, current_pos, cos_cache, sin_cache, rope_theta, short_factor.data(), long_factor.data(), original_max_position_embeddings);
        } else {
            generate_rope_embed_cache(token_ids.size(), rope_head_dim, current_pos, cos_cache, sin_cache, rope_theta);
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

        int last_token_pos = current_pos + (int)token_ids.size();

        if (rope_type == RoPE_Type::LongRoPE) {
            generate_rope_embed_cache_LongRoPE(1, rope_head_dim, last_token_pos, last_cos_cache, last_sin_cache, rope_theta, short_factor.data(), long_factor.data(), original_max_position_embeddings);
        } else {
            generate_rope_embed_cache(1, rope_head_dim, last_token_pos, last_cos_cache, last_sin_cache, rope_theta);
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
        new_ctx->position_id += ((int)token_ids.size() + 1);

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
                    generate_rope_embed_cache_LongRoPE(1, rope_head_dim, ctx->position_id, cos_cache, sin_cache, rope_theta, short_factor.data(), long_factor.data(), original_max_position_embeddings);
                } else {
                    generate_rope_embed_cache(1, rope_head_dim, ctx->position_id, cos_cache, sin_cache, rope_theta);
                }
                
                ctx->position_id++;

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
                    generate_rope_embed_cache_LongRoPE(1, rope_head_dim, bctx.position_id, cos_cache, sin_cache, rope_theta, short_factor.data(), long_factor.data(), original_max_position_embeddings);
                } else {
                    generate_rope_embed_cache(1, rope_head_dim, bctx.position_id, cos_cache, sin_cache, rope_theta);
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
                    
                    nb.ctx->position_id++;
                    
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
    static int get_scaled_image_size(float scale, int size, int patch_size) {
        // Python: patch_size = patch_size * 2
        int effective_patch_size = patch_size * 2;

        // Python: scaled_size = size * scale
        float scaled_size_f = (float)size * scale;

        // Python: math.ceil(scaled_size / patch_size) * patch_size
        // 这里使用 ceil 函数并转换类型
        int scaled_size = (int)(std::ceil(scaled_size_f / (float)effective_patch_size) * effective_patch_size);

        // Python: max(patch_size, scaled_size)
        // 这里的 patch_size 指的是已经乘过 2 的 effective_patch_size
        scaled_size = std::max(effective_patch_size, scaled_size);

        return scaled_size;
    }


    static void get_image_size_for_patches(int image_height, int image_width, int patch_size, int max_num_patches, int& target_height, int& target_width) {
        float scale = 1.0f;

        // Binary search (linear search downwards actually) for optimal scale
        while (true) {
            target_height = get_scaled_image_size(scale, image_height, patch_size);
            target_width = get_scaled_image_size(scale, image_width, patch_size);

            // 计算 patch 数量
            // 注意：这里除的是原始的 patch_size，不是乘过2的
            // num_patches = (target_height / patch_size) * (target_width / patch_size)
            // 或者是 (target_height * target_width) / (patch_size * patch_size)
            long long num_patches = ((long long)target_height / patch_size) * ((long long)target_width / patch_size);

            if (num_patches > max_num_patches) {
                scale -= 0.02f;
            } else {
                break;
            }
        }
    }

    /**
    * @brief 将 BGR 图片转换为 Vision Transformer 输入所需的 Patch Embeddings
    *
    * 逻辑：
    * 1. 按照 16x16 切分图片。
    * 2. 如果图片边缘不足 16，则视为补 0（黑色），归一化后值为 -1.0。
    * 3. 数据格式：(num_patches, 768)。每一行存放一个 Patch 的数据。
    * 4. Patch 内部排列：先放 16x16 的 B 通道，再放 G，再放 R (Planar 格式)。
    *
    * @param bgr 输入的 cv::Mat (BGR 格式, uint8)
    * @return ncnn::Mat 输出矩阵, 维度 [w=768, h=num_patches]
    */
    static ncnn::Mat bgr_to_pixel_values(const cv::Mat& bgr)
    {
        const int patch_size = 14;

        const float image_mean[3] = {0.48145466, 0.4578275, 0.40821073};
        const float image_std[3] = {0.26862954, 0.26130258, 0.27577711};

        int img_h = bgr.rows;
        int img_w = bgr.cols;

        // 1. 计算 grid 尺寸 (向上取整)
        int num_patches_h = (img_h + patch_size - 1) / patch_size;
        int num_patches_w = (img_w + patch_size - 1) / patch_size;
        int num_patches = num_patches_h * num_patches_w;

        // 2. 创建输出 Mat
        // 维度: w = 16*16*3 = 768, h = num_patches
        // NCNN Mat(w, h) 构造函数
        int embed_dim = patch_size * patch_size * 3;
        ncnn::Mat pixel_values(embed_dim, num_patches);

        // 3. 遍历每个 Patch
        // 使用 OpenMP 加速 (如果 NCNN 开启了 OpenMP，这里也可以手动加)
        // #pragma omp parallel for
        for (int p = 0; p < num_patches; p++)
        {
            // 计算当前 Patch 在 Grid 中的坐标
            int ph = p / num_patches_w;
            int pw = p % num_patches_w;

            // 计算当前 Patch 在原图中的像素起始坐标
            int start_y = ph * patch_size;
            int start_x = pw * patch_size;

            // 获取输出 Mat 当前行的指针
            float* out_ptr = pixel_values.row(p);

            // 指针偏移量：因为是 Planar 格式，B/G/R 分开存
            // [BBBB... GGGG... RRRR...]
            float* ptr_r = out_ptr;
            float* ptr_g = out_ptr + patch_size * patch_size;
            float* ptr_b = out_ptr + patch_size * patch_size * 2;

            // 遍历 Patch 内部 16x16 像素
            for (int y = 0; y < patch_size; y++) {
                // 预计算当前行的图像指针（如果在图像范围内）
                const uchar* img_row_ptr = NULL;
                int cur_img_y = start_y + y;
                if (cur_img_y < img_h) {
                    img_row_ptr = bgr.ptr<uchar>(cur_img_y);
                }

                for (int x = 0; x < patch_size; x++) {
                    int cur_img_x = start_x + x;

                    // 检查边界
                    if (img_row_ptr && cur_img_x < img_w) {
                        // 读取 BGR 像素
                        // cv::Mat 默认是 BGR 顺序: [b, g, r]
                        const uchar* pixel = img_row_ptr + cur_img_x * 3;
                        uchar b = pixel[0];
                        uchar g = pixel[1];
                        uchar r = pixel[2];

                        // 归一化并写入
                        *ptr_r++ = (r / 255.f - image_mean[0]) / image_std[0];
                        *ptr_g++ = (g / 255.f - image_mean[1]) / image_std[1];
                        *ptr_b++ = (b / 255.f - image_mean[2]) / image_std[2];
                    } else {
                        // Padding (补 0)
                        float pad_val = 0.0f;
                        *ptr_r++ = pad_val;
                        *ptr_g++ = pad_val;
                        *ptr_b++ = pad_val;
                    }
                }
            }
        }

        return pixel_values;
    }

    static ncnn::Mat reorder_patches_for_merge(const ncnn::Mat& pixel_values, int h_patches, int w_patches, int merge_size = 2) {
        int num_patches = pixel_values.h;
        int feature_dim = pixel_values.w; // 768

        // 检查输入维度是否合法
        if (num_patches != h_patches * w_patches) {
            fprintf(stderr, "Error: h_patches * w_patches (%d * %d) != pixel_values.h (%d)\n", h_patches, w_patches, num_patches);
            return ncnn::Mat();
        }

        // 计算 Block 的网格尺寸
        // 例如 h=22, w=30, merge=2 -> grid_h=11, grid_w=15
        int grid_h = h_patches / merge_size;
        int grid_w = w_patches / merge_size;

        // 创建输出 Mat
        ncnn::Mat reordered_pixel_values(feature_dim, num_patches, (size_t)4u);

        int new_row_idx = 0;

        // 1. 遍历 Block 行
        for (int gh = 0; gh < grid_h; gh++) {
            // 2. 遍历 Block 列
            for (int gw = 0; gw < grid_w; gw++) {

                // 3. 遍历 Block 内部 (2x2)
                // 顺序通常是：Block内第一行，Block内第二行...
                for (int mh = 0; mh < merge_size; mh++) {
                    for (int mw = 0; mw < merge_size; mw++) {

                        // 计算原始 Patch 的坐标
                        int original_h = gh * merge_size + mh;
                        int original_w = gw * merge_size + mw;

                        // 计算原始 Patch 在 pixel_values 中的行索引 (Row-Major)
                        int original_row_idx = original_h * w_patches + original_w;

                        // 获取源指针和目标指针
                        const float* src_ptr = pixel_values.row(original_row_idx);
                        float* dst_ptr = reordered_pixel_values.row(new_row_idx);

                        // 内存复制 (整行复制，速度快)
                        memcpy(dst_ptr, src_ptr, feature_dim * sizeof(float));

                        new_row_idx++;
                    }
                }
            }
        }

        // 如果原始图片尺寸不能被 merge_size 整除（例如 h=23），上述循环会忽略边缘。
        // 如果你的模型逻辑包含 Padding，请确保 h_patches/w_patches 已经是 Padding 后的尺寸。
        // 如果需要处理边缘剩余 Patch，需要额外的逻辑，但通常 Vision Transformer 会先 Pad 到整除。

        return reordered_pixel_values;
    }

    static void get_window_index(int num_patches_w, int num_patches_h, std::vector<int>& window_index, std::vector<int>& cu_window_seqlens)
    {
        // 2. 常量定义
        const int spatial_merge_size = 2;
        const int patch_size = 14;
        const int vit_merger_window_size = 4;

        // 3. 计算 Merged Grid 尺寸 (11, 15)
        int llm_grid_h = num_patches_h / spatial_merge_size;
        int llm_grid_w = num_patches_w / spatial_merge_size;

        // 4. 计算窗口数量 (向上取整)
        // (11 + 7) / 8 = 2
        int num_windows_h = (llm_grid_h + vit_merger_window_size - 1) / vit_merger_window_size;
        int num_windows_w = (llm_grid_w + vit_merger_window_size - 1) / vit_merger_window_size;

        // 5. 初始化输出
        window_index.clear();
        window_index.reserve(llm_grid_h * llm_grid_w); // 预分配内存

        cu_window_seqlens.clear();
        cu_window_seqlens.push_back(0); // 第一个是 0

        int current_cu_len = 0;

        // 6. 核心双重循环：遍历窗口 (Row-Major of Windows)
        for (int nh = 0; nh < num_windows_h; ++nh) {
            for (int nw = 0; nw < num_windows_w; ++nw) {

                // 计算当前窗口在 merged grid 中的起始和结束坐标
                int h_start = nh * vit_merger_window_size;
                int w_start = nw * vit_merger_window_size;

                // 裁剪边界 (相当于 Python 代码中去除 -100 的逻辑)
                int h_end = std::min(h_start + vit_merger_window_size, llm_grid_h);
                int w_end = std::min(w_start + vit_merger_window_size, llm_grid_w);

                int valid_h = h_end - h_start;
                int valid_w = w_end - w_start;

                if (valid_h <= 0 || valid_w <= 0) continue;

                // 6.1 生成 Window Index
                // 遍历窗口内的有效像素
                for (int r = h_start; r < h_end; ++r) {
                    for (int c = w_start; c < w_end; ++c) {
                        // 原始 Grid 中的线性索引 (Row-Major)
                        int original_idx = r * llm_grid_w + c;
                        window_index.push_back(original_idx);
                    }
                }

                // 6.2 生成 cu_window_seqlens
                // 注意：cu_seqlens 通常统计的是原始 token 数 (pixel 数)
                // Python: [0, 256, 480...]
                // 每一个 merged grid 点对应 spatial_merge_size^2 (2*2=4) 个原始像素
                int tokens_in_this_window = valid_h * valid_w * (spatial_merge_size * spatial_merge_size);

                current_cu_len += tokens_in_this_window;
                cu_window_seqlens.push_back(current_cu_len);
            }
        }
    }

    // 辅助函数：计算 VisionRope 的 inv_freq
    // 对应 Python: 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    static std::vector<float> compute_inv_freq(int dim, float theta = 10000.0f) {
        // 根据推导，如果最终 emb_cos 维度是 72，且由 [rot, rot] 拼接，则 rot 为 36。
        // rot 由 [h_emb, w_emb] 拼接，则 h_emb 为 18。
        // VisionRope 内部产生的向量长度为 18。
        // Python arange(0, dim, 2) 产生 18 个数，说明 dim = 36。

        // 注意：这里的 dim 参数对应 Python __init__ 中的 dim
        int num_freqs = dim / 2;
        std::vector<float> inv_freq(num_freqs);

        for (int i = 0; i < num_freqs; i++) {
            // arange(0, dim, 2) -> 0, 2, 4, ...
            float exp_val = (float)(i * 2) / (float)dim;
            inv_freq[i] = 1.0f / std::pow(theta, exp_val);
        }
        return inv_freq;
    }


    // 核心函数：生成 Cos 和 Sin Embeddings
    static void generate_rope_embeds(int num_patches_w, int num_patches_h,
                            ncnn::Mat& emb_cos,
                            ncnn::Mat& emb_sin,
                            int rope_dim = 36) { // rope_dim 对应 Python VisionRope 的 dim 参数

        int spatial_merge_size = 2;
        int seq_len = num_patches_w * num_patches_h;

        // 1. 预计算 inv_freq
        std::vector<float> inv_freq = compute_inv_freq(rope_dim);
        int half_dim = inv_freq.size(); // 18
        // 最终输出维度：(H_emb + W_emb) * 2 = (18 + 18) * 2 = 72
        int output_dim = half_dim * 4;

        // 初始化输出 Mat [w=72, h=seq_len]
        emb_cos.create(output_dim, seq_len, sizeof(float));
        emb_sin.create(output_dim, seq_len, sizeof(float));

        // 2. 模拟 Grid 遍历逻辑 (Reshape -> Permute -> Flatten)
        // Python Logic:
        // grid.reshape(h//2, 2, w//2, 2).permute(0, 2, 1, 3)
        // 意味着遍历顺序为:
        // Outer Loop: Block Row (0..h/2)
        //   Inner Loop: Block Col (0..w/2)
        //     Block Inner Row (0..2)
        //       Block Inner Col (0..2)

        int idx = 0; // 全局像素索引 0..seq_len-1

        int grid_h = num_patches_h / spatial_merge_size;
        int grid_w = num_patches_w / spatial_merge_size;

        for (int gh = 0; gh < grid_h; gh++) {
            for (int gw = 0; gw < grid_w; gw++) {
                // 在每个 2x2 的块内部遍历
                for (int bi = 0; bi < spatial_merge_size; bi++) {
                    for (int bj = 0; bj < spatial_merge_size; bj++) {

                        // 计算原始坐标 (h, w)
                        int current_h = gh * spatial_merge_size + bi;
                        int current_w = gw * spatial_merge_size + bj;

                        // 获取当前行的输出指针
                        float* cos_ptr = emb_cos.row(idx);
                        float* sin_ptr = emb_sin.row(idx);

                        // 3. 计算 Embeddings 并填充
                        // Python: cat(rotary_pos_emb, rotary_pos_emb)
                        // rotary_pos_emb = stack([hpos, wpos]).flatten()
                        // 结构: [H_freqs, W_freqs, H_freqs, W_freqs]

                        // 填充第一段 H_freqs (0 ~ 17) 和 第三段 (36 ~ 53)
                        for (int k = 0; k < half_dim; k++) {
                            float freq = inv_freq[k];
                            float angle_h = (float)current_h * freq;
                            float angle_w = (float)current_w * freq;

                            // 计算索引
                            // H part 1
                            int idx_h1 = k;
                            // W part 1
                            int idx_w1 = half_dim + k;
                            // H part 2 (Duplicate)
                            int idx_h2 = 2 * half_dim + k;
                            // W part 2 (Duplicate)
                            int idx_w2 = 3 * half_dim + k;

                            // 填入角度 (稍后统一做 cos/sin 也可以，这里直接算)
                            // 注意 Python 代码是先 cat 再 cos/sin

                            // H components
                            cos_ptr[idx_h1] = std::cos(angle_h);
                            sin_ptr[idx_h1] = std::sin(angle_h);
                            cos_ptr[idx_h2] = cos_ptr[idx_h1];
                            sin_ptr[idx_h2] = sin_ptr[idx_h1];

                            // W components
                            cos_ptr[idx_w1] = std::cos(angle_w);
                            sin_ptr[idx_w1] = std::sin(angle_w);
                            cos_ptr[idx_w2] = cos_ptr[idx_w1];
                            sin_ptr[idx_w2] = sin_ptr[idx_w1];
                        }

                        idx++;
                    }
                }
            }
        }
    }

    int get_visiual_features(const cv::Mat& bgr, ncnn::Mat& image_embeds, int& num_patches_w, int& num_patches_h) const
    {
        if (bgr.empty())
        {
            image_embeds.release();
            num_patches_w = 0;
            num_patches_h = 0;
            return 0;
        }

        const int patch_size = 14;
        const int max_num_patches = 49152;

        const int img_w = bgr.cols;
        const int img_h = bgr.rows;
        fprintf(stderr, "image %d %d\n", img_w, img_h);

        // get_image_size_for_patches
        int target_w;
        int target_h;
        get_image_size_for_patches(img_h, img_w, patch_size, max_num_patches, target_h, target_w);
        fprintf(stderr, "target %d %d\n", target_w, target_h);

        // resize
        cv::Mat bgr_resized;
        {
            // FIXME this is not pil image resize
            cv::resize(bgr, bgr_resized, cv::Size(target_w, target_h), 0, 0, cv::INTER_AREA);
        }

        num_patches_w = (target_w + patch_size - 1) / patch_size;
        num_patches_h = (target_h + patch_size - 1) / patch_size;
        fprintf(stderr, "patches %d %d\n", num_patches_w, num_patches_h);

        const int seq_len = num_patches_w * num_patches_h;
        fprintf(stderr, "seq_len %d\n", seq_len);

        ncnn::Mat pixel_values = bgr_to_pixel_values(bgr_resized);

        pixel_values = reorder_patches_for_merge(pixel_values, num_patches_h, num_patches_w, 2);

        pixel_values = pixel_values.reshape(14 * 14, 1, 3, seq_len);
        // dup depth
        {
            // TODO we can modify embed_patch weights to eliminate this depth duplication
            ncnn::Mat tmp(14 * 14, 2, 3, seq_len);
            for (int i = 0; i < seq_len; i++)
            {
                memcpy(tmp.channel(i).depth(0).row(0), pixel_values.channel(i).depth(0).row(0), 14 * 14 * sizeof(float));
                memcpy(tmp.channel(i).depth(0).row(1), pixel_values.channel(i).depth(0).row(0), 14 * 14 * sizeof(float));
                memcpy(tmp.channel(i).depth(1).row(0), pixel_values.channel(i).depth(1).row(0), 14 * 14 * sizeof(float));
                memcpy(tmp.channel(i).depth(1).row(1), pixel_values.channel(i).depth(1).row(0), 14 * 14 * sizeof(float));
                memcpy(tmp.channel(i).depth(2).row(0), pixel_values.channel(i).depth(2).row(0), 14 * 14 * sizeof(float));
                memcpy(tmp.channel(i).depth(2).row(1), pixel_values.channel(i).depth(2).row(0), 14 * 14 * sizeof(float));
            }

            pixel_values = tmp.reshape(14 * 14 * 2 * 3, seq_len);
        }

        // 2. Get Window Index
        std::vector<int> window_index;
        std::vector<int> cu_window_seqlens;
        get_window_index(num_patches_w, num_patches_h, window_index, cu_window_seqlens);

        // 3. First Network: vision_embed
        ncnn::Mat patch_embeds(1280, seq_len);
        for (int i = 0; i < seq_len; i++)
        {
            ncnn::Mat patch = pixel_values.row_range(i, 1).reshape(14, 14, 2, 3);
            ncnn::Mat patch_embed;

            ncnn::Extractor ex = vision_embed_patch->create_extractor();
            ex.input("in0", patch);
            ex.extract("out0", patch_embed);

            memcpy(patch_embeds.row(i), patch_embed.reshape(1280), 1280 * sizeof(float));
        }
        // print_mat(patch_embeds);

        // patch_embeds shape is expected to be [w=1152, h=660] (ncnn usually is h, w for 2D) or [w=1152, h=seq_len]
        // Python output shape: [seq_len, 1152]. NCNN Mat: w=1152, h=660.
        // int seq_len = patch_embeds.h;

        // 4. Emb Cos/Sin
        ncnn::Mat emb_cos, emb_sin;
        generate_rope_embeds(num_patches_w, num_patches_h, emb_cos, emb_sin, 40);

        // 5. Apply reordering based on window_index
        // Logic: patch_embeds.reshape(seq_len // 4, 4, 1152)[window_index, :, :]...
        // Essentially reordering groups of 4 rows.

        ncnn::Mat patch_embeds_reordered(patch_embeds.w, seq_len, sizeof(float));
        ncnn::Mat emb_cos_reordered(emb_cos.w, seq_len, sizeof(float));
        ncnn::Mat emb_sin_reordered(emb_sin.w, seq_len, sizeof(float));

        int group_size = 4;

        // Perform gathering
        // window_index contains indices of groups (size 165)
        for (int i = 0; i < window_index.size(); i++) {
            int src_group_idx = window_index[i];

            // Copy 4 rows (group_size)
            for (int k = 0; k < group_size; k++) {
                int src_row = src_group_idx * group_size + k;
                int dst_row = i * group_size + k;

                // Copy Patch Embeds
                const float* src_ptr = patch_embeds.row(src_row);
                float* dst_ptr = patch_embeds_reordered.row(dst_row);
                memcpy(dst_ptr, src_ptr, patch_embeds.w * sizeof(float));

                // Copy Cos
                const float* src_cos = emb_cos.row(src_row);
                float* dst_cos = emb_cos_reordered.row(dst_row);
                memcpy(dst_cos, src_cos, emb_cos.w * sizeof(float));

                // Copy Sin
                const float* src_sin = emb_sin.row(src_row);
                float* dst_sin = emb_sin_reordered.row(dst_row);
                memcpy(dst_sin, src_sin, emb_sin.w * sizeof(float));
            }
        }
        // print_mat(patch_embeds_reordered);
        // print_mat(emb_cos_reordered);
        // print_mat(emb_sin_reordered);

        // 6. Create Attention Mask
        std::vector<int> cu_seqlens = cu_window_seqlens;
        ncnn::Mat attention_mask(seq_len, seq_len);

        // Fill with -inf (using -FLT_MAX or a very small number like -1e9)
        float min_val = -1e9f;
        attention_mask.fill(min_val);

        for (size_t i = 1; i < cu_seqlens.size(); i++) {
            int start = cu_seqlens[i-1];
            int end = cu_seqlens[i];
            // Set block [start:end, start:end] to 0
            for (int r = start; r < end; r++) {
                float* row_ptr = attention_mask.row(r);
                for (int c = start; c < end; c++) {
                    row_ptr[c] = 0.f;
                }
            }
        }

        // 7. Second Network: vision_encoder
        // Python Inputs:
        // in0: patch_embeds [1, seq_len, 1152] -> ncnn: 3 dim [1152, seq_len, 1] usually handled as 2D [1152, seq_len]
        // in1: cos [1, 1, seq_len, 72] -> ncnn: [72, seq_len, 1]
        // in3: mask [1, seq_len, seq_len] -> ncnn: [seq_len, seq_len, 1]

        // Important: NCNN PNNX conversion usually keeps the dimensions correctly if using Mat(w, h, c) or Mat(w, h).
        // The previous `reordered` Mats are [w=dim, h=seq_len], which matches typical Linear layer inputs in NCNN.

        {
            ncnn::Extractor ex = vision_encoder->create_extractor();
            ex.input("in0", patch_embeds_reordered);
            ex.input("in1", emb_cos_reordered);
            ex.input("in2", emb_sin_reordered);

            // Attention mask is 3D in PyTorch [1, seq, seq], here we pass as 2D Mat,
            // ensure the NCNN graph expects shape [w=seq, h=seq]
            ex.input("in3", attention_mask);

            // ex.extract("178", image_embeds);
            ex.extract("out0", image_embeds);
        }
        // image_embeds out: [w=1152, h=seq_len]

        // print_mat(image_embeds);

        // 8. Reverse Indices (Reorder back)
        ncnn::Mat image_embeds_restored(image_embeds.w, image_embeds.h);
        for (int i = 0; i < window_index.size(); i++)
        {
            int dest_group_idx = window_index[i]; // The original location
            int src_group_idx = i;                // The current location (shuffled)

            const float* src_ptr = image_embeds.row(dest_group_idx);
            float* dst_ptr = image_embeds_restored.row(src_group_idx);
            memcpy(dst_ptr, src_ptr, image_embeds.w * sizeof(float));
        }

        image_embeds = image_embeds_restored;

        return 0;
    }
};