#include "ncnn_llm_gpt.h"

// Static RNG
static std::mt19937 rng(std::random_device{}());

// Helper functions for sampling
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
    size_t cutoff = v.size();
    for (size_t i = 0; i < v.size(); ++i) {
        cum += v[i].first;
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

static std::shared_ptr<ncnn_llm_gpt_ctx> clone_ctx(const std::shared_ptr<ncnn_llm_gpt_ctx>& src) {
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

// Class Implementation

ncnn_llm_gpt::ncnn_llm_gpt(const std::string& model_path, bool use_vulkan) {
    try {
        json config;
        {
            std::ifstream ifs(model_path + "/model.json");
            ifs >> config;
        }
        
        // Load base model
        decoder_net = std::make_shared<ncnn::Net>();
        embed_net = std::make_shared<ncnn::Net>();
        proj_out_net = std::make_shared<ncnn::Net>();

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

        // Load tokenizer
        std::string type = "bpe";
        if (config["tokenizer"].contains("type")) {
            type = config["tokenizer"]["type"].get<std::string>();
        }
        std::string vocab_file = model_path + "/" + config["tokenizer"]["vocab_file"].get<std::string>();
        std::string merges_file = model_path + "/" + config["tokenizer"]["merges_file"].get<std::string>();

        bpe = std::make_shared<BpeTokenizer>(BpeTokenizer::LoadFromFiles(
            vocab_file, merges_file, SpecialTokensConfig{}, false, true, type == "bbpe"
        ));

        std::vector<std::string> additional_special_tokens = config["tokenizer"]["additional_special_tokens"].get<std::vector<std::string>>();
        for (const auto& token : additional_special_tokens) {
            bpe->AddAdditionalSpecialToken(token);
        }

        auto eos_token = config["tokenizer"]["eos"].get<std::string>();
        eos = (eos_token != "") ? bpe->token_to_id().at(eos_token) : -1;

        auto bos_token = config["tokenizer"]["bos"].get<std::string>();
        bos = (bos_token != "") ? bpe->token_to_id().at(bos_token) : -1;

        // Model settings
        if (config["setting"].contains("attn_cnt")) {
            attn_cnt = config["setting"]["attn_cnt"].get<int>();
        }

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
            } else if (rope_cfg["type"] == "NTKRoPE") {
                // rope_scaling
                rope_type = RoPE_Type::NTK_RoPE;
            } else if (rope_cfg["type"] == "YaRNRoPE") {
                rope_type = RoPE_Type::YARN_RoPE;
            } else if (rope_cfg["type"] == "HYRoPE") {
                rope_type = RoPE_Type::HY_RoPE;
            }

            if (rope_cfg.contains("rope_scaling"))
            {
                ntk_scaling_params.alpha = rope_cfg["rope_scaling"]["alpha"].get<float>();
                ntk_scaling_params.beta_fast = rope_cfg["rope_scaling"]["beta_fast"].get<float>();
                ntk_scaling_params.beta_slow = rope_cfg["rope_scaling"]["beta_slow"].get<float>();
                ntk_scaling_params.factor = rope_cfg["rope_scaling"]["factor"].get<float>();
                ntk_scaling_params.mscale = rope_cfg["rope_scaling"]["mscale"].get<float>();
                ntk_scaling_params.mscale_all_dim = rope_cfg["rope_scaling"]["mscale_all_dim"].get<float>();
            }

            rope_theta = rope_cfg["rope_theta"].get<float>();
        }

        if (config["setting"].contains("functions")) {
            auto func_cfg = config["setting"]["functions"];
            if (func_cfg["type"].get<std::string>() == "tool_call") {
                if (func_cfg.contains("tool_call_id")) {
                    tool_call_id = bpe->token_to_id().at(func_cfg["tool_call_id"].get<std::string>());
                }
                if (func_cfg.contains("tool_call_end_id")) {
                    tool_call_end_id = bpe->token_to_id().at(func_cfg["tool_call_end_id"].get<std::string>());
                }
            }
        }

        // Vision settings
        std::string vision_type = "close";
        if (config["setting"].contains("vision")) {
            auto vision_cfg = config["setting"]["vision"];
            vision_type = vision_cfg["type"].get<std::string>();
            
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

                auto it = bpe->token_to_id().find("<|image_pad|>");
                if (it != bpe->token_to_id().end()) {
                    image_pad_id = it->second;
                }

                // Load vision config
                patch_size = vision_cfg["patch_size"].get<int>();
                patch_dim = vision_cfg["patch_dim"].get<int>();
                max_num_patches = vision_cfg["max_num_patches"].get<int>();
                spatial_merge_size = vision_cfg["spatial_merge_size"].get<int>();

                auto rope_cfg = vision_cfg["rope"];
                if (rope_cfg["type"] == "mRoPE") {
                    vision_rope_type = VisionRoPE_Type::mRoPE;
                    mrope_section = rope_cfg["mrope_section"].get<std::vector<int>>();
                }
            }
        }
    } catch (std::exception &e) {
        throw std::runtime_error(std::string("ncnn_llm_gpt load model failed: ") + e.what());
    }
}

std::shared_ptr<ncnn_llm_gpt_ctx> ncnn_llm_gpt::prefill(const std::string& input_text) const {
    auto token_ids = bpe->encode(input_text, false, false);
    if (bos >= 0) token_ids.insert(token_ids.begin(), bos);

    int last_token_id = token_ids.back();
    token_ids.pop_back();

    ncnn::Mat cos_cache, sin_cache;
    if (rope_type == RoPE_Type::LongRoPE) {
        generate_rope_embed_cache_LongRoPE(token_ids.size(), rope_head_dim, 0, cos_cache, sin_cache, rope_theta, short_factor.data(), long_factor.data(), original_max_position_embeddings);
    } else if (rope_type == RoPE_Type::NTK_RoPE) {
        generate_ntk_rope_embed_cache(token_ids.size(), rope_head_dim, 0, cos_cache, sin_cache, rope_theta, ntk_scaling_params);
    } else if (rope_type == RoPE_Type::YARN_RoPE) {
        generate_yarn_rope_embed_cache(token_ids.size(), rope_head_dim, 0, cos_cache, sin_cache, rope_theta, ntk_scaling_params);
    } else if (rope_type == RoPE_Type::HY_RoPE) {
        generate_hunyuan_rope_embed_cache(token_ids.size(), rope_head_dim, 0, cos_cache, sin_cache, rope_theta, ntk_scaling_params);
    } 
    else
    {
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
    for (int i = 0; i < (int)token_ids.size(); i++) {
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

    // Handle last token
    ncnn::Mat last_token_mat = ncnn::Mat(1, 1, (void*)&last_token_id).clone();
    ncnn::Mat last_token_embed;
    {
        ncnn::Extractor ex = embed_net->create_extractor();
        ex.input("in0", last_token_mat);
        ex.extract("out0", last_token_embed);
    }
    
    ncnn::Mat last_cos_cache, last_sin_cache;
    if (rope_type == RoPE_Type::LongRoPE) {
        generate_rope_embed_cache_LongRoPE(1, rope_head_dim, (int)token_ids.size(), last_cos_cache, last_sin_cache, rope_theta, short_factor.data(), long_factor.data(), original_max_position_embeddings);
    } else if (rope_type == RoPE_Type::NTK_RoPE) {
        generate_ntk_rope_embed_cache(1, rope_head_dim, (int)token_ids.size(), last_cos_cache, last_sin_cache, rope_theta, ntk_scaling_params);
    } else if (rope_type == RoPE_Type::YARN_RoPE) {
        generate_yarn_rope_embed_cache(1, rope_head_dim, (int)token_ids.size(), last_cos_cache, last_sin_cache, rope_theta, ntk_scaling_params);
    } else if (rope_type == RoPE_Type::HY_RoPE) {
        generate_hunyuan_rope_embed_cache(1, rope_head_dim, (int)token_ids.size(), last_cos_cache, last_sin_cache, rope_theta, ntk_scaling_params);
    }
    else {
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
        float max_val = p[0];
        for (int i = 1; i < logits.w; ++i) {
            if (p[i] > max_val) {
                max_val = p[i];
                next_token_id = i;
            }
        }
    }

    auto ctx = std::make_shared<ncnn_llm_gpt_ctx>();
    ctx->kv_cache = std::move(kv_cache);
    ctx->cur_token = next_token_id;
    ctx->position_id = (int)token_ids.size() + 1;
    return ctx;
}

std::shared_ptr<ncnn_llm_gpt_ctx> ncnn_llm_gpt::prefill(const std::string& input_text, const cv::Mat& bgr, const std::shared_ptr<ncnn_llm_gpt_ctx> ctx) const {
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

    int image_pad_index = -1;
    inject_image_embeds(token_ids, token_embed, image_pad_index, image_pad_id, image_embeds);

    ncnn::Mat cos_cache, sin_cache;
    if (image_embeds.empty()) {
        generate_rope_embed_cache(token_ids.size(), rope_head_dim, new_ctx->position_id, cos_cache, sin_cache, rope_theta);
        new_ctx->position_id += token_ids.size();
    } else {
        generate_rope_embed_cache_vision_mrope(token_ids.size(), rope_head_dim, new_ctx->position_id, image_pad_index, image_embeds_size, num_patches_w, spatial_merge_size, mrope_section, cos_cache, sin_cache, rope_theta);
        new_ctx->position_id += token_ids.size() - image_embeds_size + (num_patches_w / spatial_merge_size);
    }

    ncnn::Mat mask((int)token_ids.size() + new_ctx->kv_cache[0].first.h, (int)token_ids.size());
    mask.fill(0.0f);
    for (int i = 0; i < (int)token_ids.size(); i++) {
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
            char kname[32], vname[32];
            std::snprintf(kname, sizeof(kname), "cache_k%d", i);
            std::snprintf(vname, sizeof(vname), "cache_v%d", i);
            ex.input(kname, new_ctx->kv_cache[i].first);
            ex.input(vname, new_ctx->kv_cache[i].second);
        }

        for (int i = 0; i < attn_cnt; i++) {
            char kname[32], vname[32];
            std::snprintf(kname, sizeof(kname), "out_cache_k%d", i);
            std::snprintf(vname, sizeof(vname), "out_cache_v%d", i);
            ncnn::Mat k_cache, v_cache;
            ex.extract(kname, k_cache);
            ex.extract(vname, v_cache);
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
    
    ncnn::Mat last_cos_cache, last_sin_cache;
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
            char kname[16], vname[16];
            std::snprintf(kname, sizeof(kname), "cache_k%d", i);
            std::snprintf(vname, sizeof(vname), "cache_v%d", i);
            ex.input(kname, new_ctx->kv_cache[i].first);
            ex.input(vname, new_ctx->kv_cache[i].second);
        }

        for (int i = 0; i < attn_cnt; i++) {
            char kname[32], vname[32];
            std::snprintf(kname, sizeof(kname), "out_cache_k%d", i);
            std::snprintf(vname, sizeof(vname), "out_cache_v%d", i);
            ncnn::Mat k_cache, v_cache;
            ex.extract(kname, k_cache);
            ex.extract(vname, v_cache);
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
        float max_val = p[0];
        for (int i = 1; i < logits.w; ++i) {
            if (p[i] > max_val) {
                max_val = p[i];
                next_token_id = i;
            }
        }
    }
    new_ctx->cur_token = next_token_id;
    return new_ctx;
}

std::shared_ptr<ncnn_llm_gpt_ctx> ncnn_llm_gpt::prefill(const std::string& input_text, const std::shared_ptr<ncnn_llm_gpt_ctx> ctx) const {
    std::shared_ptr<ncnn_llm_gpt_ctx> new_ctx = clone_ctx(ctx);

    auto token_ids = bpe->encode(input_text, false, false);
    int last_token_id = token_ids.back();
    token_ids.pop_back();

    ncnn::Mat cos_cache, sin_cache;
    int current_pos = new_ctx->position_id;

    if (rope_type == RoPE_Type::LongRoPE) {
        generate_rope_embed_cache_LongRoPE(token_ids.size(), rope_head_dim, current_pos, cos_cache, sin_cache, rope_theta, short_factor.data(), long_factor.data(), original_max_position_embeddings);
    } else if (rope_type == RoPE_Type::NTK_RoPE) {
        generate_ntk_rope_embed_cache(token_ids.size(), rope_head_dim, current_pos, cos_cache, sin_cache, rope_theta, ntk_scaling_params);
    } else if (rope_type == RoPE_Type::YARN_RoPE) {
        generate_yarn_rope_embed_cache(token_ids.size(), rope_head_dim, current_pos, cos_cache, sin_cache, rope_theta, ntk_scaling_params);
    } else if (rope_type == RoPE_Type::HY_RoPE) {
        generate_hunyuan_rope_embed_cache(token_ids.size(), rope_head_dim, current_pos, cos_cache, sin_cache, rope_theta, ntk_scaling_params);
    }
    else {
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
    for (int i = 0; i < (int)token_ids.size(); i++) {
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
            char kname[32], vname[32];
            std::snprintf(kname, sizeof(kname), "cache_k%d", i);
            std::snprintf(vname, sizeof(vname), "cache_v%d", i);
            ex.input(kname, new_ctx->kv_cache[i].first);
            ex.input(vname, new_ctx->kv_cache[i].second);
        }

        for (int i = 0; i < attn_cnt; i++) {
            char kname[32], vname[32];
            std::snprintf(kname, sizeof(kname), "out_cache_k%d", i);
            std::snprintf(vname, sizeof(vname), "out_cache_v%d", i);
            ncnn::Mat k_cache, v_cache;
            ex.extract(kname, k_cache);
            ex.extract(vname, v_cache);
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
    
    ncnn::Mat last_cos_cache, last_sin_cache;
    int last_token_pos = current_pos + (int)token_ids.size();

    if (rope_type == RoPE_Type::LongRoPE) {
        generate_rope_embed_cache_LongRoPE(1, rope_head_dim, last_token_pos, last_cos_cache, last_sin_cache, rope_theta, short_factor.data(), long_factor.data(), original_max_position_embeddings);
    } else if (rope_type == RoPE_Type::NTK_RoPE) {
        generate_ntk_rope_embed_cache(1, rope_head_dim, last_token_pos, last_cos_cache, last_sin_cache, rope_theta, ntk_scaling_params);
    } else if (rope_type == RoPE_Type::YARN_RoPE) {
        generate_yarn_rope_embed_cache(1, rope_head_dim, last_token_pos, last_cos_cache, last_sin_cache, rope_theta, ntk_scaling_params);
    } else if (rope_type == RoPE_Type::HY_RoPE) {
        generate_hunyuan_rope_embed_cache(1, rope_head_dim, last_token_pos, last_cos_cache, last_sin_cache, rope_theta, ntk_scaling_params);
    }
    else {
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
            char kname[16], vname[16];
            std::snprintf(kname, sizeof(kname), "cache_k%d", i);
            std::snprintf(vname, sizeof(vname), "cache_v%d", i);
            ex.input(kname, new_ctx->kv_cache[i].first);
            ex.input(vname, new_ctx->kv_cache[i].second);
        }

        for (int i = 0; i < attn_cnt; i++) {
            char kname[32], vname[32];
            std::snprintf(kname, sizeof(kname), "out_cache_k%d", i);
            std::snprintf(vname, sizeof(vname), "out_cache_v%d", i);
            ncnn::Mat k_cache, v_cache;
            ex.extract(kname, k_cache);
            ex.extract(vname, v_cache);
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
        float max_val = p[0];
        for (int i = 1; i < logits.w; ++i) {
            if (p[i] > max_val) {
                max_val = p[i];
                next_token_id = i;
            }
        }
    }
    new_ctx->cur_token = next_token_id;
    new_ctx->position_id += ((int)token_ids.size() + 1);

    return new_ctx;
}

std::shared_ptr<ncnn_llm_gpt_ctx> ncnn_llm_gpt::generate(const std::shared_ptr<ncnn_llm_gpt_ctx>& ctx_in, const GenerateConfig& cfg, std::function<void(const std::string&)> callback) const {
    const int vocab_size = bpe->vocab_size();

    auto handle_tool = [&](const std::string& tool_call_text, std::shared_ptr<ncnn_llm_gpt_ctx>& ctx_ref) {
        nlohmann::json tool_call_json;
        try {
            tool_call_json = nlohmann::json::parse(tool_call_text);
        } catch (const std::exception& e) {
            tool_call_json = nlohmann::json::object();
        }

        nlohmann::json tool_resp;
        if (cfg.tool_callback) {
            tool_resp = cfg.tool_callback(tool_call_json);
        } else {
            tool_resp = nlohmann::json{{"tool_call", tool_call_json}};
        }

        std::string tool_response_pre = "<|im_end|>\n<|im_start|>user\n<tool_response>\n\n";
        std::string tool_response_post = "\n\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n\n";

        ctx_ref = prefill(tool_response_pre + tool_resp.dump() + tool_response_post, ctx_ref);
    };

    if (cfg.do_sample == 1 || cfg.beam_size <= 1) {
        auto ctx = clone_ctx(ctx_in);
        std::unordered_set<int> history;
        history.insert(ctx->cur_token);

        bool flag_in_tool_call = false;
        std::string tool_call_content;

        for (int step = 0; step < cfg.max_new_tokens; ++step) {
            if (ctx->cur_token == eos) break;

            if (ctx->cur_token == tool_call_id) {
                flag_in_tool_call = true;
            } else if (ctx->cur_token == tool_call_end_id) {
                flag_in_tool_call = false;
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
            } else if (rope_type == RoPE_Type::NTK_RoPE) {
                generate_ntk_rope_embed_cache(1, rope_head_dim, ctx->position_id, cos_cache, sin_cache, rope_theta, ntk_scaling_params);
            } else if (rope_type == RoPE_Type::YARN_RoPE) {
                generate_yarn_rope_embed_cache(1, rope_head_dim, ctx->position_id, cos_cache, sin_cache, rope_theta, ntk_scaling_params);
            } else if (rope_type == RoPE_Type::HY_RoPE) {
                generate_hunyuan_rope_embed_cache(1, rope_head_dim, ctx->position_id, cos_cache, sin_cache, rope_theta, ntk_scaling_params);
            }
            else {
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
                if (logits[t] < 0) logits[t] *= cfg.repetition_penalty;
                else logits[t] /= cfg.repetition_penalty;
            }

            softmax_vec(logits, cfg.temperature);
            if (cfg.top_k > 0) apply_top_k(logits, cfg.top_k);
            if (cfg.top_p < 1.0f) apply_top_p(logits, cfg.top_p);

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

    // Beam Search Implementation
    auto base_ctx = clone_ctx(ctx_in);
    std::vector<Beam> beams;
    beams.reserve(cfg.beam_size);

    Beam b0;
    b0.ctx = base_ctx;
    b0.tokens.insert(base_ctx->cur_token);
    b0.prev_token = -1; 
    beams.push_back(std::move(b0));

    if (beams[0].ctx->cur_token == tool_call_id) {
        beams[0].in_tool_call = true;
    }

    auto maybe_emit_prev = [&](const Beam& best){
        int t = best.prev_token;
        if (t != -1 && !best.prev_in_tool_call && t != eos && t != tool_call_id && t != tool_call_end_id) {
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
            } else if (rope_type == RoPE_Type::NTK_RoPE) {
                generate_ntk_rope_embed_cache(1, rope_head_dim, bctx.position_id, cos_cache, sin_cache, rope_theta, ntk_scaling_params);
            } else if (rope_type == RoPE_Type::YARN_RoPE) {
                generate_yarn_rope_embed_cache(1, rope_head_dim, bctx.position_id, cos_cache, sin_cache, rope_theta, ntk_scaling_params);
            } else if (rope_type == RoPE_Type::HY_RoPE) {
                generate_hunyuan_rope_embed_cache(1, rope_head_dim, bctx.position_id, cos_cache, sin_cache, rope_theta, ntk_scaling_params);
            }
            else {
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
                if (logits[t] < 0) logits[t] *= cfg.repetition_penalty;
                else logits[t] /= cfg.repetition_penalty;
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
                } else if (tok == tool_call_end_id && nb.in_tool_call) {
                    nb.in_tool_call = false;
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
            beams.clear();
            beams.push_back(tool_completed);
            auto& b = beams[0];
            if (!b.in_tool_call && b.ctx->cur_token != eos && b.ctx->cur_token != tool_call_id && b.ctx->cur_token != tool_call_end_id) {
                callback(bpe->decode({b.ctx->cur_token}, false));
            }
            if (b.ctx->cur_token == eos || b.finished) break;
            continue;
        }

        std::sort(candidates.begin(), candidates.end(), [](const Beam& a, const Beam& b) { return a.score > b.score; });

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
        }

        beams = std::move(next_beams);
        auto& best = beams[0];
        maybe_emit_prev(best);

        if (best.ctx->cur_token == eos || best.finished) break;
        
        bool all_finished = true;
        for (auto& b : beams) {
            if (!b.finished) { all_finished = false; break; }
        }
        if (all_finished) break;
    }

    auto best_it = std::max_element(beams.begin(), beams.end(), [](const Beam& a, const Beam& b) { return a.score < b.score; });
    return best_it->ctx;
}

std::shared_ptr<ncnn_llm_gpt_ctx> ncnn_llm_gpt::define_tools(const std::shared_ptr<ncnn_llm_gpt_ctx>& ctx, const std::vector<nlohmann::json>& tools, const std::string& system_prompt) {
    if (tool_call_id < 0 || tool_call_end_id < 0) return ctx;

    this->tools = tools;
    std::string tool_prompt = apply_chat_template({{"system", system_prompt}}, tools, false, false);

    if (ctx) return prefill(tool_prompt, ctx);
    return prefill(tool_prompt);
}

// Vision Helper Implementations

int ncnn_llm_gpt::get_scaled_image_size(float scale, int size, int effective_patch_size) const {
    float scaled_size_f = (float)size * scale;
    int scaled_size = (int)(std::ceil(scaled_size_f / (float)effective_patch_size) * effective_patch_size);
    return std::max(effective_patch_size, scaled_size);
}

void ncnn_llm_gpt::get_image_size_for_patches(int image_height, int image_width, int patch_size, int max_num_patches, int& target_height, int& target_width) const {
    float scale = 1.0f;
    int effective_patch_size = patch_size * 2;
    while (true) {
        target_height = get_scaled_image_size(scale, image_height, effective_patch_size);
        target_width = get_scaled_image_size(scale, image_width, effective_patch_size);
        long long num_patches = ((long long)target_height / patch_size) * ((long long)target_width / patch_size);
        if (num_patches > max_num_patches) {
            scale -= 0.02f;
        } else {
            break;
        }
    }
}

ncnn::Mat ncnn_llm_gpt::bgr_to_pixel_values(const cv::Mat& bgr) const {
    const float image_mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    const float image_std[3] = {0.26862954f, 0.26130258f, 0.27577711f};

    int img_h = bgr.rows;
    int img_w = bgr.cols;

    int num_patches_h = (img_h + patch_size - 1) / patch_size;
    int num_patches_w = (img_w + patch_size - 1) / patch_size;
    int num_patches = num_patches_h * num_patches_w;

    int embed_dim = patch_size * patch_size * 3;
    ncnn::Mat pixel_values(embed_dim, num_patches);

    for (int p = 0; p < num_patches; p++) {
        int ph = p / num_patches_w;
        int pw = p % num_patches_w;
        int start_y = ph * patch_size;
        int start_x = pw * patch_size;

        float* out_ptr = pixel_values.row(p);
        float* ptr_r = out_ptr;
        float* ptr_g = out_ptr + patch_size * patch_size;
        float* ptr_b = out_ptr + patch_size * patch_size * 2;

        for (int y = 0; y < patch_size; y++) {
            const uchar* img_row_ptr = NULL;
            int cur_img_y = start_y + y;
            if (cur_img_y < img_h) {
                img_row_ptr = bgr.ptr<uchar>(cur_img_y);
            }

            for (int x = 0; x < patch_size; x++) {
                int cur_img_x = start_x + x;
                if (img_row_ptr && cur_img_x < img_w) {
                    const uchar* pixel = img_row_ptr + cur_img_x * 3;
                    *ptr_r++ = (pixel[2] / 255.f - image_mean[0]) / image_std[0];
                    *ptr_g++ = (pixel[1] / 255.f - image_mean[1]) / image_std[1];
                    *ptr_b++ = (pixel[0] / 255.f - image_mean[2]) / image_std[2];
                } else {
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

ncnn::Mat ncnn_llm_gpt::reorder_patches_for_merge(const ncnn::Mat& pixel_values, int h_patches, int w_patches) const {
    int num_patches = pixel_values.h;
    int feature_dim = pixel_values.w;

    if (num_patches != h_patches * w_patches) return ncnn::Mat();

    int grid_h = h_patches / spatial_merge_size;
    int grid_w = w_patches / spatial_merge_size;

    ncnn::Mat reordered_pixel_values(feature_dim, num_patches, (size_t)4u);
    int new_row_idx = 0;

    for (int gh = 0; gh < grid_h; gh++) {
        for (int gw = 0; gw < grid_w; gw++) {
            for (int mh = 0; mh < spatial_merge_size; mh++) {
                for (int mw = 0; mw < spatial_merge_size; mw++) {
                    int original_h = gh * spatial_merge_size + mh;
                    int original_w = gw * spatial_merge_size + mw;
                    int original_row_idx = original_h * w_patches + original_w;

                    const float* src_ptr = pixel_values.row(original_row_idx);
                    float* dst_ptr = reordered_pixel_values.row(new_row_idx);
                    memcpy(dst_ptr, src_ptr, feature_dim * sizeof(float));
                    new_row_idx++;
                }
            }
        }
    }
    return reordered_pixel_values;
}

void ncnn_llm_gpt::get_window_index(int num_patches_w, int num_patches_h, std::vector<int>& window_index, std::vector<int>& cu_window_seqlens) const {
    const int vit_merger_window_size = 4;

    int llm_grid_h = num_patches_h / spatial_merge_size;
    int llm_grid_w = num_patches_w / spatial_merge_size;

    int num_windows_h = (llm_grid_h + vit_merger_window_size - 1) / vit_merger_window_size;
    int num_windows_w = (llm_grid_w + vit_merger_window_size - 1) / vit_merger_window_size;

    window_index.clear();
    window_index.reserve(llm_grid_h * llm_grid_w);

    cu_window_seqlens.clear();
    cu_window_seqlens.push_back(0);

    int current_cu_len = 0;

    for (int nh = 0; nh < num_windows_h; ++nh) {
        for (int nw = 0; nw < num_windows_w; ++nw) {
            int h_start = nh * vit_merger_window_size;
            int w_start = nw * vit_merger_window_size;
            int h_end = std::min(h_start + vit_merger_window_size, llm_grid_h);
            int w_end = std::min(w_start + vit_merger_window_size, llm_grid_w);

            int valid_h = h_end - h_start;
            int valid_w = w_end - w_start;

            if (valid_h <= 0 || valid_w <= 0) continue;

            for (int r = h_start; r < h_end; ++r) {
                for (int c = w_start; c < w_end; ++c) {
                    int original_idx = r * llm_grid_w + c;
                    window_index.push_back(original_idx);
                }
            }

            int tokens_in_this_window = valid_h * valid_w * (spatial_merge_size * spatial_merge_size);
            current_cu_len += tokens_in_this_window;
            cu_window_seqlens.push_back(current_cu_len);
        }
    }
}

std::vector<float> ncnn_llm_gpt::compute_inv_freq(int dim, float theta) {
    int num_freqs = dim / 2;
    std::vector<float> inv_freq(num_freqs);
    for (int i = 0; i < num_freqs; i++) {
        float exp_val = (float)(i * 2) / (float)dim;
        inv_freq[i] = 1.0f / std::pow(theta, exp_val);
    }
    return inv_freq;
}

void ncnn_llm_gpt::generate_rope_embeds(int num_patches_w, int num_patches_h, ncnn::Mat& emb_cos, ncnn::Mat& emb_sin, int rope_dim) const {
    int seq_len = num_patches_w * num_patches_h;
    std::vector<float> inv_freq = compute_inv_freq(rope_dim);
    int half_dim = inv_freq.size();
    int output_dim = half_dim * 4;

    emb_cos.create(output_dim, seq_len, sizeof(float));
    emb_sin.create(output_dim, seq_len, sizeof(float));

    int idx = 0;
    int grid_h = num_patches_h / spatial_merge_size;
    int grid_w = num_patches_w / spatial_merge_size;

    for (int gh = 0; gh < grid_h; gh++) {
        for (int gw = 0; gw < grid_w; gw++) {
            for (int bi = 0; bi < spatial_merge_size; bi++) {
                for (int bj = 0; bj < spatial_merge_size; bj++) {
                    int current_h = gh * spatial_merge_size + bi;
                    int current_w = gw * spatial_merge_size + bj;

                    float* cos_ptr = emb_cos.row(idx);
                    float* sin_ptr = emb_sin.row(idx);

                    for (int k = 0; k < half_dim; k++) {
                        float freq = inv_freq[k];
                        float angle_h = (float)current_h * freq;
                        float angle_w = (float)current_w * freq;

                        int idx_h1 = k;
                        int idx_w1 = half_dim + k;
                        int idx_h2 = 2 * half_dim + k;
                        int idx_w2 = 3 * half_dim + k;

                        float ch = std::cos(angle_h);
                        float sh = std::sin(angle_h);
                        float cw = std::cos(angle_w);
                        float sw = std::sin(angle_w);

                        cos_ptr[idx_h1] = ch; sin_ptr[idx_h1] = sh;
                        cos_ptr[idx_h2] = ch; sin_ptr[idx_h2] = sh;
                        cos_ptr[idx_w1] = cw; sin_ptr[idx_w1] = sw;
                        cos_ptr[idx_w2] = cw; sin_ptr[idx_w2] = sw;
                    }
                    idx++;
                }
            }
        }
    }
}

int ncnn_llm_gpt::get_visiual_features(const cv::Mat& bgr, ncnn::Mat& image_embeds, int& num_patches_w, int& num_patches_h) const {
    if (bgr.empty()) {
        image_embeds.release();
        num_patches_w = 0;
        num_patches_h = 0;
        return 0;
    }

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    int target_w, target_h;
    get_image_size_for_patches(img_h, img_w, patch_size, max_num_patches, target_h, target_w);

    cv::Mat bgr_resized;
    cv::resize(bgr, bgr_resized, cv::Size(target_w, target_h), 0, 0, cv::INTER_AREA);

    num_patches_w = (target_w + patch_size - 1) / patch_size;
    num_patches_h = (target_h + patch_size - 1) / patch_size;
    const int seq_len = num_patches_w * num_patches_h;

    ncnn::Mat pixel_values = bgr_to_pixel_values(bgr_resized);
    pixel_values = reorder_patches_for_merge(pixel_values, num_patches_h, num_patches_w);
    pixel_values = pixel_values.reshape(patch_size * patch_size, 1, 3, seq_len);

    {
        ncnn::Mat tmp(patch_size * patch_size, 2, 3, seq_len);
        for (int i = 0; i < seq_len; i++) {
            for (int c = 0; c < 3; c++) {
                const float* src = pixel_values.channel(i).depth(c).row(0);
                memcpy(tmp.channel(i).depth(c).row(0), src, patch_size * patch_size * sizeof(float));
                memcpy(tmp.channel(i).depth(c).row(1), src, patch_size * patch_size * sizeof(float));
            }
        }
        pixel_values = tmp.reshape(patch_size * patch_size * 2 * 3, seq_len);
    }

    std::vector<int> window_index;
    std::vector<int> cu_window_seqlens;
    get_window_index(num_patches_w, num_patches_h, window_index, cu_window_seqlens);

    ncnn::Mat patch_embeds(patch_dim, seq_len);
    for (int i = 0; i < seq_len; i++) {
        ncnn::Mat patch = pixel_values.row_range(i, 1).reshape(patch_size, patch_size, 2, 3);
        ncnn::Mat patch_embed;
        ncnn::Extractor ex = vision_embed_patch->create_extractor();
        ex.input("in0", patch);
        ex.extract("out0", patch_embed);
        memcpy(patch_embeds.row(i), patch_embed.reshape(patch_dim), patch_dim * sizeof(float));
    }

    ncnn::Mat emb_cos, emb_sin;
    generate_rope_embeds(num_patches_w, num_patches_h, emb_cos, emb_sin, 40);

    ncnn::Mat patch_embeds_reordered(patch_embeds.w, seq_len, sizeof(float));
    ncnn::Mat emb_cos_reordered(emb_cos.w, seq_len, sizeof(float));
    ncnn::Mat emb_sin_reordered(emb_sin.w, seq_len, sizeof(float));

    int group_size = 4;
    for (int i = 0; i < window_index.size(); i++) {
        int src_group_idx = window_index[i];
        for (int k = 0; k < group_size; k++) {
            int src_row = src_group_idx * group_size + k;
            int dst_row = i * group_size + k;

            const float* src_ptr = patch_embeds.row(src_row);
            float* dst_ptr = patch_embeds_reordered.row(dst_row);
            memcpy(dst_ptr, src_ptr, patch_embeds.w * sizeof(float));

            const float* src_cos = emb_cos.row(src_row);
            float* dst_cos = emb_cos_reordered.row(dst_row);
            memcpy(dst_cos, src_cos, emb_cos.w * sizeof(float));

            const float* src_sin = emb_sin.row(src_row);
            float* dst_sin = emb_sin_reordered.row(dst_row);
            memcpy(dst_sin, src_sin, emb_sin.w * sizeof(float));
        }
    }

    std::vector<int> cu_seqlens = cu_window_seqlens;
    ncnn::Mat attention_mask(seq_len, seq_len);
    attention_mask.fill(-1e9f);

    for (size_t i = 1; i < cu_seqlens.size(); i++) {
        int start = cu_seqlens[i-1];
        int end = cu_seqlens[i];
        for (int r = start; r < end; r++) {
            float* row_ptr = attention_mask.row(r);
            for (int c = start; c < end; c++) {
                row_ptr[c] = 0.f;
            }
        }
    }

    {
        ncnn::Extractor ex = vision_encoder->create_extractor();
        ex.input("in0", patch_embeds_reordered);
        ex.input("in1", emb_cos_reordered);
        ex.input("in2", emb_sin_reordered);
        ex.input("in3", attention_mask);
        ex.extract("out0", image_embeds);
    }

    ncnn::Mat image_embeds_restored(image_embeds.w, image_embeds.h);
    for (int i = 0; i < window_index.size(); i++) {
        int dest_group_idx = window_index[i];
        int src_group_idx = i;
        const float* src_ptr = image_embeds.row(dest_group_idx);
        float* dst_ptr = image_embeds_restored.row(src_group_idx);
        memcpy(dst_ptr, src_ptr, image_embeds.w * sizeof(float));
    }
    image_embeds = image_embeds_restored;
    return 0;
}