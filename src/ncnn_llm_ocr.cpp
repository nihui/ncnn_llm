#include "ncnn_llm_ocr.h"
#include "utils/hunyuan_ocr_prompt.h"
#include "utils/vision_rope.h"

ncnn_llm_ocr::ncnn_llm_ocr(const std::string& model_path, bool use_vulkan, int num_threads)
    : ncnn_llm_base(use_vulkan, num_threads > 0 ? num_threads : 4) {
    try {
        json config;
        {
            std::ifstream ifs(model_path + "/model.json");
            ifs >> config;
        }
        if (config.contains("model_type")) {
            model_type_ = config["model_type"].get<std::string>();
        }

        vision_net_ = std::make_shared<ncnn::Net>();
        text_embed_net_ = std::make_shared<ncnn::Net>();
        text_decoder_net_ = std::make_shared<ncnn::Net>();
        lm_head_net_ = std::make_shared<ncnn::Net>();

        if (num_threads > 0) {
            vision_net_->opt.num_threads = num_threads;
            text_embed_net_->opt.num_threads = num_threads;
            text_decoder_net_->opt.num_threads = num_threads;
            lm_head_net_->opt.num_threads = num_threads;
        }

        if (use_vulkan) {
            printf("[ncnn_llm_ocr] Vulkan enabled\n");
            vision_net_->opt.use_vulkan_compute = true;
            text_decoder_net_->opt.use_vulkan_compute = true;
            text_decoder_net_->opt.use_bf16_storage = true;
            text_decoder_net_->opt.use_fp16_arithmetic = false;
            text_decoder_net_->opt.use_fp16_storage = false;
        } else {
            printf("[ncnn_llm_ocr] Vulkan disabled, using CPU only\n");
        }

        std::string vision_param = model_path + "/" + config["params"]["vision_param"].get<std::string>();
        std::string vision_bin = model_path + "/" + config["params"]["vision_bin"].get<std::string>();
        std::string text_embed_param = model_path + "/" + config["params"]["text_embed_param"].get<std::string>();
        std::string text_embed_bin = model_path + "/" + config["params"]["text_embed_bin"].get<std::string>();
        std::string text_decoder_param = model_path + "/" + config["params"]["text_decoder_param"].get<std::string>();
        std::string text_decoder_bin = model_path + "/" + config["params"]["text_decoder_bin"].get<std::string>();
        std::string lm_head_param = model_path + "/" + config["params"]["lm_head_param"].get<std::string>();
        std::string lm_head_bin = model_path + "/" + config["params"]["lm_head_bin"].get<std::string>();

        printf("Loading OCR model (%s) from %s\n", model_type_.c_str(), model_path.c_str());
        printf("  vision param: %s\n", vision_param.c_str());
        printf("  vision bin: %s\n", vision_bin.c_str());
        printf("  text_embed param: %s\n", text_embed_param.c_str());
        printf("  text_embed bin: %s\n", text_embed_bin.c_str());
        printf("  text_decoder param: %s\n", text_decoder_param.c_str());
        printf("  text_decoder bin: %s\n", text_decoder_bin.c_str());
        printf("  lm_head param: %s\n", lm_head_param.c_str());
        printf("  lm_head bin: %s\n", lm_head_bin.c_str());

        vision_net_->load_param(vision_param.c_str());
        vision_net_->load_model(vision_bin.c_str());
        text_embed_net_->load_param(text_embed_param.c_str());
        text_embed_net_->load_model(text_embed_bin.c_str());
        text_decoder_net_->load_param(text_decoder_param.c_str());
        text_decoder_net_->load_model(text_decoder_bin.c_str());
        lm_head_net_->load_param(lm_head_param.c_str());
        lm_head_net_->load_model(lm_head_bin.c_str());

        std::string type = "bpe";
        if (config["tokenizer"].contains("type")) {
            type = config["tokenizer"]["type"].get<std::string>();
        }
        std::string vocab_file = model_path + "/" + config["tokenizer"]["vocab_file"].get<std::string>();
        std::string merges_file = model_path + "/" + config["tokenizer"]["merges_file"].get<std::string>();

        bpe_ = std::make_shared<BpeTokenizer>(BpeTokenizer::LoadFromFiles(
            vocab_file, merges_file, SpecialTokensConfig{}, false, true, type == "bbpe"
        ));

        std::vector<std::string> additional_special_tokens = config["tokenizer"]["additional_special_tokens"].get<std::vector<std::string>>();
        for (const auto& token : additional_special_tokens) {
            bpe_->AddAdditionalSpecialToken(token);
        }
        // Build set of additional special token IDs for fast lookup
        for (int id : bpe_->additional_special_token_ids()) {
            additional_special_id_set_.insert(id);
        }

        auto eos_token = config["tokenizer"]["eos"].get<std::string>();
        eos_ = (eos_token != "") ? bpe_->token_to_id().at(eos_token) : -1;
        auto it_eop = bpe_->token_to_id().find("<eop>");
        eop_ = (it_eop != bpe_->token_to_id().end()) ? it_eop->second : -1;
        if (config["tokenizer"].contains("eos_ids")) {
            for (auto& v : config["tokenizer"]["eos_ids"]) {
                eos_ids_.insert(v.get<int>());
            }
        }

        if (config["setting"].contains("attn_cnt")) {
            attn_cnt_ = config["setting"]["attn_cnt"].get<int>();
        }
        if (config["setting"].contains("hidden_size")) {
            hidden_size_ = config["setting"]["hidden_size"].get<int>();
        }
        if (config["setting"].contains("head_dim")) {
            head_dim_ = config["setting"]["head_dim"].get<int>();
        }
        if (config["setting"].contains("image_token_id")) {
            image_token_id_ = config["setting"]["image_token_id"].get<int>();
        }

        if (model_type_ == "hunyuan_ocr") {
            // ---- HunyuanOCR: 4-axis xdrope, vision without RoPE, no KV cache ----
            use_kv_cache_ = config["setting"].value("kv_cache", false);
            auto rope_cfg = config["setting"]["rope"];
            if (!rope_cfg.contains("type") || rope_cfg["type"].get<std::string>() != "xdrope") {
                throw std::runtime_error("hunyuan_ocr requires setting.rope.type == xdrope");
            }
            rope_theta_ = rope_cfg.value("rope_theta", 10000.0f);
            head_dim_ = rope_cfg.value("rope_head_dim", head_dim_);
            rope_alpha_ = rope_cfg.value("alpha", 1000.0f);
            for (auto& v : rope_cfg["xdrope_section"]) {
                xdrope_section_.push_back(v.get<int>());
            }
            int sect_sum = 0;
            for (int s : xdrope_section_) sect_sum += s;
            if (xdrope_section_.empty() || sect_sum != head_dim_ / 2) {
                throw std::runtime_error("setting.rope.xdrope_section must sum to rope_head_dim/2");
            }

            bos_id_ = config["setting"].value("bos_token_id", 120000);
            system_end_id_ = config["setting"].value("system_end_token_id", 120021);
            user_end_id_ = config["setting"].value("user_end_token_id", 120006);
            image_start_id_ = config["setting"].value("image_start_token_id", 120118);
            image_end_id_ = config["setting"].value("image_end_token_id", 120119);
            special_id_begin_ = config["setting"].value("special_token_id_begin", 120000);

            auto vision_cfg = config["setting"]["vision"];
            patch_size_ = vision_cfg.value("patch_size", 16);
            spatial_merge_size_ = vision_cfg.value("spatial_merge_size", 2);
            vision_hidden_size_ = vision_cfg.value("vision_hidden_size", 1152);
            if (vision_cfg.contains("min_pixels")) min_pixels_ = vision_cfg["min_pixels"].get<long long>();
            if (vision_cfg.contains("max_pixels")) max_pixels_ = vision_cfg["max_pixels"].get<long long>();
            if (vision_cfg.contains("image_mean")) {
                auto mean = vision_cfg["image_mean"].get<std::vector<float>>();
                image_mean_[0] = mean[0]; image_mean_[1] = mean[1]; image_mean_[2] = mean[2];
            }
            if (vision_cfg.contains("image_std")) {
                auto std_vals = vision_cfg["image_std"].get<std::vector<float>>();
                image_std_[0] = std_vals[0]; image_std_[1] = std_vals[1]; image_std_[2] = std_vals[2];
            }

            vocab_size_ = config["setting"].value("vocab_size", (int)bpe_->vocab_size());
            printf("  [hunyuan_ocr] attn_cnt: %d, hidden_size: %d, head_dim: %d, vocab_size: %d\n",
                   attn_cnt_, hidden_size_, head_dim_, vocab_size_);
            printf("  [hunyuan_ocr] patch_size: %d, merge_size: %d, min/max_pixels: %lld/%lld\n",
                   patch_size_, spatial_merge_size_, min_pixels_, max_pixels_);
            printf("  [hunyuan_ocr] rope_theta: %.1f, alpha: %.1f, xdrope_section size: %d, kv_cache: %d\n",
                   rope_theta_, rope_alpha_, (int)xdrope_section_.size(), (int)use_kv_cache_);
            printf("  [hunyuan_ocr] image_token_id: %d, image_start: %d, image_end: %d, bos: %d, user_end: %d\n",
                   image_token_id_, image_start_id_, image_end_id_, bos_id_, user_end_id_);
        } else {
            if (config["setting"].contains("rope")) {
                auto text_rope_cfg = config["setting"]["rope"];
                if (!text_rope_cfg.contains("type") || text_rope_cfg["type"].get<std::string>() != "mRoPE") {
                    throw std::runtime_error("unsupported setting.rope.type in model.json");
                }
                if (text_rope_cfg.contains("rope_theta")) {
                    rope_theta_ = text_rope_cfg["rope_theta"].get<float>();
                }
                if (text_rope_cfg.contains("rope_head_dim")) {
                    head_dim_ = text_rope_cfg["rope_head_dim"].get<int>();
                }
                if (!text_rope_cfg.contains("mrope_section")) {
                    throw std::runtime_error("missing setting.rope.mrope_section in model.json");
                }
                for (auto& v : text_rope_cfg["mrope_section"]) {
                    mrope_section_.push_back(v.get<int>());
                }
            } else if (config["setting"].contains("rope_theta") || config["setting"].contains("mrope_section")) {
                throw std::runtime_error("legacy setting.rope_theta/mrope_section is not supported; use setting.rope");
            } else {
                throw std::runtime_error("missing setting.rope in model.json");
            }
            if (mrope_section_.size() != 3) {
                throw std::runtime_error("setting.rope.mrope_section must have 3 entries");
            }

            if (config["setting"].contains("vision")) {
                auto vision_cfg = config["setting"]["vision"];
                if (vision_cfg.contains("patch_size")) {
                    patch_size_ = vision_cfg["patch_size"].get<int>();
                }
                if (vision_cfg.contains("spatial_merge_size")) {
                    spatial_merge_size_ = vision_cfg["spatial_merge_size"].get<int>();
                }
                if (vision_cfg.contains("vision_hidden_size")) {
                    vision_hidden_size_ = vision_cfg["vision_hidden_size"].get<int>();
                }
                if (vision_cfg.contains("vision_head_dim")) {
                    vision_head_dim_ = vision_cfg["vision_head_dim"].get<int>();
                }
                if (vision_cfg.contains("vision_num_heads")) {
                    vision_num_heads_ = vision_cfg["vision_num_heads"].get<int>();
                }
                if (!vision_cfg.contains("rope")) {
                    throw std::runtime_error("missing setting.vision.rope in model.json");
                }
                auto vision_rope_cfg = vision_cfg["rope"];
                if (!vision_rope_cfg.contains("type") || vision_rope_cfg["type"].get<std::string>() != "mRoPE") {
                    throw std::runtime_error("unsupported setting.vision.rope.type in model.json");
                }
                if (vision_rope_cfg.contains("rope_theta")) {
                    vision_rope_theta_ = vision_rope_cfg["rope_theta"].get<float>();
                }
                if (vision_rope_cfg.contains("rope_head_dim")) {
                    vision_rope_dim_ = vision_rope_cfg["rope_head_dim"].get<int>();
                }
                if (!vision_rope_cfg.contains("mrope_section")) {
                    throw std::runtime_error("missing setting.vision.rope.mrope_section in model.json");
                }
                for (auto& v : vision_rope_cfg["mrope_section"]) {
                    vision_mrope_section_.push_back(v.get<int>());
                }
                if (vision_rope_theta_ <= 0.0f || vision_rope_dim_ <= 0 || (vision_rope_dim_ % 2) != 0) {
                    throw std::runtime_error("invalid setting.vision.rope rope_theta/rope_head_dim in model.json");
                }
                if (vision_mrope_section_.size() != 2 ||
                    vision_mrope_section_[0] + vision_mrope_section_[1] != vision_rope_dim_) {
                    throw std::runtime_error("setting.vision.rope.mrope_section must be [h_dim,w_dim] and sum to rope_head_dim");
                }
                if (vision_cfg.contains("max_num_patches")) {
                    max_num_patches_ = vision_cfg["max_num_patches"].get<int>();
                }
                if (vision_cfg.contains("min_pixels")) {
                    min_pixels_ = vision_cfg["min_pixels"].get<long long>();
                }
                if (vision_cfg.contains("max_pixels")) {
                    max_pixels_ = vision_cfg["max_pixels"].get<long long>();
                }
                if (vision_cfg.contains("image_mean")) {
                    auto mean = vision_cfg["image_mean"].get<std::vector<float>>();
                    image_mean_[0] = mean[0]; image_mean_[1] = mean[1]; image_mean_[2] = mean[2];
                }
                if (vision_cfg.contains("image_std")) {
                    auto std_vals = vision_cfg["image_std"].get<std::vector<float>>();
                    image_std_[0] = std_vals[0]; image_std_[1] = std_vals[1]; image_std_[2] = std_vals[2];
                }
            }

            vocab_size_ = (int)bpe_->vocab_size();
            printf("  attn_cnt: %d, hidden_size: %d, head_dim: %d, vocab_size: %d\n", attn_cnt_, hidden_size_, head_dim_, vocab_size_);
            printf("  patch_size: %d, spatial_merge_size: %d, max_num_patches: %d\n", patch_size_, spatial_merge_size_, max_num_patches_);
            printf("  text mrope_section: [%d %d %d]\n", mrope_section_[0], mrope_section_[1], mrope_section_[2]);
            printf("  vision rope: type=mRoPE, rope_theta=%.1f, rope_head_dim=%d, mrope_section=[%d %d]\n",
                   vision_rope_theta_, vision_rope_dim_, vision_mrope_section_[0], vision_mrope_section_[1]);
            printf("  min_pixels: %lld, max_pixels: %lld, image_mean/std: [%.6f %.6f %.6f] / [%.6f %.6f %.6f]\n",
                   min_pixels_, max_pixels_,
                   image_mean_[0], image_mean_[1], image_mean_[2],
                   image_std_[0], image_std_[1], image_std_[2]);
            printf("  image_token_id: %d, eos: %d, eop: %d\n", image_token_id_, eos_, eop_);
        }

    } catch (std::exception &e) {
        ok_ = false;
        throw std::runtime_error(std::string("ncnn_llm_ocr load model failed: ") + e.what());
    }
}

void ncnn_llm_ocr::get_image_size_for_patches(int img_h, int img_w, int& target_h, int& target_w) const {
    int effective_patch_size = patch_size_ * spatial_merge_size_;

    auto round_by_factor = [&](double size) -> int {
        int result = (int)(std::round(size / (double)effective_patch_size) * effective_patch_size);
        return std::max(effective_patch_size, result);
    };

    double h = (double)img_h;
    double w = (double)img_w;
    double area = h * w;

    double scale = 1.0;
    if (area > (double)max_pixels_) {
        scale = std::sqrt((double)max_pixels_ / area);
    } else if (area < (double)min_pixels_) {
        scale = std::sqrt((double)min_pixels_ / area);
    }

    target_h = round_by_factor(h * scale);
    target_w = round_by_factor(w * scale);
}

ncnn::Mat ncnn_llm_ocr::bgr_to_image_strip(const ncnn::Mat& bgr, int& num_patches_h, int& num_patches_w) const {
    int img_h = bgr.h;
    int img_w = bgr.w;

    int target_h, target_w;
    get_image_size_for_patches(img_h, img_w, target_h, target_w);

    ncnn::Mat bgr_resized = ncnn_mat_resize(bgr, target_w, target_h);

    num_patches_h = target_h / patch_size_;
    num_patches_w = target_w / patch_size_;
    int num_patches = num_patches_h * num_patches_w;

    const unsigned char* bgr_data = (const unsigned char*)bgr_resized.data;

    // Match the TorchScript wrapper:
    // pv.view(1, N, 3, 2, 14, 14)[:, :, :, 0].permute(0, 2, 3, 1, 4)
    //   .reshape(1, 3, 14, 14 * N)
    // The N dimension is already grouped by 2x2 spatial merge blocks.
    ncnn::Mat image_strip(patch_size_ * num_patches, patch_size_, 3);
    image_strip.fill(0.0f);

    int grid_h = num_patches_h / spatial_merge_size_;
    int grid_w = num_patches_w / spatial_merge_size_;
    int patch_idx = 0;

    for (int gh = 0; gh < grid_h; gh++) {
        for (int gw = 0; gw < grid_w; gw++) {
            for (int mh = 0; mh < spatial_merge_size_; mh++) {
                for (int mw = 0; mw < spatial_merge_size_; mw++) {
                    int patch_h = gh * spatial_merge_size_ + mh;
                    int patch_w = gw * spatial_merge_size_ + mw;
                    int start_y = patch_h * patch_size_;
                    int start_x = patch_w * patch_size_;
                    int strip_base_x = patch_idx * patch_size_;

                    for (int y = 0; y < patch_size_; y++) {
                        const unsigned char* img_row_ptr = bgr_data + (start_y + y) * target_w * 3;
                        float* dst_r = image_strip.channel(0).row(y) + strip_base_x;
                        float* dst_g = image_strip.channel(1).row(y) + strip_base_x;
                        float* dst_b = image_strip.channel(2).row(y) + strip_base_x;

                        for (int x = 0; x < patch_size_; x++) {
                            const unsigned char* pixel = img_row_ptr + (start_x + x) * 3;
                            dst_r[x] = (pixel[2] / 255.0f - image_mean_[0]) / image_std_[0];
                            dst_g[x] = (pixel[1] / 255.0f - image_mean_[1]) / image_std_[1];
                            dst_b[x] = (pixel[0] / 255.0f - image_mean_[2]) / image_std_[2];
                        }
                    }

                    patch_idx++;
                }
            }
        }
    }

    return image_strip;
}

ncnn::Mat ncnn_llm_ocr::run_vision(const ncnn::Mat& image_strip, const ncnn::Mat& cos_cache, const ncnn::Mat& sin_cache) const {
    ncnn::Mat vision_features;
    ncnn::Extractor ex = vision_net_->create_extractor();
    ex.input("in0", image_strip);
    ex.input("in1", cos_cache);
    ex.input("in2", sin_cache);
    ex.extract("out0", vision_features);
    return vision_features;
}

void ncnn_llm_ocr::generate_text_rope_cache(int seq_len, int position_id, ncnn::Mat& cos_cache, ncnn::Mat& sin_cache) const {
    // GLM-OCR text decoder uses interleaved RoPE (head_dim=128, but cos/sin are [seq_len, 64])
    generate_rope_embed_cache(seq_len, head_dim_, position_id, cos_cache, sin_cache, rope_theta_);
}

std::shared_ptr<ncnn_llm_gpt_ctx> ncnn_llm_ocr::prefill(const std::string& prompt_text, const ncnn::Mat& bgr_image) {
    if (model_type_ == "hunyuan_ocr") {
        return prefill_hunyuan(prompt_text, bgr_image);
    }
    // Run vision model
    int num_patches_h = 0, num_patches_w = 0;
    ncnn::Mat image_strip = bgr_to_image_strip(bgr_image, num_patches_h, num_patches_w);
    int num_patches = num_patches_h * num_patches_w;

    ncnn::Mat vision_cos, vision_sin;
    generate_vision_rope_cache_2d(num_patches_h, num_patches_w, spatial_merge_size_,
                                  vision_rope_theta_, vision_mrope_section_,
                                  false, vision_cos, vision_sin);

    ncnn::Mat vision_features = run_vision(image_strip, vision_cos, vision_sin);
    int num_vision_tokens = vision_features.h;

    // Build prompt with image tokens, matching
    // tokenizer.apply_chat_template(..., add_generation_prompt=true, enable_thinking=false).
    std::string full_prompt = "[gMASK]<sop><|user|>\n<|begin_of_image|>";
    for (int i = 0; i < num_vision_tokens; i++) {
        full_prompt += "<|image|>";
    }
    full_prompt += "<|end_of_image|>" + prompt_text;
    if (prompt_text.size() < 8 || prompt_text.rfind("/nothink") != prompt_text.size() - 8) {
        full_prompt += "/nothink";
    }
    full_prompt += "<|assistant|>\n<think></think>\n";

    // Tokenize prompt
    std::vector<int> token_ids = bpe_->encode(full_prompt, false, false);

    // Get text embeddings
    ncnn::Mat token_embed = llm_run_text_embed(*text_embed_net_, token_ids);

    // Find image token positions and inject vision features
    std::vector<int> image_token_positions;
    for (int i = 0; i < (int)token_ids.size(); i++) {
        if (token_ids[i] == image_token_id_) {
            image_token_positions.push_back(i);
        }
    }
    // Replace image token embeddings with vision features
    if ((int)image_token_positions.size() == num_vision_tokens) {
        for (int i = 0; i < num_vision_tokens; i++) {
            int pos = image_token_positions[i];
            float* embed_ptr = token_embed.row(pos);
            const float* feat_ptr = vision_features.row(i);
            memcpy(embed_ptr, feat_ptr, hidden_size_ * sizeof(float));
        }
    } else {
        printf("[ncnn_llm_ocr] WARNING: image token count mismatch! vision_tokens=%d, image_tokens_in_prompt=%d\n",
               num_vision_tokens, (int)image_token_positions.size());
    }

    // Generate causal mask
    int seq_len = (int)token_ids.size();
    ncnn::Mat mask(seq_len, seq_len);
    mask.fill(0.0f);
    for (int i = 0; i < seq_len; i++) {
        float* row = mask.row(i);
        for (int j = i + 1; j < seq_len; j++) {
            row[j] = -1e38f;
        }
    }

    // Generate MRoPE cache for text decoder (image tokens have 2D positions)
    // GLM-OCR text decoder uses interleaved RoPE with cos/sin shape [seq_len, 64]
    ncnn::Mat cos_cache, sin_cache;
    int next_position_id = seq_len;
    if (!mrope_section_.empty() && !image_token_positions.empty()) {
        generate_rope_embed_cache_vision_mrope(seq_len, head_dim_, 0,
                                               image_token_positions[0], num_vision_tokens,
                                               num_patches_w, spatial_merge_size_,
                                               mrope_section_, cos_cache, sin_cache, rope_theta_);
        next_position_id = seq_len - num_vision_tokens + (num_patches_w / spatial_merge_size_);
    } else {
        generate_text_rope_cache(seq_len, 0, cos_cache, sin_cache);
    }

    // Run decoder (prefill pass, no existing KV cache)
    KVCache kv_cache;
    ncnn::Mat decode_out = llm_run_decoder_with_kv(*text_decoder_net_, token_embed, mask, cos_cache, sin_cache,
                                                   kv_cache, attn_cnt_, true);

    // Run lm_head on the last token
    ncnn::Mat last_hidden = decode_out.row_range(seq_len - 1, 1);
    ncnn::Mat logits = llm_run_lm_head(*lm_head_net_, last_hidden);

    // Get next token via argmax
    int next_token_id = argmax1d(logits);

    // Create and return context
    auto ctx = std::make_shared<ncnn_llm_gpt_base_ctx>();
    ctx->kv_cache = std::move(kv_cache);
    ctx->cur_token = next_token_id;
    ctx->position_id = next_position_id;
    return ctx;
}

std::shared_ptr<ncnn_llm_gpt_ctx> ncnn_llm_ocr::generate(
    const std::shared_ptr<ncnn_llm_gpt_ctx>& ctx_in,
    const GenerateConfig& cfg,
    std::function<void(const std::string&)> callback) {

    if (model_type_ == "hunyuan_ocr") {
        return generate_hunyuan(ctx_in, cfg, callback);
    }

    auto ctx = ctx_in->clone();
    std::unordered_set<int> history;
    history.insert(ctx->cur_token);
    std::string emitted_text;

    for (int step = 0; step < cfg.max_new_tokens; ++step) {
        if (ctx->cur_token == eos_ || ctx->cur_token == eop_) break;

        // Skip outputting the first token if it's a special token (e.g. </think|>)
        // The model may output </think|> as the first token in non-thinking mode
        bool is_special = (ctx->cur_token == eos_) ||
                          (ctx->cur_token == eop_) ||
                          (additional_special_id_set_.find(ctx->cur_token) != additional_special_id_set_.end());
        if (!is_special) {
            std::string token_text = bpe_->decode({ctx->cur_token}, false);
            if (!emitted_text.empty() && token_text.find("```") != std::string::npos) {
                break;
            }

            std::string candidate_text = emitted_text + token_text;
            int trailing_newlines = 0;
            for (auto it = candidate_text.rbegin(); it != candidate_text.rend(); ++it) {
                if (*it == '\n') {
                    trailing_newlines++;
                } else if (*it == '\r') {
                    continue;
                } else {
                    break;
                }
            }
            if (!emitted_text.empty() && trailing_newlines > 2) {
                break;
            }
            emitted_text += token_text;
            callback(token_text);
        }

        ncnn::Mat cur_embed = llm_run_text_embed(*text_embed_net_, ctx->cur_token);

        // Generate RoPE cache for single token
        ncnn::Mat cos_cache, sin_cache;
        generate_text_rope_cache(1, ctx->position_id, cos_cache, sin_cache);
        ctx->position_id++;

        // Create mask: [1, kv_len+1] all zeros (new token can attend to all previous)
        ncnn::Mat mask(1, ctx->kv_cache[0].first.h + 1);
        mask.fill(0.0f);

        ncnn::Mat decode_out = llm_run_decoder_with_kv(*text_decoder_net_, cur_embed, mask, cos_cache, sin_cache,
                                                       ctx->kv_cache, attn_cnt_, false);

        ncnn::Mat logits = llm_run_lm_head(*lm_head_net_, decode_out);

        LlmTokenSampleConfig sample_cfg;
        sample_cfg.vocab_size = vocab_size_;
        sample_cfg.temperature = cfg.temperature;
        sample_cfg.top_p = cfg.top_p;
        sample_cfg.top_k = cfg.top_k;
        sample_cfg.repetition_penalty = cfg.repetition_penalty;
        sample_cfg.do_sample = cfg.do_sample;
        int next_id = llm_select_next_token(logits, history, sample_cfg);

        ctx->cur_token = next_id;
        history.insert(next_id);
    }

    return ctx;
}

// ============================================================================ HunyuanOCR path

void ncnn_llm_ocr::smart_resize_hunyuan(int img_h, int img_w, int& target_h, int& target_w) const {
    const long long factor = (long long)patch_size_ * spatial_merge_size_; // 32

    auto round_half_even = [](double x) -> long long {
        double fl = std::floor(x);
        double diff = x - fl;
        if (diff < 0.5) return (long long)fl;
        if (diff > 0.5) return (long long)fl + 1;
        long long lo = (long long)fl;                 // exactly .5 -> round to even (Python round)
        return (lo % 2 == 0) ? lo : lo + 1;
    };

    long long h_bar = std::max(factor, round_half_even((double)img_h / (double)factor) * factor);
    long long w_bar = std::max(factor, round_half_even((double)img_w / (double)factor) * factor);
    const double area = (double)img_h * (double)img_w;

    if (h_bar * w_bar > (long long)max_pixels_) {
        double beta = std::sqrt(area / (double)max_pixels_);
        h_bar = std::max(factor, (long long)std::floor(img_h / beta / factor) * factor);
        w_bar = std::max(factor, (long long)std::floor(img_w / beta / factor) * factor);
    } else if (h_bar * w_bar < (long long)min_pixels_) {
        double beta = std::sqrt((double)min_pixels_ / area);
        h_bar = (long long)std::ceil(img_h * beta / factor) * factor;
        w_bar = (long long)std::ceil(img_w * beta / factor) * factor;
    }

    target_h = (int)h_bar;
    target_w = (int)w_bar;
}

ncnn::Mat ncnn_llm_ocr::bgr_to_chw_image_hunyuan(const ncnn::Mat& bgr, int& target_h, int& target_w) const {
    smart_resize_hunyuan(bgr.h, bgr.w, target_h, target_w);

    // PIL resizes the uint8 image with BICUBIC, then ToTensor(/255)+Normalize.
    ncnn::Mat resized = ncnn_mat_resize_bicubic(bgr, target_w, target_h);  // interleaved BGR u8
    const unsigned char* data = (const unsigned char*)resized.data;

    ncnn::Mat img(target_w, target_h, 3);  // planar float; channel 0=R, 1=G, 2=B
    for (int y = 0; y < target_h; y++) {
        const unsigned char* row = data + (size_t)y * target_w * 3;
        float* rr = img.channel(0).row(y);
        float* gg = img.channel(1).row(y);
        float* bb = img.channel(2).row(y);
        for (int x = 0; x < target_w; x++) {
            const unsigned char* px = row + (size_t)x * 3;  // stored B,G,R
            rr[x] = (px[2] / 255.0f - image_mean_[0]) / image_std_[0];
            gg[x] = (px[1] / 255.0f - image_mean_[1]) / image_std_[1];
            bb[x] = (px[0] / 255.0f - image_mean_[2]) / image_std_[2];
        }
    }
    return img;
}

ncnn::Mat ncnn_llm_ocr::run_vision_hunyuan(const ncnn::Mat& image_chw) const {
    ncnn::Mat feats;
    ncnn::Extractor ex = vision_net_->create_extractor();
    ex.input("in0", image_chw);
    ex.extract("out0", feats);
    return feats;
}

std::shared_ptr<ncnn_llm_gpt_ctx> ncnn_llm_ocr::prefill_hunyuan(const std::string& prompt_text,
                                                                const ncnn::Mat& bgr_image) {
    int target_h = 0, target_w = 0;
    ncnn::Mat image_chw = bgr_to_chw_image_hunyuan(bgr_image, target_h, target_w);
    int gh = target_h / patch_size_;
    int gw = target_w / patch_size_;
    int patch_h = gh / spatial_merge_size_;
    int patch_w = gw / spatial_merge_size_;
    int num_vision_tokens = patch_h * (patch_w + 1) + 2;

    ncnn::Mat vision_features = run_vision_hunyuan(image_chw);
    if (vision_features.h != num_vision_tokens) {
        printf("[ncnn_llm_ocr] WARNING: vision token count %d != expected %d (grid %dx%d)\n",
               vision_features.h, num_vision_tokens, gh, gw);
        num_vision_tokens = vision_features.h;
    }
    printf("[ncnn_llm_ocr] hunyuan vision: resized %dx%d, grid %dx%d, vision_tokens=%d\n",
           target_w, target_h, gh, gw, num_vision_tokens);

    // Match HunYuanVLProcessor exactly: BOS, repeated image placeholders,
    // prompt text, then UserEnd. The ViT output replaces every placeholder.
    std::vector<int> text_ids = bpe_->encode(prompt_text, false, false);
    HunyuanOcrPromptLayout layout = build_hunyuan_ocr_prompt_layout(
        bos_id_, image_token_id_, user_end_id_, text_ids, patch_h, patch_w);
    const std::vector<int>& token_ids = layout.token_ids;
    const int first_image_index = layout.first_image_index;
    const int seq_len = (int)token_ids.size();

    // Text embeddings + inject vision features into the image-token slots (in order).
    ncnn::Mat token_embed = llm_run_text_embed(*text_embed_net_, token_ids);
    for (int i = 0; i < num_vision_tokens; i++) {
        float* dst = token_embed.row(first_image_index + i);
        const float* src = vision_features.row(i);
        memcpy(dst, src, hidden_size_ * sizeof(float));
    }

    // Prefill decode with KV cache (empty cache in).
    ncnn::Mat cos_cache, sin_cache;
    generate_hunyuan_xdrope_cos_sin(layout.position_ids.data(), seq_len, head_dim_, xdrope_section_,
                                    rope_theta_, rope_alpha_, cos_cache, sin_cache);

    ncnn::Mat mask(seq_len, seq_len);
    mask.fill(0.0f);
    for (int i = 0; i < seq_len; i++) {
        float* row = mask.row(i);
        for (int j = i + 1; j < seq_len; j++) row[j] = -1e38f;
    }

    KVCache kv_cache;
    ncnn::Mat decode_out = llm_run_decoder_with_kv(*text_decoder_net_, token_embed, mask,
                                                   cos_cache, sin_cache, kv_cache, attn_cnt_, true);
    ncnn::Mat last_hidden = decode_out.row_range(seq_len - 1, 1);
    ncnn::Mat logits = llm_run_lm_head(*lm_head_net_, last_hidden);
    int next_token_id = argmax1d(logits);

    auto ctx = std::make_shared<ncnn_llm_gpt_base_ctx>();
    ctx->kv_cache = std::move(kv_cache);
    ctx->cur_token = next_token_id;
    ctx->position_id = seq_len;
    return ctx;
}

std::shared_ptr<ncnn_llm_gpt_ctx> ncnn_llm_ocr::generate_hunyuan(
    const std::shared_ptr<ncnn_llm_gpt_ctx>& ctx_in,
    const GenerateConfig& cfg,
    std::function<void(const std::string&)> callback) {

    auto ctx = ctx_in->clone();
    std::unordered_set<int> history;
    history.insert(ctx->cur_token);

    for (int step = 0; step < cfg.max_new_tokens; ++step) {
        int tok = ctx->cur_token;
        if (eos_ids_.count(tok)) break;

        // Emit the current token (skip special / added tokens, e.g. box markers).
        bool is_special = (tok >= special_id_begin_) ||
                          (additional_special_id_set_.find(tok) != additional_special_id_set_.end());
        if (!is_special) {
            std::string token_text = bpe_->decode({tok}, true);
            if (!token_text.empty()) callback(token_text);
        }

        // Single-token embedding.
        ncnn::Mat cur_embed = llm_run_text_embed(*text_embed_net_, tok);

        // Generated text tokens: all 4 xdrope axes = current linear position.
        std::vector<int> p1(1, ctx->position_id);
        const std::vector<int> pos4[4] = { p1, p1, p1, p1 };
        ncnn::Mat cos_cache, sin_cache;
        generate_hunyuan_xdrope_cos_sin(pos4, 1, head_dim_, xdrope_section_,
                                        rope_theta_, rope_alpha_, cos_cache, sin_cache);
        ctx->position_id++;

        // Mask [1, kv_len+1] all zeros: the new token attends to all cached positions + itself.
        ncnn::Mat mask(1, ctx->kv_cache[0].first.h + 1);
        mask.fill(0.0f);

        ncnn::Mat decode_out = llm_run_decoder_with_kv(*text_decoder_net_, cur_embed, mask,
                                                       cos_cache, sin_cache, ctx->kv_cache, attn_cnt_, false);
        ncnn::Mat logits = llm_run_lm_head(*lm_head_net_, decode_out);

        LlmTokenSampleConfig sample_cfg;
        sample_cfg.vocab_size = vocab_size_;
        sample_cfg.temperature = cfg.temperature;
        sample_cfg.top_p = cfg.top_p;
        sample_cfg.top_k = cfg.top_k;
        sample_cfg.repetition_penalty = cfg.repetition_penalty;
        sample_cfg.do_sample = cfg.do_sample;
        int next_id = llm_select_next_token(logits, history, sample_cfg);

        ctx->cur_token = next_id;
        history.insert(next_id);
    }

    return ctx;
}
