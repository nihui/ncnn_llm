// SPDX-License-Identifier: Apache-2.0
#include "penguin_vl.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "ncnn_llm_base.h"
#include "image_utils.h"
#include "image_preprocess.h"
#include "rope_embed.h"
#include "sampling.h"

namespace pvl {

// ---- decoder net I/O contract (must match the exported decoder graph) --------
//   inputs : in0 = hidden (hidden, seq)
//            in1 = attention mask (kv_len+seq, seq)
//            in2 = rope cos (rope_head_dim/2, seq)   [engine convention]
//            in3 = rope sin
//            cache_k{i} / cache_v{i}  (omitted on the first prefill pass)
//   outputs: out_cache_k{i} / out_cache_v{i}
//            out0 = last hidden state (hidden, seq)
// This mirrors the nihui/futz12 ncnn_llm KV-cache decoder layout.

static ncnn::Mat make_causal_mask(int seq, int past) {
    ncnn::Mat mask(past + seq, seq);
    mask.fill(0.0f);
    for (int i = 0; i < seq; ++i) {
        float* row = mask.row(i);
        for (int j = past + i + 1; j < past + seq; ++j) row[j] = -1e38f;
    }
    return mask;
}

PenguinVL::PenguinVL(const std::string& model_dir, int num_threads, bool use_vulkan)
    : model_dir_(model_dir), num_threads_(num_threads), use_vulkan_(use_vulkan) {
    cfg_ = PenguinConfig::load(model_dir);

    auto load_net = [&](std::shared_ptr<ncnn::Net>& net, const std::string& param,
                        const std::string& bin) {
        net = std::make_shared<ncnn::Net>();
        if (num_threads_ > 0) net->opt.num_threads = num_threads_;
        net->opt.use_vulkan_compute = use_vulkan_;
        if (net->load_param((model_dir_ + "/" + param).c_str()) != 0)
            throw std::runtime_error("failed to load param: " + param);
        if (net->load_model((model_dir_ + "/" + bin).c_str()) != 0)
            throw std::runtime_error("failed to load bin: " + bin);
    };
    load_net(embed_net_, cfg_.embed_param, cfg_.embed_bin);
    load_net(decoder_net_, cfg_.decoder_param, cfg_.decoder_bin);
    load_net(lm_head_net_, cfg_.lm_head_param, cfg_.lm_head_bin);

    tok_ = std::make_shared<BpeTokenizer>(BpeTokenizer::LoadFromFiles(
        model_dir_ + "/" + cfg_.vocab_file, model_dir_ + "/" + cfg_.merges_file,
        SpecialTokensConfig{}, false, true, cfg_.tokenizer_type == "bbpe"));
    for (const auto& t : cfg_.additional_special_tokens) tok_->AddAdditionalSpecialToken(t);

    const auto& t2i = tok_->token_to_id();
    if (!cfg_.eos_token.empty() && t2i.count(cfg_.eos_token)) eos_id_ = t2i.at(cfg_.eos_token);
    if (!cfg_.bos_token.empty() && t2i.count(cfg_.bos_token)) bos_id_ = t2i.at(cfg_.bos_token);

    if (cfg_.vision.enabled) {
        if (t2i.count(cfg_.vision.image_token))
            image_token_id_ = t2i.at(cfg_.vision.image_token);
        vision_ = std::make_unique<PenguinVision>(model_dir_, cfg_.vision, num_threads_, use_vulkan_);
    }
}

ncnn::Mat PenguinVL::embed_tokens(const std::vector<int>& ids) const {
    ncnn::Mat in((int)ids.size(), 1, (void*)ids.data());
    in = in.clone();
    ncnn::Mat out;
    ncnn::Extractor ex = embed_net_->create_extractor();
    ex.input("in0", in);
    ex.extract("out0", out);
    return out;
}

ncnn::Mat PenguinVL::embed_token(int id) const {
    return embed_tokens(std::vector<int>{id});
}

ncnn::Mat PenguinVL::lm_head(const ncnn::Mat& hidden) const {
    ncnn::Mat logits;
    ncnn::Extractor ex = lm_head_net_->create_extractor();
    ex.input("in0", hidden);
    ex.extract("out0", logits);
    return logits;
}

std::string PenguinVL::build_prompt(const std::string& user_text, bool has_image,
                                    bool add_think) const {
    std::string p;
    if (!cfg_.system_prompt.empty()) {
        p += "<|im_start|>system\n" + cfg_.system_prompt + "<|im_end|>\n";
    }
    p += "<|im_start|>user\n";
    if (has_image) p += cfg_.vision.image_token + "\n";
    p += user_text;
    p += "<|im_end|>\n";
    p += "<|im_start|>assistant\n";
    if (!add_think) p += "<think>\n\n</think>\n\n";
    return p;
}

ncnn::Mat PenguinVL::build_input_embeds(const std::string& prompt,
                                        const ncnn::Mat& image_embeds,
                                        std::vector<int>& token_ids_out) const {
    const int num_visual = image_embeds.empty() ? 0 : image_embeds.h;

    // Split the prompt on the image placeholder token. For a single image there
    // is exactly one boundary at which we insert `num_visual` image-token ids.
    std::vector<int> ids;
    std::vector<int> image_positions;  // indices in `ids` that hold visual tokens

    if (bos_id_ >= 0) ids.push_back(bos_id_);

    const std::string& marker = cfg_.vision.image_token;
    size_t pos = 0, prev = 0;
    auto emit_chunk = [&](const std::string& chunk) {
        if (chunk.empty()) return;
        auto sub = tok_->encode(chunk, false, false);
        ids.insert(ids.end(), sub.begin(), sub.end());
    };

    if (num_visual > 0 && !marker.empty()) {
        while ((pos = prompt.find(marker, prev)) != std::string::npos) {
            emit_chunk(prompt.substr(prev, pos - prev));
            for (int k = 0; k < num_visual; ++k) {
                image_positions.push_back((int)ids.size());
                ids.push_back(image_token_id_ >= 0 ? image_token_id_ : 0);
            }
            prev = pos + marker.size();
        }
        emit_chunk(prompt.substr(prev));
    } else {
        emit_chunk(prompt);
    }

    if ((int)image_positions.size() != num_visual) {
        throw std::runtime_error("image placeholder count does not match visual tokens");
    }

    // Embed all tokens, then overwrite the visual positions with encoder features.
    ncnn::Mat embeds = embed_tokens(ids);
    const int hidden = embeds.w;
    for (int j = 0; j < num_visual; ++j) {
        const float* src = image_embeds.row(j);
        float* dst = embeds.row(image_positions[j]);
        std::memcpy(dst, src, hidden * sizeof(float));
    }

    token_ids_out = std::move(ids);
    return embeds;
}

std::string PenguinVL::chat(const std::string& user_text, const std::string& image_path,
                            const GenerateConfig& cfg,
                            const std::function<void(const std::string&)>& on_token) {
    const bool has_image = !image_path.empty();

    ncnn::Mat image_embeds;
    if (has_image) {
        if (!vision_) throw std::runtime_error("model has no vision encoder but an image was given");
        ncnn::Mat bgr = load_image_to_ncnn_mat(image_path);
        if (ncnn_mat_empty(bgr)) throw std::runtime_error("failed to load image: " + image_path);
        std::fprintf(stderr, "[penguin-vl] image %dx%d\n", bgr.w, bgr.h);
        PreprocessResult pp = preprocess_image(bgr, cfg_.vision);
        std::fprintf(stderr, "[penguin-vl] visual grid %dx%d (%d patches)\n",
                     pp.grid_h, pp.grid_w, pp.pixel_values.h);
        image_embeds = vision_->encode(pp.pixel_values, pp.grid_h, pp.grid_w);
        std::fprintf(stderr, "[penguin-vl] encoded %d visual tokens\n", image_embeds.h);
    }

    const std::string prompt = build_prompt(user_text, has_image, cfg.add_think);
    std::vector<int> token_ids;
    ncnn::Mat token_embed = build_input_embeds(prompt, image_embeds, token_ids);
    std::fprintf(stderr, "[penguin-vl] decoding %d prompt tokens\n", token_embed.h);
    std::fflush(stderr);

    // ---- cacheless generation path ----
    // The decoder subgraph exposes only (in0..in3)->out0 and reprocesses the full
    // running sequence every step. This avoids pnnx/ncnn KV-cache concat limits at
    // the cost of O(n^2) compute; correct and fully self-contained.
    if (!cfg_.kv_cache) {
        const int head_dim = cfg_.rope_head_dim;
        const float theta = cfg_.rope_theta;
        const int H = token_embed.w;
        ncnn::Mat running = token_embed.clone();  // (H, n)
        std::string output;
        for (int step = 0; step < cfg.max_new_tokens; ++step) {
            const int n = running.h;
            // 2D causal mask (w=kv, h=seq); the graph reshapes it to (kv,seq,1)
            // internally to broadcast over attention heads.
            ncnn::Mat mask = make_causal_mask(n, 0);
            ncnn::Mat cos_c, sin_c;
            generate_rope_embed_cache_full(n, head_dim, 0, cos_c, sin_c, theta);

            ncnn::Mat hidden_all;
            {
                ncnn::Extractor ex = decoder_net_->create_extractor();
                ex.input("in0", running);
                ex.input("in1", mask);
                ex.input("in2", cos_c);
                ex.input("in3", sin_c);
                ex.extract("out0", hidden_all);  // (H, n)
            }
            ncnn::Mat last = hidden_all.row_range(n - 1, 1).clone();  // (H, 1)
            ncnn::Mat logits = lm_head(last);

            int next;
            if (!cfg.do_sample) {
                next = argmax1d(logits);
            } else {
                std::vector<float> probs(logits.w);
                const float* p = logits;
                for (int i = 0; i < logits.w; ++i) probs[i] = p[i];
                softmax_vec(probs, cfg.temperature);
                if (cfg.top_k > 0) apply_top_k(probs, cfg.top_k);
                if (cfg.top_p < 1.0f) apply_top_p(probs, cfg.top_p);
                next = sample_from_probs(probs);
            }
            if (next == eos_id_) break;

            std::string piece = tok_->decode({next}, /*skip_special_tokens=*/false);
            output += piece;
            if (on_token) on_token(piece);

            ncnn::Mat next_embed = embed_token(next);  // (H, 1)
            ncnn::Mat grown(H, n + 1);
            std::memcpy(grown.row(0), running.row(0), (size_t)H * n * sizeof(float));
            std::memcpy(grown.row(n), next_embed.row(0), (size_t)H * sizeof(float));
            running = grown;
        }
        return output;
    }

    const int total = (int)token_ids.size();
    if (total < 1) throw std::runtime_error("empty prompt");
    const int m = total - 1;  // prefill all but the last token, then step the last one

    const int head_dim = cfg_.rope_head_dim;
    const float theta = cfg_.rope_theta;
    const int L = cfg_.num_layers;

    KVCache kv;

    // ---- prefill pass A: first m tokens (no existing cache) ----
    // The exported decoder graph always concatenates cache_k{i}; ncnn's Concat
    // propagates emptiness, so a truly empty cache yields an empty output. We
    // therefore prime the concat with a single zero "dummy" cache slot (c=1),
    // mask it out in attention, and strip it from the extracted cache so the
    // downstream passes see a clean past of exactly m. Cache mat layout is
    // (w=head_dim, h=kv_heads, c=past).
    if (m > 0) {
        ncnn::Mat embeds_a = token_embed.row_range(0, m).clone();
        // mask over kv_len = 1 (dummy) + m, for m query rows. Column 0 is the
        // dummy (always masked); column 1+j is real token j (causal: row i
        // attends j <= i).
        ncnn::Mat mask(1 + m, m);
        for (int i = 0; i < m; ++i) {
            float* row = mask.row(i);
            row[0] = -1e38f;
            for (int j = 0; j < m; ++j) row[1 + j] = (j <= i) ? 0.0f : -1e38f;
        }
        ncnn::Mat cos_c, sin_c;
        generate_rope_embed_cache_full(m, head_dim, 0, cos_c, sin_c, theta);

        ncnn::Extractor ex = decoder_net_->create_extractor();
        ex.input("in0", embeds_a);
        ex.input("in1", mask);
        ex.input("in2", cos_c);
        ex.input("in3", sin_c);
        for (int i = 0; i < L; ++i) {
            char ki[24], vi[24];
            std::snprintf(ki, sizeof(ki), "cache_k%d", i);
            std::snprintf(vi, sizeof(vi), "cache_v%d", i);
            ncnn::Mat dummy(head_dim, cfg_.kv_heads, 1);
            dummy.fill(0.0f);
            ex.input(ki, dummy);
            ex.input(vi, dummy);
        }
        for (int i = 0; i < L; ++i) {
            char ko[24], vo[24];
            std::snprintf(ko, sizeof(ko), "out_cache_k%d", i);
            std::snprintf(vo, sizeof(vo), "out_cache_v%d", i);
            ncnn::Mat k, v;
            ex.extract(ko, k);
            ex.extract(vo, v);
            // Drop the leading dummy cache slot (channel 0) -> clean past = m.
            kv.emplace_back(k.channel_range(1, k.c - 1).clone(),
                            v.channel_range(1, v.c - 1).clone());
        }
    } else {
        kv.resize(L);
    }

    // ---- prefill pass B: the last prompt token -> first predicted token ----
    int cur_token;
    {
        ncnn::Mat embeds_b = token_embed.row_range(m, 1).clone();
        const int past = m;
        ncnn::Mat mask(past + 1, 1);
        mask.fill(0.0f);
        ncnn::Mat cos_c, sin_c;
        generate_rope_embed_cache_full(1, head_dim, past, cos_c, sin_c, theta);

        ncnn::Extractor ex = decoder_net_->create_extractor();
        ex.input("in0", embeds_b);
        ex.input("in1", mask);
        ex.input("in2", cos_c);
        ex.input("in3", sin_c);
        if (m > 0) {
            for (int i = 0; i < L; ++i) {
                char ki[24], vi[24];
                std::snprintf(ki, sizeof(ki), "cache_k%d", i);
                std::snprintf(vi, sizeof(vi), "cache_v%d", i);
                ex.input(ki, kv[i].first);
                ex.input(vi, kv[i].second);
            }
        }
        for (int i = 0; i < L; ++i) {
            char ko[24], vo[24];
            std::snprintf(ko, sizeof(ko), "out_cache_k%d", i);
            std::snprintf(vo, sizeof(vo), "out_cache_v%d", i);
            ncnn::Mat k, v;
            ex.extract(ko, k);
            ex.extract(vo, v);
            kv[i] = {std::move(k), std::move(v)};
        }
        ncnn::Mat hidden;
        ex.extract("out0", hidden);
        ncnn::Mat logits = lm_head(hidden);
        cur_token = argmax1d(logits);
    }

    // ---- autoregressive decode ----
    std::string output;
    int position = total;  // next position id
    for (int step = 0; step < cfg.max_new_tokens; ++step) {
        if (cur_token == eos_id_) break;

        std::string piece = tok_->decode({cur_token}, /*skip_special_tokens=*/false);
        output += piece;
        if (on_token) on_token(piece);

        ncnn::Mat cur_embed = embed_token(cur_token);
        ncnn::Mat cos_c, sin_c;
        generate_rope_embed_cache_full(1, head_dim, position, cos_c, sin_c, theta);
        ++position;

        const int past = kv[0].first.c;  // cache seq length lives on the c axis
        ncnn::Mat mask(past + 1, 1);
        mask.fill(0.0f);

        ncnn::Mat hidden;
        {
            ncnn::Extractor ex = decoder_net_->create_extractor();
            ex.input("in0", cur_embed);
            ex.input("in1", mask);
            ex.input("in2", cos_c);
            ex.input("in3", sin_c);
            for (int i = 0; i < L; ++i) {
                char ki[24], vi[24];
                std::snprintf(ki, sizeof(ki), "cache_k%d", i);
                std::snprintf(vi, sizeof(vi), "cache_v%d", i);
                ex.input(ki, kv[i].first);
                ex.input(vi, kv[i].second);
            }
            for (int i = 0; i < L; ++i) {
                char ko[24], vo[24];
                std::snprintf(ko, sizeof(ko), "out_cache_k%d", i);
                std::snprintf(vo, sizeof(vo), "out_cache_v%d", i);
                ncnn::Mat k, v;
                ex.extract(ko, k);
                ex.extract(vo, v);
                kv[i] = {std::move(k), std::move(v)};
            }
            ex.extract("out0", hidden);
        }

        ncnn::Mat logits = lm_head(hidden);
        int next;
        if (!cfg.do_sample) {
            next = argmax1d(logits);
        } else {
            std::vector<float> probs(logits.w);
            const float* p = logits;
            for (int i = 0; i < logits.w; ++i) probs[i] = p[i];
            softmax_vec(probs, cfg.temperature);
            if (cfg.top_k > 0) apply_top_k(probs, cfg.top_k);
            if (cfg.top_p < 1.0f) apply_top_p(probs, cfg.top_p);
            next = sample_from_probs(probs);
        }
        cur_token = next;
    }

    return output;
}

}  // namespace pvl
