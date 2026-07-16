// SPDX-License-Identifier: Apache-2.0
#include "penguin_config.h"

#include <stdexcept>

#include "json_min.h"

namespace pvl {

PenguinConfig PenguinConfig::load(const std::string& model_dir) {
    Json root = Json::parse_file(model_dir + "/model.json");

    PenguinConfig cfg;
    cfg.model_type = root.get_string("model_type", cfg.model_type);

    const Json& params = root["params"];
    cfg.embed_param = params["embed_token_param"].as_string();
    cfg.embed_bin = params["embed_token_bin"].as_string();
    cfg.decoder_param = params["decoder_param"].as_string();
    cfg.decoder_bin = params["decoder_bin"].as_string();
    cfg.lm_head_param = params["lm_head_param"].as_string();
    cfg.lm_head_bin = params["lm_head_bin"].as_string();

    const Json& tok = root["tokenizer"];
    cfg.tokenizer_type = tok.get_string("type", cfg.tokenizer_type);
    cfg.vocab_file = tok["vocab_file"].as_string();
    cfg.merges_file = tok["merges_file"].as_string();
    cfg.bos_token = tok.get_string("bos", "");
    cfg.eos_token = tok.get_string("eos", cfg.eos_token);
    if (tok.contains("additional_special_tokens"))
        cfg.additional_special_tokens = tok["additional_special_tokens"].as_string_vector();

    const Json& setting = root["setting"];
    cfg.hidden_size = setting.get_int("hidden_size", cfg.hidden_size);
    cfg.num_layers = setting.get_int("attn_cnt", cfg.num_layers);
    cfg.kv_heads = setting.get_int("kv_heads", cfg.kv_heads);
    cfg.kv_cache = setting.get_bool("kv_cache", cfg.kv_cache);
    cfg.system_prompt = setting.get_string("system_prompt", cfg.system_prompt);

    if (setting.contains("rope")) {
        const Json& rope = setting["rope"];
        cfg.rope_head_dim = rope.get_int("rope_head_dim", cfg.rope_head_dim);
        cfg.rope_theta = rope.get_float("rope_theta", cfg.rope_theta);
    }

    if (setting.contains("vision")) {
        const Json& v = setting["vision"];
        VisionConfig& vc = cfg.vision;
        vc.enabled = true;
        vc.patch_embed_param = v["patch_embed_param"].as_string();
        vc.patch_embed_bin = v["patch_embed_bin"].as_string();
        vc.encoder_param = v["encoder_param"].as_string();
        vc.encoder_bin = v["encoder_bin"].as_string();
        vc.projector_param = v["projector_param"].as_string();
        vc.projector_bin = v["projector_bin"].as_string();

        vc.patch_size = v.get_int("patch_size", vc.patch_size);
        vc.merge_size = v.get_int("merge_size", vc.merge_size);
        vc.min_tokens = v.get_int("min_tokens", vc.min_tokens);
        vc.max_tokens = v.get_int("max_tokens", vc.max_tokens);
        vc.vision_hidden = v.get_int("vision_hidden", vc.vision_hidden);
        vc.vision_head_dim = v.get_int("vision_head_dim", vc.vision_head_dim);
        vc.vision_rope_theta = v.get_float("vision_rope_theta", vc.vision_rope_theta);
        vc.image_token = v.get_string("image_token", vc.image_token);

        if (v.contains("image_mean")) {
            auto m = v["image_mean"].as_float_vector();
            if (m.size() != 3) throw std::runtime_error("model.json: image_mean must have 3 values");
            for (int k = 0; k < 3; ++k) vc.image_mean[k] = m[k];
        }
        if (v.contains("image_std")) {
            auto st = v["image_std"].as_float_vector();
            if (st.size() != 3) throw std::runtime_error("model.json: image_std must have 3 values");
            for (int k = 0; k < 3; ++k) vc.image_std[k] = st[k];
        }
    }

    return cfg;
}

}  // namespace pvl
