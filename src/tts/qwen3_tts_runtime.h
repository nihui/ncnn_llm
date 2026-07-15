#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "net.h"

class Qwen3TTSNcnn {
public:
    struct Options {
        int num_threads = 4;
        int decoder_num_threads = 0;
        bool use_vulkan = false;
        bool codepred_use_vulkan = true;
        bool codepred_use_fp16 = false;
        int codepred_fp16_mode = 0;
        int decoder_chunk_frames = 325;
        int decoder_context_frames = 25;
    };

    struct TalkerFiles {
        std::string prefill_param;
        std::string decode_param;
        std::string talker_bin;
        std::string codepred_body_dir;
        std::string codepred_kv_dir;
        std::string codepred_weights_dir;
        std::string tts_pad_embed;
    };

    Qwen3TTSNcnn(const std::string& front_weights_dir,
                 const std::string& decoder_param,
                 const std::string& decoder_bin,
                 const Options& opt);

    bool ok() const { return ok_; }
    bool talker_ok() const { return talker_ok_; }

    bool load_talker(const TalkerFiles& files);

    std::vector<int32_t> generate_codes(const std::string& prefill_embed,
                                        const std::string& prefill_mask,
                                        const std::string& prefill_cos,
                                        const std::string& prefill_sin,
                                        int frames) const;

    std::vector<int32_t> generate_codes_from_prompt(const ncnn::Mat& prefill_embed,
                                                    int frames) const;

    std::vector<float> codes_to_hidden(const std::vector<int32_t>& codes, int frames) const;

    std::vector<float> decode_hidden_chunk(const std::vector<float>& hidden,
                                           int hidden_frames,
                                           int context_frames,
                                           int current_frames) const;

    std::vector<float> codes_to_wav(const std::vector<int32_t>& codes,
                                    int frames,
                                    int context_frames,
                                    int current_frames) const;

    static constexpr int codebooks = 16;
    static constexpr int hidden_dim = 1024;
    static constexpr int samples_per_frame = 1920;

private:
    bool load_front_weights(const std::string& dir);
    bool load_codepred_weights(const std::string& dir);

    Options opt_;
    bool ok_ = false;
    bool talker_ok_ = false;

    std::vector<float> first_emb_;
    std::vector<std::vector<float>> rest_emb_;
    std::vector<float> first_proj_;
    std::vector<float> rest_proj_;
    std::vector<float> pre_conv_w_;
    std::vector<float> pre_conv_b_;

    std::shared_ptr<ncnn::Net> decoder_net_;

    TalkerFiles talker_files_;
    std::shared_ptr<ncnn::Net> talker_prefill_net_;
    std::shared_ptr<ncnn::Net> talker_decode_net_;
    std::vector<std::shared_ptr<ncnn::Net>> codepred_body_nets_;
    std::shared_ptr<ncnn::Net> codepred_kv_prefill_net_;
    std::vector<std::shared_ptr<ncnn::Net>> codepred_kv_decode_nets_;
    std::shared_ptr<ncnn::Net> codepred_kv_shared_prefill_body_net_;
    std::shared_ptr<ncnn::Net> codepred_kv_shared_decode_body_net_;
    std::vector<std::shared_ptr<ncnn::Net>> codepred_kv_shared_head_nets_;
    bool codepred_kv_shared_ok_ = false;
    bool codepred_kv_ok_ = false;
    std::vector<ncnn::Mat> codepred_masks_;
    std::vector<ncnn::Mat> codepred_positions_;
    std::vector<float> talker_codec_embedding_;
    std::vector<std::vector<float>> codepred_heads_;
    std::vector<std::vector<float>> codepred_embs_;
    std::vector<float> tts_pad_embed_;
};
