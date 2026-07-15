#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "net.h"
#include "utils/tokenizer/qwen_bpe.h"

struct Qwen3TTSFrontendFiles {
    std::string tokenizer;
    std::string text_embed_param;
    std::string text_embed_bin;
    std::string codec_embed_param;
    std::string codec_embed_bin;
};

class Qwen3TTSFrontend {
public:
    bool load(const Qwen3TTSFrontendFiles& files, int num_threads, bool use_vulkan);
    bool ok() const { return ok_; }

    ncnn::Mat build_customvoice_prompt(const std::string& text,
                                       const std::string& language,
                                       const std::string& speaker);

private:
    ncnn::Mat run_ids(ncnn::Net& net, const std::vector<int>& ids) const;
    static std::string lower_ascii(std::string s);
    static void copy_row(ncnn::Mat& dst, int dst_row, const ncnn::Mat& src, int src_row);
    static void add_row(ncnn::Mat& dst, int dst_row, const ncnn::Mat& src, int src_row);

    bool ok_ = false;
    q3tts::QwenBpe bpe_;
    ncnn::Net text_embed_net_;
    ncnn::Net codec_embed_net_;

    static constexpr int hidden_dim = 1024;
    static constexpr int tts_bos = 151672;
    static constexpr int tts_eos = 151673;
    static constexpr int tts_pad = 151671;
    static constexpr int codec_pad = 2148;
    static constexpr int codec_bos = 2149;
    static constexpr int codec_think = 2154;
    static constexpr int codec_think_bos = 2156;
    static constexpr int codec_think_eos = 2157;

    std::unordered_map<std::string, int> language_ids_ = {
        {"chinese", 2055},
        {"english", 2050},
        {"german", 2053},
        {"italian", 2070},
        {"portuguese", 2071},
        {"spanish", 2054},
        {"japanese", 2058},
        {"korean", 2064},
        {"french", 2061},
        {"russian", 2069},
        {"beijing_dialect", 2074},
        {"sichuan_dialect", 2062},
    };

    std::unordered_map<std::string, int> speaker_ids_ = {
        {"uncle_fu", 3010},
        {"vivian", 3065},
        {"ryan", 3061},
        {"aiden", 2861},
        {"ono_anna", 2873},
        {"sohee", 2864},
        {"eric", 2875},
        {"dylan", 2878},
        {"serena", 3066},
    };

    std::unordered_map<std::string, std::string> speaker_dialects_ = {
        {"eric", "sichuan_dialect"},
        {"dylan", "beijing_dialect"},
    };
};
