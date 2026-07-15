#include "qwen3_tts_frontend.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>

static ncnn::Mat ids_mat(const std::vector<int>& ids)
{
    ncnn::Mat m((int)ids.size(), (size_t)4u, 1);
    std::memcpy(m.data, ids.data(), ids.size() * sizeof(int));
    return m;
}

std::string Qwen3TTSFrontend::lower_ascii(std::string s)
{
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

void Qwen3TTSFrontend::copy_row(ncnn::Mat& dst, int dst_row, const ncnn::Mat& src, int src_row)
{
    std::memcpy(dst.row(dst_row), src.row(src_row), src.w * sizeof(float));
}

void Qwen3TTSFrontend::add_row(ncnn::Mat& dst, int dst_row, const ncnn::Mat& src, int src_row)
{
    float* d = dst.row(dst_row);
    const float* s = src.row(src_row);
    for (int i = 0; i < src.w; i++) d[i] += s[i];
}

bool Qwen3TTSFrontend::load(const Qwen3TTSFrontendFiles& files, int num_threads, bool use_vulkan)
{
    try
    {
        bpe_.load(files.tokenizer);
    }
    catch (...)
    {
        std::fprintf(stderr, "failed to load tokenizer %s\n", files.tokenizer.c_str());
        return false;
    }

    for (ncnn::Net* net : {&text_embed_net_, &codec_embed_net_})
    {
        net->opt.num_threads = std::max(1, num_threads);
        net->opt.use_vulkan_compute = use_vulkan;
        net->opt.use_packing_layout = false;
        net->opt.use_fp16_packed = false;
        net->opt.use_fp16_storage = false;
        net->opt.use_fp16_arithmetic = false;
    }

    if (text_embed_net_.load_param(files.text_embed_param.c_str()) != 0 ||
        text_embed_net_.load_model(files.text_embed_bin.c_str()) != 0 ||
        codec_embed_net_.load_param(files.codec_embed_param.c_str()) != 0 ||
        codec_embed_net_.load_model(files.codec_embed_bin.c_str()) != 0)
    {
        std::fprintf(stderr, "failed to load frontend ncnn nets\n");
        return false;
    }

    ok_ = true;
    return true;
}

ncnn::Mat Qwen3TTSFrontend::run_ids(ncnn::Net& net, const std::vector<int>& ids) const
{
    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", ids_mat(ids));
    ncnn::Mat out;
    if (ex.extract("out0", out) != 0) return ncnn::Mat();
    return out;
}

ncnn::Mat Qwen3TTSFrontend::build_customvoice_prompt(const std::string& text,
                                                     const std::string& language,
                                                     const std::string& speaker)
{
    if (!ok_) return ncnn::Mat();

    const std::string lang_key = lower_ascii(language);
    const std::string spk_key = lower_ascii(speaker);
    const auto sit = speaker_ids_.find(spk_key);
    if (sit == speaker_ids_.end())
    {
        std::fprintf(stderr, "unsupported language or speaker: %s / %s\n", language.c_str(), speaker.c_str());
        return ncnn::Mat();
    }
    if (lang_key == "auto")
    {
        std::fprintf(stderr, "language=Auto is not validated in this ncnn frontend package yet; pass an explicit language\n");
        return ncnn::Mat();
    }
    auto lit = language_ids_.find(lang_key);
    if (lang_key == "chinese")
    {
        const auto dit = speaker_dialects_.find(spk_key);
        if (dit != speaker_dialects_.end()) lit = language_ids_.find(dit->second);
    }
    else if (lit == language_ids_.end())
    {
        std::fprintf(stderr, "unsupported language or speaker: %s / %s\n", language.c_str(), speaker.c_str());
        return ncnn::Mat();
    }

    const std::string templ = "<|im_start|>assistant\n" + text + "<|im_end|>\n<|im_start|>assistant\n";
    std::vector<int> ids = bpe_.encode(templ);
    if (ids.size() < 9)
    {
        std::fprintf(stderr, "tokenized prompt too short\n");
        return ncnn::Mat();
    }

    ncnn::Mat special = run_ids(text_embed_net_, {tts_bos, tts_eos, tts_pad});
    ncnn::Mat codec0 = run_ids(codec_embed_net_, {codec_think, codec_think_bos, lit->second, codec_think_eos});
    ncnn::Mat speaker_embed = run_ids(codec_embed_net_, {sit->second});
    ncnn::Mat codec1 = run_ids(codec_embed_net_, {codec_pad, codec_bos});
    ncnn::Mat role = run_ids(text_embed_net_, {ids[0], ids[1], ids[2]});

    std::vector<int> text_ids(ids.begin() + 3, ids.end() - 5);
    ncnn::Mat text_embed = run_ids(text_embed_net_, text_ids);
    ncnn::Mat codec_pad_embed = run_ids(codec_embed_net_, {codec_pad});
    ncnn::Mat codec_bos_embed = run_ids(codec_embed_net_, {codec_bos});

    if (special.empty() || codec0.empty() || speaker_embed.empty() || codec1.empty() ||
        role.empty() || text_embed.empty() || codec_pad_embed.empty() || codec_bos_embed.empty())
    {
        std::fprintf(stderr, "frontend ncnn extract failed\n");
        return ncnn::Mat();
    }

    const int codec_len = codec0.h + 1 + codec1.h;
    ncnn::Mat codec_input(hidden_dim, codec_len, (size_t)4u, 1);
    for (int r = 0; r < codec0.h; r++) copy_row(codec_input, r, codec0, r);
    copy_row(codec_input, codec0.h, speaker_embed, 0);
    for (int r = 0; r < codec1.h; r++) copy_row(codec_input, codec0.h + 1 + r, codec1, r);

    const int n_text = (int)text_ids.size();
    ncnn::Mat prompt(hidden_dim, 3 + (codec_len - 1) + (n_text + 1) + 1, (size_t)4u, 1);
    for (int r = 0; r < 3; r++) copy_row(prompt, r, role, r);

    for (int r = 0; r < codec_len - 1; r++)
    {
        copy_row(prompt, 3 + r, special, r == codec_len - 2 ? 0 : 2);
        add_row(prompt, 3 + r, codec_input, r);
    }

    const int base = 3 + codec_len - 1;
    for (int r = 0; r < n_text + 1; r++)
    {
        if (r < n_text) copy_row(prompt, base + r, text_embed, r);
        else copy_row(prompt, base + r, special, 1);
        add_row(prompt, base + r, codec_pad_embed, 0);
    }
    copy_row(prompt, base + n_text + 1, special, 2);
    add_row(prompt, base + n_text + 1, codec_bos_embed, 0);

    return prompt;
}
