#include "qwen3_tts_runtime.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

static bool read_file(const std::string& path, std::vector<char>& data)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return false;
    ifs.seekg(0, std::ios::end);
    const std::streamoff size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    data.resize((size_t)size);
    if (size > 0) ifs.read(data.data(), size);
    return (bool)ifs;
}

static bool profile_enabled()
{
    const char* p = std::getenv("QWEN3_TTS_PROFILE");
    return p && p[0] && p[0] != '0';
}

static bool codepred_step_profile_enabled()
{
    const char* p = std::getenv("QWEN3_TTS_CODEPRED_STEP_PROFILE");
    return p && p[0] && p[0] != '0';
}

static double now_ms()
{
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

struct RuntimeProfile {
    double prefill_ms = 0.0;
    double codepred_total_ms = 0.0;
    double codepred_body_ms = 0.0;
    double codepred_head_ms = 0.0;
    double talker_decode_ms = 0.0;
    double codec_front_ms = 0.0;
    double speech_decoder_ms = 0.0;
    int frames = 0;
    int codepred_steps = 0;
    int talker_decode_steps = 0;
};

static bool read_f32_exact(const std::string& path, size_t count, std::vector<float>& out)
{
    std::vector<char> bytes;
    if (!read_file(path, bytes) || bytes.size() != count * sizeof(float))
    {
        std::fprintf(stderr, "bad f32 file %s: got %zu expected %zu\n", path.c_str(), bytes.size(), count * sizeof(float));
        return false;
    }
    out.resize(count);
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return true;
}

static bool read_f32_any(const std::string& path, std::vector<float>& out)
{
    std::vector<char> bytes;
    if (!read_file(path, bytes) || bytes.size() % sizeof(float) != 0) return false;
    out.resize(bytes.size() / sizeof(float));
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return true;
}

static size_t f32_count(const std::string& path)
{
    std::vector<char> bytes;
    if (!read_file(path, bytes) || bytes.size() % sizeof(float) != 0) return 0;
    return bytes.size() / sizeof(float);
}

static bool file_exists(const std::string& path)
{
    std::ifstream ifs(path, std::ios::binary);
    return (bool)ifs;
}

static ncnn::Mat mat2_from_vec(const std::vector<float>& v, int w, int h)
{
    ncnn::Mat m(w, h, (size_t)4u, 1);
    std::memcpy(m.data, v.data(), (size_t)w * h * sizeof(float));
    return m;
}

static ncnn::Mat mat2_from_ptr(const float* p, int w, int h)
{
    ncnn::Mat m(w, h, (size_t)4u, 1);
    std::memcpy(m.data, p, (size_t)w * h * sizeof(float));
    return m;
}

static void configure_net_options(ncnn::Net& net, const Qwen3TTSNcnn::Options& opt)
{
    net.opt.num_threads = std::max(1, opt.num_threads);
    net.opt.use_vulkan_compute = opt.use_vulkan;
    const char* packing = std::getenv("QWEN3_TTS_PACKING");
    net.opt.use_packing_layout = packing && packing[0] && packing[0] != '0';
    const char* fp16 = std::getenv("QWEN3_TTS_FP16");
    const bool use_fp16 = fp16 && fp16[0] && fp16[0] != '0';
    net.opt.use_fp16_packed = use_fp16;
    net.opt.use_fp16_storage = use_fp16;
    net.opt.use_fp16_arithmetic = use_fp16;
}

static void configure_codepred_net_options(ncnn::Net& net, const Qwen3TTSNcnn::Options& opt)
{
    configure_net_options(net, opt);
    if (opt.codepred_use_fp16)
    {
        const int mode = opt.codepred_fp16_mode > 0 ? opt.codepred_fp16_mode : 3;
        net.opt.use_fp16_packed = true;
        net.opt.use_fp16_storage = mode >= 2;
        net.opt.use_fp16_arithmetic = mode >= 3;
    }
}

static ncnn::Mat mat2_from_file_any(const std::string& path, int w, int h)
{
    std::vector<float> v;
    if (!read_f32_any(path, v) || v.size() != (size_t)w * h)
    {
        std::fprintf(stderr, "bad f32 file %s\n", path.c_str());
        return ncnn::Mat();
    }
    return mat2_from_vec(v, w, h);
}

static ncnn::Mat make_decode_mask(int kv_len)
{
    ncnn::Mat m(kv_len + 1, 1, (size_t)4u, 1);
    m.fill(0.0f);
    return m;
}

static ncnn::Mat make_prefill_mask(int seq_len)
{
    ncnn::Mat m(seq_len, seq_len, (size_t)4u, 1);
    m.fill(0.0f);
    for (int i = 0; i < seq_len; i++)
    {
        float* p = m.row(i);
        for (int j = i + 1; j < seq_len; j++) p[j] = -INFINITY;
    }
    return m;
}

static void make_rope_table(int seq_len, ncnn::Mat& cos, ncnn::Mat& sin)
{
    const int dim = 128;
    const int half = 64;
    cos.create(dim, seq_len, (size_t)4u, 1);
    sin.create(dim, seq_len, (size_t)4u, 1);
    for (int t = 0; t < seq_len; t++)
    {
        float* cp = cos.row(t);
        float* sp = sin.row(t);
        for (int i = 0; i < half; i++)
        {
            const float inv = std::pow(1000000.0f, -2.0f * i / (float)dim);
            const float a = t * inv;
            const float c = std::cos(a);
            const float s = std::sin(a);
            cp[i] = c;
            cp[i + half] = c;
            sp[i] = s;
            sp[i + half] = s;
        }
    }
}

static void make_rope(int position, ncnn::Mat& cos, ncnn::Mat& sin)
{
    const int dim = 128;
    const int half = 64;
    cos.create(dim, 1, (size_t)4u, 1);
    sin.create(dim, 1, (size_t)4u, 1);
    float* cp = cos;
    float* sp = sin;
    for (int i = 0; i < half; i++)
    {
        const float inv = std::pow(1000000.0f, -2.0f * i / (float)dim);
        const float a = position * inv;
        const float c = std::cos(a);
        const float s = std::sin(a);
        cp[i] = c;
        cp[i + half] = c;
        sp[i] = s;
        sp[i + half] = s;
    }
}

static int argmax_mat(const ncnn::Mat& m, int width)
{
    const float* p = (const float*)m.data;
    int best = 0;
    float bestv = p[0];
    for (int i = 1; i < width; i++)
    {
        if (p[i] > bestv)
        {
            bestv = p[i];
            best = i;
        }
    }
    return best;
}

static int argmax_head(const float* hidden, const std::vector<float>& head, int vocab, int dim)
{
    int best = 0;
    float bestv = -3.4e38f;
    for (int v = 0; v < vocab; v++)
    {
        const float* w = head.data() + (size_t)v * dim;
        float acc0 = 0.f;
        float acc1 = 0.f;
        float acc2 = 0.f;
        float acc3 = 0.f;
        int i = 0;
        for (; i + 3 < dim; i += 4)
        {
            acc0 += hidden[i] * w[i];
            acc1 += hidden[i + 1] * w[i + 1];
            acc2 += hidden[i + 2] * w[i + 2];
            acc3 += hidden[i + 3] * w[i + 3];
        }
        float fv = (acc0 + acc1) + (acc2 + acc3);
        for (; i < dim; i++) fv += hidden[i] * w[i];
        if (fv > bestv)
        {
            bestv = fv;
            best = v;
        }
    }
    return best;
}

static void append_embedding(std::vector<float>& seq, const std::vector<float>& emb, int token, int dim)
{
    const float* row = emb.data() + (size_t)token * dim;
    seq.insert(seq.end(), row, row + dim);
}

static ncnn::Mat codepred_mask(int seq_len)
{
    ncnn::Mat m(seq_len, seq_len, (size_t)4u, 1);
    float* p = m;
    const float neg = -3.4028234663852886e38f;
    for (int y = 0; y < seq_len; y++)
        for (int x = 0; x < seq_len; x++)
            p[y * seq_len + x] = x > y ? neg : 0.0f;
    return m;
}

static ncnn::Mat codepred_pos(int seq_len)
{
    ncnn::Mat m(seq_len, (size_t)4u, 1);
    float* p = m;
    for (int i = 0; i < seq_len; i++) p[i] = (float)i;
    return m;
}

static ncnn::Mat codepred_decode_mask_kv(int kv_len)
{
    ncnn::Mat m(kv_len + 1, 1, (size_t)4u, 1);
    m.fill(0.0f);
    return m;
}

static const std::array<ncnn::Mat, 15>& codepred_kv_masks()
{
    static const std::array<ncnn::Mat, 15> masks = [] {
        std::array<ncnn::Mat, 15> v;
        v[0] = codepred_mask(2);
        for (int step = 1; step <= 14; step++) v[step] = codepred_decode_mask_kv(step + 1);
        return v;
    }();
    return masks;
}

static const std::array<ncnn::Mat, 15>& codepred_kv_positions()
{
    static const std::array<ncnn::Mat, 15> positions = [] {
        std::array<ncnn::Mat, 15> v;
        v[0] = codepred_pos(2);
        for (int step = 1; step <= 14; step++)
        {
            v[step] = ncnn::Mat(1, (size_t)4u, 1);
            ((float*)v[step].data)[0] = (float)(step + 1);
        }
        return v;
    }();
    return positions;
}

struct CodepredKvCacheNames {
    std::array<std::string, 5> in_k;
    std::array<std::string, 5> in_v;
    std::array<std::string, 5> out_k;
    std::array<std::string, 5> out_v;
};

static const CodepredKvCacheNames& codepred_kv_cache_names()
{
    static const CodepredKvCacheNames names = [] {
        CodepredKvCacheNames n;
        for (int i = 0; i < 5; i++)
        {
            n.in_k[i] = "cache_k" + std::to_string(i);
            n.in_v[i] = "cache_v" + std::to_string(i);
            n.out_k[i] = "out_cache_k" + std::to_string(i);
            n.out_v[i] = "out_cache_v" + std::to_string(i);
        }
        return n;
    }();
    return names;
}

Qwen3TTSNcnn::Qwen3TTSNcnn(const std::string& front_weights_dir,
                           const std::string& decoder_param,
                           const std::string& decoder_bin,
                           const Options& opt)
    : opt_(opt)
{
    if (!load_front_weights(front_weights_dir)) return;

    decoder_net_ = std::make_shared<ncnn::Net>();
    Options decoder_opt = opt_;
    if (decoder_opt.decoder_num_threads > 0) decoder_opt.num_threads = decoder_opt.decoder_num_threads;
    configure_net_options(*decoder_net_, decoder_opt);
    if (decoder_net_->load_param(decoder_param.c_str()) != 0 || decoder_net_->load_model(decoder_bin.c_str()) != 0)
    {
        std::fprintf(stderr, "failed to load decoder graph\n");
        return;
    }

    ok_ = true;
}

bool Qwen3TTSNcnn::load_front_weights(const std::string& dir)
{
    constexpr int vocab = 2048;
    constexpr int emb = 256;
    constexpr int h = 512;
    constexpr int k = 3;

    rest_emb_.resize(15);
    if (!read_f32_exact(dir + "/first_embedding.f32.bin", (size_t)vocab * emb, first_emb_) ||
        !read_f32_exact(dir + "/first_output_proj_weight.f32.bin", (size_t)h * emb, first_proj_) ||
        !read_f32_exact(dir + "/rest_output_proj_weight.f32.bin", (size_t)h * emb, rest_proj_) ||
        !read_f32_exact(dir + "/pre_conv_weight.f32.bin", (size_t)hidden_dim * h * k, pre_conv_w_) ||
        !read_f32_exact(dir + "/pre_conv_bias.f32.bin", hidden_dim, pre_conv_b_))
    {
        return false;
    }
    for (int i = 0; i < 15; i++)
    {
        if (!read_f32_exact(dir + "/rest_embedding_" + std::to_string(i) + ".f32.bin", (size_t)vocab * emb, rest_emb_[i]))
            return false;
    }
    return true;
}

bool Qwen3TTSNcnn::load_codepred_weights(const std::string& dir)
{
    constexpr int dim = 1024;
    codepred_heads_.assign(15, {});
    codepred_embs_.assign(15, {});
    if (!read_f32_exact(dir + "/talker_codec_embedding.f32", (size_t)3072 * dim, talker_codec_embedding_))
        return false;
    for (int i = 0; i < 15; i++)
    {
        if (!read_f32_exact(dir + "/lm_head_" + std::to_string(i) + ".f32", (size_t)2048 * dim, codepred_heads_[i]))
            return false;
        if (!read_f32_exact(dir + "/codec_embedding_" + std::to_string(i) + ".f32", (size_t)2048 * dim, codepred_embs_[i]))
            return false;
    }
    return true;
}

bool Qwen3TTSNcnn::load_talker(const TalkerFiles& files)
{
    talker_files_ = files;
    talker_ok_ = false;

    if (!load_codepred_weights(files.codepred_weights_dir)) return false;
    if (!read_f32_exact(files.tts_pad_embed, hidden_dim, tts_pad_embed_)) return false;

    talker_prefill_net_ = std::make_shared<ncnn::Net>();
    talker_decode_net_ = std::make_shared<ncnn::Net>();
    for (ncnn::Net* net : {talker_prefill_net_.get(), talker_decode_net_.get()})
    {
        configure_net_options(*net, opt_);
    }

    if (talker_prefill_net_->load_param(files.prefill_param.c_str()) != 0 ||
        talker_prefill_net_->load_model(files.talker_bin.c_str()) != 0 ||
        talker_decode_net_->load_param(files.decode_param.c_str()) != 0 ||
        talker_decode_net_->load_model(files.talker_bin.c_str()) != 0)
    {
        std::fprintf(stderr, "failed to load talker graphs\n");
        return false;
    }

    codepred_body_nets_.assign(15, nullptr);
    codepred_kv_prefill_net_.reset();
    codepred_kv_decode_nets_.clear();
    codepred_kv_shared_prefill_body_net_.reset();
    codepred_kv_shared_decode_body_net_.reset();
    codepred_kv_shared_head_nets_.clear();
    codepred_kv_shared_ok_ = false;
    codepred_kv_ok_ = false;
    codepred_masks_.assign(15, ncnn::Mat());
    codepred_positions_.assign(15, ncnn::Mat());
    Options codepred_opt = opt_;
    codepred_opt.use_vulkan = opt_.codepred_use_vulkan;

    if (!files.codepred_kv_dir.empty())
    {
        const std::string shared_bin = files.codepred_kv_dir + "/codepred_body_shared.ncnn.bin";
        const std::string shared_prefill_param = files.codepred_kv_dir + "/codepred_prefill_body_s2.ncnn.param";
        const std::string shared_decode_param = files.codepred_kv_dir + "/codepred_decode_body_s1.ncnn.param";
        const std::string shared_head_param = files.codepred_kv_dir + "/codepred_head.ncnn.param";
        if (file_exists(shared_bin) && file_exists(shared_prefill_param) &&
            file_exists(shared_decode_param) && file_exists(shared_head_param))
        {
            codepred_kv_shared_prefill_body_net_ = std::make_shared<ncnn::Net>();
            codepred_kv_shared_decode_body_net_ = std::make_shared<ncnn::Net>();
            configure_codepred_net_options(*codepred_kv_shared_prefill_body_net_, codepred_opt);
            configure_codepred_net_options(*codepred_kv_shared_decode_body_net_, codepred_opt);
            if (codepred_kv_shared_prefill_body_net_->load_param(shared_prefill_param.c_str()) != 0 ||
                codepred_kv_shared_prefill_body_net_->load_model(shared_bin.c_str()) != 0 ||
                codepred_kv_shared_decode_body_net_->load_param(shared_decode_param.c_str()) != 0 ||
                codepred_kv_shared_decode_body_net_->load_model(shared_bin.c_str()) != 0)
            {
                std::fprintf(stderr, "failed to load shared code predictor kv body\n");
                return false;
            }

            codepred_kv_shared_head_nets_.assign(15, nullptr);
            for (int step = 0; step < 15; step++)
            {
                char suffix[16];
                std::snprintf(suffix, sizeof(suffix), "%02d", step);
                const std::string bin = files.codepred_kv_dir + "/codepred_head_step" + suffix + ".ncnn.bin";
                if (!file_exists(bin))
                {
                    std::fprintf(stderr, "missing shared code predictor kv head step %02d\n", step);
                    return false;
                }
                std::shared_ptr<ncnn::Net> net = std::make_shared<ncnn::Net>();
                configure_codepred_net_options(*net, codepred_opt);
                if (net->load_param(shared_head_param.c_str()) != 0 || net->load_model(bin.c_str()) != 0)
                {
                    std::fprintf(stderr, "failed to load shared code predictor kv head step %02d\n", step);
                    return false;
                }
                codepred_kv_shared_head_nets_[step] = net;
            }
            codepred_kv_shared_ok_ = true;
        }

        const std::string prefill_param = files.codepred_kv_dir + "/codepred_prefill_s2_step00.ncnn.param";
        const std::string prefill_bin = files.codepred_kv_dir + "/codepred_prefill_s2_step00.ncnn.bin";
        if (file_exists(prefill_param) && file_exists(prefill_bin))
        {
            codepred_kv_prefill_net_ = std::make_shared<ncnn::Net>();
            configure_codepred_net_options(*codepred_kv_prefill_net_, codepred_opt);
            if (codepred_kv_prefill_net_->load_param(prefill_param.c_str()) != 0 ||
                codepred_kv_prefill_net_->load_model(prefill_bin.c_str()) != 0)
            {
                std::fprintf(stderr, "failed to load code predictor kv prefill\n");
                return false;
            }

            codepred_kv_decode_nets_.assign(14, nullptr);
            for (int step = 1; step <= 14; step++)
            {
                char suffix[16];
                std::snprintf(suffix, sizeof(suffix), "%02d", step);
                const std::string param = files.codepred_kv_dir + "/codepred_decode_s1_step" + suffix + ".ncnn.param";
                const std::string bin = files.codepred_kv_dir + "/codepred_decode_s1_step" + suffix + ".ncnn.bin";
                if (!file_exists(param) || !file_exists(bin))
                {
                    std::fprintf(stderr, "missing code predictor kv decode step %02d\n", step);
                    return false;
                }
                std::shared_ptr<ncnn::Net> net = std::make_shared<ncnn::Net>();
                configure_codepred_net_options(*net, codepred_opt);
                if (net->load_param(param.c_str()) != 0 || net->load_model(bin.c_str()) != 0)
                {
                    std::fprintf(stderr, "failed to load code predictor kv decode step %02d\n", step);
                    return false;
                }
                codepred_kv_decode_nets_[step - 1] = net;
            }
            codepred_kv_ok_ = true;
        }
    }

    if (!codepred_kv_shared_ok_ && !codepred_kv_ok_)
    {
        const std::string body_bin = files.codepred_body_dir + "/code_predictor_body_shared.ncnn.bin";
        for (int i = 0; i < 15; i++)
        {
            const int seq_len = i + 2;
            char suffix[16];
            std::snprintf(suffix, sizeof(suffix), "s%02d", seq_len);
            const std::string param = files.codepred_body_dir + "/code_predictor_body_" + suffix + ".ncnn.param";
            std::shared_ptr<ncnn::Net> net = std::make_shared<ncnn::Net>();
            configure_codepred_net_options(*net, codepred_opt);
            if (net->load_param(param.c_str()) != 0 || net->load_model(body_bin.c_str()) != 0)
            {
                std::fprintf(stderr, "failed to load code predictor %s\n", suffix);
                return false;
            }
            codepred_body_nets_[i] = net;
            codepred_masks_[i] = codepred_mask(seq_len);
            codepred_positions_[i] = codepred_pos(seq_len);
        }
    }

    talker_ok_ = true;
    return true;
}

static std::vector<int32_t> run_code_predictor_runtime(const std::string& body_dir,
                                                       const std::vector<std::shared_ptr<ncnn::Net>>& body_nets,
                                                       const std::vector<ncnn::Mat>& masks,
                                                       const std::vector<ncnn::Mat>& positions,
                                                       const std::vector<std::vector<float>>& heads,
                                                       const std::vector<std::vector<float>>& embs,
                                                       const std::vector<float>& talker_codec_embedding,
                                                       const float* talker_hidden,
                                                       int first_code,
                                                       int threads,
                                                       RuntimeProfile* profile = nullptr)
{
    constexpr int dim = 1024;
    constexpr int vocab = 2048;
    std::vector<float> seq(talker_hidden, talker_hidden + dim);
    append_embedding(seq, talker_codec_embedding, first_code, dim);
    std::vector<int32_t> out;
    out.reserve(15);
    for (int step = 0; step < 15; step++)
    {
        const double step_t0 = profile ? now_ms() : 0.0;
        const int seq_len = (int)seq.size() / dim;
        char suffix[16];
        std::snprintf(suffix, sizeof(suffix), "s%02d", seq_len);
        const int net_index = seq_len - 2;
        if (net_index < 0 || net_index >= (int)body_nets.size() || !body_nets[net_index])
        {
            std::fprintf(stderr, "missing code predictor %s\n", suffix);
            return {};
        }
        (void)body_dir;
        (void)threads;
        ncnn::Extractor ex = body_nets[net_index]->create_extractor();
        ex.input("in0", mat2_from_vec(seq, dim, seq_len));
        ex.input("in1", masks[net_index]);
        ex.input("in2", positions[net_index]);
        ncnn::Mat hidden;
        if (ex.extract("out0", hidden) != 0) return {};
        if (profile) profile->codepred_body_ms += now_ms() - step_t0;

        const double head_t0 = profile ? now_ms() : 0.0;
        const float* hp = (const float*)hidden.data + (size_t)(seq_len - 1) * dim;
        const int tok = argmax_head(hp, heads[step], vocab, dim);
        if (profile)
        {
            profile->codepred_head_ms += now_ms() - head_t0;
            profile->codepred_total_ms += now_ms() - step_t0;
            profile->codepred_steps++;
        }
        out.push_back(tok);
        if (step < 14) append_embedding(seq, embs[step], tok, dim);
    }
    return out;
}

static std::vector<int32_t> run_code_predictor_kv_runtime(const std::shared_ptr<ncnn::Net>& prefill_net,
                                                          const std::vector<std::shared_ptr<ncnn::Net>>& decode_nets,
                                                          const std::vector<std::vector<float>>& embs,
                                                          const std::vector<float>& talker_codec_embedding,
                                                          const float* talker_hidden,
                                                          int first_code,
                                                          RuntimeProfile* profile = nullptr)
{
    constexpr int dim = 1024;
    std::vector<int32_t> out;
    out.reserve(15);
    if (!prefill_net || decode_nets.size() != 14) return {};

    std::vector<float> seq;
    seq.reserve(2 * dim);
    seq.insert(seq.end(), talker_hidden, talker_hidden + dim);
    append_embedding(seq, talker_codec_embedding, first_code, dim);

    ncnn::Mat prefill_mask = codepred_mask(2);
    ncnn::Mat prefill_pos = codepred_pos(2);
    std::vector<ncnn::Mat> cache_k(5), cache_v(5);
    ncnn::Mat logits;
    {
        const double t0 = profile ? now_ms() : 0.0;
        ncnn::Extractor ex = prefill_net->create_extractor();
        ex.set_light_mode(false);
        ex.input("in0", mat2_from_vec(seq, dim, 2));
        ex.input("in1", prefill_mask);
        ex.input("in2", prefill_pos);
        if (ex.extract("out1", logits) != 0) return {};
        for (int i = 0; i < 5; i++)
        {
            char kname[32], vname[32];
            std::snprintf(kname, sizeof(kname), "out_cache_k%d", i);
            std::snprintf(vname, sizeof(vname), "out_cache_v%d", i);
            if (ex.extract(kname, cache_k[i]) != 0 || ex.extract(vname, cache_v[i]) != 0) return {};
        }
        if (profile)
        {
            profile->codepred_body_ms += now_ms() - t0;
            profile->codepred_total_ms += now_ms() - t0;
            profile->codepred_steps++;
        }
    }
    int tok = argmax_mat(logits, 2048);
    out.push_back(tok);

    for (int step = 1; step <= 14; step++)
    {
        const double t0 = profile ? now_ms() : 0.0;
        ncnn::Mat input = mat2_from_ptr(embs[step - 1].data() + (size_t)tok * dim, dim, 1);
        ncnn::Mat mask = codepred_decode_mask_kv(step + 1);
        ncnn::Mat pos(1, (size_t)4u, 1);
        ((float*)pos.data)[0] = (float)(step + 1);

        ncnn::Extractor ex = decode_nets[step - 1]->create_extractor();
        ex.set_light_mode(false);
        ex.input("in0", input);
        ex.input("in1", mask);
        ex.input("in2", pos);
        for (int i = 0; i < 5; i++)
        {
            char kname[32], vname[32];
            std::snprintf(kname, sizeof(kname), "cache_k%d", i);
            std::snprintf(vname, sizeof(vname), "cache_v%d", i);
            ex.input(kname, cache_k[i]);
            ex.input(vname, cache_v[i]);
        }
        if (ex.extract("out1", logits) != 0) return {};
        for (int i = 0; i < 5; i++)
        {
            char kname[32], vname[32];
            std::snprintf(kname, sizeof(kname), "out_cache_k%d", i);
            std::snprintf(vname, sizeof(vname), "out_cache_v%d", i);
            if (ex.extract(kname, cache_k[i]) != 0 || ex.extract(vname, cache_v[i]) != 0) return {};
        }
        if (profile)
        {
            profile->codepred_body_ms += now_ms() - t0;
            profile->codepred_total_ms += now_ms() - t0;
            profile->codepred_steps++;
        }
        tok = argmax_mat(logits, 2048);
        out.push_back(tok);
    }

    return out;
}

static bool run_code_predictor_head(const std::shared_ptr<ncnn::Net>& head_net,
                                    const ncnn::Mat& hidden,
                                    ncnn::Mat& logits)
{
    if (!head_net || hidden.empty()) return false;
    ncnn::Extractor ex = head_net->create_extractor();
    ex.input("in0", hidden);
    return ex.extract("out0", logits) == 0;
}

static std::vector<int32_t> run_code_predictor_kv_shared_runtime(const std::shared_ptr<ncnn::Net>& prefill_body_net,
                                                                 const std::shared_ptr<ncnn::Net>& decode_body_net,
                                                                 const std::vector<std::shared_ptr<ncnn::Net>>& head_nets,
                                                                 const std::vector<std::vector<float>>& embs,
                                                                 const std::vector<float>& talker_codec_embedding,
                                                                 const float* talker_hidden,
                                                                 int first_code,
                                                                 RuntimeProfile* profile = nullptr)
{
    constexpr int dim = 1024;
    std::vector<int32_t> out;
    out.reserve(15);
    if (!prefill_body_net || !decode_body_net || head_nets.size() != 15) return {};

    std::vector<float> seq;
    seq.reserve(2 * dim);
    seq.insert(seq.end(), talker_hidden, talker_hidden + dim);
    append_embedding(seq, talker_codec_embedding, first_code, dim);

    const auto& fixed_masks = codepred_kv_masks();
    const auto& fixed_positions = codepred_kv_positions();
    const auto& cache_names = codepred_kv_cache_names();
    std::vector<ncnn::Mat> cache_k(5), cache_v(5);
    ncnn::Mat hidden;
    ncnn::Mat logits;
    const bool step_profile = profile && codepred_step_profile_enabled();
    double step_body_ms[15] = {};
    double step_head_ms[15] = {};
    {
        const double t0 = profile ? now_ms() : 0.0;
        ncnn::Extractor ex = prefill_body_net->create_extractor();
        ex.set_light_mode(false);
        ex.input("in0", mat2_from_vec(seq, dim, 2));
        ex.input("in1", fixed_masks[0]);
        ex.input("in2", fixed_positions[0]);
        if (ex.extract("out0", hidden) != 0) return {};
        for (int i = 0; i < 5; i++)
        {
            if (ex.extract(cache_names.out_k[i].c_str(), cache_k[i]) != 0 ||
                ex.extract(cache_names.out_v[i].c_str(), cache_v[i]) != 0) return {};
        }
        const double t1 = profile ? now_ms() : 0.0;
        if (!run_code_predictor_head(head_nets[0], hidden, logits)) return {};
        if (profile)
        {
            const double t2 = now_ms();
            profile->codepred_body_ms += t1 - t0;
            profile->codepred_head_ms += t2 - t1;
            profile->codepred_total_ms += t2 - t0;
            profile->codepred_steps++;
            if (step_profile)
            {
                step_body_ms[0] = t1 - t0;
                step_head_ms[0] = t2 - t1;
            }
        }
    }
    int tok = argmax_mat(logits, 2048);
    out.push_back(tok);

    for (int step = 1; step <= 14; step++)
    {
        const double t0 = profile ? now_ms() : 0.0;
        ncnn::Mat input = mat2_from_ptr(embs[step - 1].data() + (size_t)tok * dim, dim, 1);

        ncnn::Extractor ex = decode_body_net->create_extractor();
        ex.set_light_mode(false);
        ex.input("in0", input);
        ex.input("in1", fixed_masks[step]);
        ex.input("in2", fixed_positions[step]);
        for (int i = 0; i < 5; i++)
        {
            ex.input(cache_names.in_k[i].c_str(), cache_k[i]);
            ex.input(cache_names.in_v[i].c_str(), cache_v[i]);
        }
        if (ex.extract("out0", hidden) != 0) return {};
        for (int i = 0; i < 5; i++)
        {
            if (ex.extract(cache_names.out_k[i].c_str(), cache_k[i]) != 0 ||
                ex.extract(cache_names.out_v[i].c_str(), cache_v[i]) != 0) return {};
        }
        const double t1 = profile ? now_ms() : 0.0;
        if (!run_code_predictor_head(head_nets[step], hidden, logits)) return {};
        if (profile)
        {
            const double t2 = now_ms();
            profile->codepred_body_ms += t1 - t0;
            profile->codepred_head_ms += t2 - t1;
            profile->codepred_total_ms += t2 - t0;
            profile->codepred_steps++;
            if (step_profile)
            {
                step_body_ms[step] = t1 - t0;
                step_head_ms[step] = t2 - t1;
            }
        }
        tok = argmax_mat(logits, 2048);
        out.push_back(tok);
    }

    if (step_profile)
    {
        std::fprintf(stderr, "[qwen3_tts_codepred_steps]");
        for (int step = 0; step < 15; step++)
        {
            std::fprintf(stderr, " s%02d_body=%.3f s%02d_head=%.3f", step, step_body_ms[step], step, step_head_ms[step]);
        }
        std::fprintf(stderr, "\n");
    }

    return out;
}

std::vector<float> Qwen3TTSNcnn::codes_to_hidden(const std::vector<int32_t>& codes, int frames) const
{
    constexpr int vocab = 2048;
    constexpr int emb = 256;
    constexpr int h = 512;
    constexpr int ksize = 3;
    (void)vocab;

    std::vector<float> quant((size_t)frames * h);
#pragma omp parallel for num_threads(opt_.num_threads)
    for (int t = 0; t < frames; t++)
    {
        float rest_sum[emb];
        for (int e = 0; e < emb; e++) rest_sum[e] = 0.f;

        const int first_id = codes[(size_t)t * codebooks];
        const float* first_vec = first_emb_.data() + (size_t)first_id * emb;
        for (int q = 1; q < codebooks; q++)
        {
            const int id = codes[(size_t)t * codebooks + q];
            const float* row = rest_emb_[q - 1].data() + (size_t)id * emb;
            for (int e = 0; e < emb; e++) rest_sum[e] += row[e];
        }

        float* qout = quant.data() + (size_t)t * h;
        for (int i = 0; i < h; i++)
        {
            const float* wf = first_proj_.data() + (size_t)i * emb;
            const float* wr = rest_proj_.data() + (size_t)i * emb;
            double sum = 0.0;
            for (int e = 0; e < emb; e++) sum += (double)wf[e] * first_vec[e] + (double)wr[e] * rest_sum[e];
            qout[i] = (float)sum;
        }
    }

    std::vector<float> hidden((size_t)frames * hidden_dim);
#pragma omp parallel for num_threads(opt_.num_threads)
    for (int t = 0; t < frames; t++)
    {
        for (int oc = 0; oc < hidden_dim; oc++)
        {
            double sum = pre_conv_b_[oc];
            for (int ic = 0; ic < h; ic++)
            {
                for (int k = 0; k < ksize; k++)
                {
                    const int src_t = t + k - 2;
                    if (src_t < 0) continue;
                    sum += (double)quant[(size_t)src_t * h + ic] * pre_conv_w_[((size_t)oc * h + ic) * ksize + k];
                }
            }
            hidden[(size_t)t * hidden_dim + oc] = (float)sum;
        }
    }
    return hidden;
}

std::vector<int32_t> Qwen3TTSNcnn::generate_codes(const std::string& prefill_embed,
                                                  const std::string& prefill_mask,
                                                  const std::string& prefill_cos,
                                                  const std::string& prefill_sin,
                                                  int frames) const
{
    if (!talker_ok_ || frames <= 0) return {};
    RuntimeProfile profile;
    RuntimeProfile* prof = profile_enabled() ? &profile : nullptr;
    if (prof) prof->frames = frames;

    const size_t embed_count = f32_count(prefill_embed);
    if (embed_count == 0 || embed_count % hidden_dim != 0)
    {
        std::fprintf(stderr, "bad prefill embed shape %s\n", prefill_embed.c_str());
        return {};
    }
    const int prefill_len = (int)(embed_count / hidden_dim);
    if (f32_count(prefill_mask) != (size_t)prefill_len * prefill_len ||
        f32_count(prefill_cos) != (size_t)prefill_len * 128 ||
        f32_count(prefill_sin) != (size_t)prefill_len * 128)
    {
        std::fprintf(stderr, "bad prefill auxiliary shape for len=%d\n", prefill_len);
        return {};
    }

    std::vector<std::pair<ncnn::Mat, ncnn::Mat>> cache;
    ncnn::Mat logits;
    ncnn::Mat prefill_hidden;
    {
        const double t0 = prof ? now_ms() : 0.0;
        ncnn::Extractor ex = talker_prefill_net_->create_extractor();
        ex.set_light_mode(false);
        ex.input("in0", mat2_from_file_any(prefill_embed, hidden_dim, prefill_len));
        ex.input("in1", mat2_from_file_any(prefill_mask, prefill_len, prefill_len));
        ex.input("in2", mat2_from_file_any(prefill_cos, 128, prefill_len));
        ex.input("in3", mat2_from_file_any(prefill_sin, 128, prefill_len));
        for (int i = 0; i < 28; i++)
        {
            char kname[32], vname[32];
            std::snprintf(kname, sizeof(kname), "out_cache_k%d", i);
            std::snprintf(vname, sizeof(vname), "out_cache_v%d", i);
            ncnn::Mat k, v;
            if (ex.extract(kname, k) != 0 || ex.extract(vname, v) != 0) return {};
            cache.emplace_back(std::move(k), std::move(v));
        }
        if (ex.extract("out0", prefill_hidden) != 0 || ex.extract("out1", logits) != 0) return {};
        if (prof) prof->prefill_ms += now_ms() - t0;
    }

    std::vector<int32_t> generated;
    generated.reserve((size_t)frames * codebooks);
    ncnn::Mat hidden_last;
    int main_code = argmax_mat(logits.row_range(prefill_len - 1, 1), 3072);

    for (int frame = 0; frame < frames; frame++)
    {
        if (frame == 0)
        {
            hidden_last = prefill_hidden.row_range(prefill_len - 1, 1).clone();
        }

        std::vector<int32_t> rest = codepred_kv_shared_ok_
            ? run_code_predictor_kv_shared_runtime(codepred_kv_shared_prefill_body_net_,
                                                   codepred_kv_shared_decode_body_net_,
                                                   codepred_kv_shared_head_nets_,
                                                   codepred_embs_,
                                                   talker_codec_embedding_,
                                                   (const float*)hidden_last.data,
                                                   main_code,
                                                   prof)
            : codepred_kv_ok_
            ? run_code_predictor_kv_runtime(codepred_kv_prefill_net_,
                                            codepred_kv_decode_nets_,
                                            codepred_embs_,
                                            talker_codec_embedding_,
                                            (const float*)hidden_last.data,
                                            main_code,
                                            prof)
            : run_code_predictor_runtime(talker_files_.codepred_body_dir,
                                         codepred_body_nets_,
                                         codepred_masks_,
                                         codepred_positions_,
                                         codepred_heads_,
                                         codepred_embs_,
                                         talker_codec_embedding_,
                                         (const float*)hidden_last.data,
                                         main_code,
                                         opt_.num_threads,
                                         prof);
        if (rest.size() != 15) return {};
        generated.push_back(main_code);
        for (int32_t t : rest) generated.push_back(t);

        if (frame + 1 >= frames) break;

        std::vector<float> next_embed;
        append_embedding(next_embed, talker_codec_embedding_, main_code, hidden_dim);
        for (int i = 0; i < 15; i++) append_embedding(next_embed, codepred_embs_[i], rest[i], hidden_dim);
        std::vector<float> summed(hidden_dim, 0.0f);
        for (int i = 0; i < 16; i++)
        {
            for (int j = 0; j < hidden_dim; j++) summed[j] += next_embed[(size_t)i * hidden_dim + j];
        }
        for (int j = 0; j < hidden_dim; j++) summed[j] += tts_pad_embed_[j];

        const int position = prefill_len + frame;
        ncnn::Mat cos, sin;
        make_rope(position, cos, sin);
        ncnn::Mat decode_hidden, decode_logits;
        {
            const double t0 = prof ? now_ms() : 0.0;
            ncnn::Extractor ex = talker_decode_net_->create_extractor();
            ex.set_light_mode(false);
            ex.input("in0", mat2_from_vec(summed, hidden_dim, 1));
            ex.input("in1", make_decode_mask(position));
            ex.input("in2", cos);
            ex.input("in3", sin);
            for (int i = 0; i < 28; i++)
            {
                char kname[32], vname[32];
                std::snprintf(kname, sizeof(kname), "cache_k%d", i);
                std::snprintf(vname, sizeof(vname), "cache_v%d", i);
                ex.input(kname, cache[i].first);
                ex.input(vname, cache[i].second);
            }
            for (int i = 0; i < 28; i++)
            {
                char kname[32], vname[32];
                std::snprintf(kname, sizeof(kname), "out_cache_k%d", i);
                std::snprintf(vname, sizeof(vname), "out_cache_v%d", i);
                ncnn::Mat k, v;
                if (ex.extract(kname, k) != 0 || ex.extract(vname, v) != 0) return {};
                cache[i] = std::make_pair(std::move(k), std::move(v));
            }
            if (ex.extract("out0", decode_hidden) != 0 || ex.extract("out1", decode_logits) != 0) return {};
            if (prof)
            {
                prof->talker_decode_ms += now_ms() - t0;
                prof->talker_decode_steps++;
            }
        }
        hidden_last = decode_hidden.clone();
        main_code = argmax_mat(decode_logits, 3072);
    }

    if (prof)
    {
        std::fprintf(stderr,
                     "[qwen3_tts_profile] generate_codes frames=%d prefill_ms=%.3f codepred_total_ms=%.3f codepred_body_ms=%.3f codepred_head_ms=%.3f codepred_steps=%d talker_decode_ms=%.3f talker_decode_steps=%d\n",
                     prof->frames,
                     prof->prefill_ms,
                     prof->codepred_total_ms,
                     prof->codepred_body_ms,
                     prof->codepred_head_ms,
                     prof->codepred_steps,
                     prof->talker_decode_ms,
                     prof->talker_decode_steps);
    }

    return generated;
}

std::vector<int32_t> Qwen3TTSNcnn::generate_codes_from_prompt(const ncnn::Mat& prefill_embed,
                                                              int frames) const
{
    if (!talker_ok_ || frames <= 0 || prefill_embed.empty() || prefill_embed.w != hidden_dim) return {};
    RuntimeProfile profile;
    RuntimeProfile* prof = profile_enabled() ? &profile : nullptr;
    if (prof) prof->frames = frames;

    const int prefill_len = prefill_embed.h;
    ncnn::Mat prefill_mask = make_prefill_mask(prefill_len);
    ncnn::Mat prefill_cos, prefill_sin;
    make_rope_table(prefill_len, prefill_cos, prefill_sin);

    std::vector<std::pair<ncnn::Mat, ncnn::Mat>> cache;
    ncnn::Mat logits;
    ncnn::Mat prefill_hidden;
    {
        const double t0 = prof ? now_ms() : 0.0;
        ncnn::Extractor ex = talker_prefill_net_->create_extractor();
        ex.set_light_mode(false);
        ex.input("in0", prefill_embed);
        ex.input("in1", prefill_mask);
        ex.input("in2", prefill_cos);
        ex.input("in3", prefill_sin);
        for (int i = 0; i < 28; i++)
        {
            char kname[32], vname[32];
            std::snprintf(kname, sizeof(kname), "out_cache_k%d", i);
            std::snprintf(vname, sizeof(vname), "out_cache_v%d", i);
            ncnn::Mat k, v;
            if (ex.extract(kname, k) != 0 || ex.extract(vname, v) != 0) return {};
            cache.emplace_back(std::move(k), std::move(v));
        }
        if (ex.extract("out0", prefill_hidden) != 0 || ex.extract("out1", logits) != 0) return {};
        if (prof) prof->prefill_ms += now_ms() - t0;
    }

    std::vector<int32_t> generated;
    generated.reserve((size_t)frames * codebooks);
    ncnn::Mat hidden_last;
    int main_code = argmax_mat(logits.row_range(prefill_len - 1, 1), 3072);

    for (int frame = 0; frame < frames; frame++)
    {
        if (frame == 0)
        {
            hidden_last = prefill_hidden.row_range(prefill_len - 1, 1).clone();
        }

        std::vector<int32_t> rest = codepred_kv_shared_ok_
            ? run_code_predictor_kv_shared_runtime(codepred_kv_shared_prefill_body_net_,
                                                   codepred_kv_shared_decode_body_net_,
                                                   codepred_kv_shared_head_nets_,
                                                   codepred_embs_,
                                                   talker_codec_embedding_,
                                                   (const float*)hidden_last.data,
                                                   main_code,
                                                   prof)
            : codepred_kv_ok_
            ? run_code_predictor_kv_runtime(codepred_kv_prefill_net_,
                                            codepred_kv_decode_nets_,
                                            codepred_embs_,
                                            talker_codec_embedding_,
                                            (const float*)hidden_last.data,
                                            main_code,
                                            prof)
            : run_code_predictor_runtime(talker_files_.codepred_body_dir,
                                         codepred_body_nets_,
                                         codepred_masks_,
                                         codepred_positions_,
                                         codepred_heads_,
                                         codepred_embs_,
                                         talker_codec_embedding_,
                                         (const float*)hidden_last.data,
                                         main_code,
                                         opt_.num_threads,
                                         prof);
        if (rest.size() != 15) return {};
        generated.push_back(main_code);
        for (int32_t t : rest) generated.push_back(t);

        if (frame + 1 >= frames) break;

        std::vector<float> next_embed;
        append_embedding(next_embed, talker_codec_embedding_, main_code, hidden_dim);
        for (int i = 0; i < 15; i++) append_embedding(next_embed, codepred_embs_[i], rest[i], hidden_dim);
        std::vector<float> summed(hidden_dim, 0.0f);
        for (int i = 0; i < 16; i++)
        {
            for (int j = 0; j < hidden_dim; j++) summed[j] += next_embed[(size_t)i * hidden_dim + j];
        }
        for (int j = 0; j < hidden_dim; j++) summed[j] += tts_pad_embed_[j];

        const int position = prefill_len + frame;
        ncnn::Mat cos, sin;
        make_rope(position, cos, sin);
        ncnn::Mat decode_hidden, decode_logits;
        {
            const double t0 = prof ? now_ms() : 0.0;
            ncnn::Extractor ex = talker_decode_net_->create_extractor();
            ex.set_light_mode(false);
            ex.input("in0", mat2_from_vec(summed, hidden_dim, 1));
            ex.input("in1", make_decode_mask(position));
            ex.input("in2", cos);
            ex.input("in3", sin);
            for (int i = 0; i < 28; i++)
            {
                char kname[32], vname[32];
                std::snprintf(kname, sizeof(kname), "cache_k%d", i);
                std::snprintf(vname, sizeof(vname), "cache_v%d", i);
                ex.input(kname, cache[i].first);
                ex.input(vname, cache[i].second);
            }
            for (int i = 0; i < 28; i++)
            {
                char kname[32], vname[32];
                std::snprintf(kname, sizeof(kname), "out_cache_k%d", i);
                std::snprintf(vname, sizeof(vname), "out_cache_v%d", i);
                ncnn::Mat k, v;
                if (ex.extract(kname, k) != 0 || ex.extract(vname, v) != 0) return {};
                cache[i] = std::make_pair(std::move(k), std::move(v));
            }
            if (ex.extract("out0", decode_hidden) != 0 || ex.extract("out1", decode_logits) != 0) return {};
            if (prof)
            {
                prof->talker_decode_ms += now_ms() - t0;
                prof->talker_decode_steps++;
            }
        }
        hidden_last = decode_hidden.clone();
        main_code = argmax_mat(decode_logits, 3072);
    }

    if (prof)
    {
        std::fprintf(stderr,
                     "[qwen3_tts_profile] generate_codes_from_prompt frames=%d prefill_ms=%.3f codepred_total_ms=%.3f codepred_body_ms=%.3f codepred_head_ms=%.3f codepred_steps=%d talker_decode_ms=%.3f talker_decode_steps=%d\n",
                     prof->frames,
                     prof->prefill_ms,
                     prof->codepred_total_ms,
                     prof->codepred_body_ms,
                     prof->codepred_head_ms,
                     prof->codepred_steps,
                     prof->talker_decode_ms,
                     prof->talker_decode_steps);
    }

    return generated;
}

std::vector<float> Qwen3TTSNcnn::decode_hidden_chunk(const std::vector<float>& hidden,
                                                     int hidden_frames,
                                                     int context_frames,
                                                     int current_frames) const
{
    if (!ok_ || context_frames < 0 || current_frames <= 0 ||
        context_frames + current_frames > opt_.decoder_chunk_frames ||
        hidden_frames < context_frames + current_frames)
    {
        return {};
    }

    std::vector<float> chunk((size_t)opt_.decoder_chunk_frames * hidden_dim, 0.0f);
    std::memcpy(chunk.data(), hidden.data(), (size_t)(context_frames + current_frames) * hidden_dim * sizeof(float));

    ncnn::Mat in(hidden_dim, opt_.decoder_chunk_frames, (size_t)4u, 1);
    std::memcpy(in.data, chunk.data(), chunk.size() * sizeof(float));

    ncnn::Extractor ex = decoder_net_->create_extractor();
    if (ex.input("in0", in) != 0) return {};
    ncnn::Mat out;
    if (ex.extract("out0", out) != 0) return {};

    const int drop = context_frames * samples_per_frame;
    const int keep = current_frames * samples_per_frame;
    if ((int)out.total() < drop + keep) return {};
    const float* p = (const float*)out.data;
    return std::vector<float>(p + drop, p + drop + keep);
}

std::vector<float> Qwen3TTSNcnn::codes_to_wav(const std::vector<int32_t>& codes,
                                              int frames,
                                              int context_frames,
                                              int current_frames) const
{
    RuntimeProfile profile;
    RuntimeProfile* prof = profile_enabled() ? &profile : nullptr;
    const double t0 = prof ? now_ms() : 0.0;
    std::vector<float> hidden = codes_to_hidden(codes, frames);
    if (prof) prof->codec_front_ms = now_ms() - t0;

    const double t1 = prof ? now_ms() : 0.0;
    std::vector<float> wav = decode_hidden_chunk(hidden, frames, context_frames, current_frames);
    if (prof)
    {
        prof->speech_decoder_ms = now_ms() - t1;
        std::fprintf(stderr,
                     "[qwen3_tts_profile] codes_to_wav frames=%d codec_front_ms=%.3f speech_decoder_ms=%.3f wav_samples=%zu\n",
                     frames,
                     prof->codec_front_ms,
                     prof->speech_decoder_ms,
                     wav.size());
    }
    return wav;
}
