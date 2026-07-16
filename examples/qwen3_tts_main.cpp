#include "tts/qwen3_tts_runtime.h"
#include "tts/qwen3_tts_frontend.h"
#include "utf8_args.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

static bool profile_enabled()
{
    const char* p = std::getenv("QWEN3_TTS_PROFILE");
    return p && p[0] && p[0] != '0';
}

static double now_ms()
{
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

struct ModelJson {
    std::string front_weights_dir;
    std::string decoder_param;
    std::string decoder_bin;
    std::string talker_prefill_param;
    std::string talker_decode_param;
    std::string talker_bin;
    std::string codepred_body_dir;
    std::string codepred_kv_dir;
    std::string codepred_weights_dir;
    std::string tts_pad_embed;
    std::string prefill_embed;
    std::string prefill_mask;
    std::string prefill_cos;
    std::string prefill_sin;
    std::string tokenizer;
    std::string text_embed_param;
    std::string text_embed_bin;
    std::string codec_embed_param;
    std::string codec_embed_bin;
    int decoder_chunk_frames = 325;
    int decoder_context_frames = 25;
    int sample_rate = 24000;
};

static bool read_text(const std::string& path, std::string& out)
{
    std::ifstream ifs(path);
    if (!ifs) return false;
    out.assign((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
    return true;
}

static int env_int(const char* name, int fallback)
{
    const char* p = std::getenv(name);
    if (!p || !p[0]) return fallback;
    const int v = std::atoi(p);
    return v > 0 ? v : fallback;
}

static bool load_model_json(const std::string& path, ModelJson& cfg)
{
    std::ifstream input(path);
    if (!input) return false;

    nlohmann::json root;
    try
    {
        input >> root;
    }
    catch (const nlohmann::json::exception& e)
    {
        std::fprintf(stderr, "invalid model json %s: %s\n", path.c_str(), e.what());
        return false;
    }

    const std::filesystem::path base = std::filesystem::absolute(path).parent_path();
    auto asset = [&](const char* key) {
        const std::string value = root.value(key, std::string());
        if (value.empty()) return value;
        const std::filesystem::path candidate(value);
        return (candidate.is_absolute() ? candidate : base / candidate).lexically_normal().string();
    };

    cfg.front_weights_dir = asset("front_weights_dir");
    cfg.decoder_param = asset("decoder_param");
    cfg.decoder_bin = asset("decoder_bin");
    cfg.talker_prefill_param = asset("talker_prefill_param");
    cfg.talker_decode_param = asset("talker_decode_param");
    cfg.talker_bin = asset("talker_bin");
    cfg.codepred_body_dir = asset("codepred_body_dir");
    cfg.codepred_kv_dir = asset("codepred_kv_dir");
    cfg.codepred_weights_dir = asset("codepred_weights_dir");
    cfg.tts_pad_embed = asset("tts_pad_embed");
    cfg.prefill_embed = asset("prefill_embed");
    cfg.prefill_mask = asset("prefill_mask");
    cfg.prefill_cos = asset("prefill_cos");
    cfg.prefill_sin = asset("prefill_sin");
    cfg.tokenizer = asset("tokenizer");
    cfg.text_embed_param = asset("text_embed_param");
    cfg.text_embed_bin = asset("text_embed_bin");
    cfg.codec_embed_param = asset("codec_embed_param");
    cfg.codec_embed_bin = asset("codec_embed_bin");
    cfg.decoder_chunk_frames = root.value("decoder_chunk_frames", 325);
    cfg.decoder_context_frames = root.value("decoder_context_frames", 25);
    cfg.sample_rate = root.value("sample_rate", 24000);
    return !cfg.front_weights_dir.empty() && !cfg.decoder_param.empty() && !cfg.talker_prefill_param.empty();
}

static bool write_codes(const std::string& path, const std::vector<int32_t>& data)
{
    if (path.empty() || path == "-") return true;
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) return false;
    ofs.write((const char*)data.data(), (std::streamsize)(data.size() * sizeof(int32_t)));
    return (bool)ofs;
}

static void put_u16(std::ofstream& ofs, uint16_t v)
{
    char b[2] = {(char)(v & 255), (char)((v >> 8) & 255)};
    ofs.write(b, 2);
}

static void put_u32(std::ofstream& ofs, uint32_t v)
{
    char b[4] = {(char)(v & 255), (char)((v >> 8) & 255), (char)((v >> 16) & 255), (char)((v >> 24) & 255)};
    ofs.write(b, 4);
}

static bool write_wav16(const std::string& path, const std::vector<float>& wav, int sample_rate)
{
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) return false;
    const uint32_t data_bytes = (uint32_t)wav.size() * 2u;
    ofs.write("RIFF", 4);
    put_u32(ofs, 36u + data_bytes);
    ofs.write("WAVE", 4);
    ofs.write("fmt ", 4);
    put_u32(ofs, 16);
    put_u16(ofs, 1);
    put_u16(ofs, 1);
    put_u32(ofs, (uint32_t)sample_rate);
    put_u32(ofs, (uint32_t)sample_rate * 2u);
    put_u16(ofs, 2);
    put_u16(ofs, 16);
    ofs.write("data", 4);
    put_u32(ofs, data_bytes);
    for (float x : wav)
    {
        x = std::max(-1.0f, std::min(1.0f, x));
        const int v = (int)(x * 32767.0f);
        put_u16(ofs, (uint16_t)(int16_t)std::max(-32768, std::min(32767, v)));
    }
    return (bool)ofs;
}

struct CliArgs {
    std::string model;
    std::string out_wav;
    std::string out_codes = "-";
    std::string text;
    std::string text_file;
    std::string speaker = "Vivian";
    std::string language = "Chinese";
    int frames = 25;
    int threads = 4;
    bool use_vulkan = false;
    bool legacy = false;
    bool show_help = false;
};

static bool is_model_json_path(const std::string& path)
{
    return path.size() >= 5 && path.substr(path.size() - 5) == ".json";
}

static std::string model_json_from_arg(const std::string& model)
{
    if (is_model_json_path(model)) return model;
    if (!model.empty() && (model.back() == '/' || model.back() == '\\')) return model + "model.json";
    return model + "/model.json";
}

static void print_usage(const char* argv0)
{
    std::fprintf(stderr,
                 "Usage:\n"
                 "  %s --model <model_dir|model.json> --frames <n> --out <out.wav> [--codes <out_i32|->] [--threads <n>] [--vulkan]\n"
                 "  %s <model.json> <frames> <out.wav> <out_codes_i32|-> <threads> <use_vulkan>\n"
                 "\n"
                 "Staged frontend options accepted for ncnn_llm-style CLI compatibility:\n"
                 "  --text <text> --text-file <utf8.txt> --speaker <name> --language <name>\n"
                 "If model.json includes tokenizer/text_embed/codec_embed frontend nets, --text uses the C++ CustomVoice frontend.\n"
                 "Otherwise the runtime uses the precomputed prefill assets from model.json.\n",
                 argv0, argv0);
}

static bool parse_cli(int argc, char** argv, CliArgs& args)
{
    if (argc == 7 && argv[1][0] != '-')
    {
        args.legacy = true;
        args.model = argv[1];
        args.frames = std::atoi(argv[2]);
        args.out_wav = argv[3];
        args.out_codes = argv[4];
        args.threads = std::max(1, std::atoi(argv[5]));
        args.use_vulkan = std::atoi(argv[6]) != 0;
        return true;
    }

    for (int i = 1; i < argc; i++)
    {
        const std::string a = argv[i];
        auto need_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc)
            {
                std::fprintf(stderr, "missing value for %s\n", name);
                return nullptr;
            }
            return argv[++i];
        };

        if (a == "--model")
        {
            const char* v = need_value("--model");
            if (!v) return false;
            args.model = v;
        }
        else if (a == "--frames")
        {
            const char* v = need_value("--frames");
            if (!v) return false;
            args.frames = std::atoi(v);
        }
        else if (a == "--out")
        {
            const char* v = need_value("--out");
            if (!v) return false;
            args.out_wav = v;
        }
        else if (a == "--codes")
        {
            const char* v = need_value("--codes");
            if (!v) return false;
            args.out_codes = v;
        }
        else if (a == "--threads")
        {
            const char* v = need_value("--threads");
            if (!v) return false;
            args.threads = std::max(1, std::atoi(v));
        }
        else if (a == "--vulkan")
        {
            args.use_vulkan = true;
        }
        else if (a == "--no-vulkan")
        {
            args.use_vulkan = false;
        }
        else if (a == "--text")
        {
            const char* v = need_value("--text");
            if (!v) return false;
            args.text = v;
        }
        else if (a == "--text-file")
        {
            const char* v = need_value("--text-file");
            if (!v) return false;
            args.text_file = v;
        }
        else if (a == "--speaker")
        {
            const char* v = need_value("--speaker");
            if (!v) return false;
            args.speaker = v;
        }
        else if (a == "--language")
        {
            const char* v = need_value("--language");
            if (!v) return false;
            args.language = v;
        }
        else if (a == "-h" || a == "--help")
        {
            args.show_help = true;
            return true;
        }
        else
        {
            std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
            return false;
        }
    }

    return !args.model.empty() && !args.out_wav.empty() && args.frames > 0;
}

int main(int argc, char** argv)
{
    enable_utf8_console();
    std::vector<std::string> utf8_args = get_utf8_args(argc, argv);
    std::vector<char*> utf8_argv;
    utf8_argv.reserve(utf8_args.size());
    for (std::string& value : utf8_args) utf8_argv.push_back(value.data());
    argc = static_cast<int>(utf8_argv.size());
    argv = utf8_argv.data();

    const bool prof = profile_enabled();
    const double total_t0 = prof ? now_ms() : 0.0;
    CliArgs cli;
    if (!parse_cli(argc, argv, cli))
    {
        print_usage(argv[0]);
        return 2;
    }
    if (cli.show_help)
    {
        print_usage(argv[0]);
        return 0;
    }

    const std::string model_json = model_json_from_arg(cli.model);
    ModelJson cfg;
    if (!load_model_json(model_json, cfg))
    {
        std::fprintf(stderr, "failed to load model json %s\n", model_json.c_str());
        return 1;
    }
    if (!cli.text_file.empty() && !read_text(cli.text_file, cli.text))
    {
        std::fprintf(stderr, "failed to read text file %s\n", cli.text_file.c_str());
        return 1;
    }
    if (cli.frames > cfg.decoder_chunk_frames)
    {
        std::fprintf(stderr,
                     "requested frames=%d exceeds decoder_chunk_frames=%d in %s; "
                     "use a larger decoder graph/package or request fewer frames\n",
                     cli.frames,
                     cfg.decoder_chunk_frames,
                     model_json.c_str());
        return 2;
    }

    Qwen3TTSNcnn::Options opt;
    opt.num_threads = cli.threads;
    opt.decoder_num_threads = env_int("QWEN3_TTS_DECODER_THREADS", 0);
    opt.use_vulkan = cli.use_vulkan;
    opt.codepred_use_vulkan = cli.use_vulkan;
    if (env_int("QWEN3_TTS_CODEPRED_CPU", 0) > 0) opt.codepred_use_vulkan = false;
    opt.codepred_use_fp16 = env_int("QWEN3_TTS_CODEPRED_FP16", 0) > 0;
    opt.codepred_fp16_mode = env_int("QWEN3_TTS_CODEPRED_FP16_MODE", 0);
    opt.decoder_chunk_frames = cfg.decoder_chunk_frames;
    opt.decoder_context_frames = cfg.decoder_context_frames;

    const double construct_t0 = prof ? now_ms() : 0.0;
    Qwen3TTSNcnn tts(cfg.front_weights_dir, cfg.decoder_param, cfg.decoder_bin, opt);
    if (!tts.ok()) return 1;
    if (prof) std::fprintf(stderr, "[qwen3_tts_profile] runtime_construct_ms=%.3f\n", now_ms() - construct_t0);

    Qwen3TTSNcnn::TalkerFiles talker;
    talker.prefill_param = cfg.talker_prefill_param;
    talker.decode_param = cfg.talker_decode_param;
    talker.talker_bin = cfg.talker_bin;
    talker.codepred_body_dir = cfg.codepred_body_dir;
    talker.codepred_kv_dir = cfg.codepred_kv_dir;
    talker.codepred_weights_dir = cfg.codepred_weights_dir;
    talker.tts_pad_embed = cfg.tts_pad_embed;
    const double load_talker_t0 = prof ? now_ms() : 0.0;
    if (!tts.load_talker(talker)) return 1;
    if (prof) std::fprintf(stderr, "[qwen3_tts_profile] load_talker_ms=%.3f\n", now_ms() - load_talker_t0);

    const int frames = cli.frames;
    std::vector<int32_t> codes;
    if (!cli.text.empty() && !cfg.tokenizer.empty() && !cfg.text_embed_param.empty() && !cfg.codec_embed_param.empty())
    {
        const double frontend_t0 = prof ? now_ms() : 0.0;
        Qwen3TTSFrontend frontend;
        Qwen3TTSFrontendFiles ff;
        ff.tokenizer = cfg.tokenizer;
        ff.text_embed_param = cfg.text_embed_param;
        ff.text_embed_bin = cfg.text_embed_bin;
        ff.codec_embed_param = cfg.codec_embed_param;
        ff.codec_embed_bin = cfg.codec_embed_bin;
        if (!frontend.load(ff, cli.threads, cli.use_vulkan)) return 1;
        ncnn::Mat prompt = frontend.build_customvoice_prompt(cli.text, cli.language, cli.speaker);
        if (prompt.empty()) return 1;
        if (prof) std::fprintf(stderr, "[qwen3_tts_profile] frontend_load_build_ms=%.3f\n", now_ms() - frontend_t0);
        std::printf("frontend prompt_len=%d text=\"%s\" speaker=%s language=%s\n",
                    prompt.h, cli.text.c_str(), cli.speaker.c_str(), cli.language.c_str());
        const double gen_t0 = prof ? now_ms() : 0.0;
        codes = tts.generate_codes_from_prompt(prompt, frames);
        if (prof) std::fprintf(stderr, "[qwen3_tts_profile] generate_codes_ms=%.3f\n", now_ms() - gen_t0);
    }
    else
    {
        if (!cli.text.empty())
        {
            std::fprintf(stderr,
                         "warning: --text/--speaker/--language were passed, but model.json has no frontend nets; "
                         "using precomputed prefill assets from %s\n",
                         model_json.c_str());
        }
        const double gen_t0 = prof ? now_ms() : 0.0;
        codes = tts.generate_codes(cfg.prefill_embed, cfg.prefill_mask, cfg.prefill_cos, cfg.prefill_sin, frames);
        if (prof) std::fprintf(stderr, "[qwen3_tts_profile] generate_codes_ms=%.3f\n", now_ms() - gen_t0);
    }
    if (codes.empty() || !write_codes(cli.out_codes, codes)) return 1;

    const double wav_t0 = prof ? now_ms() : 0.0;
    std::vector<float> wav = tts.codes_to_wav(codes, frames, 0, frames);
    if (prof) std::fprintf(stderr, "[qwen3_tts_profile] codes_to_wav_ms=%.3f\n", now_ms() - wav_t0);
    if (wav.empty() || !write_wav16(cli.out_wav, wav, cfg.sample_rate)) return 1;
    if (prof) std::fprintf(stderr, "[qwen3_tts_profile] total_ms=%.3f\n", now_ms() - total_t0);

    std::printf("generated frames=%d wav_samples=%zu sample_rate=%d out=%s\n", frames, wav.size(), cfg.sample_rate, cli.out_wav.c_str());
    return 0;
}
