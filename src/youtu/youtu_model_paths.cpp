#include "youtu_model_paths.h"

#include <cstdio>

namespace youtu {

// Canonical ncnn model filenames.
//
// Keeping path construction in one place prevents the production runtime and
// historical parity modes from silently selecting different graph variants.

NetFiles embed_files(const std::string& dir, const std::string& precision) {
    const std::string release_stem = dir + "/youtu_embed_tokens_" + precision;
    if (file_exists(release_stem + ".ncnn.param")) {
        return {release_stem + ".ncnn.param", release_stem + ".ncnn.bin"};
    }
    const std::string export_stem = dir + "/youtu_embed_tokens";
    return {export_stem + ".ncnn.param", export_stem + ".ncnn.bin"};
}

NetFiles layer_files(const std::string& dir, int layer) {
    char name[128];
    std::snprintf(name, sizeof(name), "youtu_decoder_layer%02d_decode_step0_manual_kv", layer);
    return {dir + "/" + name + ".ncnn.patched.param", dir + "/" + name + ".ncnn.bin"};
}

NetFiles prefill_files(const std::string& dir, int bucket_length) {
    if (bucket_length == 33) {
        return {dir + "/youtu_decoder_prefill_manual.ncnn.patched.param", dir + "/youtu_decoder_prefill_manual.ncnn.bin"};
    }
    const std::string stem = dir + "/youtu_decoder_prefill_manual_s" + std::to_string(bucket_length);
    return {stem + ".ncnn.patched.param", stem + ".ncnn.bin"};
}

NetFiles full_decoder_files(const std::string& dir) {
    const std::string stem = dir + "/youtu_decoder_full_manual_kv_sdpa_ncnn";
    return {stem + ".ncnn.patched.param", stem + ".ncnn.bin"};
}

NetFiles full_decoder_prefill_files(const std::string& dir) {
    return {
        dir + "/youtu_decoder_full_prefill_manual_kv_sdpa_ncnn.dynamic.patched.param",
        dir + "/youtu_decoder_full_prefill_manual_kv_sdpa_ncnn_s33.ncnn.bin",
    };
}

NetFiles vision_files(const std::string& dir, const std::string& stem) {
    return {dir + "/" + stem + ".ncnn.param", dir + "/" + stem + ".ncnn.bin"};
}

NetFiles dynamic_vision_files(const std::string& dir, const std::string& stem) {
    const std::string patched = dir + "/" + stem + ".ncnn.dynamic.patched.param";
    const std::string raw = dir + "/" + stem + ".ncnn.param";
    return {file_exists(patched) ? patched : raw, dir + "/" + stem + ".ncnn.bin"};
}

} // namespace youtu
