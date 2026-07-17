#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "net.h"

namespace youtu {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

// Framework-neutral tensor used at module boundaries. Exactly one storage
// vector is populated according to the source dtype.
struct Tensor {
    std::vector<int> shape;
    std::vector<float> f32;
    std::vector<int64_t> i64;
};

// Parsed command-line configuration. Paths are resolved after parsing so the
// same structure works for source-tree and standalone release layouts.
struct Options {
    std::string root;
    std::string precision = "fp32";
    std::string npz;
    std::string export_dir;
    std::string vision_export_dir;
    std::string vision_stem = "youtu_vl_siglip2_merger_fixed_mmperm_mask1e4";
    std::string vision_encoder_stem = "youtu_vl_siglip2_encoder_dynamic";
    std::string vision_post_merger_stem = "youtu_vl_post_merger_dynamic";
    std::string tokenizer_json;
    std::string prefill_embeds_npz;
    std::string vision_npz;
    std::string image_path;
    std::string prompt;
    std::string output_path;
    std::string ids;
    int steps = 7;
    int max_new_tokens = 32;
    int image_patch_size = 16;
    int max_image_patches = 512;
    int num_threads = 4;
    bool no_packing_layout = true;
    float logits_atol = 1.0f;
    float cache_atol = 2.0f;
    int topk = 10;
    bool chat = true;
    bool echo_prompt = false;
    bool cache_from_npz = false;
    bool tokenize_only = false;
    bool vision_preprocess_only = false;
    bool vision_only = false;
    bool fixed_vision_graph = false;
    bool compare_prefill = false;
    bool full_prefill_ncnn = false;
    bool full_decoder_ncnn = false;
    std::string prefill_buckets = "33";
};

struct NetFiles {
    std::string param;
    std::string bin;
};

struct DiffStats {
    float max_abs = 0.0f;
    double mean_abs = 0.0;
    double rms = 0.0;
};

[[noreturn]] void die(const std::string& message);

std::string discover_project_root(const char* argv0);
std::string project_path(const std::string& root, const fs::path& relative);
fs::path runtime_package_root(const std::string& root);
double elapsed_ms(Clock::time_point start, Clock::time_point end);

std::vector<unsigned char> read_file(const std::string& path);
std::string read_text_file(const std::string& path);
std::string trim(const std::string& s);
void append_utf8(std::string& out, uint32_t cp);
std::string json_unescape(const std::string& s);
std::map<std::string, Tensor> load_npz(const std::string& path);
Tensor& require(std::map<std::string, Tensor>& arrays, const std::string& key);
const Tensor& require_const(const std::map<std::string, Tensor>& arrays, const std::string& key);

ncnn::Mat mat_from_f32(const Tensor& t);
ncnn::Mat mat_from_f32_shape(const std::vector<float>& data, const std::vector<int>& shape);
std::vector<float> mat_to_vector(const ncnn::Mat& m);
std::vector<int> mat_shape_nchw(const ncnn::Mat& m);
std::string shape_string(const std::vector<int>& shape);
bool file_exists(const std::string& path);
Tensor tensor_from_mat(const ncnn::Mat& m, const std::vector<int>& forced_shape = {});

void configure_net_options(ncnn::Net& net, int threads, bool no_packing, bool force_fp32 = false);
void load_net(ncnn::Net& net, const NetFiles& files, int threads, bool no_packing, bool force_fp32 = false);
Tensor run_single(const NetFiles& files, const Tensor& input, int threads, bool no_packing,
                  const std::vector<int>& forced_shape = {}, bool force_fp32 = false);
Tensor run_embed(const NetFiles& files, int token_id, int threads, bool no_packing, bool force_fp32 = false);
Tensor run_embed_sequence(const NetFiles& files, const std::vector<int>& token_ids, int threads, bool no_packing,
                          bool force_fp32 = false);
std::array<Tensor, 3> run_layer_net(const NetFiles& files, const Tensor& hidden, const Tensor& mask,
                                    const Tensor& key, const Tensor& value, const Tensor& cos,
                                    const Tensor& sin, int threads, bool no_packing);

Tensor make_mask(int total_key_length, bool mask_first_slot = false);
Tensor make_sdpa_mask(int total_key_length, bool mask_first_slot = false);
Tensor make_prefill_mask(int bucket_length, bool include_dummy_cache_column = true);
std::pair<Tensor, Tensor> make_rope(int position, float theta, int dim);
std::pair<Tensor, Tensor> make_rope_table(int start_position, int seq_length, float theta, int dim);

DiffStats diff_stats(const Tensor& actual, const Tensor& expected);
bool print_stats_line(const std::string& name, const DiffStats& st, float atol);
bool print_i64_compare_line(const std::string& name, const Tensor& actual, const Tensor& expected);
std::vector<int> topk_ids(const Tensor& t, int k);
int argmax_id(const Tensor& t);
std::string join_ids(const std::vector<int>& ids);
std::vector<int> parse_id_list(const std::string& text);
std::vector<int> parse_int_list(const std::string& text);

// JSON/token-byte helpers shared with the tokenizer implementation.
std::string byte_encoder_piece(unsigned char b);
std::unordered_map<std::string, std::string> build_byte_decoder();
size_t find_matching(const std::string& s, size_t open_pos, char open_ch, char close_ch);
std::string json_string_at(const std::string& s, size_t& pos);

} // namespace youtu
