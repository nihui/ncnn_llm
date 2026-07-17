#include "youtu_generation.h"

#include "youtu_model_paths.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <utility>

namespace youtu {

// Text-model execution and autoregressive state.
//
// RunnerContext owns ncnn networks and all 40 layers of KV cache. It provides
// the validated execution paths for legacy per-layer decode, the single
// 40-layer decode graph, fixed-bucket prefill, and dynamic-sequence prefill.

struct RunnerContext::Impl {
    std::string export_dir;
    int threads = 4;
    bool no_packing = true;
    bool force_fp32_compute = false;
    NetFiles embed_files;
    std::vector<int> prefill_buckets;
    NetFiles full_decoder_files_;
    NetFiles full_decoder_prefill_files_;
    NetFiles norm_files;
    NetFiles lm_files;
    std::vector<std::pair<Tensor, Tensor>> caches;
    std::unique_ptr<ncnn::Net> full_decoder_net;
    std::unique_ptr<ncnn::Net> full_decoder_prefill_net;

    Impl(const std::string& dir, int num_threads, bool no_packing_layout, std::vector<int> buckets,
         const std::string& precision)
        : export_dir(dir),
          threads(num_threads),
          no_packing(no_packing_layout),
          force_fp32_compute(precision == "fp16"),
          embed_files(youtu::embed_files(dir, precision)),
          prefill_buckets(std::move(buckets)),
          full_decoder_files_(full_decoder_files(dir)),
          full_decoder_prefill_files_(full_decoder_prefill_files(dir)),
          norm_files{dir + "/youtu_decoder_final_norm.ncnn.param", dir + "/youtu_decoder_final_norm.ncnn.bin"},
          lm_files{dir + "/youtu_lm_head.ncnn.param", dir + "/youtu_lm_head.ncnn.bin"} {}

    void init_empty_cache() {
        caches.clear();
        caches.reserve(40);
        for (int i = 0; i < 40; ++i) {
            Tensor key;
            key.shape = {1, 32, 0, 192};
            Tensor value;
            value.shape = {1, 32, 0, 128};
            caches.push_back({std::move(key), std::move(value)});
        }
    }

    void init_dummy_cache() {
        caches.clear();
        caches.reserve(40);
        for (int i = 0; i < 40; ++i) {
            Tensor key;
            key.shape = {1, 32, 1, 192};
            key.f32.assign(32 * 192, 0.0f);
            Tensor value;
            value.shape = {1, 32, 1, 128};
            value.f32.assign(32 * 128, 0.0f);
            caches.push_back({std::move(key), std::move(value)});
        }
    }

    static void drop_cache_prefix(Tensor& tensor, int drop) {
        if (drop <= 0 || tensor.shape.size() != 4) return;
        const int n = tensor.shape[0];
        const int heads = tensor.shape[1];
        const int length = tensor.shape[2];
        const int dim = tensor.shape[3];
        if (n != 1 || drop > length) die("unsupported cache crop shape=" + shape_string(tensor.shape));
        const size_t row = static_cast<size_t>(dim);
        const size_t old_head_stride = static_cast<size_t>(length) * row;
        const int new_length = length - drop;
        std::vector<float> cropped(static_cast<size_t>(heads) * new_length * dim);
        for (int h = 0; h < heads; ++h) {
            const float* src = tensor.f32.data() + static_cast<size_t>(h) * old_head_stride + static_cast<size_t>(drop) * row;
            float* dst = cropped.data() + static_cast<size_t>(h) * new_length * row;
            std::memcpy(dst, src, static_cast<size_t>(new_length) * row * sizeof(float));
        }
        tensor.shape[2] = new_length;
        tensor.f32 = std::move(cropped);
    }

    void drop_dummy_cache_prefix() {
        for (auto& kv : caches) {
            drop_cache_prefix(kv.first, 1);
            drop_cache_prefix(kv.second, 1);
        }
    }

    static void crop_cache_length(Tensor& tensor, int keep) {
        if (tensor.shape.size() != 4) die("unsupported cache crop-rank shape=" + shape_string(tensor.shape));
        const int n = tensor.shape[0];
        const int heads = tensor.shape[1];
        const int length = tensor.shape[2];
        const int dim = tensor.shape[3];
        if (n != 1 || keep < 0 || keep > length) die("unsupported cache crop shape=" + shape_string(tensor.shape));
        if (keep == length) return;
        const size_t row = static_cast<size_t>(dim);
        const size_t old_head_stride = static_cast<size_t>(length) * row;
        std::vector<float> cropped(static_cast<size_t>(heads) * keep * dim);
        for (int h = 0; h < heads; ++h) {
            const float* src = tensor.f32.data() + static_cast<size_t>(h) * old_head_stride;
            float* dst = cropped.data() + static_cast<size_t>(h) * keep * row;
            std::memcpy(dst, src, static_cast<size_t>(keep) * row * sizeof(float));
        }
        tensor.shape[2] = keep;
        tensor.f32 = std::move(cropped);
    }

    void init_cache_from_dump(std::map<std::string, Tensor>& arrays) {
        caches.clear();
        caches.reserve(40);
        for (int i = 0; i < 40; ++i) {
            char key_name[128];
            char value_name[128];
            std::snprintf(key_name, sizeof(key_name), "decoder_decode_step0_input_cache_layer%02d_key", i);
            std::snprintf(value_name, sizeof(value_name), "decoder_decode_step0_input_cache_layer%02d_value", i);
            caches.push_back({require(arrays, key_name), require(arrays, value_name)});
        }
    }

    static Tensor cache_to_3d(const Tensor& tensor) {
        if (tensor.shape.size() == 3) return tensor;
        if (tensor.shape.size() != 4 || tensor.shape[0] != 1) {
            die("unsupported cache_to_3d shape=" + shape_string(tensor.shape));
        }
        Tensor out = tensor;
        out.shape = {tensor.shape[1], tensor.shape[2], tensor.shape[3]};
        return out;
    }

    static Tensor cache_from_3d(Tensor tensor) {
        if (tensor.shape.size() != 3) {
            die("unsupported cache_from_3d shape=" + shape_string(tensor.shape));
        }
        tensor.shape = {1, tensor.shape[0], tensor.shape[1], tensor.shape[2]};
        return tensor;
    }

    ncnn::Net& ensure_full_decoder_net() {
        if (!full_decoder_net) {
            full_decoder_net = std::make_unique<ncnn::Net>();
            load_net(*full_decoder_net, full_decoder_files_, threads, no_packing, force_fp32_compute);
        }
        return *full_decoder_net;
    }

    ncnn::Net& ensure_full_decoder_prefill_net() {
        if (!full_decoder_prefill_net) {
            full_decoder_prefill_net = std::make_unique<ncnn::Net>();
            load_net(*full_decoder_prefill_net, full_decoder_prefill_files_, threads, no_packing, force_fp32_compute);
        }
        return *full_decoder_prefill_net;
    }

    Tensor run_full_decoder_hidden(Tensor hidden, int position, bool mask_first_cache_slot = false) {
        constexpr int hidden_size = 2560;
        constexpr int rope_dim = 64;
        constexpr float rope_theta = 500000.0f;

        hidden.shape = {1, 1, hidden_size};
        const int cache_length = caches.empty() ? 0 : caches[0].first.shape[2];
        Tensor mask = make_sdpa_mask(cache_length + 1, mask_first_cache_slot);
        auto rope = make_rope(position, rope_theta, rope_dim);

        ncnn::Extractor ex = ensure_full_decoder_net().create_extractor();
        ex.input("in0", mat_from_f32(hidden));
        ex.input("in1", mat_from_f32(mask));
        ex.input("in2", mat_from_f32(rope.first));
        ex.input("in3", mat_from_f32(rope.second));
        for (int layer = 0; layer < 40; ++layer) {
            const Tensor key = cache_to_3d(caches[layer].first);
            const Tensor value = cache_to_3d(caches[layer].second);
            ex.input(("cache_k" + std::to_string(layer)).c_str(), mat_from_f32(key));
            ex.input(("cache_v" + std::to_string(layer)).c_str(), mat_from_f32(value));
        }

        ncnn::Mat hidden_mat;
        if (ex.extract("out0", hidden_mat) != 0) die("extract full decoder out0 failed");
        hidden = tensor_from_mat(hidden_mat, {1, 1, hidden_size});

        for (int layer = 0; layer < 40; ++layer) {
            ncnn::Mat key_mat;
            ncnn::Mat value_mat;
            if (ex.extract(("out_cache_k" + std::to_string(layer)).c_str(), key_mat) != 0) {
                die("extract full decoder out_cache_k failed layer " + std::to_string(layer));
            }
            if (ex.extract(("out_cache_v" + std::to_string(layer)).c_str(), value_mat) != 0) {
                die("extract full decoder out_cache_v failed layer " + std::to_string(layer));
            }
            caches[layer].first = cache_from_3d(tensor_from_mat(key_mat));
            caches[layer].second = cache_from_3d(tensor_from_mat(value_mat));
        }

        return hidden;
    }

    Tensor project_logits(Tensor hidden) {
        constexpr int hidden_size = 2560;
        Tensor final_hidden = run_single(norm_files, hidden, threads, no_packing, {1, 1, hidden_size}, force_fp32_compute);
        return run_single(lm_files, final_hidden, threads, no_packing, {1, 283386}, force_fp32_compute);
    }

    Tensor run_full_decoder_token(int token_id, int position, bool mask_first_cache_slot = false) {
        Tensor hidden = run_embed(embed_files, token_id, threads, no_packing, force_fp32_compute);
        return project_logits(run_full_decoder_hidden(std::move(hidden), position, mask_first_cache_slot));
    }

    Tensor run_token(int token_id, int position, bool mask_first_cache_slot = false) {
        constexpr int hidden_size = 2560;
        constexpr int rope_dim = 64;
        constexpr float rope_theta = 500000.0f;

        Tensor hidden = run_embed(embed_files, token_id, threads, no_packing, force_fp32_compute);
        hidden.shape = {1, 1, hidden_size};
        const int cache_length = caches.empty() ? 0 : caches[0].first.shape[2];
        Tensor mask = make_mask(cache_length + 1, mask_first_cache_slot);
        auto rope = make_rope(position, rope_theta, rope_dim);

        for (int layer = 0; layer < 40; ++layer) {
            auto outs = run_layer_net(layer_files(export_dir, layer), hidden, mask, caches[layer].first, caches[layer].second, rope.first, rope.second, threads, no_packing);
            hidden = std::move(outs[0]);
            caches[layer].first = std::move(outs[1]);
            caches[layer].second = std::move(outs[2]);
        }

        Tensor final_hidden = run_single(norm_files, hidden, threads, no_packing, {1, 1, hidden_size}, force_fp32_compute);
        return run_single(lm_files, final_hidden, threads, no_packing, {1, 283386}, force_fp32_compute);
    }

    int select_prefill_bucket(int prompt_length) const {
        for (int bucket : prefill_buckets) {
            if (bucket < prompt_length) continue;
            const NetFiles files = prefill_files(export_dir, bucket);
            if (file_exists(files.param) && file_exists(files.bin)) return bucket;
        }
        return -1;
    }

    bool has_dynamic_full_decoder_prefill() const {
        return file_exists(full_decoder_prefill_files_.param) && file_exists(full_decoder_prefill_files_.bin);
    }

    static Tensor flatten_prefill_embeds(Tensor embeds) {
        constexpr int hidden_size = 2560;
        if (embeds.shape.size() == 3 && embeds.shape[0] == 1 && embeds.shape[2] == hidden_size) {
            embeds.shape = {embeds.shape[1], hidden_size};
            return embeds;
        }
        if (embeds.shape.size() == 2 && embeds.shape[1] == hidden_size) {
            return embeds;
        }
        die("unsupported prefill embeds shape=" + shape_string(embeds.shape));
    }

    Tensor run_full_decoder_prefill_embeds(Tensor embeds, int start_position) {
        constexpr int hidden_size = 2560;
        constexpr int rope_dim = 64;
        constexpr float rope_theta = 500000.0f;
        embeds = flatten_prefill_embeds(std::move(embeds));
        const int prompt_length = embeds.shape[0];
        if (prompt_length <= 0) die("empty prefill embeds");
        Tensor mask = make_prefill_mask(prompt_length, false);
        auto rope = make_rope_table(start_position, prompt_length, rope_theta, rope_dim);

        ncnn::Extractor ex = ensure_full_decoder_prefill_net().create_extractor();
        ex.input("in0", mat_from_f32(embeds));
        ex.input("in1", mat_from_f32(mask));
        ex.input("in2", mat_from_f32(rope.first));
        ex.input("in3", mat_from_f32(rope.second));

        ncnn::Mat hidden_mat;
        if (ex.extract("out0", hidden_mat) != 0) die("extract full decoder prefill out0 failed");
        Tensor hidden = tensor_from_mat(hidden_mat, {prompt_length, hidden_size});

        caches.clear();
        caches.reserve(40);
        for (int layer = 0; layer < 40; ++layer) {
            ncnn::Mat key_mat;
            ncnn::Mat value_mat;
            if (ex.extract(("out_cache_k" + std::to_string(layer)).c_str(), key_mat) != 0) {
                die("extract full decoder prefill out_cache_k failed layer " + std::to_string(layer));
            }
            if (ex.extract(("out_cache_v" + std::to_string(layer)).c_str(), value_mat) != 0) {
                die("extract full decoder prefill out_cache_v failed layer " + std::to_string(layer));
            }
            Tensor key = cache_from_3d(tensor_from_mat(key_mat));
            Tensor value = cache_from_3d(tensor_from_mat(value_mat));
            crop_cache_length(key, prompt_length);
            crop_cache_length(value, prompt_length);
            caches.push_back({std::move(key), std::move(value)});
        }

        Tensor last_hidden;
        last_hidden.shape = {1, 1, hidden_size};
        last_hidden.f32.resize(hidden_size);
        const size_t offset = static_cast<size_t>(prompt_length - 1) * hidden_size;
        std::memcpy(last_hidden.f32.data(), hidden.f32.data() + offset, hidden_size * sizeof(float));
        Tensor final_hidden = run_single(norm_files, last_hidden, threads, no_packing, {1, 1, hidden_size}, force_fp32_compute);
        return run_single(lm_files, final_hidden, threads, no_packing, {1, 283386}, force_fp32_compute);
    }

    Tensor run_full_decoder_prefill_embeds_tokenwise(Tensor embeds, int start_position) {
        constexpr int hidden_size = 2560;
        embeds = flatten_prefill_embeds(std::move(embeds));
        const int prompt_length = embeds.shape[0];
        if (prompt_length <= 0) die("empty prefill embeds");
        if (caches.empty()) init_dummy_cache();

        Tensor last_hidden;
        for (int i = 0; i < prompt_length; ++i) {
            Tensor hidden;
            hidden.shape = {1, 1, hidden_size};
            const size_t offset = static_cast<size_t>(i) * hidden_size;
            hidden.f32.assign(embeds.f32.begin() + offset, embeds.f32.begin() + offset + hidden_size);
            const bool first_dummy_token = i == 0 && !caches.empty() && caches[0].first.shape[2] == 1;
            last_hidden = run_full_decoder_hidden(std::move(hidden), start_position + i, first_dummy_token);
            if (first_dummy_token) drop_dummy_cache_prefix();
        }
        return project_logits(std::move(last_hidden));
    }

    Tensor run_full_decoder_prefill(const std::vector<int>& token_ids, int start_position) {
        Tensor prompt_embeds = run_embed_sequence(embed_files, token_ids, threads, no_packing, force_fp32_compute);
        return run_full_decoder_prefill_embeds(std::move(prompt_embeds), start_position);
    }

    Tensor run_full_prefill(const std::vector<int>& token_ids, int start_position, int bucket_length) {
        constexpr int hidden_size = 2560;
        constexpr int rope_dim = 64;
        constexpr float rope_theta = 500000.0f;
        const int prompt_length = static_cast<int>(token_ids.size());
        if (bucket_length < prompt_length) {
            die("prefill bucket " + std::to_string(bucket_length) + " is smaller than prompt length " + std::to_string(prompt_length));
        }
        const NetFiles files = prefill_files(export_dir, bucket_length);

        Tensor prompt_embeds = run_embed_sequence(embed_files, token_ids, threads, no_packing, force_fp32_compute);
        Tensor embeds;
        embeds.shape = {bucket_length, hidden_size};
        embeds.f32.assign(static_cast<size_t>(bucket_length) * hidden_size, 0.0f);
        std::memcpy(embeds.f32.data(), prompt_embeds.f32.data(), static_cast<size_t>(prompt_length) * hidden_size * sizeof(float));
        Tensor mask = make_prefill_mask(bucket_length);
        auto rope = make_rope_table(start_position, bucket_length, rope_theta, rope_dim);

        ncnn::Net net;
        load_net(net, files, threads, no_packing, force_fp32_compute);
        ncnn::Extractor ex = net.create_extractor();
        ex.input("in0", mat_from_f32(embeds));
        ex.input("in1", mat_from_f32(mask));
        ex.input("in2", mat_from_f32(rope.first));
        ex.input("in3", mat_from_f32(rope.second));

        ncnn::Mat hidden_mat;
        if (ex.extract("out0", hidden_mat) != 0) die("extract prefill out0 failed");
        Tensor hidden = tensor_from_mat(hidden_mat, {1, bucket_length, hidden_size});

        caches.clear();
        caches.reserve(40);
        for (int layer = 0; layer < 40; ++layer) {
            ncnn::Mat key_mat;
            ncnn::Mat value_mat;
            if (ex.extract(("out" + std::to_string(1 + layer * 2)).c_str(), key_mat) != 0) {
                die("extract prefill key failed layer " + std::to_string(layer));
            }
            if (ex.extract(("out" + std::to_string(2 + layer * 2)).c_str(), value_mat) != 0) {
                die("extract prefill value failed layer " + std::to_string(layer));
            }
            Tensor key = tensor_from_mat(key_mat, {1, 32, bucket_length, 192});
            Tensor value = tensor_from_mat(value_mat, {1, 32, bucket_length, 128});
            crop_cache_length(key, prompt_length);
            crop_cache_length(value, prompt_length);
            caches.push_back({std::move(key), std::move(value)});
        }

        Tensor last_hidden;
        last_hidden.shape = {1, 1, hidden_size};
        last_hidden.f32.resize(hidden_size);
        const size_t offset = static_cast<size_t>(prompt_length - 1) * hidden_size;
        std::memcpy(last_hidden.f32.data(), hidden.f32.data() + offset, hidden_size * sizeof(float));
        return run_single(lm_files, last_hidden, threads, no_packing, {1, 283386}, force_fp32_compute);
    }
};

RunnerContext::RunnerContext(const std::string& dir, int num_threads, bool no_packing_layout,
                             std::vector<int> buckets, const std::string& precision)
    : impl_(std::make_unique<Impl>(dir, num_threads, no_packing_layout, std::move(buckets), precision)) {}

RunnerContext::~RunnerContext() = default;
RunnerContext::RunnerContext(RunnerContext&&) noexcept = default;
RunnerContext& RunnerContext::operator=(RunnerContext&&) noexcept = default;

void RunnerContext::init_empty_cache() { impl_->init_empty_cache(); }
void RunnerContext::init_dummy_cache() { impl_->init_dummy_cache(); }
void RunnerContext::drop_dummy_cache_prefix() { impl_->drop_dummy_cache_prefix(); }
void RunnerContext::init_cache_from_dump(std::map<std::string, Tensor>& arrays) {
    impl_->init_cache_from_dump(arrays);
}
Tensor RunnerContext::run_full_decoder_token(int token_id, int position, bool mask_first_cache_slot) {
    return impl_->run_full_decoder_token(token_id, position, mask_first_cache_slot);
}
Tensor RunnerContext::run_token(int token_id, int position, bool mask_first_cache_slot) {
    return impl_->run_token(token_id, position, mask_first_cache_slot);
}
int RunnerContext::select_prefill_bucket(int prompt_length) const {
    return impl_->select_prefill_bucket(prompt_length);
}
bool RunnerContext::has_dynamic_full_decoder_prefill() const {
    return impl_->has_dynamic_full_decoder_prefill();
}
Tensor RunnerContext::run_full_decoder_prefill_embeds(Tensor embeds, int start_position) {
    return impl_->run_full_decoder_prefill_embeds(std::move(embeds), start_position);
}
Tensor RunnerContext::run_full_decoder_prefill_embeds_tokenwise(Tensor embeds, int start_position) {
    return impl_->run_full_decoder_prefill_embeds_tokenwise(std::move(embeds), start_position);
}
Tensor RunnerContext::run_full_decoder_prefill(const std::vector<int>& token_ids, int start_position) {
    return impl_->run_full_decoder_prefill(token_ids, start_position);
}
Tensor RunnerContext::run_full_prefill(const std::vector<int>& token_ids, int start_position,
                                       int bucket_length) {
    return impl_->run_full_prefill(token_ids, start_position, bucket_length);
}

bool compare_cache_sets(const RunnerContext& actual, const RunnerContext& expected, float atol) {
    if (!actual.impl_ || !expected.impl_) die("cannot compare an uninitialized runner");
    const auto& actual_caches = actual.impl_->caches;
    const auto& expected_caches = expected.impl_->caches;
    if (actual_caches.size() != expected_caches.size()) {
        std::cout << "[FAIL] prefill.cache layer_count actual=" << actual_caches.size()
                  << " expected=" << expected_caches.size() << "\n";
        return false;
    }
    bool ok = true;
    float worst = -1.0f;
    std::string worst_name;
    for (size_t layer = 0; layer < actual_caches.size(); ++layer) {
        const Tensor& actual_key = actual_caches[layer].first;
        const Tensor& expected_key = expected_caches[layer].first;
        const Tensor& actual_value = actual_caches[layer].second;
        const Tensor& expected_value = expected_caches[layer].second;
        if (actual_key.shape != expected_key.shape || actual_value.shape != expected_value.shape) {
            std::cout << "[FAIL] prefill.cache.layer" << layer
                      << " key_shape=" << shape_string(actual_key.shape)
                      << " expected_key_shape=" << shape_string(expected_key.shape)
                      << " value_shape=" << shape_string(actual_value.shape)
                      << " expected_value_shape=" << shape_string(expected_value.shape) << "\n";
            ok = false;
            continue;
        }
        const DiffStats key_stats = diff_stats(actual_key, expected_key);
        const DiffStats value_stats = diff_stats(actual_value, expected_value);
        if (key_stats.max_abs > worst) {
            worst = key_stats.max_abs;
            worst_name = "layer" + std::to_string(layer) + ".key";
        }
        if (value_stats.max_abs > worst) {
            worst = value_stats.max_abs;
            worst_name = "layer" + std::to_string(layer) + ".value";
        }
        ok = (key_stats.max_abs <= atol) && (value_stats.max_abs <= atol) && ok;
    }
    std::cout << "[" << (ok ? "PASS" : "FAIL") << "] prefill.cache.dynamic_vs_token"
              << ": max_abs=" << worst << " worst=" << worst_name
              << " atol=" << atol << "\n";
    return ok;
}

} // namespace youtu
