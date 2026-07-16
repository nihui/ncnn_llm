#include "youtu_vision.h"

#include "youtu_tokenizer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <utility>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_THREAD_LOCALS
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#include "stb_image.h"

namespace youtu {

// Youtu-VL/SigLIP2 image path.
//
// Handles image decoding, aspect-ratio-preserving antialiased resize, patch
// flattening, dynamic window/RoPE/mask inputs, vision ncnn execution, reverse
// window ordering, merger execution, and image-token embedding injection.

int ceil_div_to_multiple(int value, int multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

std::pair<int, int> get_image_size_for_patches_cpp(int image_height, int image_width, int patch_size, int max_num_patches) {
    const int patch_unit = patch_size * 2;
    // Match Python's float (C double) accumulation at resize-grid boundaries.
    double scale = 1.0;
    while (true) {
        const int target_height = std::max(patch_unit, ceil_div_to_multiple(static_cast<int>(std::ceil(image_height * scale)), patch_unit));
        const int target_width = std::max(patch_unit, ceil_div_to_multiple(static_cast<int>(std::ceil(image_width * scale)), patch_unit));
        const int num_patches = (target_height / patch_size) * (target_width / patch_size);
        if (num_patches <= max_num_patches) return {target_height, target_width};
        scale -= 0.02;
        if (scale <= 0.0) die("failed to find image resize scale");
    }
}

struct ResizeWeight {
    int index = 0;
    float weight = 0.0f;
};

std::vector<std::vector<ResizeWeight>> make_linear_resize_weights(int in_size, int out_size) {
    std::vector<std::vector<ResizeWeight>> all_weights(out_size);
    const double scale = static_cast<double>(in_size) / static_cast<double>(out_size);
    const double filter_scale = std::max(1.0, scale);
    const double support = filter_scale;
    for (int out = 0; out < out_size; ++out) {
        const double center = (static_cast<double>(out) + 0.5) * scale - 0.5;
        const int xmin = std::max(0, static_cast<int>(std::ceil(center - support)));
        const int xmax = std::min(in_size - 1, static_cast<int>(std::floor(center + support)));
        double sum = 0.0;
        std::vector<ResizeWeight> weights;
        weights.reserve(static_cast<size_t>(xmax - xmin + 1));
        for (int in = xmin; in <= xmax; ++in) {
            const double x = (static_cast<double>(in) - center) / filter_scale;
            const double w = std::max(0.0, 1.0 - std::abs(x));
            if (w <= 0.0) continue;
            weights.push_back({in, static_cast<float>(w)});
            sum += w;
        }
        if (weights.empty()) {
            const int nearest = std::max(0, std::min(static_cast<int>(std::round(center)), in_size - 1));
            weights.push_back({nearest, 1.0f});
            sum = 1.0;
        }
        for (ResizeWeight& w : weights) w.weight = static_cast<float>(static_cast<double>(w.weight) / sum);
        all_weights[out] = std::move(weights);
    }
    return all_weights;
}

std::vector<unsigned char> resize_rgb_bilinear_antialias_u8(const unsigned char* rgb, int width, int height, int target_width, int target_height) {
    const std::vector<std::vector<ResizeWeight>> x_weights = make_linear_resize_weights(width, target_width);
    const std::vector<std::vector<ResizeWeight>> y_weights = make_linear_resize_weights(height, target_height);

    std::vector<float> tmp(static_cast<size_t>(height) * target_width * 3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < target_width; ++x) {
            for (int c = 0; c < 3; ++c) {
                float value = 0.0f;
                for (const ResizeWeight& wx : x_weights[x]) {
                    value += static_cast<float>(rgb[(static_cast<size_t>(y) * width + wx.index) * 3 + c]) * wx.weight;
                }
                tmp[(static_cast<size_t>(y) * target_width + x) * 3 + c] = value;
            }
        }
    }

    std::vector<unsigned char> out(static_cast<size_t>(target_height) * target_width * 3);
    for (int y = 0; y < target_height; ++y) {
        for (int x = 0; x < target_width; ++x) {
            for (int c = 0; c < 3; ++c) {
                float value = 0.0f;
                for (const ResizeWeight& wy : y_weights[y]) {
                    value += tmp[(static_cast<size_t>(wy.index) * target_width + x) * 3 + c] * wy.weight;
                }
                value = std::max(0.0f, std::min(255.0f, value));
                out[(static_cast<size_t>(y) * target_width + x) * 3 + c] = static_cast<unsigned char>(std::round(value));
            }
        }
    }
    return out;
}

ImagePreprocessResult preprocess_siglip2_image(const std::string& image_path, int patch_size, int max_num_patches) {
    if (patch_size <= 0) die("image patch size must be positive");
    if (max_num_patches <= 0) die("max image patches must be positive");
    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* decoded = stbi_load(image_path.c_str(), &width, &height, &channels, 3);
    if (!decoded) die("failed to load image: " + image_path);
    std::unique_ptr<unsigned char, void (*)(void*)> image(decoded, stbi_image_free);

    const auto target_hw = get_image_size_for_patches_cpp(height, width, patch_size, max_num_patches);
    const int target_height = target_hw.first;
    const int target_width = target_hw.second;
    const int num_patches_height = target_height / patch_size;
    const int num_patches_width = target_width / patch_size;
    const int num_patches = num_patches_height * num_patches_width;
    if (num_patches_height % 2 != 0 || num_patches_width % 2 != 0) {
        die("preprocessed patch grid is not divisible by merge_size=2");
    }

    const std::vector<unsigned char> resized = resize_rgb_bilinear_antialias_u8(image.get(), width, height, target_width, target_height);
    std::vector<float> chw(static_cast<size_t>(3) * target_height * target_width);
    for (int y = 0; y < target_height; ++y) {
        for (int x = 0; x < target_width; ++x) {
            for (int c = 0; c < 3; ++c) {
                const float pixel = static_cast<float>(resized[(static_cast<size_t>(y) * target_width + x) * 3 + c]);
                chw[(static_cast<size_t>(c) * target_height + y) * target_width + x] = pixel / 127.5f - 1.0f;
            }
        }
    }

    Tensor patches;
    patches.shape = {1, num_patches, 3 * patch_size * patch_size};
    patches.f32.reserve(static_cast<size_t>(num_patches) * 3 * patch_size * patch_size);
    const int merge_size = 2;
    for (int gh = 0; gh < num_patches_height / merge_size; ++gh) {
        for (int gw = 0; gw < num_patches_width / merge_size; ++gw) {
            for (int mh = 0; mh < merge_size; ++mh) {
                for (int mw = 0; mw < merge_size; ++mw) {
                    for (int py = 0; py < patch_size; ++py) {
                        const int y = (gh * merge_size + mh) * patch_size + py;
                        for (int px = 0; px < patch_size; ++px) {
                            const int x = (gw * merge_size + mw) * patch_size + px;
                            for (int c = 0; c < 3; ++c) {
                                patches.f32.push_back(chw[(static_cast<size_t>(c) * target_height + y) * target_width + x]);
                            }
                        }
                    }
                }
            }
        }
    }

    Tensor mask;
    mask.shape = {1, num_patches};
    mask.i64.assign(num_patches, 1);

    Tensor spatial_shapes;
    spatial_shapes.shape = {1, 2};
    spatial_shapes.i64 = {num_patches_height, num_patches_width};

    ImagePreprocessResult result;
    result.pixel_values = std::move(patches);
    result.pixel_attention_mask = std::move(mask);
    result.spatial_shapes = std::move(spatial_shapes);
    result.input_width = width;
    result.input_height = height;
    result.resized_width = target_width;
    result.resized_height = target_height;
    result.num_patches = num_patches;
    result.image_token_count = num_patches / 4;
    return result;
}

struct VisionRuntimeInputs {
    Tensor pixel_values_windowed;
    Tensor cos;
    Tensor sin;
    Tensor mask_window;
    Tensor mask_full;
    std::vector<int> reverse_index;
    int seq_len = 0;
    int groups = 0;
};

std::vector<int> make_window_index(int grid_h, int grid_w, std::vector<int>& cu_window_seqlens) {
    constexpr int merge_size = 2;
    constexpr int spatial_merge_unit = merge_size * merge_size;
    constexpr int vit_merger_window_size = 8;
    const int llm_grid_h = grid_h / merge_size;
    const int llm_grid_w = grid_w / merge_size;
    const int pad_h = (vit_merger_window_size - llm_grid_h % vit_merger_window_size) % vit_merger_window_size;
    const int pad_w = (vit_merger_window_size - llm_grid_w % vit_merger_window_size) % vit_merger_window_size;
    const int padded_h = llm_grid_h + pad_h;
    const int padded_w = llm_grid_w + pad_w;
    const int num_windows_h = padded_h / vit_merger_window_size;
    const int num_windows_w = padded_w / vit_merger_window_size;

    std::vector<int> window_index;
    cu_window_seqlens.clear();
    cu_window_seqlens.push_back(0);
    for (int wh = 0; wh < num_windows_h; ++wh) {
        for (int ww = 0; ww < num_windows_w; ++ww) {
            int seqlen_groups = 0;
            for (int ih = 0; ih < vit_merger_window_size; ++ih) {
                for (int iw = 0; iw < vit_merger_window_size; ++iw) {
                    const int gh = wh * vit_merger_window_size + ih;
                    const int gw = ww * vit_merger_window_size + iw;
                    if (gh >= llm_grid_h || gw >= llm_grid_w) continue;
                    window_index.push_back(gh * llm_grid_w + gw);
                    seqlen_groups += 1;
                }
            }
            cu_window_seqlens.push_back(cu_window_seqlens.back() + seqlen_groups * spatial_merge_unit);
        }
    }

    std::vector<int> unique_cu;
    unique_cu.reserve(cu_window_seqlens.size());
    for (int v : cu_window_seqlens) {
        if (unique_cu.empty() || unique_cu.back() != v) unique_cu.push_back(v);
    }
    cu_window_seqlens = std::move(unique_cu);
    return window_index;
}

std::vector<int> argsort_inverse(const std::vector<int>& index) {
    std::vector<int> reverse(index.size());
    for (size_t i = 0; i < index.size(); ++i) {
        if (index[i] < 0 || index[i] >= static_cast<int>(index.size())) die("bad window index");
        reverse[static_cast<size_t>(index[i])] = static_cast<int>(i);
    }
    return reverse;
}

Tensor make_vision_mask(int seq_len, const std::vector<int>& cu_seqlens, float mask_value) {
    Tensor mask;
    mask.shape = {1, seq_len, seq_len};
    mask.f32.assign(static_cast<size_t>(seq_len) * seq_len, mask_value);
    for (size_t i = 1; i < cu_seqlens.size(); ++i) {
        const int start = cu_seqlens[i - 1];
        const int end = cu_seqlens[i];
        if (start < 0 || end < start || end > seq_len) die("bad vision cu_seqlens");
        for (int y = start; y < end; ++y) {
            float* row = mask.f32.data() + static_cast<size_t>(y) * seq_len;
            std::fill(row + start, row + end, 0.0f);
        }
    }
    return mask;
}

VisionRuntimeInputs make_vision_runtime_inputs(const ImagePreprocessResult& pp, float mask_value = -10000.0f) {
    constexpr int patch_dim = 768;
    constexpr int merge_size = 2;
    constexpr int spatial_merge_unit = merge_size * merge_size;
    constexpr int rope_half_dim = 18;
    constexpr int rotary_dim = 36;
    constexpr int rope_dim = 72;
    if (pp.spatial_shapes.i64.size() != 2) {
        die("unexpected spatial_shapes count=" + std::to_string(pp.spatial_shapes.i64.size()));
    }
    const int grid_h = static_cast<int>(pp.spatial_shapes.i64[0]);
    const int grid_w = static_cast<int>(pp.spatial_shapes.i64[1]);
    const int seq_len = grid_h * grid_w;
    const int groups = seq_len / spatial_merge_unit;
    if (pp.pixel_values.shape.size() != 3 || pp.pixel_values.shape[1] != seq_len || pp.pixel_values.shape[2] != patch_dim) {
        die("unexpected preprocessed pixel_values shape=" + shape_string(pp.pixel_values.shape));
    }
    if (pp.pixel_values.f32.size() != static_cast<size_t>(seq_len) * patch_dim) {
        die("unexpected preprocessed pixel_values count=" + std::to_string(pp.pixel_values.f32.size()));
    }
    if (grid_h % merge_size != 0 || grid_w % merge_size != 0) die("vision grid must be divisible by merge_size");

    std::vector<int> cu_window;
    const std::vector<int> window_index = make_window_index(grid_h, grid_w, cu_window);
    if (static_cast<int>(window_index.size()) != groups) die("window_index size mismatch");
    const std::vector<int> reverse_index = argsort_inverse(window_index);

    VisionRuntimeInputs rt;
    rt.seq_len = seq_len;
    rt.groups = groups;
    rt.reverse_index = reverse_index;
    rt.pixel_values_windowed.shape = {seq_len, patch_dim};
    rt.pixel_values_windowed.f32.resize(static_cast<size_t>(seq_len) * patch_dim);
    for (int dst_group = 0; dst_group < groups; ++dst_group) {
        const int src_group = window_index[dst_group];
        const float* src = pp.pixel_values.f32.data() + static_cast<size_t>(src_group) * spatial_merge_unit * patch_dim;
        float* dst = rt.pixel_values_windowed.f32.data() + static_cast<size_t>(dst_group) * spatial_merge_unit * patch_dim;
        std::memcpy(dst, src, static_cast<size_t>(spatial_merge_unit) * patch_dim * sizeof(float));
    }

    std::vector<int> hpos(seq_len);
    std::vector<int> wpos(seq_len);
    int p = 0;
    for (int gh = 0; gh < grid_h / merge_size; ++gh) {
        for (int gw = 0; gw < grid_w / merge_size; ++gw) {
            for (int mh = 0; mh < merge_size; ++mh) {
                for (int mw = 0; mw < merge_size; ++mw) {
                    hpos[p] = gh * merge_size + mh;
                    wpos[p] = gw * merge_size + mw;
                    ++p;
                }
            }
        }
    }

    std::array<float, rope_half_dim> inv_freq{};
    for (int i = 0; i < rope_half_dim; ++i) {
        inv_freq[i] = 1.0f / std::pow(10000.0f, static_cast<float>(i * 2) / static_cast<float>(rotary_dim));
    }

    rt.cos.shape = {seq_len, 1, rope_dim};
    rt.sin.shape = {seq_len, 1, rope_dim};
    rt.cos.f32.resize(static_cast<size_t>(seq_len) * rope_dim);
    rt.sin.f32.resize(static_cast<size_t>(seq_len) * rope_dim);
    for (int dst_group = 0; dst_group < groups; ++dst_group) {
        const int src_group = window_index[dst_group];
        for (int local = 0; local < spatial_merge_unit; ++local) {
            const int src_patch = src_group * spatial_merge_unit + local;
            const int dst_patch = dst_group * spatial_merge_unit + local;
            float rotary[rotary_dim];
            for (int i = 0; i < rope_half_dim; ++i) {
                rotary[i] = static_cast<float>(hpos[src_patch]) * inv_freq[i];
                rotary[rope_half_dim + i] = static_cast<float>(wpos[src_patch]) * inv_freq[i];
            }
            for (int i = 0; i < rotary_dim; ++i) {
                const float c = std::cos(rotary[i]);
                const float s = std::sin(rotary[i]);
                rt.cos.f32[static_cast<size_t>(dst_patch) * rope_dim + i] = c;
                rt.sin.f32[static_cast<size_t>(dst_patch) * rope_dim + i] = s;
                rt.cos.f32[static_cast<size_t>(dst_patch) * rope_dim + rotary_dim + i] = c;
                rt.sin.f32[static_cast<size_t>(dst_patch) * rope_dim + rotary_dim + i] = s;
            }
        }
    }

    rt.mask_window = make_vision_mask(seq_len, cu_window, mask_value);
    rt.mask_full = make_vision_mask(seq_len, {0, seq_len}, mask_value);
    return rt;
}

Tensor reverse_window_order(Tensor hidden_windowed, const std::vector<int>& reverse_index) {
    constexpr int spatial_merge_unit = 4;
    constexpr int hidden_size = 1152;
    if (hidden_windowed.shape.size() != 2 || hidden_windowed.shape[1] != hidden_size) {
        die("unexpected vision hidden shape=" + shape_string(hidden_windowed.shape));
    }
    const int seq_len = hidden_windowed.shape[0];
    const int groups = seq_len / spatial_merge_unit;
    if (static_cast<int>(reverse_index.size()) != groups) die("reverse_index size mismatch");
    Tensor hidden;
    hidden.shape = {seq_len, hidden_size};
    hidden.f32.resize(static_cast<size_t>(seq_len) * hidden_size);
    for (int dst_group = 0; dst_group < groups; ++dst_group) {
        const int src_group = reverse_index[dst_group];
        const float* src = hidden_windowed.f32.data() + static_cast<size_t>(src_group) * spatial_merge_unit * hidden_size;
        float* dst = hidden.f32.data() + static_cast<size_t>(dst_group) * spatial_merge_unit * hidden_size;
        std::memcpy(dst, src, static_cast<size_t>(spatial_merge_unit) * hidden_size * sizeof(float));
    }
    return hidden;
}

Tensor flatten_vision_pixels(Tensor pixels) {
    if (pixels.shape.size() == 3 && pixels.shape[0] == 1) {
        pixels.shape = {pixels.shape[1], pixels.shape[2]};
        return pixels;
    }
    if (pixels.shape.size() == 2) return pixels;
    die("unsupported pixel_values shape=" + shape_string(pixels.shape));
}

Tensor run_vision_net(const NetFiles& files, Tensor pixel_values, int threads, bool no_packing) {
    pixel_values = flatten_vision_pixels(std::move(pixel_values));
    ncnn::Net net;
    load_net(net, files, threads, no_packing, true);
    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", mat_from_f32(pixel_values));
    ncnn::Mat out;
    if (ex.extract("out0", out) != 0) die("extract vision out0 failed: " + files.param);
    Tensor image_embeds = tensor_from_mat(out);
    constexpr int hidden_size = 2560;
    if (image_embeds.f32.size() % hidden_size != 0) {
        die("vision output count is not divisible by hidden_size shape=" + shape_string(image_embeds.shape));
    }
    image_embeds.shape = {static_cast<int>(image_embeds.f32.size() / hidden_size), hidden_size};
    return image_embeds;
}

Tensor run_dynamic_vision_net(
    const NetFiles& encoder_files,
    const NetFiles& post_merger_files,
    const ImagePreprocessResult& pp,
    int threads,
    bool no_packing
) {
    constexpr int encoder_hidden_size = 1152;
    constexpr int output_hidden_size = 2560;
    const VisionRuntimeInputs rt = make_vision_runtime_inputs(pp);

    ncnn::Net encoder;
    load_net(encoder, encoder_files, threads, no_packing, true);
    ncnn::Extractor ex = encoder.create_extractor();
    ex.input("in0", mat_from_f32(rt.pixel_values_windowed));
    ex.input("in1", mat_from_f32(rt.cos));
    ex.input("in2", mat_from_f32(rt.sin));
    ex.input("in3", mat_from_f32(rt.mask_window));
    ex.input("in4", mat_from_f32(rt.mask_full));
    ncnn::Mat hidden_mat;
    if (ex.extract("out0", hidden_mat) != 0) die("extract dynamic vision encoder out0 failed: " + encoder_files.param);
    Tensor hidden_windowed = tensor_from_mat(hidden_mat, {rt.seq_len, encoder_hidden_size});
    Tensor hidden = reverse_window_order(std::move(hidden_windowed), rt.reverse_index);

    ncnn::Net post_merger;
    load_net(post_merger, post_merger_files, threads, no_packing, true);
    ncnn::Extractor mex = post_merger.create_extractor();
    mex.input("in0", mat_from_f32(hidden));
    ncnn::Mat out;
    if (mex.extract("out0", out) != 0) die("extract dynamic vision post-merger out0 failed: " + post_merger_files.param);
    Tensor image_embeds = tensor_from_mat(out);
    if (image_embeds.f32.size() % output_hidden_size != 0) {
        die("dynamic vision output count is not divisible by hidden_size shape=" + shape_string(image_embeds.shape));
    }
    image_embeds.shape = {static_cast<int>(image_embeds.f32.size() / output_hidden_size), output_hidden_size};
    if (image_embeds.shape[0] != rt.groups) {
        die("dynamic vision output rows=" + std::to_string(image_embeds.shape[0]) +
            " expected_groups=" + std::to_string(rt.groups));
    }
    return image_embeds;
}

bool has_nonfinite(const Tensor& t) {
    for (float v : t.f32) {
        if (!std::isfinite(v)) return true;
    }
    return false;
}

Tensor merge_image_embeds_into_text(Tensor text_embeds, const Tensor& image_embeds, const Tensor& image_token_positions) {
    constexpr int hidden_size = 2560;
    if (text_embeds.shape.size() != 3 || text_embeds.shape[0] != 1 || text_embeds.shape[2] != hidden_size) {
        die("unsupported text_inputs_embeds shape=" + shape_string(text_embeds.shape));
    }
    if (image_embeds.shape.size() != 2 || image_embeds.shape[1] != hidden_size) {
        die("unsupported image_embeds shape=" + shape_string(image_embeds.shape));
    }
    if (text_embeds.f32.size() != static_cast<size_t>(text_embeds.shape[1]) * hidden_size) {
        die("text_inputs_embeds data count does not match shape");
    }
    if (image_embeds.f32.size() != static_cast<size_t>(image_embeds.shape[0]) * hidden_size) {
        die("image_embeds data count does not match shape");
    }
    if (image_token_positions.i64.empty()) die("image_token_positions is empty");
    if (static_cast<int>(image_token_positions.i64.size()) != image_embeds.shape[0]) {
        die("image_token_positions count=" + std::to_string(image_token_positions.i64.size()) +
            " image_embeds rows=" + std::to_string(image_embeds.shape[0]));
    }
    const int seq_len = text_embeds.shape[1];
    for (size_t row = 0; row < image_token_positions.i64.size(); ++row) {
        const int pos = static_cast<int>(image_token_positions.i64[row]);
        if (pos < 0 || pos >= seq_len) {
            die("image token position out of range pos=" + std::to_string(pos) +
                " seq_len=" + std::to_string(seq_len));
        }
        float* dst = text_embeds.f32.data() + static_cast<size_t>(pos) * hidden_size;
        const float* src = image_embeds.f32.data() + row * hidden_size;
        std::memcpy(dst, src, hidden_size * sizeof(float));
    }
    return text_embeds;
}

std::string make_image_chat_prompt(const std::string& prompt, int image_token_count) {
    std::string image_tokens = "<|vision_start|>";
    for (int i = 0; i < image_token_count; ++i) image_tokens += "<|image_pad|>";
    image_tokens += "<|vision_end|>";
    return "<|begin_of_text|>system\nYou are a helpful assistant.<|end_of_text|>\n"
           "<|begin_of_text|>user\n" + image_tokens + prompt + "<|end_of_text|>\n"
           "<|begin_of_text|>assistant\n";
}

Tensor image_positions_from_ids(const std::vector<int>& prompt_ids, int image_token_id) {
    Tensor positions;
    positions.shape = {0};
    for (size_t i = 0; i < prompt_ids.size(); ++i) {
        if (prompt_ids[i] == image_token_id) positions.i64.push_back(static_cast<int64_t>(i));
    }
    positions.shape = {static_cast<int>(positions.i64.size())};
    return positions;
}

} // namespace youtu
