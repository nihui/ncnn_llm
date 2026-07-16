// SPDX-License-Identifier: Apache-2.0
#include "image_preprocess.h"

#include <cmath>
#include <stdexcept>

#include "image_utils.h"  // ncnn_mat_resize_bicubic

namespace pvl {

long py_round(double x) {
    // Round half to even (banker's rounding), matching Python's round().
    double floor_x = std::floor(x);
    double diff = x - floor_x;
    if (diff < 0.5) return (long)floor_x;
    if (diff > 0.5) return (long)floor_x + 1;
    // exactly .5 -> round to even
    long f = (long)floor_x;
    return (f % 2 == 0) ? f : f + 1;
}

void compute_target_size(int orig_h, int orig_w, const VisionConfig& vc,
                         int& target_h, int& target_w) {
    const double factor = (double)(vc.patch_size * vc.merge_size);
    const double min_pixels = (double)vc.min_tokens * factor * factor * 1.5;
    const double max_pixels = (double)vc.max_tokens * factor * factor * 0.95;

    const double h = orig_h;
    const double w = orig_w;
    const double ar = h / w;                 // aspect ratio (height / width)
    const double raw_area = h * w;

    // Single key frame: keep original size unless it exceeds the pixel budget.
    double target_area = raw_area;
    if (raw_area > max_pixels) target_area = max_pixels;

    // get_dims_from_area
    double w_new = std::sqrt(target_area / ar);
    double h_new = w_new * ar;
    long h_bar = py_round(h_new / factor) * (long)factor;
    long w_bar = py_round(w_new / factor) * (long)factor;
    if (h_bar < (long)factor) h_bar = (long)factor;
    if (w_bar < (long)factor) w_bar = (long)factor;

    // ensure_min_hw
    if ((double)h_bar * (double)w_bar < min_pixels) {
        double w2 = std::sqrt(min_pixels / ar);
        double h2 = w2 * ar;
        h_bar = (long)std::ceil(h2 / factor) * (long)factor;
        w_bar = (long)std::ceil(w2 / factor) * (long)factor;
    }

    target_h = (int)h_bar;
    target_w = (int)w_bar;
}

PreprocessResult preprocess_image(const ncnn::Mat& bgr_u8, const VisionConfig& vc) {
    if (vc.merge_size != 1)
        throw std::runtime_error("preprocess_image: only merge_size==1 (image) is supported");
    if (bgr_u8.empty())
        throw std::runtime_error("preprocess_image: empty image");

    const int ps = vc.patch_size;
    int target_h = 0, target_w = 0;
    compute_target_size(bgr_u8.h, bgr_u8.w, vc, target_h, target_w);

    // Resize (bicubic, matching PILImageResampling.BICUBIC used by the processor).
    ncnn::Mat resized = ncnn_mat_resize_bicubic(bgr_u8, target_w, target_h);

    const int grid_h = target_h / ps;
    const int grid_w = target_w / ps;
    const int num_patches = grid_h * grid_w;
    const int feat = 3 * ps * ps;

    PreprocessResult res;
    res.grid_h = grid_h;
    res.grid_w = grid_w;
    res.pixel_values.create(feat, num_patches);

    const unsigned char* img = (const unsigned char*)resized.data;  // interleaved BGR u8

    // For merge_size == 1 the flattened patch order is simple row-major over
    // (grid_h, grid_w); feature layout per patch is channel-major RGB, each
    // channel a ps*ps block in row-major order. This matches
    // patches.reshape(t, c, gh, 1, ps, gw, 1, ps).transpose(0,2,5,3,6,1,4,7).
    for (int gh = 0; gh < grid_h; ++gh) {
        for (int gw = 0; gw < grid_w; ++gw) {
            const int patch_idx = gh * grid_w + gw;
            float* out = res.pixel_values.row(patch_idx);
            float* out_r = out;
            float* out_g = out + ps * ps;
            float* out_b = out + 2 * ps * ps;

            const int y0 = gh * ps;
            const int x0 = gw * ps;
            for (int py = 0; py < ps; ++py) {
                const unsigned char* row = img + (size_t)(y0 + py) * target_w * 3;
                for (int px = 0; px < ps; ++px) {
                    const unsigned char* pix = row + (size_t)(x0 + px) * 3;  // B,G,R
                    const float b = pix[0] / 255.0f;
                    const float g = pix[1] / 255.0f;
                    const float r = pix[2] / 255.0f;
                    const int off = py * ps + px;
                    out_r[off] = (r - vc.image_mean[0]) / vc.image_std[0];
                    out_g[off] = (g - vc.image_mean[1]) / vc.image_std[1];
                    out_b[off] = (b - vc.image_mean[2]) / vc.image_std[2];
                }
            }
        }
    }

    return res;
}

}  // namespace pvl
