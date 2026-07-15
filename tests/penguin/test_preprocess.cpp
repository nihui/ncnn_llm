// SPDX-License-Identifier: Apache-2.0
#include <mat.h>

#include "image_preprocess.h"
#include "penguin_config.h"
#include "test_util.h"

using namespace pvl;

PVL_TEST(py_round_banker) {
    CHECK_EQ((int)py_round(0.5), 0);
    CHECK_EQ((int)py_round(1.5), 2);
    CHECK_EQ((int)py_round(2.5), 2);
    CHECK_EQ((int)py_round(2.4), 2);
    CHECK_EQ((int)py_round(2.6), 3);
    CHECK_EQ((int)py_round(-0.5), 0);
    CHECK_EQ((int)py_round(3.5), 4);
}

static VisionConfig default_vc() {
    VisionConfig vc;
    vc.patch_size = 14;
    vc.merge_size = 1;
    vc.min_tokens = 16;
    vc.max_tokens = 16384;
    return vc;
}

PVL_TEST(target_size_identity) {
    VisionConfig vc = default_vc();
    int th = 0, tw = 0;
    // 140x140 is a multiple of 14 and within the token budget -> unchanged.
    compute_target_size(140, 140, vc, th, tw);
    CHECK_EQ(th, 140);
    CHECK_EQ(tw, 140);
    CHECK_EQ(th % (vc.patch_size * vc.merge_size), 0);
}

PVL_TEST(target_size_min_pixels_upscale) {
    VisionConfig vc = default_vc();
    int th = 0, tw = 0;
    // 28x28 is below min_pixels -> upscaled to satisfy the minimum token budget.
    compute_target_size(28, 28, vc, th, tw);
    CHECK_EQ(th, 70);
    CHECK_EQ(tw, 70);
    // grid 5x5 = 25 tokens >= min_tokens (16)
    CHECK(( (th / vc.patch_size) * (tw / vc.patch_size)) >= vc.min_tokens);
}

PVL_TEST(square_fixture_has_exported_token_count) {
    VisionConfig vc = default_vc();
    vc.max_tokens = 196;
    int th = 0, tw = 0;
    compute_target_size(832, 832, vc, th, tw);
    CHECK_EQ(th, 196);
    CHECK_EQ(tw, 196);
    CHECK_EQ((th / vc.patch_size) * (tw / vc.patch_size), 196);
}

PVL_TEST(patchify_layout_and_normalize) {
    VisionConfig vc = default_vc();
    // Solid-color 140x140 BGR image: resize is a no-op so normalization is exact.
    const int W = 140, H = 140;
    ncnn::Mat bgr(W, H, 3, (size_t)1u);
    unsigned char* d = (unsigned char*)bgr.data;
    const unsigned char B = 10, G = 20, R = 30;
    for (int i = 0; i < W * H; ++i) {
        d[i * 3 + 0] = B;
        d[i * 3 + 1] = G;
        d[i * 3 + 2] = R;
    }

    PreprocessResult pp = preprocess_image(bgr, vc);
    CHECK_EQ(pp.grid_h, 10);
    CHECK_EQ(pp.grid_w, 10);
    CHECK_EQ(pp.num_patches(), 100);
    CHECK_EQ(pp.pixel_values.w, 3 * 14 * 14);
    CHECK_EQ(pp.pixel_values.h, 100);

    const float exp_r = (R / 255.0f - vc.image_mean[0]) / vc.image_std[0];
    const float exp_g = (G / 255.0f - vc.image_mean[1]) / vc.image_std[1];
    const float exp_b = (B / 255.0f - vc.image_mean[2]) / vc.image_std[2];

    const int ps2 = 14 * 14;
    const float* row0 = pp.pixel_values.row(0);
    // channel-major RGB layout
    CHECK_NEAR(row0[0], exp_r, 1e-4);
    CHECK_NEAR(row0[ps2], exp_g, 1e-4);
    CHECK_NEAR(row0[2 * ps2], exp_b, 1e-4);
    // a later patch should carry the same solid color
    const float* row50 = pp.pixel_values.row(50);
    CHECK_NEAR(row50[0], exp_r, 1e-4);
}
