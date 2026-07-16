// SPDX-License-Identifier: Apache-2.0
#include <cmath>

#include <mat.h>

#include "penguin_vision.h"
#include "test_util.h"

PVL_TEST(vision_rope_shape_and_values) {
    const int grid_h = 2, grid_w = 3, head_dim = 4;
    const float theta = 10000.0f;
    ncnn::Mat cos_c, sin_c;
    pvl::generate_penguin_vision_rope(grid_h, grid_w, head_dim, theta, cos_c, sin_c);

    CHECK_EQ(cos_c.w, head_dim);
    CHECK_EQ(cos_c.h, grid_h * grid_w);
    CHECK_EQ(sin_c.w, head_dim);
    CHECK_EQ(sin_c.h, grid_h * grid_w);

    const int half = head_dim / 2;
    // inv_freq[0] = 1, inv_freq[1] = 1/theta^(2/4) = 1/100
    const float if0 = 1.0f;
    const float if1 = 1.0f / std::pow(theta, 2.0f / head_dim);

    for (int gh = 0; gh < grid_h; ++gh) {
        for (int gw = 0; gw < grid_w; ++gw) {
            const int idx = gh * grid_w + gw;  // row-major order (merge_size == 1)
            const float* c = cos_c.row(idx);
            const float* s = sin_c.row(idx);
            // first half -> height coordinate, second half -> width coordinate
            CHECK_NEAR(c[0], std::cos(gh * if0), 1e-5);
            CHECK_NEAR(c[1], std::cos(gh * if1), 1e-5);
            CHECK_NEAR(c[half + 0], std::cos(gw * if0), 1e-5);
            CHECK_NEAR(c[half + 1], std::cos(gw * if1), 1e-5);
            CHECK_NEAR(s[0], std::sin(gh * if0), 1e-5);
            CHECK_NEAR(s[half + 1], std::sin(gw * if1), 1e-5);
        }
    }
}
