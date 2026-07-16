// SPDX-License-Identifier: Apache-2.0
//
// Faithful C++ re-implementation of Penguin-VL's image preprocessing
// (penguinvl/model/penguinvl_encoder/image_processing_penguinvl.py) for the
// single-image path. Produces flattened patches identical in layout to the
// PyTorch processor so that vision-encoder inputs match bit-for-bit (modulo the
// resize kernel, see docs/ALIGNMENT.md).
#pragma once

#include <mat.h>

#include "penguin_config.h"

namespace pvl {

struct PreprocessResult {
    ncnn::Mat pixel_values;  // (w = 3*patch*patch, h = grid_h*grid_w), channel-major RGB per patch
    int grid_h = 0;          // number of patches along height
    int grid_w = 0;          // number of patches along width
    int num_patches() const { return grid_h * grid_w; }
};

// Python-compatible round-half-to-even (matches numpy/round used in the
// reference target-size computation).
long py_round(double x);

// Compute the resized (height, width) for one image, replicating
// simple_batched_resize() for a single key frame.
void compute_target_size(int orig_h, int orig_w, const VisionConfig& vc,
                         int& target_h, int& target_w);

// Full preprocessing for a single interleaved BGR u8 image (as returned by
// load_image_to_ncnn_mat). merge_size > 1 (video) is not yet supported.
PreprocessResult preprocess_image(const ncnn::Mat& bgr_u8, const VisionConfig& vc);

}  // namespace pvl
