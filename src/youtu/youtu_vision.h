#pragma once

#include <string>
#include <vector>

#include "youtu_runner_common.h"

namespace youtu {

struct ImagePreprocessResult {
    Tensor pixel_values;
    Tensor pixel_attention_mask;
    Tensor spatial_shapes;
    int input_width = 0;
    int input_height = 0;
    int resized_width = 0;
    int resized_height = 0;
    int num_patches = 0;
    int image_token_count = 0;
};

ImagePreprocessResult preprocess_siglip2_image(const std::string& image_path, int patch_size,
                                               int max_num_patches);
Tensor run_vision_net(const NetFiles& files, Tensor pixel_values, int threads, bool no_packing);
Tensor run_dynamic_vision_net(const NetFiles& encoder_files, const NetFiles& post_merger_files,
                              const ImagePreprocessResult& pp, int threads, bool no_packing);
bool has_nonfinite(const Tensor& t);
Tensor merge_image_embeds_into_text(Tensor text_embeds, const Tensor& image_embeds,
                                    const Tensor& image_token_positions);
std::string make_image_chat_prompt(const std::string& prompt, int image_token_count);
Tensor image_positions_from_ids(const std::vector<int>& prompt_ids, int image_token_id);

} // namespace youtu
