#pragma once

#include <array>
#include <stdexcept>
#include <vector>

struct HunyuanOcrPromptLayout {
    std::vector<int> token_ids;
    std::array<std::vector<int>, 4> position_ids;
    int first_image_index = 0;
};

inline HunyuanOcrPromptLayout build_hunyuan_ocr_prompt_layout(
    int bos_id,
    int image_token_id,
    int user_end_id,
    const std::vector<int>& text_ids,
    int patch_h,
    int patch_w) {
    if (patch_h <= 0 || patch_w <= 0) {
        throw std::invalid_argument("HunyuanOCR patch grid must be positive");
    }

    const int num_vision_tokens = patch_h * (patch_w + 1) + 2;
    HunyuanOcrPromptLayout layout;
    layout.token_ids.reserve(1 + num_vision_tokens + text_ids.size() + 1);
    layout.token_ids.push_back(bos_id);
    layout.first_image_index = static_cast<int>(layout.token_ids.size());
    for (int i = 0; i < num_vision_tokens; i++) {
        layout.token_ids.push_back(image_token_id);
    }
    layout.token_ids.insert(layout.token_ids.end(), text_ids.begin(), text_ids.end());
    layout.token_ids.push_back(user_end_id);

    const int seq_len = static_cast<int>(layout.token_ids.size());
    for (std::vector<int>& axis : layout.position_ids) {
        axis.resize(seq_len);
        for (int i = 0; i < seq_len; i++) axis[i] = i;
    }

    // The first and last vision features retain their linear positions. The
    // intervening patch/newline features use the same W/H/T coordinates as
    // HunYuanVLProcessor.
    int index = layout.first_image_index + 1;
    for (int row = 0; row < patch_h; row++) {
        for (int column = 0; column < patch_w + 1; column++, index++) {
            layout.position_ids[1][index] = column;
            layout.position_ids[2][index] = row;
            layout.position_ids[3][index] = 0;
        }
    }

    return layout;
}
