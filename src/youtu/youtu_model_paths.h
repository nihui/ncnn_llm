#pragma once

#include <string>

#include "youtu_runner_common.h"

namespace youtu {

NetFiles layer_files(const std::string& dir, int layer);
NetFiles embed_files(const std::string& dir, const std::string& precision);
NetFiles prefill_files(const std::string& dir, int bucket_length);
NetFiles full_decoder_files(const std::string& dir);
NetFiles full_decoder_prefill_files(const std::string& dir);
NetFiles vision_files(const std::string& dir, const std::string& stem);
NetFiles dynamic_vision_files(const std::string& dir, const std::string& stem);

} // namespace youtu
