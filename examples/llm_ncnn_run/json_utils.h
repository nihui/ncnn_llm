#pragma once

#include "utils/prompt.h"

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

using nlohmann::json;

std::string extract_content(const json& content);
std::vector<Message> parse_messages(const json& messages_json);
std::string make_response_id();

json truncate_large_strings(json v, size_t max_bytes);
json strip_image_payloads(json v);
std::string sanitize_utf8(const std::string& s);
json make_error(int status, const std::string& message);

bool looks_like_base64(std::string_view s);
size_t base64_fingerprint(const std::string& s);
