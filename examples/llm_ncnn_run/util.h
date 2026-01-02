#pragma once

#include <cstdint>
#include <optional>
#include <string>

std::optional<int> parse_int(const std::string& s);
int64_t now_ms_epoch();
