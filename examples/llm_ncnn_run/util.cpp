#include "util.h"

#include <chrono>
#include <climits>

std::optional<int> parse_int(const std::string& s) {
    try {
        size_t idx = 0;
        long v = std::stol(s, &idx, 10);
        if (idx != s.size()) return std::nullopt;
        if (v < 0 || v > INT32_MAX) return std::nullopt;
        return (int)v;
    } catch (...) {
        return std::nullopt;
    }
}

int64_t now_ms_epoch() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}
