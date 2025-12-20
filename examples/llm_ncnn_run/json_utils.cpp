#include "json_utils.h"

#include <chrono>
#include <functional>
#include <sstream>
#include <string>

bool looks_like_base64(std::string_view s) {
    size_t n = 0;
    size_t valid = 0;
    for (char c : s) {
        if (c == '\r' || c == '\n' || c == ' ' || c == '\t') continue;
        ++n;
        if (n > 8192) break;
        bool ok = (c >= 'A' && c <= 'Z') ||
                  (c >= 'a' && c <= 'z') ||
                  (c >= '0' && c <= '9') ||
                  c == '+' || c == '/' || c == '=';
        if (ok) ++valid;
        if (n > 4096 && valid < n * 98 / 100) return false;
    }
    if (n < 1024) return false;
    return valid >= n * 98 / 100;
}

size_t base64_fingerprint(const std::string& s) {
    constexpr size_t k = 1024;
    std::hash<std::string_view> h;
    if (s.size() <= k) {
        return h(std::string_view(s)) ^ (s.size() << 1);
    }
    std::string_view prefix(s.data(), k);
    std::string_view suffix(s.data() + (s.size() - k), k);
    size_t hp = h(prefix);
    size_t hs = h(suffix);
    return hp ^ (hs << 1) ^ (s.size() << 2);
}

std::string image_artifact_key(const json& a) {
    if (a.is_object() && a.contains("url") && a["url"].is_string()) {
        return std::string("url:") + a["url"].get<std::string>();
    }
    if (a.is_object() && a.contains("data_base64") && a["data_base64"].is_string()) {
        const std::string b64 = a["data_base64"].get<std::string>();
        return "b64:" + std::to_string(b64.size()) + ":" + std::to_string(base64_fingerprint(b64));
    }
    return {};
}

std::string extract_content(const json& content) {
    if (content.is_string()) {
        return content.get<std::string>();
    }
    if (content.is_array()) {
        std::string merged;
        for (const auto& part : content) {
            if (part.is_string()) {
                merged += part.get<std::string>();
            } else if (part.is_object()) {
                auto type = part.value("type", "");
                if (type == "text" && part.contains("text")) {
                    merged += part["text"].get<std::string>();
                }
            }
        }
        return merged;
    }
    return "";
}

std::vector<Message> parse_messages(const json& messages_json) {
    std::vector<Message> messages;
    for (const auto& m : messages_json) {
        if (!m.contains("role")) continue;
        Message msg;
        msg.role = m.value("role", "");
        if (m.contains("content")) {
            msg.content = extract_content(m["content"]);
        }
        if (m.contains("tool_calls") && m["tool_calls"].is_array()) {
            msg.tool_calls = m["tool_calls"].get<std::vector<json>>();
        }
        msg.reasoning_content = m.value("reasoning_content", "");
        messages.push_back(std::move(msg));
    }
    return messages;
}

std::string make_response_id() {
    using namespace std::chrono;
    auto now = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    std::stringstream ss;
    ss << "chatcmpl-" << now;
    return ss.str();
}

json truncate_large_strings(json v, size_t max_bytes) {
    if (max_bytes == 0) return v;
    if (v.is_string()) {
        const std::string s = v.get<std::string>();
        if (s.size() <= max_bytes) return v;
        std::string replaced = "<omitted " + std::to_string(s.size()) + " bytes>";
        return replaced;
    }
    if (v.is_array()) {
        for (auto& el : v) el = truncate_large_strings(el, max_bytes);
        return v;
    }
    if (v.is_object()) {
        for (auto it = v.begin(); it != v.end(); ++it) {
            it.value() = truncate_large_strings(it.value(), max_bytes);
        }
        return v;
    }
    return v;
}

json strip_image_payloads(json v) {
    if (v.is_array()) {
        for (auto& el : v) el = strip_image_payloads(el);
        return v;
    }
    if (v.is_object()) {
        if (v.value("type", "") == "image") {
            if (v.contains("data") && v["data"].is_string()) {
                std::string s = v["data"].get<std::string>();
                v["data"] = "<omitted " + std::to_string(s.size()) + " bytes>";
            }
        }
        for (auto it = v.begin(); it != v.end(); ++it) {
            if (it.value().is_string()) {
                std::string s = it.value().get<std::string>();
                if (looks_like_base64(s)) {
                    it.value() = "<omitted " + std::to_string(s.size()) + " bytes>";
                }
            } else {
                it.value() = strip_image_payloads(it.value());
            }
        }
        return v;
    }
    return v;
}

void collect_mcp_image_artifacts(const json& v, std::vector<json>& out, std::unordered_set<size_t>& seen_b64) {
    if (v.is_array()) {
        for (const auto& el : v) collect_mcp_image_artifacts(el, out, seen_b64);
        return;
    }
    if (!v.is_object()) return;

    if (v.value("type", "") == "image" && v.contains("data") && v["data"].is_string()) {
        std::string data = v["data"].get<std::string>();
        if (looks_like_base64(data)) {
            size_t fp = base64_fingerprint(data);
            if (seen_b64.insert(fp).second) {
                std::string mime = v.value("mimeType", v.value("mime_type", std::string("image/png")));
                out.push_back(json{{"kind", "image"}, {"mime_type", mime}, {"data_base64", data}});
            }
        }
    }

    for (auto it = v.begin(); it != v.end(); ++it) {
        if (v.value("type", "") == "image" && it.key() == "data") continue;
        if (it.value().is_string()) {
            std::string s = it.value().get<std::string>();
            if (looks_like_base64(s)) {
                size_t fp = base64_fingerprint(s);
                if (seen_b64.insert(fp).second) {
                    out.push_back(json{{"kind", "image"}, {"mime_type", "image/png"}, {"data_base64", s}});
                }
            }
        } else {
            collect_mcp_image_artifacts(it.value(), out, seen_b64);
        }
    }
}

std::string sanitize_utf8(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    auto is_cont = [&](unsigned char c) { return (c & 0xC0) == 0x80; };
    for (size_t i = 0; i < s.size();) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        if (c < 0x80) { out.push_back(static_cast<char>(c)); ++i; continue; }
        if ((c >> 5) == 0x6 && i + 1 < s.size() && is_cont(static_cast<unsigned char>(s[i+1]))) {
            out.append(s, i, 2); i += 2; continue;
        }
        if ((c >> 4) == 0xE && i + 2 < s.size() &&
            is_cont(static_cast<unsigned char>(s[i+1])) &&
            is_cont(static_cast<unsigned char>(s[i+2]))) {
            out.append(s, i, 3); i += 3; continue;
        }
        if ((c >> 3) == 0x1E && i + 3 < s.size() &&
            is_cont(static_cast<unsigned char>(s[i+1])) &&
            is_cont(static_cast<unsigned char>(s[i+2])) &&
            is_cont(static_cast<unsigned char>(s[i+3]))) {
            out.append(s, i, 4); i += 4; continue;
        }
        out.push_back('?');
        ++i;
    }
    return out;
}

json make_error(int status, const std::string& message) {
    json err;
    err["error"] = {{"type", "invalid_request_error"}, {"message", message}};
    err["status"] = status;
    return err;
}
