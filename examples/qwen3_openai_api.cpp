#include "qwen3_0.6b.h"
#include "utils/prompt.h"

#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <httplib.h>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

using nlohmann::json;

namespace {

int64_t now_ms_epoch() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

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

struct Options {
    int port = 8080;
    std::string mcp_server_cmdline;
    bool mcp_merge_tools = true;
    int mcp_timeout_ms = 15000;
    bool mcp_debug = false;
    std::string mcp_transport = "lsp"; // lsp|jsonl
    size_t mcp_max_string_bytes_in_prompt = 4096;
};

void print_usage(const char* argv0) {
    std::cout
        << "Usage: " << (argv0 ? argv0 : "qwen3_openai_api") << " [options]\n"
        << "\n"
        << "Options:\n"
        << "  --port <n>                 Listen port (default: 8080)\n"
        << "  --mcp-server <cmdline>     Launch an MCP server over stdio\n"
        << "  --mcp-transport <mode>     MCP stdio framing: lsp|jsonl (default: lsp)\n"
        << "  --no-mcp-merge-tools       Do not merge MCP tools into request tools\n"
        << "  --mcp-timeout-ms <n>       MCP request timeout in ms (default: 15000)\n"
        << "  --mcp-max-string-bytes <n> Truncate huge tool strings in prompt (default: 4096)\n"
        << "  --mcp-debug                Enable verbose MCP logs\n"
        << "  --help                     Show this help\n"
        << "\n"
        << "Example:\n"
        << "  " << (argv0 ? argv0 : "qwen3_openai_api")
        << " --mcp-server \"./my_mcp_server --flag\" --port 8080\n";
}

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

Options parse_options(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--") {
            break;
        }
        if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (a == "--port") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --port\n";
                std::exit(2);
            }
            auto v = parse_int(argv[++i]);
            if (!v) {
                std::cerr << "Invalid --port value\n";
                std::exit(2);
            }
            opt.port = *v;
        } else if (a == "--mcp-server") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --mcp-server\n";
                std::exit(2);
            }
            opt.mcp_server_cmdline = argv[++i];
        } else if (a == "--mcp-transport") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --mcp-transport\n";
                std::exit(2);
            }
            opt.mcp_transport = argv[++i];
            if (opt.mcp_transport != "lsp" && opt.mcp_transport != "jsonl") {
                std::cerr << "Invalid --mcp-transport value (expected lsp|jsonl)\n";
                std::exit(2);
            }
        } else if (a == "--no-mcp-merge-tools") {
            opt.mcp_merge_tools = false;
        } else if (a == "--mcp-timeout-ms") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --mcp-timeout-ms\n";
                std::exit(2);
            }
            auto v = parse_int(argv[++i]);
            if (!v) {
                std::cerr << "Invalid --mcp-timeout-ms value\n";
                std::exit(2);
            }
            opt.mcp_timeout_ms = *v;
        } else if (a == "--mcp-max-string-bytes") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --mcp-max-string-bytes\n";
                std::exit(2);
            }
            auto v = parse_int(argv[++i]);
            if (!v) {
                std::cerr << "Invalid --mcp-max-string-bytes value\n";
                std::exit(2);
            }
            opt.mcp_max_string_bytes_in_prompt = (size_t)*v;
        } else if (a == "--mcp-debug") {
            opt.mcp_debug = true;
        } else {
            std::cerr << "Unknown option: " << a << "\n";
            print_usage(argv[0]);
            std::exit(2);
        }
    }
    return opt;
}

std::vector<std::string> split_cmdline(const std::string& cmdline) {
    std::vector<std::string> out;
    std::string cur;
    bool in_single = false;
    bool in_double = false;
    bool escape = false;
    auto push = [&]() {
        if (!cur.empty()) out.push_back(cur);
        cur.clear();
    };
    for (size_t i = 0; i < cmdline.size(); ++i) {
        char c = cmdline[i];
        if (escape) {
            cur.push_back(c);
            escape = false;
            continue;
        }
        if (c == '\\' && !in_single) {
            escape = true;
            continue;
        }
        if (c == '\'' && !in_double) {
            in_single = !in_single;
            continue;
        }
        if (c == '"' && !in_single) {
            in_double = !in_double;
            continue;
        }
        if (!in_single && !in_double && (c == ' ' || c == '\t' || c == '\n' || c == '\r')) {
            push();
            continue;
        }
        cur.push_back(c);
    }
    push();
    return out;
}

#ifndef _WIN32
#include <errno.h>
#include <poll.h>
#include <signal.h>
#include <cstring>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

class StdioProcess {
public:
    ~StdioProcess() { stop(); }

    bool start(const std::vector<std::string>& argv) {
        if (argv.empty()) return false;
        stop();

        int in_pipe[2] = {-1, -1};   // parent writes -> child reads
        int out_pipe[2] = {-1, -1};  // child writes -> parent reads
        if (pipe(in_pipe) != 0) return false;
        if (pipe(out_pipe) != 0) {
            ::close(in_pipe[0]);
            ::close(in_pipe[1]);
            return false;
        }

        pid_ = fork();
        if (pid_ == -1) {
            ::close(in_pipe[0]); ::close(in_pipe[1]);
            ::close(out_pipe[0]); ::close(out_pipe[1]);
            pid_ = -1;
            return false;
        }

        if (pid_ == 0) {
            ::dup2(in_pipe[0], STDIN_FILENO);
            ::dup2(out_pipe[1], STDOUT_FILENO);
            // keep stderr attached (so users can see MCP server logs)

            ::close(in_pipe[0]); ::close(in_pipe[1]);
            ::close(out_pipe[0]); ::close(out_pipe[1]);

            std::vector<char*> cargv;
            cargv.reserve(argv.size() + 1);
            for (const auto& a : argv) cargv.push_back(const_cast<char*>(a.c_str()));
            cargv.push_back(nullptr);
            execvp(cargv[0], cargv.data());
            _exit(127);
        }

        // parent
        child_stdin_ = in_pipe[1];
        child_stdout_ = out_pipe[0];
        ::close(in_pipe[0]);
        ::close(out_pipe[1]);

        // avoid SIGPIPE crashing the whole server if child dies
        signal(SIGPIPE, SIG_IGN);
        return true;
    }

    void stop() {
        if (child_stdin_ != -1) {
            ::close(child_stdin_);
            child_stdin_ = -1;
        }
        if (child_stdout_ != -1) {
            ::close(child_stdout_);
            child_stdout_ = -1;
        }
        if (pid_ > 0) {
            int status = 0;
            // Try to reap without blocking too long; if it's still running, terminate.
            pid_t r = waitpid(pid_, &status, WNOHANG);
            if (r == 0) {
                kill(pid_, SIGTERM);
                waitpid(pid_, &status, 0);
            }
            pid_ = -1;
        }
        read_buf_.clear();
    }

    bool write_all(const std::string& s) {
        if (child_stdin_ == -1) return false;
        const char* p = s.data();
        size_t n = s.size();
        while (n > 0) {
            ssize_t w = ::write(child_stdin_, p, n);
            if (w <= 0) return false;
            p += (size_t)w;
            n -= (size_t)w;
        }
        return true;
    }

    bool read_exact(size_t n, std::string& out, int timeout_ms, std::string* err) {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        out.clear();
        out.reserve(n);
        while (out.size() < n) {
            if (!read_buf_.empty()) {
                size_t take = std::min(n - out.size(), read_buf_.size());
                out.append(read_buf_.data(), take);
                read_buf_.erase(0, take);
                continue;
            }
            if (!read_some_until(deadline, err)) return false;
        }
        return true;
    }

    bool read_until(const std::string& delim, std::string& out, int timeout_ms, std::string* err) {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (true) {
            size_t pos = read_buf_.find(delim);
            if (pos != std::string::npos) {
                out = read_buf_.substr(0, pos);
                read_buf_.erase(0, pos + delim.size());
                return true;
            }
            if (!read_some_until(deadline, err)) return false;
        }
    }

    // Reads a JSON message framed as:
    //   Content-Length: N\r\n\r\n<json-bytes>
    // Also tolerates JSON-lines (one JSON object per line) for non-compliant servers.
    bool read_json(json& out, int timeout_ms, std::string* err) {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        auto skip_ws = [&]() -> bool {
            while (true) {
                while (!read_buf_.empty()) {
                    char c = read_buf_[0];
                    if (c == '\r' || c == '\n' || c == ' ' || c == '\t') read_buf_.erase(0, 1);
                    else return true;
                }
                if (!read_some_until(deadline, err)) return false;
            }
        };

        if (!skip_ws()) return false;

        if (!read_buf_.empty() && read_buf_[0] == '{') {
            // JSON-lines mode: read until we can parse.
            std::string body;
            while (true) {
                size_t nl = read_buf_.find('\n');
                if (nl == std::string::npos) {
                    if (!read_some_until(deadline, err)) return false;
                    continue;
                }
                body.append(read_buf_.substr(0, nl + 1));
                read_buf_.erase(0, nl + 1);
                std::string trimmed = body;
                while (!trimmed.empty() && (trimmed.back() == '\r' || trimmed.back() == '\n')) trimmed.pop_back();
                try {
                    out = json::parse(trimmed);
                    return true;
                } catch (...) {
                    // keep reading more lines (in case JSON is multi-line)
                }
                if (body.size() > 1 * 1024 * 1024) return false;
            }
        }

        std::string header;
        if (!read_until("\r\n\r\n", header, timeout_ms, err)) return false;
        size_t len = 0;
        {
            std::istringstream iss(header);
            std::string line;
            bool found = false;
            while (std::getline(iss, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                const std::string prefix = "Content-Length:";
                if (line.rfind(prefix, 0) == 0) {
                    std::string v = line.substr(prefix.size());
                    while (!v.empty() && (v[0] == ' ' || v[0] == '\t')) v.erase(0, 1);
                    auto iv = parse_int(v);
                    if (!iv || *iv < 0) return false;
                    len = (size_t)*iv;
                    found = true;
                }
            }
            if (!found) return false;
        }

        std::string body;
        if (!read_exact(len, body, timeout_ms, err)) return false;
        try {
            out = json::parse(body);
            return true;
        } catch (...) {
            return false;
        }
    }

    int stdout_fd() const { return child_stdout_; }

private:
    bool read_some_until(const std::chrono::steady_clock::time_point deadline, std::string* err) {
        while (true) {
            int timeout_ms = (int)std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count();
            if (timeout_ms < 0) timeout_ms = 0;

            struct pollfd pfd;
            pfd.fd = child_stdout_;
            pfd.events = POLLIN;
            pfd.revents = 0;

            int pr = ::poll(&pfd, 1, timeout_ms);
            if (pr == 0) {
                if (err) *err = "timeout waiting for MCP server output";
                return false;
            }
            if (pr < 0) {
                if (errno == EINTR) continue;
                if (err) *err = std::string("poll failed: ") + std::strerror(errno);
                return false;
            }
            if ((pfd.revents & POLLIN) == 0) {
                if (err) *err = "poll returned without POLLIN";
                return false;
            }
            char tmp[4096];
            ssize_t r = ::read(child_stdout_, tmp, sizeof(tmp));
            if (r <= 0) {
                if (err) *err = "MCP server closed stdout";
                return false;
            }
            read_buf_.append(tmp, (size_t)r);
            return true;
        }
    }

    int child_stdin_ = -1;
    int child_stdout_ = -1;
    pid_t pid_ = -1;
    std::string read_buf_;
};

class McpStdioClient {
public:
    enum class Transport {
        Lsp,   // Content-Length: N\r\n\r\n<json>
        Jsonl  // <json>\n
    };

    void set_timeout_ms(int timeout_ms) { timeout_ms_ = timeout_ms; }
    void set_debug(bool debug) { debug_ = debug; }
    void set_transport(Transport t) { transport_ = t; }

    bool start(const std::string& cmdline, std::string* err = nullptr) {
        if (cmdline.empty()) return false;
        auto argv = split_cmdline(cmdline);
        if (argv.empty()) return false;
        if (debug_) {
            std::cerr << "[MCP] start: " << cmdline << "\n";
            std::cerr << "[MCP] argv:";
            for (const auto& a : argv) std::cerr << " [" << a << "]";
            std::cerr << "\n";
        }
        if (!proc_.start(argv)) {
            if (err) *err = "failed to start process";
            return false;
        }
        if (!initialize(err)) {
            proc_.stop();
            return false;
        }
        return true;
    }

    json list_tools(std::string* err = nullptr) {
        json resp = request("tools/list", json::object(), err);
        if (resp.is_null()) return json::object();
        return resp.value("tools", json::array());
    }

    json call_tool(const std::string& name, const json& arguments, std::string* err = nullptr) {
        json params = {{"name", name}, {"arguments", arguments}};
        return request("tools/call", params, err);
    }

private:
    bool initialize(std::string* err) {
        json params = {
            {"protocolVersion", "2024-11-05"},
            {"capabilities", {{"tools", json::object()}}},
            {"clientInfo", {{"name", "ncnn_llm"}, {"version", "demo"}}}
        };
        json resp = request("initialize", params, err);
        if (resp.is_null()) return false;

        // Best-effort initialized notification.
        json notif = {{"jsonrpc", "2.0"}, {"method", "notifications/initialized"}, {"params", json::object()}};
        send(notif);
        return true;
    }

    bool send(const json& msg) {
        std::string body = msg.dump();
        std::string framed;
        if (transport_ == Transport::Jsonl) {
            framed = body;
            framed.push_back('\n');
        } else {
            framed = "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
        }
        if (debug_) {
            std::string preview = body;
            if (preview.size() > 800) preview.resize(800), preview += "...";
            std::cerr << "[MCP] => " << preview << "\n";
        }
        return proc_.write_all(framed);
    }

    json request(const std::string& method, const json& params, std::string* err) {
        int64_t id = next_id_++;
        json req = {{"jsonrpc", "2.0"}, {"id", id}, {"method", method}, {"params", params}};
        if (!send(req)) {
            if (err) *err = "failed to write request";
            return nullptr;
        }

        // Serialize: one in-flight request at a time.
        while (true) {
            json msg;
            std::string read_err;
            if (!proc_.read_json(msg, timeout_ms_, &read_err)) {
                if (err) *err = read_err.empty() ? "failed to read response" : read_err;
                return nullptr;
            }
            if (!msg.is_object()) continue;
            if (debug_) {
                std::string preview = msg.dump();
                if (preview.size() > 1200) preview.resize(1200), preview += "...";
                std::cerr << "[MCP] <= " << preview << "\n";
            }
            if (msg.contains("id") && msg["id"].is_number_integer() && msg["id"].get<int64_t>() == id) {
                if (msg.contains("error")) {
                    if (err) *err = msg["error"].dump();
                    return nullptr;
                }
                return msg.value("result", json::object());
            }
            // ignore notifications/other responses
        }
    }

    StdioProcess proc_;
    int64_t next_id_ = 1;
    int timeout_ms_ = 15000;
    bool debug_ = false;
    Transport transport_ = Transport::Lsp;
};
#else
class McpStdioClient {
public:
    bool start(const std::string&, std::string* err = nullptr) {
        if (err) *err = "MCP stdio is not supported on Windows in this demo";
        return false;
    }
    json list_tools(std::string* err = nullptr) {
        if (err) *err = "MCP stdio is not supported on Windows in this demo";
        return json::array();
    }
    json call_tool(const std::string&, const json&, std::string* err = nullptr) {
        if (err) *err = "MCP stdio is not supported on Windows in this demo";
        return nullptr;
    }
};
#endif

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
        if (v.value("type", "") == "image" && it.key() == "data") continue; // already handled above
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

// Replace malformed UTF-8 sequences with '?', to avoid nlohmann::json throwing.
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
        out.push_back('?'); // replace malformed byte
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

} // namespace

int main(int argc, char** argv) {
    Options opt = parse_options(argc, argv);
    if (opt.mcp_server_cmdline.empty()) {
        if (const char* env = std::getenv("NCNN_LLM_MCP_SERVER")) {
            opt.mcp_server_cmdline = env;
        }
    }
    if (!opt.mcp_debug) {
        if (const char* env = std::getenv("NCNN_LLM_MCP_DEBUG")) {
            opt.mcp_debug = (std::string(env) == "1" || std::string(env) == "true" || std::string(env) == "TRUE");
        }
    }
    if (const char* env = std::getenv("NCNN_LLM_MCP_TRANSPORT")) {
        std::string v = env;
        if (v == "lsp" || v == "jsonl") opt.mcp_transport = v;
    }
    if (const char* env = std::getenv("NCNN_LLM_MCP_TIMEOUT_MS")) {
        if (auto v = parse_int(env)) opt.mcp_timeout_ms = *v;
    }
    if (const char* env = std::getenv("NCNN_LLM_MCP_MAX_STRING_BYTES")) {
        if (auto v = parse_int(env)) opt.mcp_max_string_bytes_in_prompt = (size_t)*v;
    }

    auto mcp_client = std::make_shared<McpStdioClient>();
    std::mutex mcp_mutex;
    std::vector<json> mcp_openai_tools;
    std::unordered_set<std::string> mcp_tool_names;

    if (!opt.mcp_server_cmdline.empty()) {
        mcp_client->set_timeout_ms(opt.mcp_timeout_ms);
        mcp_client->set_debug(opt.mcp_debug);
        mcp_client->set_transport(opt.mcp_transport == "jsonl" ? McpStdioClient::Transport::Jsonl : McpStdioClient::Transport::Lsp);
        std::string err;
        std::cerr << "[MCP] launching stdio server...\n";
        if (!mcp_client->start(opt.mcp_server_cmdline, &err)) {
            std::cerr << "Warning: failed to initialize MCP server: " << err << "\n";
        } else {
            std::cerr << "[MCP] connected; listing tools...\n";
            std::string list_err;
            json tools = mcp_client->list_tools(&list_err);
            if (!list_err.empty()) {
                std::cerr << "Warning: MCP tools/list failed: " << list_err << "\n";
            } else if (tools.is_array()) {
                for (const auto& t : tools) {
                    if (!t.is_object()) continue;
                    std::string name = t.value("name", "");
                    if (name.empty()) continue;
                    mcp_tool_names.insert(name);
                    json openai_tool = {
                        {"type", "function"},
                        {"function", {
                            {"name", name},
                            {"description", t.value("description", "")},
                            {"parameters", t.value("inputSchema", json::object())}
                        }}
                    };
                    mcp_openai_tools.push_back(std::move(openai_tool));
                }
                std::cerr << "Loaded " << mcp_openai_tools.size() << " MCP tool(s) from stdio server.\n";
            }
        }
    }

    qwen3_0_6b model(
        "./assets/qwen3_0.6b/qwen3_embed_token.ncnn.param",
        "./assets/qwen3_0.6b/qwen3_embed_token.ncnn.bin",
        "./assets/qwen3_0.6b/qwen3_proj_out.ncnn.param",
        "./assets/qwen3_0.6b/qwen3_decoder.ncnn.param",
        "./assets/qwen3_0.6b/qwen3_decoder.ncnn.bin",
        "./assets/qwen3_0.6b/vocab.txt",
        "./assets/qwen3_0.6b/merges.txt",
        /*use_vulkan=*/false);

    std::mutex model_mutex;

    httplib::Server server;

    if (!server.set_mount_point("/", "./examples/web")) {
        std::cerr << "Warning: failed to mount ./examples/web for static frontend.\n";
    }

    server.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(make_error(400, std::string("Invalid JSON: ") + e.what()).dump(), "application/json");
            return;
        }

        if (!body.contains("messages") || !body["messages"].is_array()) {
            res.status = 400;
            res.set_content(make_error(400, "`messages` must be an array").dump(), "application/json");
            return;
        }

        auto messages = parse_messages(body["messages"]);
        if (messages.empty() || messages[0].role != "system") {
            messages.insert(messages.begin(), Message{"system", "You are a helpful assistant."});
        }

        std::vector<json> tools;
        if (body.contains("tools") && body["tools"].is_array()) {
            for (const auto& t : body["tools"]) {
                if (t.is_object()) tools.push_back(t);
            }
        }
        if (!mcp_openai_tools.empty() && opt.mcp_merge_tools) {
            std::unordered_set<std::string> existing;
            existing.reserve(tools.size());
            for (const auto& t : tools) {
                if (t.is_object() && t.contains("function") && t["function"].is_object()) {
                    existing.insert(t["function"].value("name", ""));
                }
            }
            for (const auto& t : mcp_openai_tools) {
                std::string name = t["function"].value("name", "");
                if (!name.empty() && existing.find(name) == existing.end()) {
                    tools.push_back(t);
                }
            }
        } else if (!mcp_openai_tools.empty() && tools.empty()) {
            // If caller didn't provide tools, still expose MCP tools to the model.
            tools = mcp_openai_tools;
        }

        std::unordered_set<std::string> mcp_tools_in_prompt;
        if (!mcp_openai_tools.empty()) {
            mcp_tools_in_prompt.reserve(tools.size());
            for (const auto& t : tools) {
                if (!t.is_object() || !t.contains("function") || !t["function"].is_object()) continue;
                std::string name = t["function"].value("name", "");
                if (!name.empty() && mcp_tool_names.find(name) != mcp_tool_names.end()) {
                    mcp_tools_in_prompt.insert(std::move(name));
                }
            }
        }

        GenerateConfig cfg;
        cfg.max_new_tokens = body.value("max_tokens", cfg.max_new_tokens);
        cfg.temperature = body.value("temperature", cfg.temperature);
        cfg.top_p = body.value("top_p", cfg.top_p);
        cfg.top_k = body.value("top_k", cfg.top_k);
        cfg.repetition_penalty = body.value("repetition_penalty", cfg.repetition_penalty);
        cfg.beam_size = body.value("beam_size", cfg.beam_size);
        cfg.debug = body.value("debug", false);
        if (body.contains("do_sample") && body["do_sample"].is_boolean()) {
            cfg.do_sample = body["do_sample"].get<bool>() ? 1 : 0;
        } else if (cfg.temperature <= 0.0f) {
            cfg.do_sample = 0;
        }

        auto artifacts_out = std::make_shared<std::vector<json>>();
        auto artifacts_seen = std::make_shared<std::unordered_set<std::string>>();
        std::string mcp_image_delivery = body.value("mcp_image_delivery", std::string("base64")); // file|base64|both
        if (mcp_image_delivery != "file" && mcp_image_delivery != "base64" && mcp_image_delivery != "both") {
            mcp_image_delivery = "file";
        }

        if (!mcp_tools_in_prompt.empty()) {
            auto mcp = mcp_client;
            const size_t max_tool_string_bytes = opt.mcp_max_string_bytes_in_prompt;
            auto allowed = std::make_shared<std::unordered_set<std::string>>(mcp_tools_in_prompt);
            cfg.tool_callback = [mcp, &mcp_mutex, allowed, max_tool_string_bytes, artifacts_out, artifacts_seen, mcp_image_delivery](
                                    const nlohmann::json& call) -> nlohmann::json {
                std::string name = call.value("name", "");
                json args = call.value("arguments", json::object());
                if (name.empty()) {
                    return json{{"error", "missing tool name"}, {"call", call}};
                }
                if (allowed->find(name) == allowed->end()) {
                    return json{{"error", "tool not available"}, {"name", name}, {"call", call}};
                }

                // For image outputs: send base64/URL to the HTTP client (artifacts_out),
                // but never inject huge base64 payloads back into the text-only model prompt.
                json artifact_summaries = json::array();
                std::optional<std::string> forced_image_url;
                std::optional<std::string> forced_image_path;
                if (name == "sd_txt2img") {
                    if (mcp_image_delivery == "file" || mcp_image_delivery == "both") {
                        try {
                            std::filesystem::path outdir = std::filesystem::path("./examples/web/generated");
                            std::error_code ec;
                            std::filesystem::create_directories(outdir, ec);

                            std::string filename = "sd_txt2img_" + std::to_string(now_ms_epoch()) + ".png";
                            std::filesystem::path outpath = outdir / filename;
                            args["output"] = mcp_image_delivery;
                            args["out_path"] = outpath.string();
                            forced_image_url = std::string("/generated/") + filename;
                            forced_image_path = outpath.string();
                        } catch (...) {
                            args["output"] = mcp_image_delivery;
                        }
                    } else {
                        args["output"] = "base64";
                        args.erase("out_path");
                    }
                }

                std::string err;
                json result;
                {
                    std::lock_guard<std::mutex> lock(mcp_mutex);
                    result = mcp->call_tool(name, args, &err);
                }
                if (!err.empty() || result.is_null()) {
                    return json{{"error", "mcp tools/call failed"}, {"detail", err}, {"call", call}};
                }

                if (forced_image_url) {
                    json a = {{"kind", "image"},
                              {"mime_type", "image/png"},
                              {"tool", name},
                              {"url", *forced_image_url}};
                    if (forced_image_path) a["path"] = *forced_image_path;
                    std::string k = image_artifact_key(a);
                    if (k.empty() || artifacts_seen->insert(k).second) {
                        artifacts_out->push_back(a);
                        artifact_summaries.push_back(json{{"kind", "image"}, {"url", *forced_image_url}});
                    }
                }

                std::vector<json> images;
                std::unordered_set<size_t> seen_b64;
                collect_mcp_image_artifacts(result, images, seen_b64);
                for (auto& img : images) {
                    img["tool"] = name;
                    if (forced_image_url && !img.contains("url")) img["url"] = *forced_image_url;
                    std::string k = image_artifact_key(img);
                    if (k.empty() || artifacts_seen->insert(k).second) {
                        artifacts_out->push_back(img);

                        json summary = {{"kind", "image"}};
                        if (img.contains("url")) summary["url"] = img["url"];
                        artifact_summaries.push_back(std::move(summary));
                    }
                }

                // Strip base64 payloads before injecting into the prompt.
                json safe_result = strip_image_payloads(result);
                safe_result = truncate_large_strings(safe_result, max_tool_string_bytes);
                json resp = json{{"result", safe_result}, {"call", call}};
                if (!artifact_summaries.empty()) resp["artifacts"] = artifact_summaries;
                return resp;
            };
        }

        bool stream = body.value("stream", false);
        bool enable_thinking = body.value("enable_thinking", false);
        std::string model_name = body.value("model", std::string("qwen3-0.6b"));
        std::string prompt = apply_chat_template(messages, tools, true, enable_thinking);
        std::string resp_id = make_response_id();

        if (stream) {
            res.set_header("Content-Type", "text/event-stream");
            res.set_header("Cache-Control", "no-cache");
            res.set_header("Connection", "keep-alive");

            res.set_chunked_content_provider(
                "text/event-stream",
                [&, prompt, cfg, resp_id, model_name, artifacts_out](size_t, httplib::DataSink& sink) mutable {
                    std::lock_guard<std::mutex> lock(model_mutex);

                    auto ctx = model.prefill(prompt);
                    model.generate(ctx, cfg, [&](const std::string& token) {
                        std::string safe_token = sanitize_utf8(token);
                        json chunk = {
                            {"id", resp_id},
                            {"object", "chat.completion.chunk"},
                            {"model", model_name},
                            {"choices", json::array({
                                json{
                                    {"index", 0},
                                    {"delta", {{"role", "assistant"}, {"content", safe_token}}},
                                    {"finish_reason", nullptr}
                                }
                            })}
                        };
                        std::string data = "data: " + chunk.dump() + "\n\n";
                        sink.write(data.data(), data.size());
                    });

                    json done_chunk = {
                        {"id", resp_id},
                        {"object", "chat.completion.chunk"},
                        {"model", model_name},
                        {"choices", json::array({
                            json{
                                {"index", 0},
                                {"delta", json::object()},
                                {"finish_reason", "stop"}
                            }
                        })}
                    };
                    if (!artifacts_out->empty()) {
                        done_chunk["artifacts"] = *artifacts_out;
                    }
                    std::string end_data = "data: " + done_chunk.dump() + "\n\n";
                    sink.write(end_data.data(), end_data.size());

                    const char done[] = "data: [DONE]\n\n";
                    sink.write(done, sizeof(done) - 1);
                    return false;
                },
                [](bool) {});
            return;
        }

        std::string generated;
        {
            std::lock_guard<std::mutex> lock(model_mutex);
            auto ctx = model.prefill(prompt);
            model.generate(ctx, cfg, [&](const std::string& token) {
                generated += sanitize_utf8(token);
            });
        }

        json resp = {
            {"id", resp_id},
            {"object", "chat.completion"},
            {"model", model_name},
            {"choices", json::array({
                json{
                    {"index", 0},
                    {"message", {{"role", "assistant"}, {"content", generated}}},
                    {"finish_reason", "stop"}
                }
            })},
            {"usage", {{"prompt_tokens", 0}, {"completion_tokens", 0}}}
        };
        if (!artifacts_out->empty()) {
            resp["artifacts"] = *artifacts_out;
        }

        res.set_content(resp.dump(), "application/json");
    });

    const int port = opt.port;
    std::cout << "Qwen3 OpenAI-style API server listening on http://0.0.0.0:" << port << std::endl;
    std::cout << "POST /v1/chat/completions with OpenAI-format payloads." << std::endl;
    server.listen("0.0.0.0", port);

    return 0;
}
