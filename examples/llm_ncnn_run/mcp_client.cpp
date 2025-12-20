#include "mcp_client.h"

#include "util.h"

#include <algorithm>
#include <climits>
#include <chrono>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

#ifndef _WIN32
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
#endif

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

        int in_pipe[2] = {-1, -1};
        int out_pipe[2] = {-1, -1};
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

        child_stdin_ = in_pipe[1];
        child_stdout_ = out_pipe[0];
        ::close(in_pipe[0]);
        ::close(out_pipe[1]);

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
#else
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

static std::wstring utf8_to_wide(const std::string& s) {
    if (s.empty()) return std::wstring();
    int len = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
    if (len <= 0) return std::wstring();
    std::wstring w(len, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), w.data(), len);
    return w;
}

static std::string win_last_error(DWORD code) {
    LPSTR msg = nullptr;
    DWORD flags = FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS;
    DWORD n = FormatMessageA(flags, nullptr, code, 0, (LPSTR)&msg, 0, nullptr);
    std::string out = (n && msg) ? std::string(msg, n) : std::string("win32 error ") + std::to_string(code);
    if (msg) LocalFree(msg);
    while (!out.empty() && (out.back() == '\r' || out.back() == '\n')) out.pop_back();
    return out;
}

class StdioProcess {
public:
    ~StdioProcess() { stop(); }

    bool start_cmdline(const std::string& cmdline) {
        if (cmdline.empty()) return false;
        stop();

        SECURITY_ATTRIBUTES sa;
        sa.nLength = sizeof(sa);
        sa.lpSecurityDescriptor = nullptr;
        sa.bInheritHandle = TRUE;

        HANDLE child_stdin_read = nullptr;
        HANDLE child_stdin_write = nullptr;
        HANDLE child_stdout_read = nullptr;
        HANDLE child_stdout_write = nullptr;

        if (!CreatePipe(&child_stdin_read, &child_stdin_write, &sa, 0)) return false;
        if (!CreatePipe(&child_stdout_read, &child_stdout_write, &sa, 0)) {
            CloseHandle(child_stdin_read);
            CloseHandle(child_stdin_write);
            return false;
        }

        SetHandleInformation(child_stdin_write, HANDLE_FLAG_INHERIT, 0);
        SetHandleInformation(child_stdout_read, HANDLE_FLAG_INHERIT, 0);

        STARTUPINFOW si;
        ZeroMemory(&si, sizeof(si));
        si.cb = sizeof(si);
        si.hStdInput = child_stdin_read;
        si.hStdOutput = child_stdout_write;
        si.hStdError = GetStdHandle(STD_ERROR_HANDLE);
        si.dwFlags |= STARTF_USESTDHANDLES;

        PROCESS_INFORMATION pi;
        ZeroMemory(&pi, sizeof(pi));

        std::wstring wcmd = utf8_to_wide(cmdline);
        if (!CreateProcessW(nullptr, wcmd.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &si, &pi)) {
            CloseHandle(child_stdin_read);
            CloseHandle(child_stdin_write);
            CloseHandle(child_stdout_read);
            CloseHandle(child_stdout_write);
            return false;
        }

        CloseHandle(child_stdin_read);
        CloseHandle(child_stdout_write);

        child_process_ = pi.hProcess;
        child_thread_ = pi.hThread;
        child_stdin_ = child_stdin_write;
        child_stdout_ = child_stdout_read;

        return true;
    }

    void stop() {
        if (child_stdin_) {
            CloseHandle(child_stdin_);
            child_stdin_ = nullptr;
        }
        if (child_stdout_) {
            CloseHandle(child_stdout_);
            child_stdout_ = nullptr;
        }
        if (child_process_) {
            TerminateProcess(child_process_, 1);
            CloseHandle(child_process_);
            child_process_ = nullptr;
        }
        if (child_thread_) {
            CloseHandle(child_thread_);
            child_thread_ = nullptr;
        }
        read_buf_.clear();
    }

    bool write_all(const std::string& s) {
        if (!child_stdin_) return false;
        DWORD written = 0;
        const char* p = s.data();
        size_t n = s.size();
        while (n > 0) {
            if (!WriteFile(child_stdin_, p, (DWORD)std::min(n, (size_t)INT_MAX), &written, nullptr)) {
                return false;
            }
            p += written;
            n -= written;
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

private:
    bool read_some_until(const std::chrono::steady_clock::time_point deadline, std::string* err) {
        while (true) {
            int timeout_ms = (int)std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count();
            if (timeout_ms < 0) timeout_ms = 0;

            DWORD avail = 0;
            if (!PeekNamedPipe(child_stdout_, nullptr, 0, nullptr, &avail, nullptr)) {
                if (err) *err = win_last_error(GetLastError());
                return false;
            }
            if (avail == 0) {
                if (timeout_ms == 0) {
                    if (err) *err = "timeout waiting for MCP server output";
                    return false;
                }
                Sleep(std::min(timeout_ms, 10));
                continue;
            }

            char tmp[4096];
            DWORD read = 0;
            if (!ReadFile(child_stdout_, tmp, (DWORD)sizeof(tmp), &read, nullptr) || read == 0) {
                if (err) *err = win_last_error(GetLastError());
                return false;
            }
            read_buf_.append(tmp, (size_t)read);
            return true;
        }
    }

    HANDLE child_stdin_ = nullptr;
    HANDLE child_stdout_ = nullptr;
    HANDLE child_process_ = nullptr;
    HANDLE child_thread_ = nullptr;
    std::string read_buf_;
};
#endif

} // namespace

struct McpStdioClient::Impl {
    using Transport = McpStdioClient::Transport;

    StdioProcess proc;
    int64_t next_id = 1;
    int timeout_ms = 15000;
    bool debug = false;
    Transport transport = Transport::Lsp;

    bool start(const std::string& cmdline, std::string* err) {
        if (cmdline.empty()) return false;
#ifndef _WIN32
        auto argv = split_cmdline(cmdline);
        if (argv.empty()) return false;
        if (debug) {
            std::cerr << "[MCP] start: " << cmdline << "\n";
            std::cerr << "[MCP] argv:";
            for (const auto& a : argv) std::cerr << " [" << a << "]";
            std::cerr << "\n";
        }
        if (!proc.start(argv)) {
            if (err) *err = "failed to start process";
            return false;
        }
#else
        if (debug) {
            std::cerr << "[MCP] start: " << cmdline << "\n";
        }
        if (!proc.start_cmdline(cmdline)) {
            if (err) *err = "failed to start process";
            return false;
        }
#endif
        if (!initialize(err)) {
            proc.stop();
            return false;
        }
        return true;
    }

    json list_tools(std::string* err) {
        json resp = request("tools/list", json::object(), err);
        if (resp.is_null()) return json::object();
        return resp.value("tools", json::array());
    }

    json call_tool(const std::string& name, const json& arguments, std::string* err) {
        json params = {{"name", name}, {"arguments", arguments}};
        return request("tools/call", params, err);
    }

    void set_timeout_ms(int timeout_ms_in) { timeout_ms = timeout_ms_in; }
    void set_debug(bool debug_in) { debug = debug_in; }
    void set_transport(Transport t) { transport = t; }

private:
    bool initialize(std::string* err) {
        json params = {
            {"protocolVersion", "2024-11-05"},
            {"capabilities", {{"tools", json::object()}}},
            {"clientInfo", {{"name", "ncnn_llm"}, {"version", "demo"}}}
        };
        json resp = request("initialize", params, err);
        if (resp.is_null()) return false;

        json notif = {{"jsonrpc", "2.0"}, {"method", "notifications/initialized"}, {"params", json::object()}};
        send(notif);
        return true;
    }

    bool send(const json& msg) {
        std::string body = msg.dump();
        std::string framed;
        if (transport == Transport::Jsonl) {
            framed = body;
            framed.push_back('\n');
        } else {
            framed = "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
        }
        if (debug) {
            std::string preview = body;
            if (preview.size() > 800) preview.resize(800), preview += "...";
            std::cerr << "[MCP] => " << preview << "\n";
        }
        return proc.write_all(framed);
    }

    json request(const std::string& method, const json& params, std::string* err) {
        int64_t id = next_id++;
        json req = {{"jsonrpc", "2.0"}, {"id", id}, {"method", method}, {"params", params}};
        if (!send(req)) {
            if (err) *err = "failed to write request";
            return nullptr;
        }

        while (true) {
            json msg;
            std::string read_err;
            if (!proc.read_json(msg, timeout_ms, &read_err)) {
                if (err) *err = read_err.empty() ? "failed to read response" : read_err;
                return nullptr;
            }
            if (!msg.is_object()) continue;
            if (debug) {
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
        }
    }
};

McpStdioClient::McpStdioClient() : impl_(std::make_unique<Impl>()) {}
McpStdioClient::~McpStdioClient() = default;
McpStdioClient::McpStdioClient(McpStdioClient&&) noexcept = default;
McpStdioClient& McpStdioClient::operator=(McpStdioClient&&) noexcept = default;

void McpStdioClient::set_timeout_ms(int timeout_ms) {
    impl_->set_timeout_ms(timeout_ms);
}

void McpStdioClient::set_debug(bool debug) {
    impl_->set_debug(debug);
}

void McpStdioClient::set_transport(Transport t) {
    impl_->set_transport(t);
}

bool McpStdioClient::start(const std::string& cmdline, std::string* err) {
    return impl_->start(cmdline, err);
}

json McpStdioClient::list_tools(std::string* err) {
    return impl_->list_tools(err);
}

json McpStdioClient::call_tool(const std::string& name, const json& arguments, std::string* err) {
    return impl_->call_tool(name, arguments, err);
}
