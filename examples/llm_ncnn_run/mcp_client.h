#pragma once

#include <memory>
#include <string>

#include <nlohmann/json.hpp>

using nlohmann::json;

class McpStdioClient {
public:
    enum class Transport {
        Lsp,
        Jsonl
    };

    McpStdioClient();
    ~McpStdioClient();

    McpStdioClient(const McpStdioClient&) = delete;
    McpStdioClient& operator=(const McpStdioClient&) = delete;
    McpStdioClient(McpStdioClient&&) noexcept;
    McpStdioClient& operator=(McpStdioClient&&) noexcept;

    void set_timeout_ms(int timeout_ms);
    void set_debug(bool debug);
    void set_transport(Transport t);

    bool start(const std::string& cmdline, std::string* err = nullptr);
    json list_tools(std::string* err = nullptr);
    json call_tool(const std::string& name, const json& arguments, std::string* err = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
