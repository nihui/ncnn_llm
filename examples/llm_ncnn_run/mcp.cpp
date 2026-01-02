#include "mcp.h"

#include <iostream>

McpState init_mcp(const Options& opt) {
    McpState mcp;
    if (opt.mcp_server_cmdline.empty()) return mcp;

    mcp.client = std::make_shared<McpStdioClient>();
    mcp.client->set_timeout_ms(opt.mcp_timeout_ms);
    mcp.client->set_debug(opt.mcp_debug);
    mcp.client->set_transport(opt.mcp_transport == "jsonl" ? McpStdioClient::Transport::Jsonl : McpStdioClient::Transport::Lsp);

    std::string err;
    std::cerr << "[MCP] launching stdio server...\n";
    if (!mcp.client->start(opt.mcp_server_cmdline, &err)) {
        std::cerr << "Warning: failed to initialize MCP server: " << err << "\n";
        mcp.client.reset();
        return mcp;
    }

    std::cerr << "[MCP] connected; listing tools...\n";
    std::string list_err;
    json tools = mcp.client->list_tools(&list_err);
    if (!list_err.empty()) {
        std::cerr << "Warning: MCP tools/list failed: " << list_err << "\n";
        return mcp;
    }
    if (!tools.is_array()) return mcp;

    for (const auto& t : tools) {
        if (!t.is_object()) continue;
        std::string name = t.value("name", "");
        if (name.empty()) continue;
        mcp.tool_names.insert(name);
        json openai_tool = {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", t.value("description", "")},
                {"parameters", t.value("inputSchema", json::object())}
            }}
        };
        mcp.openai_tools.push_back(std::move(openai_tool));
    }
    std::cerr << "Loaded " << mcp.openai_tools.size() << " MCP tool(s) from stdio server.\n";
    return mcp;
}
