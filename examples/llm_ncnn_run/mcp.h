#pragma once

#include "mcp_client.h"
#include "options.h"

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

using nlohmann::json;

struct McpState {
    std::shared_ptr<McpStdioClient> client;
    std::vector<json> openai_tools;
    std::unordered_set<std::string> tool_names;
};

McpState init_mcp(const Options& opt);
