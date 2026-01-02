#pragma once

#include "mcp.h"
#include "options.h"

#include "ncnn_llm_gpt.h"

#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

using nlohmann::json;

int run_openai_server(const Options& opt,
                      ncnn_llm_gpt& model,
                      const std::vector<json>& builtin_tools,
                      const std::unordered_map<std::string, std::function<json(const json&)>>& builtin_router,
                      const McpState& mcp,
                      std::mutex& mcp_mutex);
