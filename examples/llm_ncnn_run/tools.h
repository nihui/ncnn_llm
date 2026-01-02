#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

using nlohmann::json;

std::string tool_name_from_openai_tool(const json& tool);
std::vector<json> merge_tools_by_name(const std::vector<json>& base, const std::vector<json>& extra);
std::vector<json> make_builtin_tools();
std::unordered_map<std::string, std::function<json(const json&)>> make_builtin_router();
