#pragma once
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct Message {
    std::string role;
    std::string content;
    std::string reasoning_content;
    std::vector<json> tool_calls;

    Message() = default;
    Message(std::string r, std::string c, std::string rc = "", std::vector<json> tc = {})
        : role(std::move(r)), content(std::move(c)), reasoning_content(std::move(rc)), tool_calls(std::move(tc)) {}
};

std::string apply_chat_template(
    const std::vector<Message>& messages,
    const std::vector<json>& tools = {},
    bool add_generation_prompt = true,
    bool enable_thinking = true
);
