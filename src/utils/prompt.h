#pragma once
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// 定义支持的模板类型
enum class TemplateType {
    CHATML,   // 原始的 <|im_start|> 格式 (Qwen/Yi 等)
    HUNYUAN   // 混元 XML 格式
};

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
    bool enable_thinking = true,
    TemplateType template_type = TemplateType::CHATML // 新增参数，默认使用 ChatML
);
