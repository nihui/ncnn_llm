#include "prompt.h"
#include <sstream>
#include <algorithm>
#include <iostream>

static std::string lstrip_newlines(const std::string& s) {
    size_t start = s.find_first_not_of('\n');
    return (start == std::string::npos) ? "" : s.substr(start);
}

static std::string rstrip_newlines(const std::string& s) {
    size_t end = s.find_last_not_of('\n');
    return (end == std::string::npos) ? "" : s.substr(0, end + 1);
}

std::string apply_chat_template(
    const std::vector<Message>& messages,
    const std::vector<json>& tools,
    bool add_generation_prompt,
    bool enable_thinking,
    TemplateType template_type
) {
    std::stringstream prompt;
    bool has_tools = !tools.empty();

    if (template_type == TemplateType::HUNYUAN) {
        // ==========================================
        // HUNYUAN LOGIC (aligned with provided jinja)
        // - No BOS (per your requirement)
        // - Placeholder/no3 behavior aligned
        // - Tool formatting aligned
        // - Empty <think> is appended ONLY when we are positioned to generate assistant text
        // ==========================================
        static const std::string HY_USER = "<｜hy_User｜>";
        static const std::string HY_ASSISTANT = "<｜hy_Assistant｜>";
        static const std::string HY_PH3 = "<｜hy_place▁holder▁no▁3｜>";
        static const std::string HY_EOS = "<｜hy_place▁holder▁no▁2｜>";

        // Track whether the current prompt ends right after an assistant tag,
        // i.e. the model is expected to generate assistant content next.
        bool ends_with_assistant_tag = false;

        // 1) Merge system prompts: join all system messages with "\n\n"
        std::string system_prompt;
        bool first_sp = true;
        for (const auto& m : messages) {
            if (m.role == "system") {
                if (first_sp) {
                    system_prompt += m.content;
                    first_sp = false;
                } else {
                    system_prompt += "\n\n";
                    system_prompt += m.content;
                }
            }
        }

        if (!system_prompt.empty()) {
            prompt << system_prompt;
            ends_with_assistant_tag = false;
        }

        // 2) Tools header block (same spirit as the jinja)
        if (has_tools) {
            if (!system_prompt.empty()) {
                prompt << "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.";
            } else {
                prompt << "# Tools\n\nYou may call one or more functions to assist with the user query.";
            }

            prompt << "\n\nYou are provided with function signatures within <tools></tools> XML tags:";
            prompt << "\n<tools>\n";
            for (size_t i = 0; i < tools.size(); ++i) {
                if (i > 0) prompt << "\n";
                prompt << tools[i].dump();
            }
            prompt << "\n</tools>\n\n";

            prompt << "For function call returns, you should first print <tool_calls>";
            prompt << "For each function call, you should return object like:\n";
            prompt << "<tool_call>function_name\n```json\nfunction_arguments_in_json_format\n```</tool_call>";
            prompt << "At the end of function call returns, you should print </tool_calls>";
            ends_with_assistant_tag = false;
        }

        // 3) placeholder no3 if system_prompt or tools exists
        if (!system_prompt.empty() || has_tools) {
            prompt << HY_PH3; // IMPORTANT: no newline
            ends_with_assistant_tag = false;
        }

        // 4) Iterate messages (skip system, already merged)
        bool is_tool = false;          // ns.is_tool
        bool is_output_first = true;   // ns.is_output_first
        bool is_last_user = false;     // ns.is_last_user

        for (size_t i = 0; i < messages.size(); ++i) {
            const auto& msg = messages[i];
            if (msg.role == "system") continue;

            if (msg.role == "user") {
                // NOTE: we assume tool blocks are normally followed by assistant, like OpenAI tool flow.
                is_last_user = true;
                is_tool = false; // jinja sets ns.is_tool=false on user
                is_output_first = true;

                // '<｜hy_User｜>' + user + '<｜hy_Assistant｜>'
                prompt << HY_USER << msg.content << HY_ASSISTANT;
                ends_with_assistant_tag = true;
                continue;
            }

            if (msg.role == "tool") {
                is_last_user = false;
                is_tool = true;

                if (is_output_first) {
                    prompt << HY_USER
                           << "<tool_responses><tool_response>"
                           << msg.content
                           << "</tool_response>";
                    is_output_first = false;
                } else {
                    prompt << "\n<tool_response>" << msg.content << "</tool_response>";
                }

                ends_with_assistant_tag = false;
                continue;
            }

            if (msg.role == "assistant") {
                is_last_user = false;

                // If we were outputting tool responses, close them and open assistant
                if (is_tool) {
                    prompt << "</tool_responses>" << HY_ASSISTANT;
                    is_tool = false;
                    is_output_first = true;
                    ends_with_assistant_tag = true;
                }

                // Assistant WITH tool_calls
                if (!msg.tool_calls.empty()) {
                    bool first_call = true;

                    for (const auto& raw_tc : msg.tool_calls) {
                        json tc = raw_tc;
                        if (tc.contains("function")) tc = tc["function"];

                        const std::string name = tc.value("name", "");
                        json args_j = tc.contains("arguments") ? tc["arguments"] : json::object();

                        std::string args;
                        if (args_j.is_string()) args = args_j.get<std::string>();
                        else args = args_j.dump();

                        if (first_call) {
                            if (!msg.content.empty()) {
                                prompt << msg.content;
                            }
                            prompt << "<tool_calls><tool_call>"
                                   << name << "\n"
                                   << "```json\n"
                                   << args << "\n"
                                   << "```"
                                   << "</tool_call>";
                            first_call = false;
                        } else {
                            prompt << "\n<tool_call>"
                                   << name << "\n"
                                   << "```json\n"
                                   << args << "\n"
                                   << "```"
                                   << "</tool_call>";
                        }
                    }

                    prompt << "</tool_calls>" << HY_EOS;
                    ends_with_assistant_tag = false; // EOS printed
                    continue;
                }

                // Assistant WITHOUT tool_calls
                std::string content = msg.content;

                // if '<answer>' in content and not loop.last => keep only answer inner
                const bool is_last_message = (i == messages.size() - 1);
                if (!is_last_message) {
                    size_t pos = content.rfind("<answer>");
                    if (pos != std::string::npos) {
                        content = content.substr(pos + std::string("<answer>").size());
                        size_t endpos = content.rfind("</answer>");
                        if (endpos != std::string::npos) {
                            content = content.substr(0, endpos);
                        }
                        auto l = content.find_first_not_of(" \t\r\n");
                        auto r = content.find_last_not_of(" \t\r\n");
                        if (l == std::string::npos) content.clear();
                        else content = content.substr(l, r - l + 1);
                    }
                }

                prompt << content << HY_EOS;
                ends_with_assistant_tag = false; // EOS printed
                continue;
            }

            // Unknown roles: ignore (or handle as needed)
        }

        // 5) If ended while still inside tool_responses, close and open assistant
        if (is_tool) {
            prompt << "</tool_responses>" << HY_ASSISTANT;
            is_tool = false;
            ends_with_assistant_tag = true;
        }

        // 6) add_generation_prompt tail
        if (add_generation_prompt && !is_last_user && !is_tool) {
            prompt << HY_ASSISTANT;
            ends_with_assistant_tag = true;
        }

        // 7) FIX: only append empty think when we're positioned to generate assistant text
        if (!enable_thinking && ends_with_assistant_tag) {
            prompt << "<think>\n\n</think>\n";
        }

    } else if (template_type == TemplateType::CHATML) {
        // ==========================================
        // CHATML LOGIC (Original)
        // ==========================================

        if (has_tools) {
            prompt << "<|im_start|>system\n";
            if (!messages.empty() && messages[0].role == "system") {
                prompt << messages[0].content << "\n\n";
            }
            prompt << "# Tools\n\n"
                   << "You may call one or more functions to assist with the user query.\n\n"
                   << "You are provided with function signatures within <tools></tools> XML tags:\n"
                   << "<tools>";
            for (const auto& tool : tools) {
                prompt << "\n" << tool.dump();
            }
            prompt << "\n</tools>\n\n"
                   << "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
                   << "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n";
        } else {
            if (!messages.empty() && messages[0].role == "system") {
                prompt << "<|im_start|>system\n" << messages[0].content << "<|im_end|>\n";
            }
        }

        bool multi_step_tool = true;
        int last_query_index = (int)messages.size() - 1;

        // Logic to determine if we are in a multi-step tool sequence
        for (int i = (int)messages.size() - 1; i >= 0; --i) {
            const auto& msg = messages[i];
            bool is_tool_response = false;
            if (msg.content.size() >= 15) {
                if (msg.content.rfind("<tool_response>", 0) == 0 &&
                    msg.content.find("</tool_response>") != std::string::npos) {
                    is_tool_response = true;
                }
            }
            if (multi_step_tool && msg.role == "user" && !is_tool_response) {
                multi_step_tool = false;
                last_query_index = i;
            }
        }

        for (size_t i = 0; i < messages.size(); ++i) {
            const auto& msg = messages[i];
            std::string content = msg.content;

            if (msg.role == "system" && i == 0) continue;

            if (msg.role == "user" || msg.role == "system") {
                prompt << "<|im_start|>" << msg.role << "\n" << content << "<|im_end|>\n";
            }
            else if (msg.role == "assistant") {
                std::string reasoning_content = msg.reasoning_content;
                std::string final_content = content;

                if (reasoning_content.empty() && final_content.find("</think>") != std::string::npos) {
                    size_t start_think = final_content.find("<think>");
                    size_t end_think = final_content.find("</think>");
                    if (start_think != std::string::npos && end_think != std::string::npos) {
                        std::string extracted = final_content.substr(start_think + 7, end_think - (start_think + 7));
                        reasoning_content = lstrip_newlines(rstrip_newlines(extracted));
                        std::string remainder = final_content.substr(end_think + 8);
                        final_content = lstrip_newlines(remainder);
                    }
                }

                bool is_after_last_query = ((int)i > last_query_index);
                bool is_last_message = (i == messages.size() - 1);
                bool has_reasoning = !reasoning_content.empty();

                bool show_thinking = enable_thinking && is_after_last_query && (is_last_message || has_reasoning);

                prompt << "<|im_start|>" << msg.role << "\n";

                if (show_thinking && has_reasoning) {
                    prompt << "<think>\n" << reasoning_content << "\n</think>\n\n";
                }

                if (!final_content.empty()) {
                    prompt << final_content;
                }

                if (!msg.tool_calls.empty()) {
                    if (show_thinking || !final_content.empty()) prompt << "\n";
                    for (size_t t = 0; t < msg.tool_calls.size(); ++t) {
                        if (t > 0) prompt << "\n";
                        json tc_obj = msg.tool_calls[t];
                        if (tc_obj.contains("function")) tc_obj = tc_obj["function"];

                        prompt << "<tool_call>\n"
                               << "{\"name\": \"" << tc_obj["name"].get<std::string>() << "\", "
                               << "\"arguments\": " << tc_obj["arguments"].dump() << "}\n"
                               << "</tool_call>";
                    }
                }
                prompt << "<|im_end|>\n";
            }
            else if (msg.role == "tool") {
                if (i == 0 || messages[i-1].role != "tool") prompt << "<|im_start|>user";
                prompt << "\n<tool_response>\n" << content << "\n</tool_response>";
                if (i == messages.size() - 1 || messages[i+1].role != "tool") prompt << "<|im_end|>\n";
            }
        }

        if (add_generation_prompt) {
            prompt << "<|im_start|>assistant\n";

            if (!enable_thinking) {
                prompt << "<think>\n\n</think>\n\n";
            }
        }
    }

    return prompt.str();
}
