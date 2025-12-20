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
        // HUNYUAN LOGIC
        // ==========================================
        
        std::string system_prompt = "";
        bool first_system = true;

        // 1. Accumulate all System Messages first
        for (const auto& msg : messages) {
            if (msg.role == "system") {
                if (!first_system) system_prompt += "\n\n";
                system_prompt += msg.content;
                first_system = false;
            }
        }

        prompt << system_prompt;

        // 2. Add Tools Definition if present
        if (has_tools) {
            if (!system_prompt.empty()) prompt << "\n\n";
            prompt << "# Tools\n\n"
                   << "You may call one or more functions to assist with the user query.\n\n"
                   << "You are provided with function signatures within <tools></tools> XML tags:\n"
                   << "<tools>\n";
            
            for (size_t i = 0; i < tools.size(); ++i) {
                if (i > 0) prompt << "\n";
                prompt << tools[i].dump();
            }
            
            prompt << "\n</tools>\n\n"
                   << "For function call returns, you should first print <tool_calls>\n"
                   << "For each function call, you should return object like:\n"
                   << "<tool_call>function_name\n```json\nfunction_arguments_in_json_format\n```</tool_call>\n"
                   << "At the end of function call returns, you should print </tool_calls>";
        }

        // 3. Add Separator (Placeholder) if system or tools existed
        if (!system_prompt.empty() || has_tools) {
            prompt << "<｜hy_place▁holder▁no▁3｜>";
        }

        // 4. Conversation Loop
        for (size_t i = 0; i < messages.size(); ++i) {
            const auto& msg = messages[i];

            if (msg.role == "system") continue; // Already handled

            if (msg.role == "user") {
                // User message wraps the Assistant start token at the end
                prompt << "\n<｜hy_User｜>" << msg.content << "<｜hy_Assistant｜>\n";
            } 
            else if (msg.role == "assistant") {
                // Handle Reasoning/Thinking extraction
                std::string reasoning = msg.reasoning_content;
                std::string content = msg.content;
                
                // Fallback: extract thinking from content if not explicit in struct
                if (reasoning.empty() && content.find("</think>") != std::string::npos) {
                    size_t start_think = content.find("<think>");
                    size_t end_think = content.find("</think>");
                    if (start_think != std::string::npos && end_think != std::string::npos) {
                        reasoning = content.substr(start_think + 7, end_think - (start_think + 7));
                        reasoning = lstrip_newlines(rstrip_newlines(reasoning));
                        content = content.substr(end_think + 8);
                        content = lstrip_newlines(content);
                    }
                }

                if (!reasoning.empty()) {
                    prompt << "<think>\n" << reasoning << "\n</think>\n";
                }

                // Handle Tool Calls vs Normal Content
                if (!msg.tool_calls.empty()) {
                    prompt << "<tool_calls>";
                    for (const auto& tool : msg.tool_calls) {
                        json tc_obj = tool;
                        if (tc_obj.contains("function")) tc_obj = tc_obj["function"];
                        
                        // Hunyuan expects arguments in a markdown json block
                        std::string args_str = tc_obj["arguments"].dump();
                        
                        prompt << "<tool_call>" << tc_obj["name"].get<std::string>() << "\n"
                               << "```json\n" << args_str << "\n```"
                               << "</tool_call>";
                    }
                    prompt << "</tool_calls>"; 
                } else {
                    prompt << content;
                }
            } 
            else if (msg.role == "tool") {
                // Tool output is wrapped in User tags but with specific tool_response tags
                prompt << "<｜hy_User｜><tool_responses><tool_response>" 
                       << msg.content 
                       << "</tool_response></tool_responses><｜hy_Assistant｜>";
            }
        }

        // 5. Generation Prompt
        // If we need to force the start of a generation (usually implied by the last <｜hy_Assistant｜>)
        // We mainly check this to inject the empty <think> block if thinking is disabled
        if (add_generation_prompt) {
             if (!enable_thinking) {
                 prompt << "<think>\n\n</think>\n";
             }
        }

    } else {
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