#pragma once

#include <array>
#include <cassert>
#include <functional>
#include <locale>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>
#include <nlohmann/json.hpp>

struct GenerateConfig {
    int max_new_tokens = 4096;
    float temperature = 0.3f;
    float top_p = 0.8f;
    int top_k = 50;
    float repetition_penalty = 1.1f;
    int beam_size = 1;
    int do_sample = 1;

    // 当模型输出 </tool_call> 时，会调用回调，返回的 JSON 将作为 tool_response 注入对话。
    // 若为空，则会返回 {"tool_call": <原始调用>} 作为占位。
    std::function<nlohmann::json(const nlohmann::json&)> tool_callback = nullptr;

    // 调试开关：若为 true，在工具调用过程中输出中间信息到 stderr
    bool debug = false;
};

struct qwen3_0_6b_ctx;

class qwen3_0_6b {
public:
    qwen3_0_6b(std::string embed_param,
              std::string embed_bin,
              std::string proj_out_param,
              std::string decoder_param,
              std::string decoder_bin,
              std::string vocab_file,
              std::string merges_file,
              bool use_vulkan);

    ~qwen3_0_6b();

    std::shared_ptr<qwen3_0_6b_ctx> prefill(const std::string& input_text);

    std::shared_ptr<qwen3_0_6b_ctx> prefill(const std::string& input_text,
                                         const std::shared_ptr<qwen3_0_6b_ctx> ctx);

    // 定义工具：内部生成包含 tools 段落的 prompt，对 ctx 进行 prefill 后返回。
    // 工具只需定义一次，后续对话直接复用返回的 ctx。
    std::shared_ptr<qwen3_0_6b_ctx> define_tools(
        const std::shared_ptr<qwen3_0_6b_ctx>& ctx,
        const std::vector<nlohmann::json>& tools,
        const std::string& system_prompt = "You are a helpful assistant.");

    std::shared_ptr<qwen3_0_6b_ctx> generate(const std::shared_ptr<qwen3_0_6b_ctx>& ctx, const GenerateConfig& cfg, std::function<void(const std::string&)> callback);

    bool decode(std::shared_ptr<qwen3_0_6b_ctx> ctx,
                std::function<void(const std::string&)> callback);

    // ---------- 工具定义辅助模板 ----------
    template<typename T>
    static constexpr const char* json_type_name() {
        if constexpr (std::is_same_v<T, int> || std::is_same_v<T, long> || std::is_same_v<T, long long>) return "integer";
        else if constexpr (std::is_same_v<T, bool>) return "boolean";
        else if constexpr (std::is_floating_point_v<T>) return "number";
        else return "string";
    }

    // 根据 C++ 函数形参类型与名称自动生成 JSON schema。
    template<typename Ret, typename... Args>
    static nlohmann::json make_function_tool(
        const std::string& name,
        const std::string& description,
        const std::array<std::string, sizeof...(Args)>& arg_names)
    {
        // 运行时防御（MSVC 对参数 constexpr 要求较严格）
        assert(arg_names.size() == sizeof...(Args));
        nlohmann::json properties = nlohmann::json::object();
        size_t idx = 0;
        // 展开参数类型与名称
        ((
            properties[arg_names[idx]] = nlohmann::json{
                {"type", json_type_name<Args>()},
                {"description", ""}
            },
            ++idx
        ), ...);

        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", description},
                {"parameters", {
                    {"type", "object"},
                    {"properties", properties},
                    {"required", arg_names}
                }}
            }}
        };
    }

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};