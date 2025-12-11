#include "qwen3_0.6b.h"
#include "utils/prompt.h"

#include <chrono>
#include <httplib.h>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

using nlohmann::json;

namespace {

std::string extract_content(const json& content) {
    if (content.is_string()) {
        return content.get<std::string>();
    }
    if (content.is_array()) {
        std::string merged;
        for (const auto& part : content) {
            if (part.is_string()) {
                merged += part.get<std::string>();
            } else if (part.is_object()) {
                auto type = part.value("type", "");
                if (type == "text" && part.contains("text")) {
                    merged += part["text"].get<std::string>();
                }
            }
        }
        return merged;
    }
    return "";
}

std::vector<Message> parse_messages(const json& messages_json) {
    std::vector<Message> messages;
    for (const auto& m : messages_json) {
        if (!m.contains("role")) continue;
        Message msg;
        msg.role = m.value("role", "");
        if (m.contains("content")) {
            msg.content = extract_content(m["content"]);
        }
        if (m.contains("tool_calls") && m["tool_calls"].is_array()) {
            msg.tool_calls = m["tool_calls"].get<std::vector<json>>();
        }
        msg.reasoning_content = m.value("reasoning_content", "");
        messages.push_back(std::move(msg));
    }
    return messages;
}

std::string make_response_id() {
    using namespace std::chrono;
    auto now = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    std::stringstream ss;
    ss << "chatcmpl-" << now;
    return ss.str();
}

json make_error(int status, const std::string& message) {
    json err;
    err["error"] = {{"type", "invalid_request_error"}, {"message", message}};
    err["status"] = status;
    return err;
}

} // namespace

int main() {
    qwen3_0_6b model(
        "./assets/qwen3_0.6b/qwen3_embed_token.ncnn.param",
        "./assets/qwen3_0.6b/qwen3_embed_token.ncnn.bin",
        "./assets/qwen3_0.6b/qwen3_proj_out.ncnn.param",
        "./assets/qwen3_0.6b/qwen3_decoder.ncnn.param",
        "./assets/qwen3_0.6b/qwen3_decoder.ncnn.bin",
        "./assets/qwen3_0.6b/vocab.txt",
        "./assets/qwen3_0.6b/merges.txt",
        /*use_vulkan=*/false);

    std::mutex model_mutex;

    httplib::Server server;

    if (!server.set_mount_point("/", "./examples/web")) {
        std::cerr << "Warning: failed to mount ./examples/web for static frontend.\n";
    }

    server.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(make_error(400, std::string("Invalid JSON: ") + e.what()).dump(), "application/json");
            return;
        }

        if (!body.contains("messages") || !body["messages"].is_array()) {
            res.status = 400;
            res.set_content(make_error(400, "`messages` must be an array").dump(), "application/json");
            return;
        }

        auto messages = parse_messages(body["messages"]);
        if (messages.empty() || messages[0].role != "system") {
            messages.insert(messages.begin(), Message{"system", "You are a helpful assistant."});
        }

        std::vector<json> tools;
        if (body.contains("tools") && body["tools"].is_array()) {
            for (const auto& t : body["tools"]) {
                if (t.is_object()) tools.push_back(t);
            }
        }

        GenerateConfig cfg;
        cfg.max_new_tokens = body.value("max_tokens", cfg.max_new_tokens);
        cfg.temperature = body.value("temperature", cfg.temperature);
        cfg.top_p = body.value("top_p", cfg.top_p);
        cfg.top_k = body.value("top_k", cfg.top_k);
        cfg.repetition_penalty = body.value("repetition_penalty", cfg.repetition_penalty);
        cfg.beam_size = body.value("beam_size", cfg.beam_size);
        if (body.contains("do_sample") && body["do_sample"].is_boolean()) {
            cfg.do_sample = body["do_sample"].get<bool>() ? 1 : 0;
        } else if (cfg.temperature <= 0.0f) {
            cfg.do_sample = 0;
        }

        bool stream = body.value("stream", false);
        bool enable_thinking = body.value("enable_thinking", false);
        std::string model_name = body.value("model", std::string("qwen3-0.6b"));
        std::string prompt = apply_chat_template(messages, tools, true, enable_thinking);
        std::string resp_id = make_response_id();

        if (stream) {
            res.set_header("Content-Type", "text/event-stream");
            res.set_header("Cache-Control", "no-cache");
            res.set_header("Connection", "keep-alive");

            res.set_chunked_content_provider(
                "text/event-stream",
                [&, prompt, cfg, resp_id, model_name](size_t, httplib::DataSink& sink) mutable {
                    std::lock_guard<std::mutex> lock(model_mutex);

                    auto ctx = model.prefill(prompt);
                    model.generate(ctx, cfg, [&](const std::string& token) {
                        json chunk = {
                            {"id", resp_id},
                            {"object", "chat.completion.chunk"},
                            {"model", model_name},
                            {"choices", json::array({
                                json{
                                    {"index", 0},
                                    {"delta", {{"role", "assistant"}, {"content", token}}},
                                    {"finish_reason", nullptr}
                                }
                            })}
                        };
                        std::string data = "data: " + chunk.dump() + "\n\n";
                        sink.write(data.data(), data.size());
                    });

                    json done_chunk = {
                        {"id", resp_id},
                        {"object", "chat.completion.chunk"},
                        {"model", model_name},
                        {"choices", json::array({
                            json{
                                {"index", 0},
                                {"delta", json::object()},
                                {"finish_reason", "stop"}
                            }
                        })}
                    };
                    std::string end_data = "data: " + done_chunk.dump() + "\n\n";
                    sink.write(end_data.data(), end_data.size());

                    const char done[] = "data: [DONE]\n\n";
                    sink.write(done, sizeof(done) - 1);
                    return false;
                },
                [](bool) {});
            return;
        }

        std::string generated;
        {
            std::lock_guard<std::mutex> lock(model_mutex);
            auto ctx = model.prefill(prompt);
            model.generate(ctx, cfg, [&](const std::string& token) {
                generated += token;
            });
        }

        json resp = {
            {"id", resp_id},
            {"object", "chat.completion"},
            {"model", model_name},
            {"choices", json::array({
                json{
                    {"index", 0},
                    {"message", {{"role", "assistant"}, {"content", generated}}},
                    {"finish_reason", "stop"}
                }
            })},
            {"usage", {{"prompt_tokens", 0}, {"completion_tokens", 0}}}
        };

        res.set_content(resp.dump(), "application/json");
    });

    const int port = 8080;
    std::cout << "Qwen3 OpenAI-style API server listening on http://0.0.0.0:" << port << std::endl;
    std::cout << "POST /v1/chat/completions with OpenAI-format payloads." << std::endl;
    server.listen("0.0.0.0", port);

    return 0;
}
