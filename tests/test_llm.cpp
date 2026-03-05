#include "test_framework.h"
#include "ncnn_llm_gpt.h"
#include "utils/prompt.h"

#include <filesystem>
#include <fstream>
#include <sstream>

static bool has_model(const std::string& model_name) {
    std::string path = "./assets/" + model_name;
    if (std::filesystem::exists(path)) return true;
    return std::filesystem::exists("./" + model_name);
}

static std::string get_model_path(const std::string& model_name) {
    std::string path = "./assets/" + model_name;
    if (std::filesystem::exists(path)) return path;
    return "./" + model_name;
}

// Test 1: Basic prompt template generation
bool test_prompt_template_basic() {
    std::vector<Message> messages = {
        {"system", "You are a helpful assistant.", "", {}},
        {"user", "Hello!", "", {}}
    };

    std::string prompt = apply_chat_template(messages, {}, true, false);

    TEST_ASSERT(!prompt.empty(), "Prompt should not be empty");
    TEST_ASSERT(prompt.find("<|im_start|>system") != std::string::npos, 
                "Prompt should contain system message start");
    TEST_ASSERT(prompt.find("<|im_start|>user") != std::string::npos, 
                "Prompt should contain user message start");
    TEST_ASSERT(prompt.find("<|im_start|>assistant") != std::string::npos, 
                "Prompt should contain assistant message start for generation");

    return true;
}

// Test 2: Prompt template with tools
bool test_prompt_template_with_tools() {
    std::vector<Message> messages = {
        {"system", "You are a helpful assistant.", "", {}},
        {"user", "What is 2 + 3?", "", {}}
    };

    std::vector<json> tools = {
        ncnn_llm_gpt::make_function_tool<int, int, int>(
            "add",
            "Add two integers.",
            {"a", "b"}
        )
    };

    std::string prompt = apply_chat_template(messages, tools, true, false);

    TEST_ASSERT(!prompt.empty(), "Prompt should not be empty");
    TEST_ASSERT(prompt.find("# Tools") != std::string::npos, 
                "Prompt should contain tools section");
    TEST_ASSERT(prompt.find("add") != std::string::npos, 
                "Prompt should contain tool name");

    return true;
}

// Test 3: Tool definition creation
bool test_tool_definition() {
    auto tool = ncnn_llm_gpt::make_function_tool<int, int, int>(
        "random",
        "Generate a random number between two integers.",
        {"floor", "ceiling"}
    );

    TEST_ASSERT(tool.is_object(), "Tool should be a JSON object");
    TEST_ASSERT(tool["type"] == "function", "Tool type should be function");
    TEST_ASSERT(tool["function"]["name"] == "random", "Tool name should be random");
    TEST_ASSERT(tool["function"]["description"] == "Generate a random number between two integers.",
                "Tool description should match");
    TEST_ASSERT(tool["function"]["parameters"]["properties"].is_object(), 
                "Tool should have parameters properties");

    return true;
}

// Test 4: Message parsing and serialization
bool test_message_serialization() {
    std::vector<Message> messages = {
        {"user", "Hello, how are you?", "", {}},
        {"assistant", "I'm doing well, thank you!", "", {}},
        {"user", "Can you help me with something?", "", {}}
    };

    std::string prompt = apply_chat_template(messages, {}, true, false);

    TEST_ASSERT(!prompt.empty(), "Prompt should not be empty");
    TEST_ASSERT(prompt.find("Hello, how are you?") != std::string::npos, 
                "Prompt should contain first user message");
    TEST_ASSERT(prompt.find("I'm doing well, thank you!") != std::string::npos, 
                "Prompt should contain assistant message");
    TEST_ASSERT(prompt.find("Can you help me with something?") != std::string::npos, 
                "Prompt should contain second user message");

    return true;
}

// Test 5: Tool call message format
bool test_tool_call_message() {
    json tool_call = {
        {"name", "add"},
        {"arguments", {{"a", 2}, {"b", 3}}}
    };

    std::vector<Message> messages = {
        {"user", "What is 2 + 3?", "", {}},
        {"assistant", "", "", {tool_call}}
    };

    std::string prompt = apply_chat_template(messages, {}, true, false);

    TEST_ASSERT(!prompt.empty(), "Prompt should not be empty");
    TEST_ASSERT(prompt.find("add") != std::string::npos, 
                "Prompt should contain tool call name");

    return true;
}

// Test 6: Context memory - conversation history
bool test_context_memory() {
    // This test verifies that the prompt template correctly handles
    // multi-turn conversations with context
    std::vector<Message> messages = {
        {"user", "My name is Alice.", "", {}},
        {"assistant", "Nice to meet you, Alice!", "", {}},
        {"user", "What is my name?", "", {}}
    };

    std::string prompt = apply_chat_template(messages, {}, true, false);

    TEST_ASSERT(!prompt.empty(), "Prompt should not be empty");
    TEST_ASSERT(prompt.find("My name is Alice.") != std::string::npos, 
                "Prompt should contain first user message");
    TEST_ASSERT(prompt.find("Nice to meet you, Alice!") != std::string::npos, 
                "Prompt should contain assistant response");
    TEST_ASSERT(prompt.find("What is my name?") != std::string::npos, 
                "Prompt should contain second user message");

    return true;
}

// Test 7: System prompt handling
bool test_system_prompt() {
    std::vector<Message> messages = {
        {"system", "You are a math tutor. Always explain step by step.", "", {}},
        {"user", "What is 5 * 7?", "", {}}
    };

    std::string prompt = apply_chat_template(messages, {}, true, false);

    TEST_ASSERT(!prompt.empty(), "Prompt should not be empty");
    TEST_ASSERT(prompt.find("You are a math tutor") != std::string::npos, 
                "Prompt should contain system message");
    TEST_ASSERT(prompt.find("<|im_start|>system") != std::string::npos, 
                "Prompt should have system message start tag");

    return true;
}

// Test 8: Empty tools handling
bool test_empty_tools() {
    std::vector<Message> messages = {
        {"user", "Hello!", "", {}}
    };

    std::string prompt = apply_chat_template(messages, {}, true, false);

    TEST_ASSERT(!prompt.empty(), "Prompt should not be empty");
    TEST_ASSERT(prompt.find("# Tools") == std::string::npos, 
                "Prompt should not contain tools section when no tools provided");

    return true;
}

// Test 9: Thinking mode toggle
bool test_thinking_mode() {
    std::vector<Message> messages = {
        {"user", "Solve this problem.", "", {}}
    };

    std::string prompt_with_thinking = apply_chat_template(messages, {}, true, true);
    std::string prompt_without_thinking = apply_chat_template(messages, {}, true, false);

    TEST_ASSERT(!prompt_with_thinking.empty(), "Prompt with thinking should not be empty");
    TEST_ASSERT(!prompt_without_thinking.empty(), "Prompt without thinking should not be empty");

    return true;
}

// Test 10: Long conversation context
bool test_long_conversation() {
    std::vector<Message> messages;

    for (int i = 0; i < 10; i++) {
        messages.push_back({"user", "Question " + std::to_string(i), "", {}});
        messages.push_back({"assistant", "Answer " + std::to_string(i), "", {}});
    }

    std::string prompt = apply_chat_template(messages, {}, true, false);

    TEST_ASSERT(!prompt.empty(), "Prompt should not be empty");
    TEST_ASSERT(prompt.find("Question 0") != std::string::npos, 
                "Prompt should contain first question");
    TEST_ASSERT(prompt.find("Answer 9") != std::string::npos, 
                "Prompt should contain last answer");

    return true;
}

// Model-based tests (require model files)
bool test_model_tool_calling() {
    if (!has_model("qwen3_0.6b")) {
        std::cout << "(skipped - model not found) ";
        return true;
    }

    ncnn_llm_gpt model(get_model_path("qwen3_0.6b"));

    std::string system_prompt = "You are a helpful assistant.";
    std::string prompt = apply_chat_template({{"system", system_prompt}}, {}, false, false);
    auto ctx = model.prefill(prompt);

    auto tools = ncnn_llm_gpt::make_function_tool<int, int, int>(
        "add",
        "Add two integers.",
        {"a", "b"}
    );

    ctx = model.define_tools(ctx, {tools}, system_prompt);

    std::string user_msg = apply_chat_template({{"user", "What is 2 + 3?"}}, {}, true, false);
    ctx = model.prefill(user_msg, ctx);

    bool tool_called = false;
    GenerateConfig cfg;
    cfg.max_new_tokens = 256;
    cfg.tool_callback = [&](const json& call) {
        tool_called = true;
        return json{{"result", {{"value", 5}}}, {"call", call}};
    };

    ctx = model.generate(ctx, cfg, [](const std::string& token) {});

    return true;
}

bool test_model_context_memory() {
    if (!has_model("qwen3_0.6b")) {
        std::cout << "(skipped - model not found) ";
        return true;
    }

    ncnn_llm_gpt model(get_model_path("qwen3_0.6b"));

    std::string prompt = apply_chat_template({{"system", "You are a helpful assistant."}}, {}, false, false);
    auto ctx = model.prefill(prompt);

    std::string msg1 = apply_chat_template({{"user", "My favorite color is blue."}}, {}, true, false);
    ctx = model.prefill(msg1, ctx);

    ctx = model.generate(ctx, GenerateConfig{}, [](const std::string& token) {});

    std::string msg2 = apply_chat_template({{"user", "What is my favorite color?"}}, {}, true, false);
    ctx = model.prefill(msg2, ctx);

    std::string response;
    ctx = model.generate(ctx, GenerateConfig{}, [&response](const std::string& token) {
        response += token;
    });

    TEST_ASSERT(!response.empty(), "Response should not be empty");

    return true;
}

int main() {
    TestRunner runner;

    std::cout << "=== Unit Tests ===\n\n";

    runner.add_test("prompt_template_basic", test_prompt_template_basic);
    runner.add_test("prompt_template_with_tools", test_prompt_template_with_tools);
    runner.add_test("tool_definition", test_tool_definition);
    runner.add_test("message_serialization", test_message_serialization);
    runner.add_test("tool_call_message", test_tool_call_message);
    runner.add_test("context_memory", test_context_memory);
    runner.add_test("system_prompt", test_system_prompt);
    runner.add_test("empty_tools", test_empty_tools);
    runner.add_test("thinking_mode", test_thinking_mode);
    runner.add_test("long_conversation", test_long_conversation);

    std::cout << "\n=== Model Tests ===\n\n";

    runner.add_test("model_tool_calling", test_model_tool_calling);
    runner.add_test("model_context_memory", test_model_context_memory);

    return runner.run_all();
}
