#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <functional>

struct TestResult {
    std::string name;
    bool passed;
    std::string message;
};

class TestRunner {
public:
    void add_test(const std::string& name, std::function<bool()> test_func) {
        tests_.push_back({name, test_func});
    }

    int run_all() {
        int passed = 0;
        int failed = 0;

        std::cout << "=== Running Tests ===\n\n";

        for (const auto& test : tests_) {
            std::cout << "Running: " << test.name << "... ";
            try {
                bool result = test.func();
                if (result) {
                    std::cout << "PASSED\n";
                    passed++;
                } else {
                    std::cout << "FAILED\n";
                    failed++;
                }
            } catch (const std::exception& e) {
                std::cout << "FAILED (exception: " << e.what() << ")\n";
                failed++;
            }
        }

        std::cout << "\n=== Results ===\n";
        std::cout << "Passed: " << passed << "\n";
        std::cout << "Failed: " << failed << "\n";
        std::cout << "Total:  " << tests_.size() << "\n";

        return failed > 0 ? 1 : 0;
    }

private:
    struct Test {
        std::string name;
        std::function<bool()> func;
    };
    std::vector<Test> tests_;
};

#define TEST_ASSERT(cond, msg) \
    if (!(cond)) { \
        std::cerr << "Assertion failed: " << msg << "\n"; \
        return false; \
    }
