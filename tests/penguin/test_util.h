// SPDX-License-Identifier: Apache-2.0
// Minimal self-contained test harness (no third-party test framework).
#pragma once

#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace pvltest {

struct TestCase {
    std::string name;
    std::function<void()> fn;
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> r;
    return r;
}

inline int& failures() {
    static int f = 0;
    return f;
}

struct Registrar {
    Registrar(const std::string& name, std::function<void()> fn) {
        registry().push_back({name, std::move(fn)});
    }
};

inline void fail(const char* file, int line, const std::string& msg) {
    std::printf("    FAIL %s:%d  %s\n", file, line, msg.c_str());
    ++failures();
}

}  // namespace pvltest

#define PVL_TEST(name)                                                      \
    static void name();                                                     \
    static ::pvltest::Registrar registrar_##name(#name, name);             \
    static void name()

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) ::pvltest::fail(__FILE__, __LINE__, "CHECK(" #cond ")"); \
    } while (0)

#define CHECK_EQ(a, b)                                                      \
    do {                                                                    \
        auto _va = (a);                                                     \
        auto _vb = (b);                                                     \
        if (!(_va == _vb))                                                  \
            ::pvltest::fail(__FILE__, __LINE__,                            \
                            std::string(#a " == " #b " (") +               \
                                std::to_string(_va) + " vs " +            \
                                std::to_string(_vb) + ")");                \
    } while (0)

#define CHECK_NEAR(a, b, eps)                                               \
    do {                                                                    \
        double _da = (a);                                                   \
        double _db = (b);                                                   \
        if (std::fabs(_da - _db) > (eps))                                   \
            ::pvltest::fail(__FILE__, __LINE__,                            \
                            std::string(#a " ~= " #b " (") +               \
                                std::to_string(_da) + " vs " +            \
                                std::to_string(_db) + ")");                \
    } while (0)
