// SPDX-License-Identifier: Apache-2.0
#include "json_min.h"
#include "test_util.h"

using pvl::Json;

PVL_TEST(json_parse_scalars) {
    Json j = Json::parse(R"({"a": 1, "b": 2.5, "c": "hi", "d": true, "e": null})");
    CHECK(j.is_object());
    CHECK_EQ(j["a"].as_int(), 1);
    CHECK_NEAR(j["b"].as_double(), 2.5, 1e-9);
    CHECK(j["c"].as_string() == "hi");
    CHECK(j["d"].as_bool() == true);
    CHECK(j["e"].is_null());
    CHECK(j.contains("a"));
    CHECK(!j.contains("z"));
}

PVL_TEST(json_parse_arrays) {
    Json j = Json::parse(R"({"nums": [1, 2, 3], "strs": ["x", "y"], "f": [0.5, 1.5]})");
    auto nums = j["nums"].as_int_vector();
    CHECK_EQ((int)nums.size(), 3);
    CHECK_EQ(nums[2], 3);
    auto strs = j["strs"].as_string_vector();
    CHECK(strs[1] == "y");
    auto f = j["f"].as_float_vector();
    CHECK_NEAR(f[0], 0.5, 1e-6);
}

PVL_TEST(json_nested_and_escapes) {
    Json j = Json::parse(R"({"outer": {"inner": {"v": -7}}, "s": "a\n\t\"b\"", "u": "\u4e2d"})");
    CHECK_EQ(j["outer"]["inner"]["v"].as_int(), -7);
    CHECK(j["s"].as_string() == "a\n\t\"b\"");
    // \u4e2d is the Chinese character 涓?(3 UTF-8 bytes)
    CHECK_EQ((int)j["u"].as_string().size(), 3);
}

PVL_TEST(json_defaults) {
    Json j = Json::parse(R"({"x": 42})");
    CHECK_EQ(j.get_int("x", 0), 42);
    CHECK_EQ(j.get_int("y", 99), 99);
    CHECK(j.get_string("z", "def") == "def");
}
