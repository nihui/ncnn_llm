// Minimal, dependency-free JSON reader for Penguin-VL-ncnn model configuration.
//
// Only the subset required to read `model.json` is implemented: objects, arrays,
// strings (with \uXXXX and standard escapes), numbers, booleans and null.
// It is intentionally small so the project keeps zero third-party dependencies
// beyond ncnn itself (see the issue requirement "鐏忎粙鍣洪崙蹇撶毌缁楊兛绗侀弬閫涚贩鐠?).
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace pvl {

class Json {
public:
    enum class Type { Null, Bool, Number, String, Array, Object };

    Json() : type_(Type::Null) {}

    Type type() const { return type_; }
    bool is_null() const { return type_ == Type::Null; }
    bool is_object() const { return type_ == Type::Object; }
    bool is_array() const { return type_ == Type::Array; }
    bool is_string() const { return type_ == Type::String; }
    bool is_number() const { return type_ == Type::Number; }
    bool is_bool() const { return type_ == Type::Bool; }

    bool contains(const std::string& key) const {
        return type_ == Type::Object && object_.find(key) != object_.end();
    }

    // Object access. Throws if the key is missing.
    const Json& at(const std::string& key) const {
        auto it = object_.find(key);
        if (type_ != Type::Object || it == object_.end())
            throw std::runtime_error("json: missing key '" + key + "'");
        return it->second;
    }
    const Json& operator[](const std::string& key) const { return at(key); }

    // Array access.
    size_t size() const {
        if (type_ == Type::Array) return array_.size();
        if (type_ == Type::Object) return object_.size();
        return 0;
    }
    const Json& operator[](size_t idx) const {
        if (type_ != Type::Array || idx >= array_.size())
            throw std::runtime_error("json: array index out of range");
        return array_[idx];
    }

    // Scalar extraction with type checking.
    std::string as_string() const {
        if (type_ != Type::String) throw std::runtime_error("json: not a string");
        return string_;
    }
    double as_double() const {
        if (type_ != Type::Number) throw std::runtime_error("json: not a number");
        return number_;
    }
    int as_int() const { return static_cast<int>(as_double()); }
    float as_float() const { return static_cast<float>(as_double()); }
    bool as_bool() const {
        if (type_ != Type::Bool) throw std::runtime_error("json: not a bool");
        return bool_;
    }

    // Convenience extractors with defaults.
    std::string get_string(const std::string& key, const std::string& def = "") const {
        return contains(key) ? at(key).as_string() : def;
    }
    int get_int(const std::string& key, int def = 0) const {
        return contains(key) ? at(key).as_int() : def;
    }
    float get_float(const std::string& key, float def = 0.f) const {
        return contains(key) ? at(key).as_float() : def;
    }
    bool get_bool(const std::string& key, bool def = false) const {
        return contains(key) ? at(key).as_bool() : def;
    }

    std::vector<float> as_float_vector() const {
        if (type_ != Type::Array) throw std::runtime_error("json: not an array");
        std::vector<float> out;
        out.reserve(array_.size());
        for (const auto& e : array_) out.push_back(e.as_float());
        return out;
    }
    std::vector<int> as_int_vector() const {
        if (type_ != Type::Array) throw std::runtime_error("json: not an array");
        std::vector<int> out;
        out.reserve(array_.size());
        for (const auto& e : array_) out.push_back(e.as_int());
        return out;
    }
    std::vector<std::string> as_string_vector() const {
        if (type_ != Type::Array) throw std::runtime_error("json: not an array");
        std::vector<std::string> out;
        out.reserve(array_.size());
        for (const auto& e : array_) out.push_back(e.as_string());
        return out;
    }

    static Json parse(const std::string& text) {
        Parser p(text);
        p.skip_ws();
        Json v = p.parse_value();
        p.skip_ws();
        if (!p.eof()) throw std::runtime_error("json: trailing characters after root value");
        return v;
    }

    static Json parse_file(const std::string& path);

private:
    Type type_;
    bool bool_ = false;
    double number_ = 0.0;
    std::string string_;
    std::vector<Json> array_;
    std::map<std::string, Json> object_;

    struct Parser {
        const std::string& s;
        size_t i = 0;
        explicit Parser(const std::string& str) : s(str) {}

        bool eof() const { return i >= s.size(); }
        char peek() const { return i < s.size() ? s[i] : '\0'; }
        char get() { return i < s.size() ? s[i++] : '\0'; }

        void skip_ws() {
            while (i < s.size()) {
                char c = s[i];
                if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ++i;
                else break;
            }
        }

        Json parse_value() {
            skip_ws();
            if (eof()) throw std::runtime_error("json: unexpected end of input");
            char c = peek();
            switch (c) {
                case '{': return parse_object();
                case '[': return parse_array();
                case '"': { Json v; v.type_ = Type::String; v.string_ = parse_string(); return v; }
                case 't': case 'f': return parse_bool();
                case 'n': return parse_null();
                default: return parse_number();
            }
        }

        Json parse_object() {
            Json v; v.type_ = Type::Object;
            expect('{');
            skip_ws();
            if (peek() == '}') { get(); return v; }
            while (true) {
                skip_ws();
                if (peek() != '"') throw std::runtime_error("json: expected string key");
                std::string key = parse_string();
                skip_ws();
                expect(':');
                v.object_[key] = parse_value();
                skip_ws();
                char c = get();
                if (c == ',') continue;
                if (c == '}') break;
                throw std::runtime_error("json: expected ',' or '}' in object");
            }
            return v;
        }

        Json parse_array() {
            Json v; v.type_ = Type::Array;
            expect('[');
            skip_ws();
            if (peek() == ']') { get(); return v; }
            while (true) {
                v.array_.push_back(parse_value());
                skip_ws();
                char c = get();
                if (c == ',') continue;
                if (c == ']') break;
                throw std::runtime_error("json: expected ',' or ']' in array");
            }
            return v;
        }

        std::string parse_string() {
            expect('"');
            std::string out;
            while (true) {
                if (eof()) throw std::runtime_error("json: unterminated string");
                char c = get();
                if (c == '"') break;
                if (c == '\\') {
                    char e = get();
                    switch (e) {
                        case '"': out.push_back('"'); break;
                        case '\\': out.push_back('\\'); break;
                        case '/': out.push_back('/'); break;
                        case 'b': out.push_back('\b'); break;
                        case 'f': out.push_back('\f'); break;
                        case 'n': out.push_back('\n'); break;
                        case 'r': out.push_back('\r'); break;
                        case 't': out.push_back('\t'); break;
                        case 'u': {
                            uint32_t cp = parse_hex4();
                            if (cp >= 0xD800 && cp <= 0xDBFF) {
                                // surrogate pair
                                if (get() != '\\' || get() != 'u')
                                    throw std::runtime_error("json: invalid surrogate pair");
                                uint32_t lo = parse_hex4();
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            }
                            append_utf8(out, cp);
                            break;
                        }
                        default: throw std::runtime_error("json: invalid escape");
                    }
                } else {
                    out.push_back(c);
                }
            }
            return out;
        }

        uint32_t parse_hex4() {
            uint32_t v = 0;
            for (int k = 0; k < 4; ++k) {
                char c = get();
                v <<= 4;
                if (c >= '0' && c <= '9') v |= (c - '0');
                else if (c >= 'a' && c <= 'f') v |= (c - 'a' + 10);
                else if (c >= 'A' && c <= 'F') v |= (c - 'A' + 10);
                else throw std::runtime_error("json: invalid \\u escape");
            }
            return v;
        }

        static void append_utf8(std::string& out, uint32_t cp) {
            if (cp <= 0x7F) {
                out.push_back(static_cast<char>(cp));
            } else if (cp <= 0x7FF) {
                out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
                out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            } else if (cp <= 0xFFFF) {
                out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
                out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            } else {
                out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
                out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
                out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            }
        }

        Json parse_bool() {
            Json v; v.type_ = Type::Bool;
            if (s.compare(i, 4, "true") == 0) { v.bool_ = true; i += 4; }
            else if (s.compare(i, 5, "false") == 0) { v.bool_ = false; i += 5; }
            else throw std::runtime_error("json: invalid literal");
            return v;
        }

        Json parse_null() {
            Json v; v.type_ = Type::Null;
            if (s.compare(i, 4, "null") == 0) i += 4;
            else throw std::runtime_error("json: invalid literal");
            return v;
        }

        Json parse_number() {
            size_t start = i;
            if (peek() == '-' || peek() == '+') ++i;
            while (i < s.size()) {
                char c = s[i];
                if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') ++i;
                else break;
            }
            if (i == start) throw std::runtime_error("json: invalid number");
            Json v; v.type_ = Type::Number;
            v.number_ = std::stod(s.substr(start, i - start));
            return v;
        }

        void expect(char c) {
            if (get() != c) throw std::runtime_error(std::string("json: expected '") + c + "'");
        }
    };
};

}  // namespace pvl
