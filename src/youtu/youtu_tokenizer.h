#pragma once

#include <memory>
#include <string>
#include <vector>

namespace youtu {

// Byte-level BPE tokenizer backed by Hugging Face tokenizer.json.
class Tokenizer {
public:
    Tokenizer();
    ~Tokenizer();
    Tokenizer(Tokenizer&&) noexcept;
    Tokenizer& operator=(Tokenizer&&) noexcept;
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;

    static Tokenizer load(const std::string& path);
    std::vector<int> encode_with_specials(const std::string& text) const;
    std::string decode(const std::vector<int>& ids, bool skip_special = true) const;
    int token_id(const std::string& token) const;
    int eos_id() const;

private:
    struct Impl;
    explicit Tokenizer(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace youtu
