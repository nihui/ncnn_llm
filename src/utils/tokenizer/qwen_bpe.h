// Qwen2 byte-level BPE tokenizer (encode only).
// Adapted from mingshi2333/Qwen3-TTS-ncnn (Apache-2.0) for this runtime.
//
// Pipeline (mirrors HF Qwen2Tokenizer):
//   1. split out added/special tokens (<|im_start|> ...) as whole ids
//   2. GPT-2 style pre-tokenizer regex, implemented as a hand-written scanner:
//      (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N}
//      |  ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
//   3. per pre-token: bytes -> GPT-2 byte-unicode string -> BPE merges -> ids
//
// Unicode note: \p{L}/\p{N} are approximated with range tables covering Latin
// (incl. extensions), Greek, Cyrillic, CJK, Kana, Hangul, fullwidth forms and
// common number forms. Parity is VERIFIED against HF on a mixed corpus
// (examples/test_tokenizer.cpp); extend the tables if a new script fails.
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace q3tts {

class QwenBpe {
public:
    void load(const std::string& tokenizer_txt);
    std::vector<int> encode(const std::string& utf8_text) const;

private:
    std::vector<int> bpe_word(const std::string& byte_unicode) const;

    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<std::string, int> merge_rank_;  // "left right" -> rank
    std::vector<std::pair<std::string, int>> added_;   // content -> id
    std::string byte_enc_[256];                        // byte -> utf8 of mapped codepoint
};

}  // namespace q3tts
