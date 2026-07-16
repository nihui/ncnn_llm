// SPDX-License-Identifier: Apache-2.0
#include <cstdio>
#include <fstream>
#include <string>

#include "bpe_tokenizer.h"
#include "test_util.h"

static void write_file(const std::string& path, const std::string& content) {
    std::ofstream ofs(path, std::ios::binary);
    ofs << content;
}

PVL_TEST(tokenizer_load_and_decode) {
    const std::string vocab_path = "pvl_test_vocab.txt";
    const std::string merges_path = "pvl_test_merges.txt";

    // 4 tokens: 'a','b','c' and the sentencepiece piece "\u2581hello".
    write_file(vocab_path, "a\nb\nc\n\xE2\x96\x81hello\n");
    write_file(merges_path, "#version: 0.2\n");

    BpeTokenizer tok = BpeTokenizer::LoadFromFiles(
        vocab_path, merges_path, SpecialTokensConfig{},
        /*add_special_if_missing=*/false, /*fallback_to_chars=*/true,
        /*use_byte_encoder=*/false);

    CHECK_EQ((int)tok.vocab_size(), 4);
    const auto& t2i = tok.token_to_id();
    CHECK(t2i.count("a") == 1);
    CHECK_EQ(t2i.at("a"), 0);
    CHECK_EQ(t2i.at("c"), 2);

    // Decoding the sentencepiece piece converts U+2581 to a space and trims it.
    CHECK(tok.decode({3}, false) == "hello");
    // Plain tokens concatenate.
    CHECK(tok.decode({0, 1, 2}, false) == "abc");

    // Round-trip invariant: decode(encode(x)) reconstructs the visible text.
    CHECK(tok.decode(tok.encode("abc", false, false), false) == "abc");

    std::remove(vocab_path.c_str());
    std::remove(merges_path.c_str());
}
