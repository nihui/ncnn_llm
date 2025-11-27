#include <iostream>
#include "utils/tokenizer/bpe_tokenizer.h"

int main()
{
    BpeTokenizer tokenizer = BpeTokenizer::LoadFromFiles(
        "assets/qwen3_0.6b/vocab.txt",
        "assets/qwen3_0.6b/merges.txt",
        SpecialTokensConfig{
            .bos_token = "<bos>"
        },
        true,
        true,
        true
    );

    auto encoded = tokenizer.encode("你好，世界");
    for (const auto& token : encoded) {
        std::cout << token << " ";
    }
    auto decoded = tokenizer.decode(encoded);
    std::cout << decoded << std::endl;
}