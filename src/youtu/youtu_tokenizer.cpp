#include "youtu_tokenizer.h"

#include "youtu_runner_common.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <unordered_map>

namespace youtu {

// Hugging Face tokenizer.json reader and byte-level BPE implementation.
//
// This module is deliberately independent of ncnn. It converts prompts to
// token IDs, preserves Youtu-VL special tokens, and decodes generated IDs back
// to UTF-8 text.

struct Tokenizer::Impl {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int> token_to_id;
    std::unordered_map<std::string, int> merge_rank;
    std::unordered_map<std::string, std::string> byte_decoder;
    int bos_id = 128000;
    int eos_id = 128001;

    static std::unique_ptr<Impl> load(const std::string& path) {
        Impl tok;
        tok.byte_decoder = build_byte_decoder();
        const std::string json = read_text_file(path);

        const size_t model_pos = json.find("\"model\"");
        if (model_pos == std::string::npos) die("tokenizer.json missing model");
        const size_t vocab_key = json.find("\"vocab\"", model_pos);
        if (vocab_key == std::string::npos) die("tokenizer.json missing model.vocab");
        const size_t vocab_open = json.find('{', vocab_key);
        const size_t vocab_close = find_matching(json, vocab_open, '{', '}');
        size_t p = vocab_open + 1;
        while (p < vocab_close) {
            while (p < vocab_close && (std::isspace(static_cast<unsigned char>(json[p])) || json[p] == ',')) ++p;
            if (p >= vocab_close) break;
            std::string token = json_string_at(json, p);
            while (p < vocab_close && (std::isspace(static_cast<unsigned char>(json[p])) || json[p] == ':')) ++p;
            size_t n0 = p;
            while (p < vocab_close && std::isdigit(static_cast<unsigned char>(json[p]))) ++p;
            int id = std::stoi(json.substr(n0, p - n0));
            if (id >= static_cast<int>(tok.id_to_token.size())) tok.id_to_token.resize(id + 1);
            tok.id_to_token[id] = token;
            tok.token_to_id[token] = id;
        }

        const size_t added_key = json.find("\"added_tokens\"");
        if (added_key != std::string::npos) {
            const size_t arr_open = json.find('[', added_key);
            const size_t arr_close = find_matching(json, arr_open, '[', ']');
            p = arr_open + 1;
            while (p < arr_close) {
                const size_t obj_open = json.find('{', p);
                if (obj_open == std::string::npos || obj_open >= arr_close) break;
                const size_t obj_close = find_matching(json, obj_open, '{', '}');
                const std::string obj = json.substr(obj_open, obj_close - obj_open + 1);
                size_t cp = obj.find("\"content\"");
                size_t ip = obj.find("\"id\"");
                if (cp != std::string::npos && ip != std::string::npos) {
                    cp = obj.find(':', cp) + 1;
                    std::string token = json_string_at(obj, cp);
                    ip = obj.find(':', ip) + 1;
                    while (ip < obj.size() && std::isspace(static_cast<unsigned char>(obj[ip]))) ++ip;
                    size_t ie = ip;
                    while (ie < obj.size() && std::isdigit(static_cast<unsigned char>(obj[ie]))) ++ie;
                    int id = std::stoi(obj.substr(ip, ie - ip));
                    if (id >= static_cast<int>(tok.id_to_token.size())) tok.id_to_token.resize(id + 1);
                    tok.id_to_token[id] = token;
                    tok.token_to_id[token] = id;
                }
                p = obj_close + 1;
            }
        }

        const size_t merges_key = json.find("\"merges\"", model_pos);
        if (merges_key != std::string::npos) {
            const size_t arr_open = json.find('[', merges_key);
            const size_t arr_close = find_matching(json, arr_open, '[', ']');
            p = arr_open + 1;
            int rank = 0;
            while (p < arr_close) {
                while (p < arr_close && (std::isspace(static_cast<unsigned char>(json[p])) || json[p] == ',')) ++p;
                if (p >= arr_close) break;
                std::string merge;
                if (json[p] == '"') {
                    merge = json_string_at(json, p);
                } else if (json[p] == '[') {
                    const size_t pair_close = find_matching(json, p, '[', ']');
                    size_t q = p + 1;
                    std::string a = json_string_at(json, q);
                    q = json.find(',', q) + 1;
                    std::string b = json_string_at(json, q);
                    merge = a + " " + b;
                    p = pair_close + 1;
                } else {
                    die("unsupported merge entry");
                }
                tok.merge_rank[merge] = rank++;
            }
        }

        auto bos = tok.token_to_id.find("<|begin_of_text|>");
        auto eos = tok.token_to_id.find("<|end_of_text|>");
        if (bos != tok.token_to_id.end()) tok.bos_id = bos->second;
        if (eos != tok.token_to_id.end()) tok.eos_id = eos->second;
        return std::make_unique<Impl>(std::move(tok));
    }

    std::vector<std::string> split_basic(const std::string& text) const {
        std::vector<std::string> pieces;
        std::string cur;
        std::string pending_space;
        auto flush = [&]() {
            if (!cur.empty()) {
                pieces.push_back(cur);
                cur.clear();
            }
        };
        for (size_t i = 0; i < text.size();) {
            unsigned char c = static_cast<unsigned char>(text[i]);
            if (c == '\n' || c == '\r') {
                flush();
                if (!pending_space.empty()) {
                    pieces.push_back(pending_space);
                    pending_space.clear();
                }
                pieces.push_back(std::string(1, text[i++]));
            } else if (c < 128 && (c == ' ' || c == '\t')) {
                flush();
                pending_space.push_back(text[i++]);
            } else if (c < 128 && std::isalnum(c)) {
                if (!pending_space.empty()) {
                    cur += pending_space;
                    pending_space.clear();
                }
                cur.push_back(text[i++]);
            } else if (c == '-' && i + 1 < text.size() && static_cast<unsigned char>(text[i + 1]) < 128 && std::isalnum(static_cast<unsigned char>(text[i + 1]))) {
                flush();
                cur += pending_space;
                pending_space.clear();
                cur.push_back(text[i++]);
            } else if (c < 128) {
                flush();
                std::string piece = pending_space + std::string(1, text[i++]);
                pending_space.clear();
                pieces.push_back(piece);
            } else {
                flush();
                size_t start = i++;
                while (i < text.size() && (static_cast<unsigned char>(text[i]) & 0xc0) == 0x80) ++i;
                std::string piece = pending_space + text.substr(start, i - start);
                pending_space.clear();
                pieces.push_back(piece);
            }
        }
        flush();
        if (!pending_space.empty()) pieces.push_back(pending_space);
        return pieces;
    }

    std::vector<int> bpe_piece(const std::string& text) const {
        std::vector<std::string> parts;
        for (unsigned char b : text) parts.push_back(byte_encoder_piece(b));
        if (parts.empty()) return {};

        while (parts.size() > 1) {
            int best_rank = std::numeric_limits<int>::max();
            size_t best = static_cast<size_t>(-1);
            for (size_t i = 0; i + 1 < parts.size(); ++i) {
                auto it = merge_rank.find(parts[i] + " " + parts[i + 1]);
                if (it != merge_rank.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best = i;
                }
            }
            if (best == static_cast<size_t>(-1)) break;
            parts[best] += parts[best + 1];
            parts.erase(parts.begin() + static_cast<long>(best + 1));
        }

        std::vector<int> ids;
        for (const std::string& part : parts) {
            auto it = token_to_id.find(part);
            if (it == token_to_id.end()) {
                for (unsigned char b : text) {
                    auto bit = token_to_id.find(byte_encoder_piece(b));
                    if (bit == token_to_id.end()) die("byte token missing from vocab");
                    ids.push_back(bit->second);
                }
                return ids;
            }
            ids.push_back(it->second);
        }
        return ids;
    }

    std::vector<int> encode_plain(const std::string& text) const {
        std::vector<int> ids;
        for (const std::string& piece : split_basic(text)) {
            std::vector<int> part = bpe_piece(piece);
            ids.insert(ids.end(), part.begin(), part.end());
        }
        return ids;
    }

    std::vector<int> encode_with_specials(const std::string& text) const {
        std::vector<int> ids;
        size_t pos = 0;
        while (pos < text.size()) {
            int special_id = -1;
            size_t special_len = 0;
            if (text.compare(pos, 2, "<|") == 0) {
                size_t end = text.find("|>", pos);
                if (end != std::string::npos) {
                    std::string tok = text.substr(pos, end + 2 - pos);
                    auto it = token_to_id.find(tok);
                    if (it != token_to_id.end()) {
                        special_id = it->second;
                        special_len = tok.size();
                    }
                }
            }
            if (special_id >= 0) {
                ids.push_back(special_id);
                pos += special_len;
            } else {
                size_t next = text.find("<|", pos + 1);
                std::string chunk = text.substr(pos, next == std::string::npos ? std::string::npos : next - pos);
                std::vector<int> part = encode_plain(chunk);
                ids.insert(ids.end(), part.begin(), part.end());
                if (next == std::string::npos) break;
                pos = next;
            }
        }
        return ids;
    }

    std::string decode(const std::vector<int>& ids, bool skip_special = true) const {
        std::string bytes;
        for (int id : ids) {
            if (id < 0 || id >= static_cast<int>(id_to_token.size())) continue;
            const std::string& token = id_to_token[id];
            if (skip_special && token.size() >= 4 && token.substr(0, 2) == "<|" && token.substr(token.size() - 2) == "|>") continue;
            size_t i = 0;
            while (i < token.size()) {
                std::string match;
                size_t match_len = 0;
                for (size_t len = 1; len <= 4 && i + len <= token.size(); ++len) {
                    std::string sub = token.substr(i, len);
                    if (byte_decoder.find(sub) != byte_decoder.end()) {
                        match = sub;
                        match_len = len;
                    }
                }
                if (match_len) {
                    bytes += byte_decoder.at(match);
                    i += match_len;
                } else {
                    bytes.push_back(token[i++]);
                }
            }
        }
        return bytes;
    }
};

Tokenizer::Tokenizer() = default;
Tokenizer::~Tokenizer() = default;
Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;

Tokenizer::Tokenizer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

Tokenizer Tokenizer::load(const std::string& path) {
    return Tokenizer(Impl::load(path));
}

std::vector<int> Tokenizer::encode_with_specials(const std::string& text) const {
    if (!impl_) die("tokenizer is not initialized");
    return impl_->encode_with_specials(text);
}

std::string Tokenizer::decode(const std::vector<int>& ids, bool skip_special) const {
    if (!impl_) die("tokenizer is not initialized");
    return impl_->decode(ids, skip_special);
}

int Tokenizer::token_id(const std::string& token) const {
    if (!impl_) die("tokenizer is not initialized");
    const auto it = impl_->token_to_id.find(token);
    if (it == impl_->token_to_id.end()) die("tokenizer missing special token " + token);
    return it->second;
}

int Tokenizer::eos_id() const {
    if (!impl_) die("tokenizer is not initialized");
    return impl_->eos_id;
}

} // namespace youtu
