#include "bpe_tokenizer.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <map>

// ---------------- I/O helpers ----------------
static std::string Trim(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e-1]))) --e;
    return s.substr(b, e - b);
}

std::vector<std::string> BpeTokenizer::LoadVocab(const std::string& vocab_path) {
    std::ifstream fin(vocab_path);
    if (!fin) throw std::runtime_error("Failed to open vocab file: " + vocab_path);
    std::vector<std::string> v;
    v.reserve(50000);
    std::string line;
    while (std::getline(fin, line)) {
        std::string tok = Trim(line);
        if (tok.empty()) continue;
        v.push_back(tok);
    }
    return v;
}

std::unordered_map<std::string, int> BpeTokenizer::BuildTokenToId(const std::vector<std::string>& id_to_token) {
    std::unordered_map<std::string, int> m;
    m.reserve(id_to_token.size() * 2);
    for (int i = 0; i < static_cast<int>(id_to_token.size()); ++i) {
        m.emplace(id_to_token[i], i);
    }
    return m;
}

std::unordered_map<std::string, int> BpeTokenizer::LoadMergesRank(const std::string& merges_path) {
    std::ifstream fin(merges_path);
    if (!fin) throw std::runtime_error("Failed to open merges file: " + merges_path);
    std::unordered_map<std::string, int> rank;
    rank.reserve(100000);
    std::string line;
    while (std::getline(fin, line)) {
        std::string s = Trim(line);
        if (s.empty()) continue;
        std::string a, b;
        {
            std::istringstream iss(s);
            if (!(iss >> a >> b)) continue;
        }
        std::string key = PairKey(a, b);
        if (rank.find(key) == rank.end()) {
            rank.emplace(std::move(key), static_cast<int>(rank.size()));
        }
    }
    return rank;
}

// ---------------- UTF-8 helpers ----------------
bool BpeTokenizer::IsAsciiSpace(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

bool BpeTokenizer::IsUnicodeSpace(uint32_t cp) {
    if (cp <= 0x7F) return IsAsciiSpace(static_cast<unsigned char>(cp));
    switch (cp) {
        case 0x00A0: case 0x1680:
        case 0x2000: case 0x2001: case 0x2002: case 0x2003:
        case 0x2004: case 0x2005: case 0x2006: case 0x2007:
        case 0x2008: case 0x2009: case 0x200A:
        case 0x2028: case 0x2029:
        case 0x202F: case 0x205F: case 0x3000:
            return true;
        default:
            return false;
    }
}

bool BpeTokenizer::NextUtf8(const std::string& s, size_t& i, uint32_t& cp, size_t& cp_len) {
    if (i >= s.size()) return false;
    unsigned char c0 = static_cast<unsigned char>(s[i]);
    if (c0 < 0x80) {
        cp = c0; cp_len = 1; ++i; return true;
    } else if ((c0 >> 5) == 0x6) {
        if (i + 1 >= s.size()) return false;
        unsigned char c1 = static_cast<unsigned char>(s[i+1]);
        if ((c1 & 0xC0) != 0x80) return false;
        cp = ((c0 & 0x1F) << 6) | (c1 & 0x3F);
        cp_len = 2; i += 2; return true;
    } else if ((c0 >> 4) == 0xE) {
        if (i + 2 >= s.size()) return false;
        unsigned char c1 = static_cast<unsigned char>(s[i+1]);
        unsigned char c2 = static_cast<unsigned char>(s[i+2]);
        if (((c1 & 0xC0) != 0x80) || ((c2 & 0xC0) != 0x80)) return false;
        cp = ((c0 & 0x0F) << 12) | ((c1 & 0x3F) << 6) | (c2 & 0x3F);
        cp_len = 3; i += 3; return true;
    } else if ((c0 >> 3) == 0x1E) {
        if (i + 3 >= s.size()) return false;
        unsigned char c1 = static_cast<unsigned char>(s[i+1]);
        unsigned char c2 = static_cast<unsigned char>(s[i+2]);
        unsigned char c3 = static_cast<unsigned char>(s[i+3]);
        if (((c1 & 0xC0) != 0x80) || ((c2 & 0xC0) != 0x80) || ((c3 & 0xC0) != 0x80)) return false;
        cp = ((c0 & 0x07) << 18) | ((c1 & 0x3F) << 12) | ((c2 & 0x3F) << 6) | (c3 & 0x3F);
        cp_len = 4; i += 4; return true;
    } else {
        return false;
    }
}

std::vector<std::string> BpeTokenizer::Utf8Chars(const std::string& s) {
    std::vector<std::string> chars;
    chars.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        uint32_t cp; size_t len;
        if (!NextUtf8(s, i, cp, len)) break;
        chars.emplace_back(s.substr(i - len, len));
    }
    return chars;
}

// ---------------- Byte Encoder/Decoder Logic ----------------

void BpeTokenizer::InitByteMaps() {
    byte_encoder_.resize(256, 0);
    byte_decoder_.clear();

    // Logic from byte_converter.cpp: generate_byte_encoder
    // "printable" chars map to themselves, others map to 256+
    auto is_printable = [](int b) {
        return (b >= '!' && b <= '~')
            || (b >= 161 && b <= 172)
            || (b >= 174 && b <= 255);
    };

    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (is_printable(b)) {
            byte_encoder_[b] = static_cast<uint32_t>(b);
        } else {
            byte_encoder_[b] = static_cast<uint32_t>(256 + n);
            n++;
        }
        byte_decoder_[byte_encoder_[b]] = static_cast<uint8_t>(b);
    }
}

std::string BpeTokenizer::ByteEncode(const std::string& text) const {
    // Convert raw byte stream -> sequence of Mapped Codepoints -> UTF-8 string
    std::string out;
    out.reserve(text.size() * 2); // conservative reserve

    for (unsigned char b : text) {
        uint32_t cp = byte_encoder_[b];

        // Append codepoint as UTF-8
        if (cp < 0x80) {
            out += static_cast<char>(cp);
        } else if (cp < 0x800) {
            out += static_cast<char>((cp >> 6) | 0xC0);
            out += static_cast<char>((cp & 0x3F) | 0x80);
        } else if (cp < 0x10000) {
            out += static_cast<char>((cp >> 12) | 0xE0);
            out += static_cast<char>(((cp >> 6) & 0x3F) | 0x80);
            out += static_cast<char>((cp & 0x3F) | 0x80);
        } else {
            out += static_cast<char>((cp >> 18) | 0xF0);
            out += static_cast<char>(((cp >> 12) & 0x3F) | 0x80);
            out += static_cast<char>(((cp >> 6) & 0x3F) | 0x80);
            out += static_cast<char>((cp & 0x3F) | 0x80);
        }
    }
    return out;
}

std::string BpeTokenizer::ByteDecode(const std::string& text) const {
    // Convert UTF-8 string -> sequence of Mapped Codepoints -> raw byte stream
    std::string out;
    out.reserve(text.size());

    size_t i = 0;
    uint32_t cp; size_t len;
    while (i < text.size()) {
        if (!NextUtf8(text, i, cp, len)) break; // Should check return logic

        auto it = byte_decoder_.find(cp);
        if (it != byte_decoder_.end()) {
            out += static_cast<char>(it->second);
        } else {
            // If a codepoint is not in our map (e.g. some special token artifacts),
            // we might decide to keep it or ignore it.
            // Usually, strict BPE output should only contain mapped chars + special tokens.
            // We can try to output as is if it's ascii, or ignore.
            // For now, let's fallback to standard UTF-8 bytes of the codepoint if not found?
            // Or strictly skip. Let's strictly skip or handle 0x00 replacement.
        }
    }
    return out;
}

// ---------------- Pretokenizer ----------------
std::vector<std::string> BpeTokenizer::PretokenizeSentencePiece(const std::string& text) {
    static const std::string ws_mark = "▁";
    std::vector<std::string> out;
    std::string curr;
    curr.reserve(text.size());

    size_t i = 0;
    while (i < text.size()) {
        uint32_t cp; size_t len;
        if (!NextUtf8(text, i, cp, len)) break;
        if (IsUnicodeSpace(cp)) {
            if (!curr.empty()) {
                out.emplace_back(ws_mark + curr);
                curr.clear();
            }
        } else {
            curr.append(text, i - len, len);
        }
    }
    if (!curr.empty()) {
        out.emplace_back(ws_mark + curr);
    }
    return out;
}

// ---------------- Pair key ----------------
std::string BpeTokenizer::PairKey(const std::string& a, const std::string& b) {
    std::string k;
    k.reserve(a.size() + b.size() + 1);
    k.append(a);
    k.push_back('\t');
    k.append(b);
    return k;
}

// ---------------- BPE core ----------------
const std::vector<std::string>& BpeTokenizer::BpeForPieceCached(const std::string& piece) const {
    {
        std::lock_guard<std::mutex> g(cache_mu_);
        auto it = bpe_cache_.find(piece);
        if (it != bpe_cache_.end()) return it->second;
    }
    auto tokens = BpeForPiece(piece);
    {
        std::lock_guard<std::mutex> g(cache_mu_);
        auto [it, _] = bpe_cache_.emplace(piece, std::move(tokens));
        return it->second;
    }
}

std::vector<std::string> BpeTokenizer::BpeForPiece(const std::string& piece) const {
    std::vector<std::string> symbols = Utf8Chars(piece);
    if (symbols.size() <= 1) return symbols;

    while (symbols.size() >= 2) {
        int best_rank = std::numeric_limits<int>::max();
        int best_i = -1;

        for (int i = 0; i + 1 < static_cast<int>(symbols.size()); ++i) {
            std::string key = PairKey(symbols[i], symbols[i+1]);
            auto it = merges_rank_.find(key);
            if (it != merges_rank_.end()) {
                int r = it->second;
                if (r < best_rank) {
                    best_rank = r;
                    best_i = i;
                }
            }
        }
        if (best_i < 0) break;

        symbols[best_i] += symbols[best_i + 1];
        symbols.erase(symbols.begin() + best_i + 1);
    }
    return symbols;
}

// ---------------- Token -> id mapping with fallback ----------------
void BpeTokenizer::TokensToIds(const std::vector<std::string>& tokens, std::vector<int>& out) const {
    out.reserve(out.size() + tokens.size());
    for (const auto& t : tokens) {
        auto it = token_to_id_.find(t);
        if (it != token_to_id_.end()) {
            out.push_back(it->second);
        } else if (fallback_to_chars_) {
            auto chars = Utf8Chars(t);
            for (const auto& ch : chars) {
                auto itc = token_to_id_.find(ch);
                if (itc != token_to_id_.end()) {
                    out.push_back(itc->second);
                } else if (special_ids_.unk_id >= 0) {
                    out.push_back(special_ids_.unk_id);
                }
            }
        } else if (special_ids_.unk_id >= 0) {
            out.push_back(special_ids_.unk_id);
        }
    }
}

// ---------------- Move semantics ----------------
BpeTokenizer::BpeTokenizer(BpeTokenizer&& other) noexcept
    : id_to_token_(std::move(other.id_to_token_)),
      token_to_id_(std::move(other.token_to_id_)),
      merges_rank_(std::move(other.merges_rank_)),
      special_ids_(other.special_ids_),
      fallback_to_chars_(other.fallback_to_chars_),
      use_byte_encoder_(other.use_byte_encoder_),
      byte_encoder_(std::move(other.byte_encoder_)),
      byte_decoder_(std::move(other.byte_decoder_))
{
    // 迁移缓存（为安全起见加锁）
    std::lock_guard<std::mutex> lk(other.cache_mu_);
    bpe_cache_ = std::move(other.bpe_cache_);
    // cache_mu_ 自身是新构造的 mutex
}

BpeTokenizer& BpeTokenizer::operator=(BpeTokenizer&& other) noexcept {
    if (this != &other) {
        // 同时锁两边，避免数据竞争
        std::scoped_lock lk(cache_mu_, other.cache_mu_);
        id_to_token_ = std::move(other.id_to_token_);
        token_to_id_ = std::move(other.token_to_id_);
        merges_rank_ = std::move(other.merges_rank_);
        special_ids_ = other.special_ids_;
        fallback_to_chars_ = other.fallback_to_chars_;
        use_byte_encoder_ = other.use_byte_encoder_;
        byte_encoder_ = std::move(other.byte_encoder_);
        byte_decoder_ = std::move(other.byte_decoder_);
        bpe_cache_ = std::move(other.bpe_cache_);
        // cache_mu_ 仍为当前对象自己的 mutex
    }
    return *this;
}

// ---------------- Public APIs ----------------
BpeTokenizer BpeTokenizer::LoadFromFiles(const std::string& vocab_path,
                                         const std::string& merges_path,
                                         const SpecialTokensConfig& spec,
                                         bool add_special_if_missing,
                                         bool fallback_to_chars,
                                         bool use_byte_encoder) {
    BpeTokenizer tok;
    tok.id_to_token_ = LoadVocab(vocab_path);
    if (tok.id_to_token_.empty()) {
        throw std::runtime_error("Vocab is empty: " + vocab_path);
    }
    tok.token_to_id_ = BuildTokenToId(tok.id_to_token_);
    tok.merges_rank_ = LoadMergesRank(merges_path);
    tok.fallback_to_chars_ = fallback_to_chars;
    tok.use_byte_encoder_ = use_byte_encoder;

    if (tok.use_byte_encoder_) {
        tok.InitByteMaps();
    }

    tok.EnsureSpecialTokens(spec, add_special_if_missing);
    // 显式移动，避免 MSVC 尝试拷贝
    return std::move(tok);
}

void BpeTokenizer::EnsureSpecialTokens(const SpecialTokensConfig& spec, bool add_if_missing) {
    auto ensure = [&](const std::optional<std::string>& name, int& id_slot) {
        if (!name.has_value()) { id_slot = -1; return; }
        auto it = token_to_id_.find(*name);
        if (it != token_to_id_.end()) {
            id_slot = it->second;
            return;
        }
        if (!add_if_missing) {
            id_slot = -1;
            return;
        }
        id_slot = static_cast<int>(id_to_token_.size());
        id_to_token_.push_back(*name);
        token_to_id_.emplace(*name, id_slot);
    };

    ensure(spec.bos_token, special_ids_.bos_id);
    ensure(spec.eos_token, special_ids_.eos_id);
    ensure(spec.unk_token, special_ids_.unk_id);
    ensure(spec.sep_token, special_ids_.sep_id);
    ensure(spec.pad_token, special_ids_.pad_id);
    ensure(spec.cls_token, special_ids_.cls_id);
    ensure(spec.mask_token, special_ids_.mask_id);
}

std::vector<int> BpeTokenizer::encode(const std::string& text,
                                      bool add_bos,
                                      bool add_eos,
                                      bool add_cls,
                                      bool add_sep) const {
    std::vector<int> ids;
    ids.reserve(text.size() / 2 + 8);

    if (add_cls && special_ids_.cls_id >= 0) ids.push_back(special_ids_.cls_id);
    if (add_bos && special_ids_.bos_id >= 0) ids.push_back(special_ids_.bos_id);

    if (use_byte_encoder_) {
        // New Logic: Byte-Level BPE
        // 1. 将整个文本进行 ByteEncode (Raw Bytes -> Unicode Chars)
        std::string encoded_text = ByteEncode(text);

        // 2. 直接进行 BPE (通常不需要再次 split by space，除非模仿 GPT-2 的 regex split)
        // 简单起见，这里暂不引入复杂的 Regex split，直接对转码后的整串做 BPE
        // 如果需要完全复刻 GPT-2，需要在 ByteEncode 前先做 Regex split
        const auto& toks = BpeForPieceCached(encoded_text);
        TokensToIds(toks, ids);

    } else {
        // Original Logic: SentencePiece-like Pretokenization
        auto pieces = PretokenizeSentencePiece(text);
        for (const auto& p : pieces) {
            const auto& toks = BpeForPieceCached(p);
            TokensToIds(toks, ids);
        }
    }

    if (add_sep && special_ids_.sep_id >= 0) ids.push_back(special_ids_.sep_id);
    if (add_eos && special_ids_.eos_id >= 0) ids.push_back(special_ids_.eos_id);
    return ids;
}

std::string BpeTokenizer::decode(const std::vector<int>& ids, bool skip_special_tokens) const {
    std::string s;
    s.reserve(ids.size() * 3);
    for (int id : ids) {
        if (id < 0 || id >= static_cast<int>(id_to_token_.size())) continue;
        const std::string& tok = id_to_token_[id];

        bool is_special =
            id == special_ids_.bos_id || id == special_ids_.eos_id ||
            id == special_ids_.unk_id || id == special_ids_.sep_id ||
            id == special_ids_.pad_id || id == special_ids_.cls_id ||
            id == special_ids_.mask_id;

        if (skip_special_tokens && is_special) continue;

        s += tok;
    }

    if (use_byte_encoder_) {
        // New Logic: Convert Unicode Chars back to Raw Bytes
        if (!s.empty()) {
            return ByteDecode(s);
        }
        return s;
    } else {
        // Original Logic: Handle 0x2581 space marker
        if (!s.empty()) {
            std::string out;
            out.reserve(s.size());
            size_t i = 0;
            while (i < s.size()) {
                uint32_t cp; size_t len;
                if (!NextUtf8(s, i, cp, len)) break;
                if (cp == 0x2581) {
                    out.push_back(' ');
                } else {
                    out.append(s, i - len, len);
                }
            }
            size_t b = 0;
            while (b < out.size() && out[b] == ' ') ++b;
            if (b > 0) out.erase(0, b);
            return out;
        }
    }
    return s;
}
