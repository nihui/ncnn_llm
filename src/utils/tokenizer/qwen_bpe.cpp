#include "utils/tokenizer/qwen_bpe.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

namespace q3tts {

// ---------- utf-8 / unicode helpers ----------

static std::string cp_to_utf8(uint32_t cp) {
    std::string s;
    if (cp < 0x80) s += (char)cp;
    else if (cp < 0x800) {
        s += (char)(0xC0 | (cp >> 6));
        s += (char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += (char)(0xE0 | (cp >> 12));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    } else {
        s += (char)(0xF0 | (cp >> 18));
        s += (char)(0x80 | ((cp >> 12) & 0x3F));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    }
    return s;
}

// decode utf-8 at position i, advance i
static uint32_t utf8_next(const std::string& s, size_t& i) {
    const unsigned char c = s[i];
    uint32_t cp;
    int n;
    if (c < 0x80) { cp = c; n = 1; }
    else if ((c >> 5) == 0x6) { cp = c & 0x1F; n = 2; }
    else if ((c >> 4) == 0xE) { cp = c & 0x0F; n = 3; }
    else { cp = c & 0x07; n = 4; }
    for (int k = 1; k < n && i + k < s.size(); k++)
        cp = (cp << 6) | (s[i + k] & 0x3F);
    i += n;
    return cp;
}

struct Range { uint32_t lo, hi; };

// \p{L} approximation â extend when a script fails parity (see header note)
static const Range kLetter[] = {
    {0x41, 0x5A}, {0x61, 0x7A}, {0xAA, 0xAA}, {0xB5, 0xB5}, {0xBA, 0xBA},
    {0xC0, 0xD6}, {0xD8, 0xF6}, {0xF8, 0x2C1}, {0x2C6, 0x2D1}, {0x2E0, 0x2E4},
    {0x370, 0x374}, {0x376, 0x377}, {0x37A, 0x37D}, {0x37F, 0x37F}, {0x386, 0x386},
    {0x388, 0x3FF}, {0x400, 0x481}, {0x48A, 0x52F}, {0x531, 0x556}, {0x561, 0x587},
    {0x5D0, 0x5EA}, {0x620, 0x64A}, {0x66E, 0x66F}, {0x671, 0x6D3}, {0x904, 0x939},
    {0x958, 0x961}, {0x971, 0x980}, {0xE01, 0xE30}, {0xE40, 0xE46},
    {0x10A0, 0x10C5}, {0x10D0, 0x10FA}, {0x1100, 0x1159}, {0x115F, 0x11A2},
    {0x1E00, 0x1F15}, {0x1F18, 0x1F1D}, {0x1F20, 0x1F45}, {0x1F48, 0x1F4D},
    {0x1F50, 0x1FFC}, {0x2071, 0x2071}, {0x207F, 0x207F}, {0x2090, 0x209C},
    {0x2C60, 0x2C7F}, {0x3005, 0x3006}, {0x3031, 0x3035}, {0x303B, 0x303C},
    {0x3041, 0x3096}, {0x309D, 0x309F}, {0x30A1, 0x30FA}, {0x30FC, 0x30FF},
    {0x3105, 0x312D}, {0x3131, 0x318E}, {0x31A0, 0x31BA}, {0x31F0, 0x31FF},
    {0x3400, 0x4DBF}, {0x4E00, 0x9FFF}, {0xA000, 0xA48C}, {0xA4D0, 0xA4FD},
    {0xA500, 0xA60C}, {0xA610, 0xA61F}, {0xA62A, 0xA62B}, {0xA640, 0xA66E},
    {0xAC00, 0xD7A3}, {0xF900, 0xFA6D}, {0xFB00, 0xFB06}, {0xFF21, 0xFF3A},
    {0xFF41, 0xFF5A}, {0xFF66, 0xFFBE}, {0x20000, 0x2A6DF}, {0x2A700, 0x2B738},
};

// \p{N} approximation (Nd + common No/Nl)
static const Range kNumber[] = {
    {0x30, 0x39}, {0xB2, 0xB3}, {0xB9, 0xB9}, {0xBC, 0xBE},
    {0x660, 0x669}, {0x6F0, 0x6F9}, {0x966, 0x96F}, {0xE50, 0xE59},
    {0x2070, 0x2070}, {0x2074, 0x2079}, {0x2080, 0x2089},
    {0x2150, 0x2182}, {0x2185, 0x2189}, {0x2460, 0x249B}, {0x24EA, 0x24FF},
    {0x2776, 0x2793}, {0x3007, 0x3007}, {0x3021, 0x3029}, {0x3192, 0x3195},
    {0x3220, 0x3229}, {0x3248, 0x324F}, {0x3251, 0x325F}, {0x3280, 0x3289},
    {0x32B1, 0x32BF}, {0xFF10, 0xFF19},
};

static bool in_ranges(uint32_t cp, const Range* r, size_t n) {
    for (size_t i = 0; i < n; i++)
        if (cp >= r[i].lo && cp <= r[i].hi) return true;
    return false;
}

static bool is_letter(uint32_t cp) { return in_ranges(cp, kLetter, sizeof(kLetter) / sizeof(Range)); }
static bool is_number(uint32_t cp) { return in_ranges(cp, kNumber, sizeof(kNumber) / sizeof(Range)); }
static bool is_space(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == 0x0B || cp == 0x0C ||
           cp == 0x85 || cp == 0xA0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) ||
           cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}
static bool is_newline(uint32_t cp) { return cp == '\r' || cp == '\n'; }

// ---------- GPT-2 pre-tokenizer scanner ----------
// operates on decoded codepoints; returns byte spans of the original string
static std::vector<std::pair<size_t, size_t>> pretokenize(const std::string& s) {
    // decode with byte offsets
    std::vector<uint32_t> cp;
    std::vector<size_t> off;  // byte offset of each cp; off[n] = s.size()
    for (size_t i = 0; i < s.size();) {
        off.push_back(i);
        cp.push_back(utf8_next(s, i));
    }
    off.push_back(s.size());
    const size_t n = cp.size();

    std::vector<std::pair<size_t, size_t>> spans;
    size_t i = 0;
    auto flush = [&](size_t a, size_t b) { if (b > a) spans.emplace_back(off[a], off[b]); };

    while (i < n) {
        // 1) (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if (cp[i] == '\'' && i + 1 < n) {
            uint32_t c1 = cp[i + 1] | 0x20;  // ascii lowercase
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
                flush(i, i + 2); i += 2; continue;
            }
            if (i + 2 < n) {
                uint32_t c2 = cp[i + 2] | 0x20;
                if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) {
                    flush(i, i + 3); i += 3; continue;
                }
            }
        }
        // 2) [^\r\n\p{L}\p{N}]?\p{L}+
        {
            size_t j = i;
            bool prefix = !is_newline(cp[j]) && !is_letter(cp[j]) && !is_number(cp[j]);
            size_t k = j + (prefix ? 1 : 0);
            if (k < n && is_letter(cp[k])) {
                while (k < n && is_letter(cp[k])) k++;
                flush(i, k); i = k; continue;
            }
        }
        // 3) \p{N}
        if (is_number(cp[i])) { flush(i, i + 1); i++; continue; }
        // 4)  ?[^\s\p{L}\p{N}]+[\r\n]*
        {
            size_t j = i;
            if (cp[j] == ' ' && j + 1 < n) j++;
            size_t k = j;
            while (k < n && !is_space(cp[k]) && !is_letter(cp[k]) && !is_number(cp[k])) k++;
            if (k > j) {
                while (k < n && is_newline(cp[k])) k++;
                flush(i, k); i = k; continue;
            }
        }
        // 5) \s*[\r\n]+
        {
            size_t k = i;
            while (k < n && is_space(cp[k]) && !is_newline(cp[k])) k++;
            if (k < n && is_newline(cp[k])) {
                while (k < n && is_newline(cp[k])) k++;
                flush(i, k); i = k; continue;
            }
        }
        // 6) \s+(?!\S)  |  7) \s+
        if (is_space(cp[i])) {
            size_t k = i;
            while (k < n && is_space(cp[k])) k++;
            // if a non-space follows, leave the last space for its branch-2/4 prefix
            if (k < n && k - i > 1) k--;
            else if (k < n && k - i == 1) { /* single space before non-space: still \s+ per alternation? */
                // regex: branch 6 fails (followed by \S), branch 7 \s+ matches the single space
            }
            flush(i, k > i ? k : i + 1); i = (k > i ? k : i + 1); continue;
        }
        // fallback: single cp (should not happen)
        flush(i, i + 1); i++;
    }
    return spans;
}

// ---------- QwenBpe ----------

void QwenBpe::load(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::string tag;
    size_t count;

    // GPT-2 byte -> unicode table
    std::vector<int> bs;
    for (int b = '!'; b <= '~'; b++) bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; b++) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; b++) bs.push_back(b);
    std::vector<uint32_t> cs(256);
    std::vector<bool> seen(256, false);
    for (int b : bs) { cs[b] = b; seen[b] = true; }
    int extra = 0;
    for (int b = 0; b < 256; b++)
        if (!seen[b]) cs[b] = 256 + extra++;
    for (int b = 0; b < 256; b++) byte_enc_[b] = cp_to_utf8(cs[b]);

    f >> tag >> count;
    f.ignore();
    vocab_.reserve(count * 2);
    std::string line;
    for (size_t i = 0; i < count; i++) {
        std::getline(f, line);
        vocab_[line] = (int)i;
    }
    f >> tag >> count;
    f.ignore();
    merge_rank_.reserve(count * 2);
    for (size_t i = 0; i < count; i++) {
        std::getline(f, line);
        merge_rank_[line] = (int)i;
    }
    f >> tag >> count;
    f.ignore();
    for (size_t i = 0; i < count; i++) {
        std::getline(f, line);
        size_t sp = line.find(' ');
        added_.emplace_back(line.substr(sp + 1), atoi(line.c_str()));
    }
}

std::vector<int> QwenBpe::bpe_word(const std::string& w) const {
    // split into unicode chars
    std::vector<std::string> parts;
    for (size_t i = 0; i < w.size();) {
        size_t j = i;
        utf8_next(w, j);
        parts.push_back(w.substr(i, j - i));
        i = j;
    }
    while (parts.size() > 1) {
        int best_rank = INT32_MAX, best = -1;
        for (size_t i = 0; i + 1 < parts.size(); i++) {
            auto it = merge_rank_.find(parts[i] + " " + parts[i + 1]);
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best = (int)i;
            }
        }
        if (best < 0) break;
        parts[best] += parts[best + 1];
        parts.erase(parts.begin() + best + 1);
    }
    std::vector<int> ids;
    for (const auto& p : parts) {
        auto it = vocab_.find(p);
        if (it != vocab_.end()) ids.push_back(it->second);
        // unknown pieces are silently dropped (Qwen2 has full byte coverage,
        // so this only happens on table bugs)
    }
    return ids;
}

std::vector<int> QwenBpe::encode(const std::string& text) const {
    std::vector<int> out;
    size_t pos = 0;
    while (pos < text.size()) {
        // find earliest added-token occurrence
        size_t best_at = std::string::npos, best_len = 0;
        int best_id = -1;
        for (const auto& [content, id] : added_) {
            size_t at = text.find(content, pos);
            if (at != std::string::npos &&
                (at < best_at || (at == best_at && content.size() > best_len))) {
                best_at = at;
                best_len = content.size();
                best_id = id;
            }
        }
        const size_t chunk_end = best_at == std::string::npos ? text.size() : best_at;
        if (chunk_end > pos) {
            const std::string chunk = text.substr(pos, chunk_end - pos);
            for (auto [a, b] : pretokenize(chunk)) {
                std::string bu;
                for (size_t i = a; i < b; i++)
                    bu += byte_enc_[(unsigned char)chunk[i]];
                std::vector<int> ids = bpe_word(bu);
                out.insert(out.end(), ids.begin(), ids.end());
            }
        }
        if (best_at == std::string::npos) break;
        out.push_back(best_id);
        pos = best_at + best_len;
    }
    return out;
}

}  // namespace q3tts
