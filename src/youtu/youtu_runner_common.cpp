#include "youtu_runner_common.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include <zlib.h>

namespace youtu {

// Shared runner primitives.
//
// Owns the lightweight Tensor container, command options, root discovery,
// NPY/NPZ parsing, ncnn Mat conversion, model loading helpers, RoPE/mask
// construction, and numerical diagnostics. Higher-level modules build on this
// layer and should not duplicate tensor-layout conversions.

[[noreturn]] void die(const std::string& message) {
    throw std::runtime_error(message);
}

bool is_project_root(const fs::path& path) {
    std::error_code ec;
    const bool source_layout = fs::is_regular_file(path / "cpp" / "CMakeLists.txt", ec) &&
                               fs::is_directory(path / "export", ec);
    ec.clear();
    const bool runtime_layout = fs::is_directory(path / "assets" / "youtu_text_export", ec) &&
                                fs::is_directory(path / "assets" / "youtu_vl_export", ec);
    ec.clear();
    const bool release_layout = fs::is_directory(path / "models" / "text", ec) &&
                                fs::is_directory(path / "models" / "vision", ec);
    return source_layout || runtime_layout || release_layout;
}

std::string find_project_root_from(fs::path path) {
    std::error_code ec;
    path = fs::absolute(path, ec);
    if (ec) return {};
    if (!fs::is_directory(path, ec)) path = path.parent_path();

    while (!path.empty()) {
        if (is_project_root(path)) return path.lexically_normal().string();
        const fs::path parent = path.parent_path();
        if (parent == path) break;
        path = parent;
    }
    return {};
}

std::string discover_project_root(const char* argv0) {
    if (const char* env_root = std::getenv("YOUTU_VL_ROOT")) {
        if (*env_root != '\0') return fs::path(env_root).lexically_normal().string();
    }

    if (argv0 && *argv0 != '\0') {
        const std::string executable_root = find_project_root_from(fs::path(argv0));
        if (!executable_root.empty()) return executable_root;
    }

    std::error_code ec;
    const fs::path cwd = fs::current_path(ec);
    if (!ec) {
        const std::string cwd_root = find_project_root_from(cwd);
        if (!cwd_root.empty()) return cwd_root;
        return cwd.lexically_normal().string();
    }
    return ".";
}

std::string project_path(const std::string& root, const fs::path& relative) {
    return (fs::path(root) / relative).lexically_normal().string();
}

fs::path runtime_package_root(const std::string& root) {
    const fs::path root_path(root);
    if (fs::is_directory(root_path / "models" / "text")) return root_path;
    return {};
}

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::vector<unsigned char> read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) die("failed to open " + path);
    in.seekg(0, std::ios::end);
    const std::streamoff size = in.tellg();
    if (size < 0) die("failed to determine file size " + path);
    in.seekg(0, std::ios::beg);
    std::vector<unsigned char> data(static_cast<size_t>(size));
    if (!data.empty() && !in.read(reinterpret_cast<char*>(data.data()), size)) {
        die("failed to read " + path);
    }
    return data;
}

std::string read_text_file(const std::string& path) {
    const std::vector<unsigned char> data = read_file(path);
    return std::string(reinterpret_cast<const char*>(data.data()), data.size());
}

uint16_t le16(const unsigned char* p) {
    return static_cast<uint16_t>(p[0] | (p[1] << 8));
}

uint32_t le32(const unsigned char* p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

uint64_t le64(const unsigned char* p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) v = (v << 8) | p[i];
    return v;
}

std::string trim(const std::string& s) {
    const size_t b = s.find_first_not_of(" \t\n\r");
    if (b == std::string::npos) return "";
    const size_t e = s.find_last_not_of(" \t\n\r");
    return s.substr(b, e - b + 1);
}

void append_utf8(std::string& out, uint32_t cp) {
    if (cp <= 0x7f) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7ff) {
        out.push_back(static_cast<char>(0xc0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    } else if (cp <= 0xffff) {
        out.push_back(static_cast<char>(0xe0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    } else {
        out.push_back(static_cast<char>(0xf0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    }
}

std::string json_unescape(const std::string& s) {
    std::string out;
    for (size_t i = 0; i < s.size(); ++i) {
        char c = s[i];
        if (c != '\\') {
            out.push_back(c);
            continue;
        }
        if (++i >= s.size()) die("bad json escape");
        c = s[i];
        if (c == '"' || c == '\\' || c == '/') out.push_back(c);
        else if (c == 'b') out.push_back('\b');
        else if (c == 'f') out.push_back('\f');
        else if (c == 'n') out.push_back('\n');
        else if (c == 'r') out.push_back('\r');
        else if (c == 't') out.push_back('\t');
        else if (c == 'u') {
            if (i + 4 >= s.size()) die("bad json unicode escape");
            uint32_t cp = 0;
            for (int k = 0; k < 4; ++k) {
                char h = s[++i];
                cp <<= 4;
                if (h >= '0' && h <= '9') cp |= h - '0';
                else if (h >= 'a' && h <= 'f') cp |= h - 'a' + 10;
                else if (h >= 'A' && h <= 'F') cp |= h - 'A' + 10;
                else die("bad json unicode hex");
            }
            if (cp >= 0xd800 && cp <= 0xdbff && i + 6 < s.size() && s[i + 1] == '\\' && s[i + 2] == 'u') {
                i += 2;
                uint32_t lo = 0;
                for (int k = 0; k < 4; ++k) {
                    char h = s[++i];
                    lo <<= 4;
                    if (h >= '0' && h <= '9') lo |= h - '0';
                    else if (h >= 'a' && h <= 'f') lo |= h - 'a' + 10;
                    else if (h >= 'A' && h <= 'F') lo |= h - 'A' + 10;
                    else die("bad json unicode hex");
                }
                cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00);
            }
            append_utf8(out, cp);
        } else {
            die("unsupported json escape");
        }
    }
    return out;
}

std::vector<int> parse_shape(const std::string& header) {
    const size_t l = header.find('(');
    const size_t r = header.find(')', l);
    if (l == std::string::npos || r == std::string::npos) die("npy header has no shape");
    std::string body = header.substr(l + 1, r - l - 1);
    std::vector<int> shape;
    size_t start = 0;
    while (start < body.size()) {
        const size_t comma = body.find(',', start);
        const std::string item = trim(body.substr(start, comma == std::string::npos ? std::string::npos : comma - start));
        if (!item.empty()) shape.push_back(std::stoi(item));
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return shape;
}

std::string parse_descr(const std::string& header) {
    const std::string key = "'descr'";
    size_t p = header.find(key);
    if (p == std::string::npos) p = header.find("\"descr\"");
    if (p == std::string::npos) die("npy header has no descr");
    p = header.find(':', p);
    p = header.find_first_of("'\"", p);
    const char quote = header[p];
    const size_t q = header.find(quote, p + 1);
    return header.substr(p + 1, q - p - 1);
}

Tensor parse_npy(const std::vector<unsigned char>& bytes) {
    if (bytes.size() < 16 || std::memcmp(bytes.data(), "\x93NUMPY", 6) != 0) die("invalid npy payload");
    const int major = bytes[6];
    size_t offset = 8;
    uint32_t header_len = 0;
    if (major == 1) {
        header_len = le16(bytes.data() + offset);
        offset += 2;
    } else {
        header_len = le32(bytes.data() + offset);
        offset += 4;
    }
    if (offset + header_len > bytes.size()) die("truncated npy header");
    const std::string header(reinterpret_cast<const char*>(bytes.data() + offset), header_len);
    offset += header_len;

    Tensor t;
    t.shape = parse_shape(header);
    const std::string descr = parse_descr(header);
    size_t count = 1;
    for (int d : t.shape) count *= static_cast<size_t>(d);
    if (descr == "<f4" || descr == "|f4") {
        if (offset + count * sizeof(float) > bytes.size()) die("truncated f32 npy data");
        t.f32.resize(count);
        std::memcpy(t.f32.data(), bytes.data() + offset, count * sizeof(float));
    } else if (descr == "<i8" || descr == "|i8") {
        if (offset + count * sizeof(int64_t) > bytes.size()) die("truncated i64 npy data");
        t.i64.resize(count);
        std::memcpy(t.i64.data(), bytes.data() + offset, count * sizeof(int64_t));
    } else if (descr == "<i4" || descr == "|i4") {
        if (offset + count * sizeof(int32_t) > bytes.size()) die("truncated i32 npy data");
        t.i64.resize(count);
        const int32_t* src = reinterpret_cast<const int32_t*>(bytes.data() + offset);
        for (size_t i = 0; i < count; ++i) t.i64[i] = static_cast<int64_t>(src[i]);
    } else {
        die("unsupported npy dtype " + descr);
    }
    return t;
}

std::map<std::string, Tensor> load_npz(const std::string& path) {
    const std::vector<unsigned char> z = read_file(path);
    if (z.size() < 22) die("invalid zip");

    size_t eocd = std::string::npos;
    for (size_t i = z.size() - 22; i + 4 <= z.size(); --i) {
        if (le32(z.data() + i) == 0x06054b50) {
            eocd = i;
            break;
        }
        if (i == 0) break;
    }
    if (eocd == std::string::npos) die("zip eocd not found");
    const uint16_t entries = le16(z.data() + eocd + 10);
    const uint32_t cd_offset = le32(z.data() + eocd + 16);

    std::map<std::string, Tensor> out;
    size_t pos = cd_offset;
    for (uint16_t i = 0; i < entries; ++i) {
        if (pos + 46 > z.size() || le32(z.data() + pos) != 0x02014b50) die("invalid zip central directory");
        const uint16_t method = le16(z.data() + pos + 10);
        const uint32_t comp_size = le32(z.data() + pos + 20);
        const uint32_t uncomp_size = le32(z.data() + pos + 24);
        const uint16_t name_len = le16(z.data() + pos + 28);
        const uint16_t extra_len = le16(z.data() + pos + 30);
        const uint16_t comment_len = le16(z.data() + pos + 32);
        const uint32_t local_offset = le32(z.data() + pos + 42);
        const std::string name(reinterpret_cast<const char*>(z.data() + pos + 46), name_len);
        pos += 46 + name_len + extra_len + comment_len;

        if (local_offset + 30 > z.size() || le32(z.data() + local_offset) != 0x04034b50) die("invalid zip local header");
        const uint16_t lname_len = le16(z.data() + local_offset + 26);
        const uint16_t lextra_len = le16(z.data() + local_offset + 28);
        const size_t data_offset = local_offset + 30 + lname_len + lextra_len;
        if (data_offset + comp_size > z.size()) die("truncated zip member");

        std::vector<unsigned char> payload;
        if (method == 0) {
            payload.assign(z.begin() + data_offset, z.begin() + data_offset + comp_size);
        } else if (method == 8) {
            payload.resize(uncomp_size);
            z_stream zs{};
            zs.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(z.data() + data_offset));
            zs.avail_in = comp_size;
            zs.next_out = reinterpret_cast<Bytef*>(payload.data());
            zs.avail_out = uncomp_size;
            if (inflateInit2(&zs, -MAX_WBITS) != Z_OK) die("inflateInit2 failed");
            const int ret = inflate(&zs, Z_FINISH);
            inflateEnd(&zs);
            if (ret != Z_STREAM_END) die("inflate failed for " + name);
        } else {
            die("unsupported zip compression method " + std::to_string(method));
        }

        std::string key = name;
        if (key.size() > 4 && key.substr(key.size() - 4) == ".npy") key.resize(key.size() - 4);
        out[key] = parse_npy(payload);
    }
    return out;
}

Tensor& require(std::map<std::string, Tensor>& arrays, const std::string& key) {
    auto it = arrays.find(key);
    if (it == arrays.end()) die("npz missing " + key);
    return it->second;
}

const Tensor& require_const(const std::map<std::string, Tensor>& arrays, const std::string& key) {
    auto it = arrays.find(key);
    if (it == arrays.end()) die("npz missing " + key);
    return it->second;
}

ncnn::Mat mat_from_f32(const Tensor& t) {
    ncnn::Mat m;
    const std::vector<int>& s = t.shape;
    if (s.size() == 1) m.create(s[0], static_cast<size_t>(4u));
    else if (s.size() == 2) m.create(s[1], s[0], static_cast<size_t>(4u));
    else if (s.size() == 3) m.create(s[2], s[1], s[0], static_cast<size_t>(4u));
    else if (s.size() == 4) m.create(s[3], s[2], s[1], s[0], static_cast<size_t>(4u));
    else die("unsupported tensor rank");
    if (!t.f32.empty()) std::memcpy(m.data, t.f32.data(), t.f32.size() * sizeof(float));
    return m;
}

ncnn::Mat mat_from_f32_shape(const std::vector<float>& data, const std::vector<int>& shape) {
    Tensor t;
    t.shape = shape;
    t.f32 = data;
    return mat_from_f32(t);
}

std::vector<float> mat_to_vector(const ncnn::Mat& m) {
    const size_t count = m.total();
    std::vector<float> v(count);
    std::memcpy(v.data(), m.data, count * sizeof(float));
    return v;
}

std::vector<int> mat_shape_nchw(const ncnn::Mat& m) {
    if (m.dims == 1) return {m.w};
    if (m.dims == 2) return {m.h, m.w};
    if (m.dims == 3) return {m.c, m.h, m.w};
    if (m.dims == 4) return {m.c, m.d, m.h, m.w};
    return {};
}

std::string shape_string(const std::vector<int>& shape) {
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) s += ", ";
        s += std::to_string(shape[i]);
    }
    s += "]";
    return s;
}

bool file_exists(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return static_cast<bool>(in);
}

Tensor tensor_from_mat(const ncnn::Mat& m, const std::vector<int>& forced_shape) {
    Tensor t;
    t.shape = forced_shape.empty() ? mat_shape_nchw(m) : forced_shape;
    t.f32 = mat_to_vector(m);
    if (!forced_shape.empty()) {
        size_t wanted = 1;
        for (int d : forced_shape) wanted *= static_cast<size_t>(d);
        if (t.f32.size() < wanted) {
            die("mat has fewer values than forced shape actual_count=" + std::to_string(t.f32.size()) +
                " forced_shape=" + shape_string(forced_shape));
        }
        t.f32.resize(wanted);
    }
    return t;
}

void configure_net_options(ncnn::Net& net, int threads, bool no_packing, bool force_fp32) {
    net.opt.use_vulkan_compute = false;
    net.opt.num_threads = threads;
    if (no_packing) net.opt.use_packing_layout = false;
    if (force_fp32) {
        net.opt.use_fp16_packed = false;
        net.opt.use_fp16_storage = false;
        net.opt.use_fp16_arithmetic = false;
        net.opt.use_bf16_packed = false;
        net.opt.use_bf16_storage = false;
    }
}

void load_net(ncnn::Net& net, const NetFiles& files, int threads, bool no_packing, bool force_fp32) {
    configure_net_options(net, threads, no_packing, force_fp32);
    if (net.load_param(files.param.c_str()) != 0) die("load_param failed: " + files.param);
    if (net.load_model(files.bin.c_str()) != 0) die("load_model failed: " + files.bin);
}

Tensor run_single(const NetFiles& files, const Tensor& input, int threads, bool no_packing,
                  const std::vector<int>& forced_shape, bool force_fp32) {
    ncnn::Net net;
    load_net(net, files, threads, no_packing, force_fp32);
    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", mat_from_f32(input));
    ncnn::Mat out;
    if (ex.extract("out0", out) != 0) die("extract out0 failed: " + files.param);
    return tensor_from_mat(out, forced_shape);
}

bool read_fp16_embed_rows(const std::string& bin_path, const std::vector<int>& token_ids, Tensor& out) {
    constexpr std::uint32_t fp16_tag = 0x01306B47u;
    constexpr int hidden_size = 2560;
    std::ifstream input(bin_path, std::ios::binary | std::ios::ate);
    if (!input) return false;
    const std::streamoff bytes = input.tellg();
    if (bytes < 4 || (bytes - 4) % (hidden_size * 2) != 0) return false;
    const int vocab_size = static_cast<int>((bytes - 4) / (hidden_size * 2));
    input.seekg(0);
    std::uint32_t tag = 0;
    input.read(reinterpret_cast<char*>(&tag), sizeof(tag));
    if (!input || tag != fp16_tag) return false;

    std::vector<unsigned short> half(static_cast<size_t>(token_ids.size()) * hidden_size);
    for (size_t row = 0; row < token_ids.size(); ++row) {
        const int token_id = token_ids[row];
        if (token_id < 0 || token_id >= vocab_size) {
            die("embedding token id out of range id=" + std::to_string(token_id) +
                " vocab_size=" + std::to_string(vocab_size));
        }
        const std::streamoff offset = 4 + static_cast<std::streamoff>(token_id) * hidden_size * 2;
        input.seekg(offset);
        input.read(reinterpret_cast<char*>(half.data() + row * hidden_size), hidden_size * 2);
        if (!input) die("failed to read fp16 embedding row from " + bin_path);
    }

    ncnn::Mat decoded = ncnn::Mat::from_float16(half.data(), half.size());
    out.shape = {1, static_cast<int>(token_ids.size()), hidden_size};
    out.f32.resize(half.size());
    std::memcpy(out.f32.data(), decoded.data, out.f32.size() * sizeof(float));
    return true;
}

Tensor run_embed(const NetFiles& files, int token_id, int threads, bool no_packing, bool force_fp32) {
    Tensor direct;
    if (read_fp16_embed_rows(files.bin, {token_id}, direct)) {
        direct.shape = {1, 1, 2560};
        return direct;
    }
    ncnn::Net net;
    load_net(net, files, threads, no_packing, force_fp32);
    ncnn::Extractor ex = net.create_extractor();
    int ids[1] = {token_id};
    ncnn::Mat in(1, ids, 4u);
    ex.input("in0", in);
    ncnn::Mat out;
    if (ex.extract("out0", out) != 0) die("extract embed out0 failed");
    return tensor_from_mat(out, {1, 1, 2560});
}

Tensor run_embed_sequence(const NetFiles& files, const std::vector<int>& token_ids, int threads, bool no_packing,
                          bool force_fp32) {
    Tensor direct;
    if (read_fp16_embed_rows(files.bin, token_ids, direct)) return direct;
    ncnn::Net net;
    load_net(net, files, threads, no_packing, force_fp32);
    ncnn::Extractor ex = net.create_extractor();
    ncnn::Mat in(static_cast<int>(token_ids.size()), const_cast<int*>(token_ids.data()), 4u);
    ex.input("in0", in);
    ncnn::Mat out;
    if (ex.extract("out0", out) != 0) die("extract embed sequence out0 failed");
    return tensor_from_mat(out, {1, static_cast<int>(token_ids.size()), 2560});
}

std::array<Tensor, 3> run_layer_net(const NetFiles& files, const Tensor& hidden, const Tensor& mask, const Tensor& key, const Tensor& value, const Tensor& cos, const Tensor& sin, int threads, bool no_packing) {
    ncnn::Net net;
    load_net(net, files, threads, no_packing);
    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", mat_from_f32(hidden));
    ex.input("in1", mat_from_f32(mask));
    ex.input("in2", mat_from_f32(key));
    ex.input("in3", mat_from_f32(value));
    ex.input("in4", mat_from_f32(cos));
    ex.input("in5", mat_from_f32(sin));
    ncnn::Mat out0, out1, out2;
    if (ex.extract("out0", out0) != 0) die("extract layer out0 failed");
    if (ex.extract("out1", out1) != 0) die("extract layer out1 failed");
    if (ex.extract("out2", out2) != 0) die("extract layer out2 failed");
    return {tensor_from_mat(out0, {1, 1, 2560}), tensor_from_mat(out1), tensor_from_mat(out2)};
}

Tensor make_mask(int total_key_length, bool mask_first_slot) {
    Tensor t;
    t.shape = {1, 1, 1, total_key_length};
    t.f32.assign(total_key_length, 0.0f);
    if (mask_first_slot && !t.f32.empty()) t.f32[0] = -3.4028234663852886e38f;
    return t;
}

Tensor make_sdpa_mask(int total_key_length, bool mask_first_slot) {
    Tensor t;
    t.shape = {1, total_key_length};
    t.f32.assign(total_key_length, 0.0f);
    if (mask_first_slot && !t.f32.empty()) t.f32[0] = -3.4028234663852886e38f;
    return t;
}

Tensor make_prefill_mask(int bucket_length, bool include_dummy_cache_column) {
    const int key_length = bucket_length + (include_dummy_cache_column ? 1 : 0);
    Tensor t;
    t.shape = {1, bucket_length, key_length};
    t.f32.assign(static_cast<size_t>(bucket_length) * key_length, 0.0f);
    const float masked = -3.4028234663852886e38f;
    for (int q = 0; q < bucket_length; ++q) {
        const size_t row = static_cast<size_t>(q) * key_length;
        if (include_dummy_cache_column) {
            t.f32[row] = masked;
        }
        for (int k = 0; k < bucket_length; ++k) {
            if (k > q) {
                const int column = (include_dummy_cache_column ? 1 : 0) + k;
                t.f32[row + column] = masked;
            }
        }
    }
    return t;
}

std::pair<Tensor, Tensor> make_rope(int position, float theta, int dim) {
    Tensor cos;
    Tensor sin;
    cos.shape = {1, 1, dim};
    sin.shape = {1, 1, dim};
    cos.f32.resize(dim);
    sin.f32.resize(dim);
    for (int i = 0; i < dim / 2; ++i) {
        const float inv = 1.0f / std::pow(theta, static_cast<float>(i * 2) / static_cast<float>(dim));
        const float f = inv * static_cast<float>(position);
        const float c = std::cos(f);
        const float s = std::sin(f);
        cos.f32[i] = c;
        cos.f32[i + dim / 2] = c;
        sin.f32[i] = s;
        sin.f32[i + dim / 2] = s;
    }
    return {cos, sin};
}

std::pair<Tensor, Tensor> make_rope_table(int start_position, int seq_length, float theta, int dim) {
    Tensor cos;
    Tensor sin;
    cos.shape = {1, seq_length, dim};
    sin.shape = {1, seq_length, dim};
    cos.f32.resize(static_cast<size_t>(seq_length) * dim);
    sin.f32.resize(static_cast<size_t>(seq_length) * dim);
    for (int p = 0; p < seq_length; ++p) {
        const int position = start_position + p;
        for (int i = 0; i < dim / 2; ++i) {
            const float inv = 1.0f / std::pow(theta, static_cast<float>(i * 2) / static_cast<float>(dim));
            const float f = inv * static_cast<float>(position);
            const float c = std::cos(f);
            const float s = std::sin(f);
            const size_t base = static_cast<size_t>(p) * dim;
            cos.f32[base + i] = c;
            cos.f32[base + i + dim / 2] = c;
            sin.f32[base + i] = s;
            sin.f32[base + i + dim / 2] = s;
        }
    }
    return {cos, sin};
}

DiffStats diff_stats(const Tensor& actual, const Tensor& expected) {
    if (actual.f32.size() != expected.f32.size()) {
        die("diff size mismatch actual_shape=" + shape_string(actual.shape) +
            " actual_count=" + std::to_string(actual.f32.size()) +
            " expected_shape=" + shape_string(expected.shape) +
            " expected_count=" + std::to_string(expected.f32.size()));
    }
    DiffStats st;
    double sum = 0.0;
    double sum2 = 0.0;
    for (size_t i = 0; i < actual.f32.size(); ++i) {
        const float d = std::fabs(actual.f32[i] - expected.f32[i]);
        st.max_abs = std::max(st.max_abs, d);
        sum += d;
        sum2 += static_cast<double>(d) * d;
    }
    st.mean_abs = sum / static_cast<double>(actual.f32.size());
    st.rms = std::sqrt(sum2 / static_cast<double>(actual.f32.size()));
    return st;
}

bool print_stats_line(const std::string& name, const DiffStats& st, float atol) {
    const bool ok = st.max_abs <= atol;
    std::cout << "[" << (ok ? "PASS" : "FAIL") << "] " << name
              << ": max_abs=" << st.max_abs
              << " mean_abs=" << st.mean_abs
              << " rms=" << st.rms << "\n";
    return ok;
}

bool print_i64_compare_line(const std::string& name, const Tensor& actual, const Tensor& expected) {
    if (actual.shape != expected.shape || actual.i64.size() != expected.i64.size()) {
        std::cout << "[FAIL] " << name
                  << ": shape=" << shape_string(actual.shape)
                  << " expected_shape=" << shape_string(expected.shape) << "\n";
        return false;
    }
    size_t mismatch = 0;
    for (size_t i = 0; i < actual.i64.size(); ++i) {
        if (actual.i64[i] != expected.i64[i]) ++mismatch;
    }
    std::cout << "[" << (mismatch == 0 ? "PASS" : "FAIL") << "] " << name
              << ": mismatches=" << mismatch << "/" << actual.i64.size() << "\n";
    return mismatch == 0;
}

std::vector<int> topk_ids(const Tensor& t, int k) {
    std::vector<int> idx(t.f32.size());
    std::iota(idx.begin(), idx.end(), 0);
    if (k < static_cast<int>(idx.size())) {
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return t.f32[a] > t.f32[b]; });
        idx.resize(k);
    } else {
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return t.f32[a] > t.f32[b]; });
    }
    return idx;
}

int argmax_id(const Tensor& t) {
    return static_cast<int>(std::max_element(t.f32.begin(), t.f32.end()) - t.f32.begin());
}

std::string join_ids(const std::vector<int>& ids) {
    std::string s = "[";
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i) s += ", ";
        s += std::to_string(ids[i]);
    }
    s += "]";
    return s;
}

std::vector<int> parse_id_list(const std::string& text) {
    std::vector<int> ids;
    std::string item;
    std::stringstream ss(text);
    while (std::getline(ss, item, ',')) {
        item = trim(item);
        if (!item.empty()) ids.push_back(std::stoi(item));
    }
    if (ids.empty()) die("--ids produced an empty token list");
    return ids;
}

std::vector<int> parse_int_list(const std::string& text) {
    std::vector<int> values;
    std::string item;
    std::stringstream ss(text);
    while (std::getline(ss, item, ',')) {
        item = trim(item);
        if (!item.empty()) values.push_back(std::stoi(item));
    }
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}

std::string byte_encoder_piece(unsigned char b) {
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
        return std::string(1, static_cast<char>(b));
    }
    uint32_t cp = 0;
    if (b <= 32) cp = static_cast<uint32_t>(b) + 256;
    else if (b == 127) cp = 289;
    else if (b >= 128 && b <= 160) cp = static_cast<uint32_t>(b) + 162;
    else if (b == 173) cp = 323;
    else die("unexpected byte encoder value");
    std::string out;
    append_utf8(out, cp);
    return out;
}

std::unordered_map<std::string, std::string> build_byte_decoder() {
    std::unordered_map<std::string, std::string> m;
    for (int i = 0; i < 256; ++i) {
        m[byte_encoder_piece(static_cast<unsigned char>(i))] = std::string(1, static_cast<char>(i));
    }
    return m;
}

size_t find_matching(const std::string& s, size_t open_pos, char open_ch, char close_ch) {
    int depth = 0;
    bool in_string = false;
    bool esc = false;
    for (size_t i = open_pos; i < s.size(); ++i) {
        const char c = s[i];
        if (in_string) {
            if (esc) esc = false;
            else if (c == '\\') esc = true;
            else if (c == '"') in_string = false;
            continue;
        }
        if (c == '"') in_string = true;
        else if (c == open_ch) ++depth;
        else if (c == close_ch && --depth == 0) return i;
    }
    die("json matching delimiter not found");
}

std::string json_string_at(const std::string& s, size_t& pos) {
    while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) ++pos;
    if (pos >= s.size() || s[pos] != '"') die("expected json string");
    ++pos;
    std::string raw;
    bool esc = false;
    for (; pos < s.size(); ++pos) {
        const char c = s[pos];
        if (esc) {
            raw.push_back('\\');
            raw.push_back(c);
            esc = false;
        } else if (c == '\\') {
            esc = true;
        } else if (c == '"') {
            ++pos;
            return json_unescape(raw);
        } else {
            raw.push_back(c);
        }
    }
    die("unterminated json string");
}

} // namespace youtu
