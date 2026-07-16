// SPDX-License-Identifier: Apache-2.0
#include "json_min.h"

#include <fstream>
#include <sstream>

namespace pvl {

Json Json::parse_file(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) throw std::runtime_error("json: cannot open file: " + path);
    std::ostringstream ss;
    ss << ifs.rdbuf();
    return Json::parse(ss.str());
}

}  // namespace pvl
