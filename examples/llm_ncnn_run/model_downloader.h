#pragma once

#include <string>

bool ensure_model_present(const std::string& model_dir, const std::string& base_url, std::string* err = nullptr);
bool ensure_model_present(const std::string& model_dir, std::string* err = nullptr);
