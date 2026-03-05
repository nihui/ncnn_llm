#pragma once

#include "options.h"

#include "ncnn_llm_gpt.h"

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#if NCNN_LLM_WITH_OPENCV
#include <opencv2/core/core.hpp>
#endif

using nlohmann::json;

int run_cli(const Options& opt,
            ncnn_llm_gpt& model,
            const std::vector<json>& builtin_tools,
            const std::unordered_map<std::string, std::function<json(const json&)>>& builtin_router
#if NCNN_LLM_WITH_OPENCV
            , const cv::Mat& image = cv::Mat()
#endif
            );
