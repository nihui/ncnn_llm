#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "youtu_runner_common.h"

namespace youtu {

// Owns loaded text networks and the mutable 40-layer KV cache.
class RunnerContext {
public:
    RunnerContext(const std::string& dir, int num_threads, bool no_packing_layout,
                  std::vector<int> buckets, const std::string& precision = "fp32");
    ~RunnerContext();
    RunnerContext(RunnerContext&&) noexcept;
    RunnerContext& operator=(RunnerContext&&) noexcept;
    RunnerContext(const RunnerContext&) = delete;
    RunnerContext& operator=(const RunnerContext&) = delete;

    void init_empty_cache();
    void init_dummy_cache();
    void drop_dummy_cache_prefix();
    void init_cache_from_dump(std::map<std::string, Tensor>& arrays);
    Tensor run_full_decoder_token(int token_id, int position, bool mask_first_cache_slot = false);
    Tensor run_token(int token_id, int position, bool mask_first_cache_slot = false);
    int select_prefill_bucket(int prompt_length) const;
    bool has_dynamic_full_decoder_prefill() const;
    Tensor run_full_decoder_prefill_embeds(Tensor embeds, int start_position);
    Tensor run_full_decoder_prefill_embeds_tokenwise(Tensor embeds, int start_position);
    Tensor run_full_decoder_prefill(const std::vector<int>& token_ids, int start_position);
    Tensor run_full_prefill(const std::vector<int>& token_ids, int start_position, int bucket_length);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend bool compare_cache_sets(const RunnerContext&, const RunnerContext&, float);
};

bool compare_cache_sets(const RunnerContext& actual, const RunnerContext& expected, float atol);

} // namespace youtu
