#include "youtu_app.h"

#include "youtu_cli.h"
#include "youtu_generation.h"
#include "youtu_model_paths.h"
#include "youtu_runner_common.h"
#include "youtu_tokenizer.h"
#include "youtu_vision.h"

#include <algorithm>
#include <exception>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace youtu {

// Top-level application orchestration.
//
// Selects tokenizer-only, preprocessing parity, text generation, multimodal
// generation, prefill comparison, or historical NPZ parity mode. Neural
// network details remain in the lower-level modules included by the entry TU.

int run_app(int argc, char** argv) {
    try {
        const Options opt = parse_args(argc, argv);

        if (opt.vision_preprocess_only) {
            if (opt.image_path.empty()) die("--vision-preprocess-only requires --image");
            const ImagePreprocessResult pp = preprocess_siglip2_image(opt.image_path, opt.image_patch_size, opt.max_image_patches);
            std::cout << "[image] path=" << opt.image_path
                      << " input=" << pp.input_width << "x" << pp.input_height
                      << " resized=" << pp.resized_width << "x" << pp.resized_height
                      << " patches=" << pp.num_patches
                      << " image_tokens=" << pp.image_token_count
                      << " pixel_values=" << shape_string(pp.pixel_values.shape)
                      << " spatial_shapes=[" << pp.spatial_shapes.i64[0] << ", " << pp.spatial_shapes.i64[1] << "]\n";
            if (!opt.vision_npz.empty()) {
                auto arrays = load_npz(opt.vision_npz);
                print_stats_line("preprocess.pixel_values", diff_stats(pp.pixel_values, require_const(arrays, "pixel_values")), opt.logits_atol);
                print_i64_compare_line("preprocess.pixel_attention_mask", pp.pixel_attention_mask, require_const(arrays, "pixel_attention_mask"));
                print_i64_compare_line("preprocess.spatial_shapes", pp.spatial_shapes, require_const(arrays, "spatial_shapes"));
            }
            return 0;
        }

        if (!opt.ids.empty() || !opt.prompt.empty() || opt.cache_from_npz || !opt.prefill_embeds_npz.empty() || !opt.vision_npz.empty() || !opt.image_path.empty()) {
            Tokenizer tokenizer = Tokenizer::load(opt.tokenizer_json);
            std::vector<int> prompt_ids;
            int start_position = 0;
            Tensor external_prefill_embeds;
            bool has_external_prefill_embeds = false;
            if (!opt.prefill_embeds_npz.empty() && !opt.vision_npz.empty()) {
                die("--prefill-embeds-npz and --vision-npz are mutually exclusive");
            }
            if (!opt.vision_npz.empty()) {
                auto arrays = load_npz(opt.vision_npz);
                const Tensor& input_ids = require_const(arrays, "input_ids");
                if (input_ids.i64.empty()) die("vision npz input_ids is empty");
                prompt_ids.reserve(input_ids.i64.size());
                for (int64_t id : input_ids.i64) prompt_ids.push_back(static_cast<int>(id));

                ImagePreprocessResult pp;
                if (!opt.image_path.empty()) {
                    pp = preprocess_siglip2_image(opt.image_path, opt.image_patch_size, opt.max_image_patches);
                    std::cout << "[image] path=" << opt.image_path
                              << " input=" << pp.input_width << "x" << pp.input_height
                              << " resized=" << pp.resized_width << "x" << pp.resized_height
                              << " patches=" << pp.num_patches
                              << " image_tokens=" << pp.image_token_count
                              << " spatial_shapes=[" << pp.spatial_shapes.i64[0] << ", " << pp.spatial_shapes.i64[1] << "]\n";
                    auto expected_pixels = arrays.find("pixel_values");
                    if (expected_pixels != arrays.end() && !expected_pixels->second.f32.empty()) {
                        print_stats_line("preprocess.pixel_values", diff_stats(pp.pixel_values, expected_pixels->second), opt.logits_atol);
                    }
                    auto expected_mask = arrays.find("pixel_attention_mask");
                    if (expected_mask != arrays.end()) {
                        print_i64_compare_line("preprocess.pixel_attention_mask", pp.pixel_attention_mask, expected_mask->second);
                    }
                    auto expected_shapes = arrays.find("spatial_shapes");
                    if (expected_shapes != arrays.end()) {
                        print_i64_compare_line("preprocess.spatial_shapes", pp.spatial_shapes, expected_shapes->second);
                    }
                } else {
                    pp.pixel_values = require(arrays, "pixel_values");
                    pp.spatial_shapes = require(arrays, "spatial_shapes");
                    pp.num_patches = pp.pixel_values.shape.size() >= 2 ? pp.pixel_values.shape[1] : 0;
                    pp.image_token_count = pp.num_patches / 4;
                }

                const auto vision_start = Clock::now();
                Tensor image_embeds;
                if (opt.fixed_vision_graph) {
                    image_embeds = run_vision_net(
                        vision_files(opt.vision_export_dir, opt.vision_stem),
                        std::move(pp.pixel_values),
                        opt.num_threads,
                        opt.no_packing_layout
                    );
                } else {
                    image_embeds = run_dynamic_vision_net(
                        dynamic_vision_files(opt.vision_export_dir, opt.vision_encoder_stem),
                        dynamic_vision_files(opt.vision_export_dir, opt.vision_post_merger_stem),
                        pp,
                        opt.num_threads,
                        opt.no_packing_layout
                    );
                }
                const double vision_ms = elapsed_ms(vision_start, Clock::now());
                if (has_nonfinite(image_embeds)) die("vision ncnn produced non-finite image_embeds");
                std::cout << "[vision] ncnn image_embeds=" << shape_string(image_embeds.shape)
                          << " ms=" << vision_ms << "\n";

                auto expected_it = arrays.find("image_embeds");
                if (expected_it != arrays.end() && !expected_it->second.f32.empty()) {
                    print_stats_line("vision.image_embeds", diff_stats(image_embeds, expected_it->second), opt.logits_atol);
                }

                external_prefill_embeds = merge_image_embeds_into_text(
                    require(arrays, "text_inputs_embeds"),
                    image_embeds,
                    require_const(arrays, "image_token_positions")
                );
                auto merged_it = arrays.find("merged_inputs_embeds");
                if (merged_it != arrays.end() && !merged_it->second.f32.empty()) {
                    print_stats_line("vision.merged_inputs_embeds", diff_stats(external_prefill_embeds, merged_it->second), opt.logits_atol);
                }
                if (opt.vision_only) {
                    std::cout << "[vision] completed without loading the text decoder\n";
                    return 0;
                }
                has_external_prefill_embeds = true;
            } else if (!opt.image_path.empty()) {
                if (opt.prompt.empty()) die("--image without --vision-npz requires --prompt");
                const ImagePreprocessResult pp = preprocess_siglip2_image(opt.image_path, opt.image_patch_size, opt.max_image_patches);
                std::cout << "[image] path=" << opt.image_path
                          << " input=" << pp.input_width << "x" << pp.input_height
                          << " resized=" << pp.resized_width << "x" << pp.resized_height
                          << " patches=" << pp.num_patches
                          << " image_tokens=" << pp.image_token_count
                          << " spatial_shapes=[" << pp.spatial_shapes.i64[0] << ", " << pp.spatial_shapes.i64[1] << "]\n";

                const std::string prompt_text = opt.chat ? make_image_chat_prompt(opt.prompt, pp.image_token_count) : opt.prompt;
                prompt_ids = tokenizer.encode_with_specials(prompt_text);
                const int image_token_id = tokenizer.token_id("<|image_pad|>");
                Tensor image_positions = image_positions_from_ids(prompt_ids, image_token_id);
                if (static_cast<int>(image_positions.i64.size()) != pp.image_token_count) {
                    die("image token count mismatch prompt_tokens=" + std::to_string(image_positions.i64.size()) +
                        " image_embeds=" + std::to_string(pp.image_token_count));
                }

                const auto vision_start = Clock::now();
                Tensor image_embeds = run_dynamic_vision_net(
                    dynamic_vision_files(opt.vision_export_dir, opt.vision_encoder_stem),
                    dynamic_vision_files(opt.vision_export_dir, opt.vision_post_merger_stem),
                    pp,
                    opt.num_threads,
                    opt.no_packing_layout
                );
                const double vision_ms = elapsed_ms(vision_start, Clock::now());
                if (has_nonfinite(image_embeds)) die("vision ncnn produced non-finite image_embeds");
                std::cout << "[vision] ncnn image_embeds=" << shape_string(image_embeds.shape)
                          << " ms=" << vision_ms << "\n";

                Tensor text_embeds = run_embed_sequence(
                    embed_files(opt.export_dir, opt.precision),
                    prompt_ids,
                    opt.num_threads,
                    opt.no_packing_layout,
                    opt.precision == "fp16"
                );
                external_prefill_embeds = merge_image_embeds_into_text(std::move(text_embeds), image_embeds, image_positions);
                has_external_prefill_embeds = true;
            } else if (!opt.prefill_embeds_npz.empty()) {
                auto arrays = load_npz(opt.prefill_embeds_npz);
                const Tensor& input_ids = require_const(arrays, "input_ids");
                if (input_ids.i64.empty()) die("prefill embeds npz input_ids is empty");
                prompt_ids.reserve(input_ids.i64.size());
                for (int64_t id : input_ids.i64) prompt_ids.push_back(static_cast<int>(id));
                external_prefill_embeds = require(arrays, "merged_inputs_embeds");
                if (external_prefill_embeds.f32.empty()) die("merged_inputs_embeds is empty");
                has_external_prefill_embeds = true;
            } else if (opt.cache_from_npz && opt.ids.empty() && opt.prompt.empty()) {
                auto arrays = load_npz(opt.npz);
                const Tensor& input_ids = require_const(arrays, "decoder_decode_step0_input_ids");
                const Tensor& cache_pos = require_const(arrays, "decoder_decode_step0_cache_position");
                if (input_ids.i64.empty() || cache_pos.i64.empty()) die("missing input ids or cache position");
                prompt_ids.push_back(static_cast<int>(input_ids.i64[0]));
                start_position = static_cast<int>(cache_pos.i64[0]);
            } else if (!opt.ids.empty()) {
                prompt_ids = parse_id_list(opt.ids);
            } else {
                std::string prompt_text = opt.prompt;
                if (opt.chat) {
                    prompt_text = "<|begin_of_text|>system\nYou are a helpful assistant.<|end_of_text|>\n"
                                  "<|begin_of_text|>user\n" + prompt_text + "<|end_of_text|>\n"
                                  "<|begin_of_text|>assistant\n";
                }
                prompt_ids = tokenizer.encode_with_specials(prompt_text);
            }
            if (prompt_ids.empty()) die("empty prompt ids");
            if (opt.tokenize_only) {
                std::cout << "[ids] prompt=" << join_ids(prompt_ids) << "\n";
                std::cout << "[text] " << tokenizer.decode(prompt_ids) << "\n";
                return 0;
            }
            if (opt.compare_prefill) {
                if (opt.cache_from_npz) die("--compare-prefill does not support --cache-from-npz");
                RunnerContext dynamic_runner(opt.export_dir, opt.num_threads, opt.no_packing_layout, parse_int_list(opt.prefill_buckets), opt.precision);
                if (!dynamic_runner.has_dynamic_full_decoder_prefill()) {
                    die("dynamic full decoder prefill param/bin not found");
                }
                const auto dynamic_start = Clock::now();
                Tensor dynamic_logits = dynamic_runner.run_full_decoder_prefill(prompt_ids, start_position);
                const double dynamic_ms = elapsed_ms(dynamic_start, Clock::now());

                RunnerContext token_runner(opt.export_dir, opt.num_threads, opt.no_packing_layout, parse_int_list(opt.prefill_buckets), opt.precision);
                token_runner.init_dummy_cache();
                Tensor token_logits;
                const auto token_start = Clock::now();
                for (size_t i = 0; i < prompt_ids.size(); ++i) {
                    const bool first_dummy_token = i == 0;
                    token_logits = token_runner.run_full_decoder_token(prompt_ids[i], start_position + static_cast<int>(i), first_dummy_token);
                    if (first_dummy_token) token_runner.drop_dummy_cache_prefix();
                }
                const double token_ms = elapsed_ms(token_start, Clock::now());

                const DiffStats logits_stats = diff_stats(dynamic_logits, token_logits);
                const bool logits_ok = print_stats_line("prefill.logits.dynamic_vs_token", logits_stats, opt.logits_atol);
                const int dynamic_next = argmax_id(dynamic_logits);
                const int token_next = argmax_id(token_logits);
                const bool next_ok = dynamic_next == token_next;
                std::cout << "[" << (next_ok ? "PASS" : "FAIL") << "] prefill.next_id dynamic="
                          << dynamic_next << " token=" << token_next << "\n";
                std::cout << "[info] prefill.top" << opt.topk
                          << " dynamic=" << join_ids(topk_ids(dynamic_logits, opt.topk))
                          << " token=" << join_ids(topk_ids(token_logits, opt.topk)) << "\n";
                const bool cache_ok = compare_cache_sets(dynamic_runner, token_runner, opt.cache_atol);
                std::cout << "[time] prefill.dynamic_ms=" << dynamic_ms
                          << " token_ms=" << token_ms
                          << " speedup=" << (dynamic_ms > 0.0 ? token_ms / dynamic_ms : 0.0) << "\n";
                std::cout << "[compare] tokens=" << prompt_ids.size() << " dynamic_seq=1"
                          << " logits_atol=" << opt.logits_atol
                          << " cache_atol=" << opt.cache_atol << "\n";
                return (logits_ok && next_ok && cache_ok) ? 0 : 1;
            }

            RunnerContext runner(opt.export_dir, opt.num_threads, opt.no_packing_layout, parse_int_list(opt.prefill_buckets), opt.precision);
            if (opt.cache_from_npz) {
                auto arrays = load_npz(opt.npz);
                runner.init_cache_from_dump(arrays);
                if (start_position == 0) {
                    const Tensor& cache_pos = require_const(arrays, "decoder_decode_step0_cache_position");
                    if (cache_pos.i64.empty()) die("missing cache position");
                    start_position = static_cast<int>(cache_pos.i64[0]);
                }
            } else {
                runner.init_dummy_cache();
            }
            Tensor logits;
            if (has_external_prefill_embeds) {
                if (!opt.full_decoder_ncnn) die("--prefill-embeds-npz requires --full-decoder-ncnn");
                const auto prefill_start = Clock::now();
                const bool dynamic_prefill = runner.has_dynamic_full_decoder_prefill();
                logits = dynamic_prefill
                    ? runner.run_full_decoder_prefill_embeds(std::move(external_prefill_embeds), start_position)
                    : runner.run_full_decoder_prefill_embeds_tokenwise(std::move(external_prefill_embeds), start_position);
                const double prefill_ms = elapsed_ms(prefill_start, Clock::now());
                std::cout << "[prefill] external_merged_inputs_embeds tokens=" << prompt_ids.size()
                          << " mode=" << (dynamic_prefill ? "dynamic_graph" : "tokenwise_decode_graph")
                          << " ms=" << prefill_ms << "\n";
            } else if (opt.full_prefill_ncnn && opt.full_decoder_ncnn && !opt.cache_from_npz) {
                if (runner.has_dynamic_full_decoder_prefill()) {
                    const auto prefill_start = Clock::now();
                    logits = runner.run_full_decoder_prefill(prompt_ids, start_position);
                    const double prefill_ms = elapsed_ms(prefill_start, Clock::now());
                    std::cout << "[prefill] full_decoder_prefill_ncnn tokens=" << prompt_ids.size()
                              << " dynamic_seq=1"
                              << " ms=" << prefill_ms << "\n";
                } else {
                    for (size_t i = 0; i < prompt_ids.size(); ++i) {
                        const bool first_dummy_token = i == 0;
                        logits = runner.run_full_decoder_token(prompt_ids[i], start_position + static_cast<int>(i), first_dummy_token);
                        if (first_dummy_token) runner.drop_dummy_cache_prefix();
                        std::cout << "[prefill] full_decoder_ncnn " << (i + 1) << "/" << prompt_ids.size()
                                  << " token_id=" << prompt_ids[i] << "\n";
                    }
                }
            } else if (opt.full_prefill_ncnn && !opt.cache_from_npz) {
                const int bucket = runner.select_prefill_bucket(static_cast<int>(prompt_ids.size()));
                if (bucket < 0) {
                    die("no usable full prefill bucket for prompt length " + std::to_string(prompt_ids.size()) +
                        "; pass --prefill-buckets with an exported bucket or omit --full-prefill-ncnn");
                }
                logits = runner.run_full_prefill(prompt_ids, start_position, bucket);
                std::cout << "[prefill] full_ncnn tokens=" << prompt_ids.size() << " bucket=" << bucket << "\n";
            } else {
                for (size_t i = 0; i < prompt_ids.size(); ++i) {
                    const bool first_dummy_token = !opt.cache_from_npz && i == 0;
                    if (opt.full_decoder_ncnn) {
                        logits = runner.run_full_decoder_token(prompt_ids[i], start_position + static_cast<int>(i), first_dummy_token);
                    } else {
                        logits = runner.run_token(prompt_ids[i], start_position + static_cast<int>(i), first_dummy_token);
                    }
                    if (first_dummy_token) runner.drop_dummy_cache_prefix();
                    std::cout << "[prefill] " << (i + 1) << "/" << prompt_ids.size() << " token_id=" << prompt_ids[i] << "\n";
                }
            }

            std::vector<int> generated;
            for (int step = 0; step < opt.max_new_tokens; ++step) {
                const int next_id = argmax_id(logits);
                generated.push_back(next_id);
                std::cout << "[decode] step=" << step << " token_id=" << next_id
                          << " text=\"" << tokenizer.decode({next_id}) << "\"\n";
                if (next_id == tokenizer.eos_id()) break;
                if (opt.full_decoder_ncnn) {
                    logits = runner.run_full_decoder_token(next_id, start_position + static_cast<int>(prompt_ids.size() + generated.size() - 1));
                } else {
                    logits = runner.run_token(next_id, start_position + static_cast<int>(prompt_ids.size() + generated.size() - 1));
                }
            }

            std::vector<int> all = prompt_ids;
            all.insert(all.end(), generated.begin(), generated.end());
            std::cout << "[ids] prompt=" << join_ids(prompt_ids) << "\n";
            std::cout << "[ids] generated=" << join_ids(generated) << "\n";
            const std::string final_text = opt.echo_prompt
                ? tokenizer.decode(all)
                : tokenizer.decode(generated);
            std::cout << "[text] " << final_text << "\n";
            if (!opt.output_path.empty()) {
                std::ofstream output(opt.output_path, std::ios::binary);
                if (!output) die("failed to open output file " + opt.output_path);
                output.write(final_text.data(), static_cast<std::streamsize>(final_text.size()));
            }
            return 0;
        }

        auto arrays = load_npz(opt.npz);
        RunnerContext runner(opt.export_dir, opt.num_threads, opt.no_packing_layout, parse_int_list(opt.prefill_buckets), opt.precision);
        runner.init_cache_from_dump(arrays);

        const Tensor& input_ids = require_const(arrays, "decoder_decode_step0_input_ids");
        const Tensor& cache_pos = require_const(arrays, "decoder_decode_step0_cache_position");
        const Tensor& expected_logits_all = require_const(arrays, "decode_step_logits");
        const Tensor& expected_next = require_const(arrays, "decode_next_ids");
        if (input_ids.i64.empty() || cache_pos.i64.empty()) die("missing input ids or cache position");
        int token_id = static_cast<int>(input_ids.i64[0]);
        const int prompt_len = static_cast<int>(cache_pos.i64[0]);
        const int steps = std::min(opt.steps, expected_logits_all.shape.empty() ? 0 : expected_logits_all.shape[0]);

        std::cout << "[data] npz=" << opt.npz << " steps=" << steps << "\n";
        std::cout << "[ncnn] export_dir=" << opt.export_dir << "\n";

        bool ok = true;
        std::vector<int> produced;
        for (int step = 0; step < steps; ++step) {
            Tensor logits = opt.full_decoder_ncnn
                ? runner.run_full_decoder_token(token_id, prompt_len + step)
                : runner.run_token(token_id, prompt_len + step);

            Tensor expected_step;
            expected_step.shape = {1, expected_logits_all.shape[1]};
            expected_step.f32.assign(
                expected_logits_all.f32.begin() + static_cast<size_t>(step) * expected_logits_all.shape[1],
                expected_logits_all.f32.begin() + static_cast<size_t>(step + 1) * expected_logits_all.shape[1]
            );

            const bool logits_ok = print_stats_line("step" + std::to_string(step) + ".logits", diff_stats(logits, expected_step), opt.logits_atol);
            const int next_id = argmax_id(logits);
            const int expected_id = static_cast<int>(expected_next.i64[step]);
            const bool next_ok = next_id == expected_id;
            const std::vector<int> actual_top = topk_ids(logits, opt.topk);
            const std::vector<int> expected_top = topk_ids(expected_step, opt.topk);
            std::cout << "[" << (next_ok ? "PASS" : "FAIL") << "] step" << step << ".next_id actual=" << next_id << " expected=" << expected_id << "\n";
            std::cout << "[info] step" << step << ".top" << opt.topk << " actual=" << join_ids(actual_top) << " expected=" << join_ids(expected_top) << "\n";
            produced.push_back(next_id);
            token_id = next_id;
            ok = logits_ok && next_ok && ok;
        }

        std::vector<int> expected_seq;
        for (int i = 0; i < steps; ++i) expected_seq.push_back(static_cast<int>(expected_next.i64[i]));
        const bool seq_ok = produced == expected_seq;
        std::cout << "[" << (seq_ok ? "PASS" : "FAIL") << "] decode_next_ids actual=" << join_ids(produced) << " expected=" << join_ids(expected_seq) << "\n";
        ok = seq_ok && ok;
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }
}

} // namespace youtu
