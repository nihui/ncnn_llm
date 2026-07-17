#include "youtu_cli.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace youtu {

// Command-line parsing and defaults.
//
// Parsing is kept separate from execution so model orchestration can evolve
// without spreading option handling through the inference code.

Options parse_args(int argc, char** argv) {
    Options opt;
    opt.root = discover_project_root(argc > 0 ? argv[0] : nullptr);
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) die("missing value for " + a);
            return argv[++i];
        };
        if (a == "--root") opt.root = next();
        else if (a == "--precision") opt.precision = next();
        else if (a == "--npz") opt.npz = next();
        else if (a == "--export-dir") opt.export_dir = next();
        else if (a == "--vision-export-dir") opt.vision_export_dir = next();
        else if (a == "--vision-stem") opt.vision_stem = next();
        else if (a == "--vision-encoder-stem") opt.vision_encoder_stem = next();
        else if (a == "--vision-post-merger-stem") opt.vision_post_merger_stem = next();
        else if (a == "--tokenizer") opt.tokenizer_json = next();
        else if (a == "--prefill-embeds-npz") opt.prefill_embeds_npz = next();
        else if (a == "--vision-npz") opt.vision_npz = next();
        else if (a == "--image") opt.image_path = next();
        else if (a == "--prompt") opt.prompt = next();
        else if (a == "--output") opt.output_path = next();
        else if (a == "--ids") opt.ids = next();
        else if (a == "--steps") opt.steps = std::stoi(next());
        else if (a == "--max-new-tokens") opt.max_new_tokens = std::stoi(next());
        else if (a == "--image-patch-size") opt.image_patch_size = std::stoi(next());
        else if (a == "--max-image-patches") opt.max_image_patches = std::stoi(next());
        else if (a == "--threads") opt.num_threads = std::stoi(next());
        else if (a == "--packing-layout") opt.no_packing_layout = false;
        else if (a == "--logits-atol") opt.logits_atol = std::stof(next());
        else if (a == "--cache-atol") opt.cache_atol = std::stof(next());
        else if (a == "--topk") opt.topk = std::stoi(next());
        else if (a == "--no-chat") opt.chat = false;
        else if (a == "--echo-prompt") opt.echo_prompt = true;
        else if (a == "--cache-from-npz") opt.cache_from_npz = true;
        else if (a == "--tokenize-only") opt.tokenize_only = true;
        else if (a == "--vision-preprocess-only") opt.vision_preprocess_only = true;
        else if (a == "--vision-only") opt.vision_only = true;
        else if (a == "--fixed-vision-graph") opt.fixed_vision_graph = true;
        else if (a == "--compare-prefill") opt.compare_prefill = true;
        else if (a == "--full-prefill-ncnn") opt.full_prefill_ncnn = true;
        else if (a == "--full-decoder-ncnn") opt.full_decoder_ncnn = true;
        else if (a == "--prefill-buckets") opt.prefill_buckets = next();
        else if (a == "--help") {
            std::cout << "Usage:\n"
                      << "  parity:   youtu_vl_main [--steps N] [--npz FILE]\n"
                      << "  generate: youtu_vl_main --ids 128000,882,... [--max-new-tokens N]\n"
                      << "            youtu_vl_main --cache-from-npz [--max-new-tokens N]\n"
                      << "            youtu_vl_main --prompt \"hello\" [--max-new-tokens N] [--no-chat]\n"
                      << "            youtu_vl_main --prompt \"hello\" --full-prefill-ncnn --full-decoder-ncnn\n"
                      << "            youtu_vl_main --prefill-embeds-npz vision_embeds.npz --full-decoder-ncnn\n"
                      << "            youtu_vl_main --vision-npz vision_embeds.npz [--image image.jpg] --full-decoder-ncnn\n"
                      << "            youtu_vl_main --image image.jpg --prompt \"describe\" --full-decoder-ncnn\n"
                      << "            youtu_vl_main --image image.jpg --prompt \"describe\" --output result.txt\n"
                      << "            youtu_vl_main --image image.jpg --vision-preprocess-only\n"
                      << "            youtu_vl_main --vision-npz vision_embeds.npz --vision-only\n"
                      << "            youtu_vl_main --ids 128000,198 --compare-prefill\n"
                      << "            youtu_vl_main --prompt \"hello\" --tokenize-only\n"
                      << "  precision: --precision fp32|fp16 (FP16 weights use FP32 activations)\n";
            std::exit(0);
        } else {
            die("unknown arg " + a);
        }
    }
    opt.root = fs::path(opt.root).lexically_normal().string();
    if (opt.precision != "fp32" && opt.precision != "fp16") {
        die("--precision must be fp32 or fp16");
    }
    if (opt.steps < 0) die("--steps must be non-negative");
    if (opt.max_new_tokens < 0) die("--max-new-tokens must be non-negative");
    if (opt.image_patch_size <= 0) die("--image-patch-size must be positive");
    if (opt.max_image_patches <= 0) die("--max-image-patches must be positive");
    if (opt.num_threads <= 0) die("--threads must be positive");
    if (opt.topk <= 0) die("--topk must be positive");
    if (opt.logits_atol < 0.0f) die("--logits-atol must be non-negative");
    if (opt.cache_atol < 0.0f) die("--cache-atol must be non-negative");
    const fs::path package_root = runtime_package_root(opt.root);
    if (opt.npz.empty()) opt.npz = project_path(opt.root, "assets/youtu_text_test/dumps/en_short.npz");
    if (opt.export_dir.empty()) {
        opt.export_dir = package_root.empty()
            ? project_path(opt.root, opt.precision == "fp32" ? "assets/youtu_text_export" : "assets/youtu_text_export/" + opt.precision)
            : (package_root / "models" / "text").string();
    }
    if (opt.vision_export_dir.empty()) {
        opt.vision_export_dir = package_root.empty()
            ? project_path(opt.root, opt.precision == "fp32" ? "assets/youtu_vl_export" : "assets/youtu_vl_export/" + opt.precision)
            : (package_root / "models" / "vision").string();
    }
    if (opt.tokenizer_json.empty()) {
        opt.tokenizer_json = package_root.empty()
            ? project_path(opt.root, "assets/hf/Youtu-VL-4B-Instruct/tokenizer.json")
            : (package_root / "tokenizer" / "tokenizer.json").string();
    }
    return opt;
}

} // namespace youtu
