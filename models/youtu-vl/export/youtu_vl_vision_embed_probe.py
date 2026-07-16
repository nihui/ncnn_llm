#!/usr/bin/env python3
"""Dump Youtu-VL image-side embeddings before LLM prefill.

This isolates the first vision integration boundary:

  processor image tensors -> siglip2 -> merger -> masked_scatter into token embeddings

The resulting merged_inputs_embeds can later be fed to the text decoder path,
while siglip2/merger can be exported and checked independently.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from youtu_vl_reference import (
    DEFAULT_MANIFEST,
    build_messages,
    die,
    first_parameter_device,
    get_required_inputs,
    iter_samples,
    load_json,
    load_model_and_processor,
    maybe_download_image,
    tensor_to_numpy,
    to_model_device,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "assets" / "youtu_vl_test" / "vision_embeds"


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == ROOT.name:
        return (ROOT.parent / path).resolve()
    return (ROOT / path).resolve()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_npz(np, path: Path, arrays: dict[str, Any], compress: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)


def token_positions(torch, input_ids, token_id: int) -> list[int]:
    return torch.nonzero(input_ids[0] == int(token_id), as_tuple=False).reshape(-1).detach().cpu().tolist()


def merge_image_embeddings(torch, model, input_ids, inputs_embeds, image_embeds):
    image_token_id = int(model.config.image_token_id)
    mask = input_ids == image_token_id
    image_token_count = int(mask.sum().item())
    image_feature_count = int(image_embeds.shape[0])
    if image_token_count > image_feature_count:
        die(f"image token/features mismatch: tokens={image_token_count} features={image_feature_count}")
    if int(input_ids.shape[0]) != 1:
        die("only batch size 1 is supported")
    image_mask = mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
    image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype).unsqueeze(0)
    return inputs_embeds.masked_scatter(image_mask, image_embeds), image_token_count, image_feature_count


def collect_one(np, torch, model, processor, image_info: dict[str, Any], prompt_info: dict[str, Any], image_path: Path, args):
    image = Image.open(image_path).convert("RGB")
    messages = build_messages(image_path, prompt_info["text"])
    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=prompt_text,
        images=image,
        return_tensors="pt",
        max_image_patches=args.max_image_patches,
    )

    input_device = first_parameter_device(model)
    inputs = to_model_device(inputs, input_device)
    model_inputs = dict(inputs)
    required = get_required_inputs(
        model_inputs,
        (
            "input_ids",
            "attention_mask",
            "pixel_values",
            "pixel_attention_mask",
            "spatial_shapes",
        ),
    )
    input_ids = required["input_ids"]
    inputs_embeds = model.model.embed_tokens(input_ids)
    pixel_values = required["pixel_values"].type(model.siglip2.dtype)

    with torch.inference_mode():
        vision_outputs = model.siglip2(
            pixel_values,
            required.get("pixel_attention_mask"),
            required.get("spatial_shapes"),
        )
        vision_last_hidden_state = vision_outputs.last_hidden_state
        image_embeds = model.merger(vision_last_hidden_state, required["spatial_shapes"])
        merged_inputs_embeds, image_token_count, image_feature_count = merge_image_embeddings(
            torch,
            model,
            input_ids,
            inputs_embeds,
            image_embeds,
        )

    image_token_id = int(model.config.image_token_id)
    arrays: dict[str, Any] = {
        "input_ids": tensor_to_numpy(np, torch, input_ids),
        "attention_mask": tensor_to_numpy(np, torch, model_inputs.get("attention_mask")),
        "pixel_values": tensor_to_numpy(np, torch, required["pixel_values"]),
        "pixel_attention_mask": tensor_to_numpy(np, torch, required.get("pixel_attention_mask")),
        "spatial_shapes": tensor_to_numpy(np, torch, required["spatial_shapes"]),
        "text_inputs_embeds": tensor_to_numpy(np, torch, inputs_embeds),
        "vision_last_hidden_state": tensor_to_numpy(np, torch, vision_last_hidden_state),
        "image_embeds": tensor_to_numpy(np, torch, image_embeds),
        "merged_inputs_embeds": tensor_to_numpy(np, torch, merged_inputs_embeds),
        "image_token_positions": np.asarray(token_positions(torch, input_ids, image_token_id), dtype=np.int64),
    }
    arrays = {k: v for k, v in arrays.items() if v is not None}
    meta = {
        "sample_id": f"{image_info['id']}:{prompt_info['id']}",
        "image_id": image_info["id"],
        "prompt_id": prompt_info["id"],
        "image_path": str(image_path),
        "prompt": prompt_info["text"],
        "prompt_text": prompt_text,
        "image_token_id": image_token_id,
        "image_token_count": image_token_count,
        "image_feature_count": image_feature_count,
        "image_token_feature_match": image_token_count <= image_feature_count,
        "array_shapes": {k: list(v.shape) for k, v in arrays.items() if hasattr(v, "shape")},
        "array_dtypes": {k: str(v.dtype) for k, v in arrays.items() if hasattr(v, "dtype")},
    }
    return arrays, meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump Youtu-VL vision embeddings and merged LLM input embeddings.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--model", default="")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--max-image-patches", type=int, default=2048)
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device-map", default="none")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--download-missing-images", action="store_true")
    parser.add_argument("--no-compress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = resolve_path(args.manifest)
    if not manifest_path.exists():
        die(f"manifest not found: {manifest_path}")
    manifest = load_json(manifest_path)
    samples = list(iter_samples(manifest, args.only))
    if not samples:
        die(f"no samples matched --only={args.only}" if args.only else "manifest has no runnable samples")

    np, torch, transformers, model_path, model, processor, model_device, pydensecrf_stubbed = load_model_and_processor(args, manifest)
    output_dir = resolve_path(args.output_dir)
    test_root = manifest_path.parent

    metadata: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "manifest": str(manifest_path),
        "model": {
            "path_or_id": str(model_path),
            "image_token_id": int(getattr(model.config, "image_token_id", -1)),
            "vision_hidden_size": int(getattr(model.config.vision_config, "hidden_size", 0)),
            "hidden_size": int(getattr(model.config, "hidden_size", 0)),
        },
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "model_device": str(model_device),
            "dtype_arg": args.dtype,
            "pydensecrf_stubbed": pydensecrf_stubbed,
        },
        "args": {
            "max_image_patches": args.max_image_patches,
            "attn_implementation": args.attn_implementation,
        },
        "samples": [],
    }

    for image_info, prompt_info in samples:
        sample_name = f"{image_info['id']}_{prompt_info['id']}"
        image_path = (test_root / image_info["filename"]).resolve()
        maybe_download_image(image_info, image_path, args.download_missing_images)
        print(f"[sample] {image_info['id']}:{prompt_info['id']}")
        arrays, sample_meta = collect_one(np, torch, model, processor, image_info, prompt_info, image_path, args)
        npz_path = output_dir / f"{sample_name}.npz"
        save_npz(np, npz_path, arrays, compress=not args.no_compress)
        sample_meta["npz_path"] = str(npz_path)
        metadata["samples"].append(sample_meta)
        print(f"[write] {npz_path}")
        print(f"[shape] image_embeds={sample_meta['array_shapes'].get('image_embeds')} merged_inputs_embeds={sample_meta['array_shapes'].get('merged_inputs_embeds')}")

    metadata_path = output_dir / "metadata.json"
    write_json(metadata_path, metadata)
    print(f"[write] {metadata_path}")


if __name__ == "__main__":
    main()
