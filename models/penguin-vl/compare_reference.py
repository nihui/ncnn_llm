#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# Produce the PyTorch "golden" reference for a fixed (prompt, image) pair so the
# ncnn port can be checked for exact text equality (the core issue requirement:
# "鍦ㄧ浉鍚岃緭鍏ヤ笅锛宯cnn 杈撳嚭鐨勬渶缁堟枃鏈』涓?PyTorch 鍘熺増涓€鑷?).
#
# It runs greedy decoding (do_sample=False) and also dumps intermediate tensors
# used by docs/ALIGNMENT.md (pixel_values, vision features, first logits).
#
# Usage:
#   python tools/compare_reference.py --model tencent/Penguin-VL-2B \
#       --image assets/demo.jpg --prompt "Describe this image." --out golden/

import argparse
import json
import os

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", default=None)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", default="golden")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from penguinvl import disable_torch_init, model_init, mm_infer
    disable_torch_init()
    model, processor = model_init(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32)

    modal = "image" if args.image else "text"
    if args.image:
        images = processor.load_images(args.image)
        image_inputs = processor.process_images(images, merge_size=1, return_tensors="pt")
        content = [{"type": "image"}, {"type": "text", "text": args.prompt}]
        np.save(os.path.join(args.out, "pixel_values.npy"),
                image_inputs["pixel_values"].float().cpu().numpy())
        np.save(os.path.join(args.out, "grid_sizes.npy"),
                image_inputs["grid_sizes"].cpu().numpy())
    else:
        image_inputs = {}
        content = args.prompt

    prompt = processor.apply_chat_template(
        conversation=[{"role": "user", "content": content}],
        tokenize=False, add_system_prompt=True, add_generation_prompt=True)
    with open(os.path.join(args.out, "prompt.txt"), "w", encoding="utf-8") as f:
        f.write(prompt)

    text_inputs = processor.process_text(text=prompt, image_inputs=image_inputs, return_tensors="pt")

    output = mm_infer({**text_inputs, **image_inputs}, model=model,
                      tokenizer=processor.tokenizer, do_sample=False, modal=modal,
                      max_new_tokens=args.max_new_tokens)

    with open(os.path.join(args.out, "output.txt"), "w", encoding="utf-8") as f:
        f.write(output)
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "prompt": args.prompt, "image": args.image,
                   "modal": modal, "greedy": True}, f, indent=2, ensure_ascii=False)
    print("=== GOLDEN OUTPUT ===")
    print(output)
    print(f"[ok] wrote reference to {args.out}/")


if __name__ == "__main__":
    main()
