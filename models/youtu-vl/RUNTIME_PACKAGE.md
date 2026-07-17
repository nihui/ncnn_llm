# Youtu-VL-4B-Instruct ncnn runtime package

Runtime-only pnnx/ncnn assets for the official
`tencent/Youtu-VL-4B-Instruct` checkpoint pinned at
`8d30a0e49662a1d628a472b12df264dbcd768753`.

The package contains the SigLIP2 vision encoder, projector/merger, YoutuLLM
embedding, 40-layer decoder with KV cache, final norm, LM head, and tokenizer.
Inference uses the C++ `youtu_vl_main` executable and does not require Python,
PyTorch, Transformers, or the original safetensors.

Example after building the `model/youtu-vl` branch:

```bash
youtu_vl_main \
  --root /path/to/this/package \
  --precision fp16 \
  --image testdata/coco_cats.jpg \
  --prompt "Describe this image in one sentence." \
  --full-decoder-ncnn \
  --max-new-tokens 3
```

The pinned regression fixture produces token IDs `10392, 5240, 1896` and the
strict final text `Two tabby` in both the PyTorch baseline and ncnn runtime.
See `manifest.json` for per-file sizes and SHA-256 digests.
