# Youtu-VL-4B-Instruct on ncnn

Independent implementation for [Tencent/ncnn#6789](https://github.com/Tencent/ncnn/issues/6789)
on branch `model/youtu-vl`.  It reuses the shared `ncnn_llm` tokenizer,
embedding, KV-cache, sampling and CPU backend, and adds the Youtu-VL image
preprocessor, SigLIP2 vision encoder, merger/projector and 40-layer YoutuLLM
decoder graph.

## Fixed checkpoint and package

- Source: `tencent/Youtu-VL-4B-Instruct`
- Revision: `8d30a0e49662a1d628a472b12df264dbcd768753`
- Source hashes: [`checkpoint.json`](checkpoint.json)
- Runtime package (pending upload and public verification):
  <https://huggingface.co/March-77/Youtu-VL-4B-Instruct-ncnn>

The runtime package is assembled by `package_runtime.py`, contains only ncnn
graphs, tokenizer data and the pinned test fixture, and carries a SHA-256
manifest.  The executable does not load PyTorch, Python, Transformers or the
original safetensors.

## Reproduce conversion

The complete conversion deliberately runs the 17 GB decoder trace and pnnx in
separate processes.  This keeps the peak within a 24 GB WSL machine plus swap;
running pnnx while the 4B PyTorch checkpoint is still resident is not viable.

```bash
python models/youtu-vl/convert.py \
  --checkpoint /path/to/Youtu-VL-4B-Instruct \
  --output build/models/youtu-vl-4b-ncnn \
  --work build/conversion/youtu-vl \
  --pnnx /path/to/pnnx \
  --python /path/to/python \
  --device cuda:0 \
  --precision fp16

python models/youtu-vl/package_runtime.py \
  --conversion-output build/models/youtu-vl-4b-ncnn \
  --output build/packages/Youtu-VL-4B-Instruct-ncnn
```

`youtu_vl_export.py` forces BF16 embedding weights through an FP32 trace before
pnnx FP16 storage and verifies the exact binary payload size.  This catches a
real failure mode where pnnx exits successfully but writes a half-sized Embed
payload.  `youtu_full_decoder_pnnx.py` rejects residual `pnnx.*`, `Tensor.*`
or `aten::` operators after graph patching.

## Build

Linux/WSL:

```bash
cmake -S . -B build/youtu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DNCNN_SOURCE_DIR=/path/to/ncnn \
  -DNCNN_LLM_FETCH_DEPS=ON \
  -DNCNN_LLM_ENABLE_VULKAN=OFF
cmake --build build/youtu -j
ctest --test-dir build/youtu --output-on-failure
```

The same CMake project builds on Windows with LLVM-MinGW or MSVC.  The committed
evidence records the exact compiler, build, CTest and inference commands.

## Run and compare

Runtime-only image inference:

```bash
build/youtu/youtu_vl_main \
  --root /path/to/Youtu-VL-4B-Instruct-ncnn \
  --precision fp16 \
  --image models/youtu-vl/testdata/coco_cats.jpg \
  --prompt "Describe this image in one sentence." \
  --full-decoder-ncnn \
  --max-new-tokens 3 \
  --output actual.txt

python models/youtu-vl/verify_text.py --actual actual.txt
```

For a boundary-level regression using the saved official tensors:

```bash
build/youtu/youtu_vl_main \
  --root /path/to/Youtu-VL-4B-Instruct-ncnn \
  --precision fp16 \
  --vision-npz /path/to/testdata/vision-boundary/coco_cats_describe.npz \
  --full-decoder-ncnn \
  --max-new-tokens 3
```

The production fallback reuses the single-token 40-layer decoder for multimodal
prefill when no separate dynamic prefill graph is installed.  It builds the
same causal KV cache token by token and projects logits only for the final
prefill token.

## Fresh regression result

| Runtime | Generated IDs | Final text | Result |
|---|---:|---|---|
| PyTorch 2.11 / Transformers 4.57.3 | `10392, 5240, 1896` | `Two tabby` | baseline |
| ncnn Linux x86_64 | `10392, 5240, 1896` | `Two tabby` | exact |
| ncnn Windows x86_64 | `10392, 5240, 1896` | `Two tabby` | exact |

The Linux ncnn vision boundary has `max_abs=0.0132732` and
`mean_abs=9.59808e-05`; the final text is compared byte-for-byte.  Decoder
greedy top-1 and all 40 KV-cache layers also pass the saved PyTorch boundary.
Raw commands, timings, memory peaks and failure diagnostics are retained in
[`evidence/`](evidence/README.md), including the unsuccessful half-sized Embed
artifact that led to the guarded conversion fix.

## Known limitations

The package does not include a separate large dynamic prefill graph.  Its
runtime fallback reuses the decode graph token by token; for this 136-token
fixture, prefill took 85.7 seconds on Linux and 81.2--89.7 seconds on native
Windows.  This is causally equivalent and produced the exact final IDs/text,
but it is not a throughput-optimized prefill path.

One near-boundary candidate changed in the decoder top-10 diagnostic.  Greedy
top-1 and the final three generated IDs/text are exact; this result must not be
described as full top-10 equality.

## License and provenance

See [`PROVENANCE.md`](PROVENANCE.md).  The official Youtu-VL checkpoint remains
subject to its own license and geographic restriction; this branch does not
redistribute the original safetensors.
