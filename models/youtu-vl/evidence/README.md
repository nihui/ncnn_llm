# Evidence index

All files below are raw outputs from the current branch and pinned checkpoint,
not copied success summaries from an earlier repository.

- `reference/pytorch-reference.log`: official PyTorch image inference,
  `Two tabby`, exit 0, peak RSS 23,673,520 KiB.
- `reference/vision-embeddings-boundary.log`: fresh official vision boundary
  dump used by the C++ comparison.
- `conversion/embed-fp16-repaired.log`: guarded FP32-trace/FP16-storage Embed
  conversion after detecting a truncated pnnx payload.
- `conversion/full-decoder-pnnx.log`: direct 17 GB TorchScript to 40-layer ncnn
  graph conversion after the PyTorch process exited.
- `conversion/full-decoder-greedy-parity.log`: 40 KV-cache layers, final norm,
  logits and greedy top-1 comparison.
- `conversion/full-decoder-ncnn-parity.log`: the corresponding full top-10
  diagnostic; top-1 passes while one near-boundary top-10 candidate differs.
- `conversion/vision-pnnx.log`: SigLIP2 encoder and post-merger pnnx conversion.
- `linux/cmake-configure.log` and `linux/cmake-build.log`: fresh Release/Ninja
  configure and incremental rebuild output after the zlib integration.
- `linux/ctest.log`: three of three tests passed.
- `linux/vision-only.log`: ncnn vision boundary `max_abs=0.0132732`.
- `linux/e2e.log`: complete vision + tokenwise prefill + decode; generated IDs
  `10392, 5240, 1896`, final text `Two tabby`, exit 0, peak RSS 22,908,812 KiB.
- `package.log`: runtime-only package receipt, 22 files and 12,177,556,868
  bytes before `manifest.json`.

- `windows/cmake-configure.log` and `windows/cmake-build.log`: native
  LLVM-MinGW UCRT build (`Clang 22.1.8`, x86_64 Windows).
- `windows/ctest.log`: three of three native Windows tests passed.
- `windows/vision-only.log`: native Windows ncnn vision boundary
  `max_abs=0.0132427`, exit 0.
- `windows/e2e.log`: first native end-to-end run with peak working set
  23,206,891,520 bytes plus strict output-file verification.
- `windows/e2e-direct-exit.log`: second direct native invocation recording
  `exit_code=0`, the same three token IDs and exact `Two tabby` text.

The machine-readable expected result lives in `../testdata/reference.json`.
