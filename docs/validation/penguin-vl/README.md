# Penguin-VL validation evidence

This directory records a fresh conversion and real end-to-end inference of
`tencent/Penguin-VL-2B@26ac2ceac1179ec3eda106cbb88e9eb909e68d4e`.

## Passing regression

- Fixed prompt: `What month is printed at the top? Answer with the month only.`
- PyTorch baseline: `models/penguin-vl/testdata/pytorch.json`
- Expected UTF-8 bytes: `July` (4 bytes)
- Expected SHA-256: `e9cf7a1ca4a760843ff30e1fbb125e6a066c63e883c58523bdbe04c9daf5dbae`
- WSL ncnn: `linux-ncnn.txt` and `linux-parity.json`
- Native Windows ncnn: `windows-ncnn.txt` and `windows-parity.json`

The `linux-inference-*` and `windows-inference-*` files preserve stdout,
stderr, command timing, peak memory (Linux), and process exit status. The full
CMake build and CTest transcripts are in the two `*-build-test.txt` files.

## Conversion and package integrity

- `conversion-core.*.log`: patch embedding, vision encoder, projector,
  embedding and tied LM-head pnnx conversion.
- `conversion-decoder.*.log`: independent 28-layer Qwen3 decoder conversion.
- `independent-sha256.txt`: hashes of the independently generated package.
- `windows-asset-verify.json`: all files checked against
  `models/penguin-vl/assets.json`; every entry has `ok: true`.
- `reference-package-sha256.txt`: a post-conversion cross-check against an
  existing public package. The BIN hashes match; no source, commit, or converted
  file from that package was used to create this branch.

## Retained negative diagnostic

`pytorch-baseline.txt` / `pytorch-time.txt` record the official PIL result for
the more pixel-sensitive date prompt (`July 4, 1776`). The corresponding older
ncnn observation was `1764`. This is explicitly diagnostic-only in
`models/penguin-vl/testdata/pytorch-date-diagnostic.json` and is not counted as
a parity pass. The less interpolation-sensitive month prompt above is the sole
passing regression case.
