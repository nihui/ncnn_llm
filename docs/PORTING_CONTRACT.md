# ncnn model porting contract

Each model port lives under `models/<model-id>/` and uses the shared tokenizer,
prompt, decoder, KV-cache, sampling, CPU/Vulkan and model-loading facilities in
`src/`. A port is complete only when the following artifacts are committed:

1. `checkpoint.json` pins an official public checkpoint and immutable revision.
2. `convert.py` exports every runtime graph through pnnx and accepts
   `--checkpoint`, `--output` and `--pnnx`.
3. `assets.json` lists every converted/runtime file with SHA-256 and byte size.
4. `tests/` contains a redistributable input, PyTorch baseline, ncnn output and
   an exact comparison receipt.
5. `README.md` records exact conversion/build/run/compare commands for Linux and
   Windows, plus measured platform/tool versions.

The C++ executable may depend on ncnn and the C/C++ standard libraries. Python,
PyTorch and pnnx are conversion/reference-test dependencies only and must not be
required by the deployed executable.

## Conversion

Run a model converter through `tools/run_conversion.py`. The wrapper records the
command, checkpoint metadata, pnnx version, platform, timestamps, exit code and
SHA-256 of generated files. Conversion scripts must set deterministic seeds,
call `eval()`, disable gradients and declare all dynamic axes or concrete input
shapes. Post-processing a pnnx graph is allowed only through a committed,
deterministic script documented by the receipt.

## Parity

`tools/compare_outputs.py` is the final gate. Text and token modes require byte-
for-byte/element-for-element equality. WAV mode requires identical format and PCM
samples. Model branches may additionally record diagnostic tolerances, but a
tolerance result does not replace the exact final gate.

Receipts are evidence, not claims: keep raw stdout/stderr logs, input hashes,
checkpoint hashes and the commands used to produce them. Do not commit model
weights when their license or GitHub file-size limits forbid it; publish the
package separately and pin it through `assets.json`.
