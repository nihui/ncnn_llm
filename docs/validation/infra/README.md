# Shared infrastructure validation

Validated from clean build directories on 2026-07-16 against:

- `ncnn_llm`: `f2f29e41be164c788c36cc44bdbf2d0d4810477e`
- `ncnn`: `13b6d5318c73be53bb386fa51e8067615d0eb7c1`

The Windows build used native MSYS2 UCRT64 (CMake 4.4.0, Ninja 1.13.2,
GCC 16.1.0) and completed 819 build steps. The WSL Ubuntu build used CMake
3.28.3, Ninja 1.11.1 and GCC 13.3. Both built the full library and every
non-benchmark example, passed the C++ and Python tests, and installed eight
executables plus `libncnn_llm` and `libncnn_tokenizer`.

Raw rerun logs are committed next to this file:

- `windows-build-test.txt`
- `linux-build-test.txt`

The logs deliberately show the tested source revisions and toolchain versions.
The initial full build transcripts remain GitHub Actions artifacts; the compact
rerun logs prove that the resulting build trees are complete and tests pass.
