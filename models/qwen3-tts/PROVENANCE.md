# Provenance

This branch is an independent commit series rooted at
nihui/ncnn_llm revision f2f29e41be164c788c36cc44bdbf2d0d4810477e plus the shared
infrastructure branch. It does not carry Git history from another activity
submission.

The Qwen3-TTS runtime and export research preserve Apache-2.0 code previously
validated in LudovicoYIN/Qwen3-TTS-ncnn revision
9bc18b5794e7648a7a410abac457d3d9f045f4b8. The Qwen byte-level BPE
implementation also credits the earlier mingshi2333/Qwen3-TTS-ncnn adaptation
in its source header. Those sources were migrated into the common ncnn_llm
build, corrected for model-relative paths and Windows UTF-8 arguments, and
paired with a pinned reproducible conversion and regression workflow. Their
licenses and attribution are retained; this submission does not claim those
portions as wholly original.

The model architecture and PyTorch baseline use the official
QwenLM/Qwen3-TTS source revision recorded in checkpoint.json.
