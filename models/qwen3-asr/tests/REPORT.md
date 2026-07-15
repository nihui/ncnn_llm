# Qwen3-ASR exact-parity evidence

The fixed test uses Qwen's public `asr_en.wav`. `prepare_audio.py` verifies the
official 48 kHz PCM24 SHA-256 and converts it, without trimming, to the runtime's
16 kHz mono PCM16 contract. The checked-in fixture is 15.05125 seconds and has
SHA-256 `9c33540cbba2cf82c2b724f733b002466b87f83c06e20fbe3c4eed83c141edc9`.

PyTorch, native Windows ncnn, and WSL ncnn all reported language `English` and
the same final transcript. Each final-text file is 186 bytes with SHA-256
`817cafe879a57eb4b7f2e786eb766ae14ec5e0b3f5cef483ff4ac25bd54b3dc6`.

- `parity-wsl/comparison.json`: one-process PyTorch+ncnn language/text gate.
- `windows-comparison.json`: independent exact Windows text receipt.
- `linux-comparison.json`: independent exact WSL text receipt.
- `windows.stdout.txt`: full native model-load/audio-token/result log.
- `linux.stdout.txt` and `linux.time.txt`: full WSL log, timing and maximum RSS.
- `pytorch-report.json`: official qwen-asr output, sample metadata and timing.

The deployed executable uses no Python or PyTorch.
