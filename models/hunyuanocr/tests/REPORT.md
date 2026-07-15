# HunyuanOCR parity evidence

The checked-in files are the unedited outputs of a fixed, public checkpoint and
test image. They are deliberately small enough to review in Git.

| Artifact | SHA-256 / observation |
|---|---|
| `guwan-right-column.png` | `c8faf31d71fea69a2aa35e02436d86b3408b3c810cfdedf8fff760015ef5c4a5` |
| PyTorch final text | `清江浦河庫道` |
| Linux/WSL ncnn final text | `清江浦河庫道` |
| Windows ncnn final text | `清江浦河庫道` |
| each UTF-8 text file | `171c58169bf5fb3a5c6f7df69bd972da294ab8d494e15cc46167f9a7fb44738a` (19 bytes) |

`pytorch-ncnn-report.json` records the reference checkpoint/config hashes,
runtime versions, input hash, command, and exact-match result. The two platform
comparison receipts were then generated independently by
`tools/compare_outputs.py text`; both report `"exact": true`.

The native Windows output and WSL output include complete ncnn model-load,
vision-resize/token-count, generation, and timing logs in `windows.stdout.txt`
and `linux.stdout.txt`.
