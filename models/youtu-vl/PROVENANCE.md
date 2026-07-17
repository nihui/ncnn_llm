# Provenance

This branch is an independent commit on `March-77/ncnn_llm`, created directly
from the shared infrastructure branch. It does not contain another submission's
Git history.

The working Youtu-VL conversion and runtime modules preserve the useful code
from March-77's superseded standalone experiment. That experiment studied the
official `TencentCloudADP/youtu-vl` implementation, `nihui/ncnn_llm`, and the
public `AtomAlpaca/Youtu-VL-ncnn` project (source snapshot
`43ecf19c1281100f4e52469bf36535b321dea2fc`). The ancestry is disclosed here;
referenced work is not presented as newly authored code.

This migration adds shared-repository CMake integration, native Windows UTF-8
handling, exact output files, pinned checkpoint metadata, reproducible tools,
fresh Linux and Windows inference, and retained machine-readable evidence. The
model remains subject to the Youtu-VL license, including its stated geographic
restriction. ncnn and stb_image retain their upstream licenses and notices.
