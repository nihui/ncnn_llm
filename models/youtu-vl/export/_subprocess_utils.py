"""Cross-platform subprocess helpers for ncnn parity tools."""

from __future__ import annotations

import subprocess
from typing import Any, Sequence


def run_utf8(args: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    """Run a command and decode its textual output as UTF-8 on every host."""
    kwargs.setdefault("text", True)
    kwargs.setdefault("encoding", "utf-8")
    kwargs.setdefault("errors", "replace")
    return subprocess.run(args, **kwargs)
