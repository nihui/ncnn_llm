#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import sys
import unittest

from _subprocess_utils import run_utf8


class RunUtf8Test(unittest.TestCase):
    def test_decodes_utf8_output_independently_of_host_locale(self) -> None:
        script = "import sys; sys.stdout.buffer.write('中文输出'.encode('utf-8'))"
        completed = run_utf8(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )
        self.assertEqual(completed.stdout, "中文输出")

    def test_replaces_an_invalid_byte_without_losing_ascii_markers(self) -> None:
        script = "import sys; sys.stdout.buffer.write(b'PASS\\xff')"
        completed = run_utf8([sys.executable, "-c", script], capture_output=True, check=True)
        self.assertEqual(completed.stdout, "PASS�")


if __name__ == "__main__":
    unittest.main()
