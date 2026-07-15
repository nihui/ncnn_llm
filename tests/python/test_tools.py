from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import wave


ROOT = Path(__file__).resolve().parents[2]


def load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


assets = load_tool("model_assets")
compare = load_tool("compare_outputs")


class ToolTests(unittest.TestCase):
    def test_asset_verification_and_reproducible_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "weights.bin"
            payload.write_bytes(b"ncnn-model")
            manifest = {
                "schema_version": 1,
                "files": [{
                    "path": "weights.bin",
                    "size": payload.stat().st_size,
                    "sha256": assets.sha256(payload),
                }],
            }
            self.assertTrue(assets.verify(manifest, root)[0]["ok"])
            first, second = root / "a.zip", root / "b.zip"
            assets.package(manifest, root, first)
            assets.package(manifest, root, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_path_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                assets.safe_path(Path(directory), "../outside")

    def test_text_and_token_exactness(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b = root / "a.txt", root / "b.txt"
            a.write_text("exact\n", encoding="utf-8")
            b.write_text("exact\n", encoding="utf-8")
            self.assertTrue(compare.text_result(a, b)["exact"])
            b.write_text("exact", encoding="utf-8")
            self.assertFalse(compare.text_result(a, b)["exact"])
            a.write_text(json.dumps({"tokens": [1, 2]}), encoding="utf-8")
            b.write_text(json.dumps([1, 2]), encoding="utf-8")
            self.assertTrue(compare.token_result(a, b)["exact"])

    def test_wav_requires_identical_pcm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / "a.wav", root / "b.wav"]
            for path in paths:
                with wave.open(str(path), "wb") as stream:
                    stream.setnchannels(1)
                    stream.setsampwidth(2)
                    stream.setframerate(24000)
                    stream.writeframes(b"\x01\x00\x02\x00")
            self.assertTrue(compare.wav_result(*paths)["exact"])
            with wave.open(str(paths[1]), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(24000)
                stream.writeframes(b"\x01\x00\x03\x00")
            self.assertFalse(compare.wav_result(*paths)["exact"])


if __name__ == "__main__":
    unittest.main()
