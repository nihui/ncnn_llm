#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.torch import safe_open


PREFIX = "talker.code_predictor."


def write_tensor(out_dir: Path, key: str, array, manifest: dict) -> None:
    arr = np.asarray(array, dtype=np.float32)
    name = key[len(PREFIX):].replace(".", "_") + ".f32.bin"
    arr.tofile(out_dir / name)
    manifest[key[len(PREFIX):]] = {
        "file": name,
        "shape": list(arr.shape),
        "dtype": "float32",
        "layout": "row-major",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--safetensors", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "qwen3_tts_code_predictor_structured_weights_v1",
        "source": str(Path(args.safetensors).resolve()),
        "constants": {
            "hidden_size": 1024,
            "intermediate_size": 3072,
            "vocab_size": 2048,
            "num_layers": 5,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "num_code_groups": 16,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 1e-6,
        },
        "tensors": {},
    }

    with safe_open(args.safetensors, framework="pt", device="cpu") as f:
        keys = [k for k in f.keys() if k.startswith(PREFIX)]
        for key in sorted(keys):
            write_tensor(out_dir, key, f.get_tensor(key).float().cpu().numpy(), manifest["tensors"])

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(out_dir)
    print(f"tensors={len(manifest['tensors'])}")


if __name__ == "__main__":
    main()
