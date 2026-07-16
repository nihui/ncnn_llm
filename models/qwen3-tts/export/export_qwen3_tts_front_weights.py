#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.torch import safe_open


def write_tensor(out_dir: Path, name: str, array: np.ndarray, manifest: dict) -> None:
    array = np.asarray(array, dtype=np.float32)
    filename = name.replace(".", "_").replace("/", "_") + ".f32.bin"
    array.tofile(out_dir / filename)
    manifest[name] = {"file": filename, "shape": list(array.shape), "dtype": "float32"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--safetensors", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}

    with safe_open(args.safetensors, framework="pt", device="cpu") as f:
        def tensor(name: str) -> np.ndarray:
            return f.get_tensor(name).cpu().numpy()

        first_sum = tensor("decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum")
        first_usage = np.maximum(tensor("decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage"), 1e-5)
        write_tensor(out_dir, "first_embedding", first_sum / first_usage[:, None], manifest)

        for i in range(15):
            prefix = f"decoder.quantizer.rvq_rest.vq.layers.{i}._codebook"
            emb_sum = tensor(prefix + ".embedding_sum")
            usage = np.maximum(tensor(prefix + ".cluster_usage"), 1e-5)
            write_tensor(out_dir, f"rest_embedding_{i}", emb_sum / usage[:, None], manifest)

        write_tensor(out_dir, "first_output_proj_weight", tensor("decoder.quantizer.rvq_first.output_proj.weight"), manifest)
        write_tensor(out_dir, "rest_output_proj_weight", tensor("decoder.quantizer.rvq_rest.output_proj.weight"), manifest)
        write_tensor(out_dir, "pre_conv_weight", tensor("decoder.pre_conv.conv.weight"), manifest)
        write_tensor(out_dir, "pre_conv_bias", tensor("decoder.pre_conv.conv.bias"), manifest)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
