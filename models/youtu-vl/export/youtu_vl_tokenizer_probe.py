#!/usr/bin/env python3

from __future__ import annotations

from PIL import Image
import argparse
import json
import sys
import types
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "assets" / "youtu_vl_test" / "image_manifest.json"
DEFAULT_DUMP_DIR = ROOT / "assets" / "youtu_tokenizer_probe"


def die(message: str, code: int = 2) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def import_runtime_deps():
    try:
        import numpy as np
        import torch
        import transformers
        from transformers import AutoProcessor
    except ModuleNotFoundError as exc:
        missing = exc.name or "a required module"
        die(
            "missing Python dependency "
            f"`{missing}`. Install the HF runtime deps first, for example:\n"
            "  pip install -U torch transformers accelerate pillow numpy "
            "safetensors opencv-python-headless",
            code=3,
        )

    return np, torch, transformers, AutoProcessor


def install_pydensecrf_import_stub() -> bool:
    """Install a tiny pydensecrf stub when the real package is unavailable.

    Youtu-VL's remote modeling file imports pydensecrf at module load time, but
    P0 reference dumps do not call dense CRF post-processing.  The real
    pydensecrf package currently fails to build on Python 3.13, so this stub
    keeps model import usable for VQA/OCR/logit dumps while still failing loudly
    if dense CRF functionality is accidentally invoked.
    """
    try:
        import pydensecrf.densecrf  # type: ignore  # noqa: F401
        import pydensecrf.utils  # type: ignore  # noqa: F401
        return False
    except ModuleNotFoundError:
        pass

    def unavailable(*_args, **_kwargs):
        raise RuntimeError(
            "pydensecrf is not installed. It is optional for P0 reference dumps, "
            "but required for dense prediction CRF post-processing."
        )

    package = types.ModuleType("pydensecrf")
    densecrf = types.ModuleType("pydensecrf.densecrf")
    utils = types.ModuleType("pydensecrf.utils")

    class DenseCRF2D:  # pragma: no cover - should not be used in P0
        def __init__(self, *_args, **_kwargs):
            unavailable()

    densecrf.DenseCRF2D = DenseCRF2D
    densecrf.DIAG_KERNEL = 0
    densecrf.NORMALIZE_SYMMETRIC = 0
    utils.unary_from_softmax = unavailable

    package.densecrf = densecrf
    package.utils = utils
    sys.modules.setdefault("pydensecrf", package)
    sys.modules.setdefault("pydensecrf.densecrf", densecrf)
    sys.modules.setdefault("pydensecrf.utils", utils)
    return True


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def resolve_project_path(path_like: str, base: Path = ROOT) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def tensor_to_numpy(np, torch, value: Any):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        x = value.detach().cpu()
        if x.dtype is torch.bfloat16:
            x = x.float()
        return x.numpy()
    return np.asarray(value)


def build_messages(image_path: Path, prompt: str) -> List[Dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def iter_samples(manifest: Dict[str, Any], only: Optional[Sequence[str]]) -> Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]:
    wanted = set(only or [])
    for image in manifest.get("images", []):
        image_id = image["id"]
        for prompt in image.get("prompts", []):
            sample_id = f"{image_id}:{prompt['id']}"
            if wanted and sample_id not in wanted:
                continue
            yield image, prompt


def maybe_download_image(image: Dict[str, Any], image_path: Path, enabled: bool) -> None:
    if image_path.exists():
        return
    url = image.get("url") or ""
    if not url:
        die(f"missing image file and no url is configured: {image_path}")
    if not enabled:
        die(
            f"missing image file: {image_path}\n"
            "rerun with --download-missing-images to fetch configured URLs"
        )
    image_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url} -> {image_path}")
    urllib.request.urlretrieve(url, image_path)


def load_processor(args, manifest: Dict[str, Any]):
    np, torch, transformers, AutoProcessor = import_runtime_deps()
    pydensecrf_stubbed = install_pydensecrf_import_stub()

    model_cfg = manifest.get("model", {})
    model_arg = args.model or model_cfg.get("local_dir") or model_cfg.get("hf_model_id")
    if not model_arg:
        die("no model path/id provided")
    model_path = resolve_project_path(model_arg) if not model_arg.startswith("tencent/") else model_arg

    print(f"[load] processor: {model_path}")
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        use_fast=True,
        trust_remote_code=True,
    )
    return np, torch, processor


def collect_sample_dump(
    *,
    np,
    torch,
    processor,
    image: Dict[str, Any],
    prompt: Dict[str, Any],
    image_path: Path,
    args,
) -> Dict[str, Any]:
    arrays: Dict[str, Any] = {}

    messages = build_messages(image_path, prompt["text"])
    image = Image.open(image_path).convert("RGB")

    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=prompt_text,
        images=image,
        return_tensors="pt",
        max_image_patches=args.max_image_patches,
    )

    input_ids = inputs["input_ids"]
    spatial_shapes = inputs["spatial_shapes"]

    special_token_names = [
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
    ]

    special_token_positions : Dict[str, List[int]] = {}
    input_ids_list = input_ids[0].tolist()
    tokens = processor.tokenizer.convert_ids_to_tokens(input_ids_list)

    special_token_ids: Dict[str, int] = {}
    for token in special_token_names:
        special_token_ids[token] = processor.tokenizer.convert_tokens_to_ids(token)
        special_token_positions[token] = []

    for index, token_id in enumerate(input_ids_list):
        for token_name, special_id in special_token_ids.items():
            if token_id == special_id:
                special_token_positions[token_name].append(index)

    arrays["input_ids"] = input_ids_list
    arrays["spatial_shapes"] = tensor_to_numpy(np, torch, spatial_shapes).tolist()
    arrays["tokens"] = tokens
    arrays["prompt_text"] = prompt_text
    arrays["special_token_positions"] = special_token_positions
    arrays["special_token_ids"] = special_token_ids
    arrays["image_pad_count"] = len(special_token_positions["<|image_pad|>"])
    h, w = arrays["spatial_shapes"][0]
    arrays["expected_image_pad_count"] = h * w // 4
    arrays["image_pad_count_matches"] = (arrays["image_pad_count"] == arrays["expected_image_pad_count"])
    return arrays

def save_json(path : Path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path=path, data = array)

def main() -> None:
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Path to image_manifest.json.")
    parser.add_argument("--model", default="", help="Local HF model dir or model id. Defaults to manifest model.local_dir.")
    parser.add_argument("--output-dir", default=str(DEFAULT_DUMP_DIR), help="Directory for output.")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Run only one image_id:prompt_id sample. Can be repeated.",
    )
    parser.add_argument("--max-image-patches", type=int, default=36864)
    parser.add_argument("--download-missing-images", action="store_true")
    args = parser.parse_args()

    manifest_path = resolve_project_path(args.manifest)
    if not manifest_path.exists():
        die(f"manifest not found: {manifest_path}")
    manifest = load_json(manifest_path)

    samples = list(iter_samples(manifest, args.only))
    if not samples:
        if args.only:
            die(f"no samples matched --only={args.only}")
        die("manifest has no runnable samples")

    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        np,
        torch,
        processor,
    ) = load_processor(args, manifest)

    test_root = manifest_path.parent

    for image, prompt in samples:
        sample_name = f"{image['id']}_{prompt['id']}"
        image_path = (test_root / image["filename"]).resolve()
        maybe_download_image(image, image_path, args.download_missing_images)

        print(f"[sample] {image['id']}:{prompt['id']}")

        arrays = collect_sample_dump(
            np=np,
            torch=torch,
            processor=processor,
            image=image,
            prompt=prompt,
            image_path=image_path,
            args=args,
        )

        json_path = output_dir / f"{sample_name}.json"
        save_json(json_path, arrays)
        print(f"[write] {json_path}")



if __name__ == "__main__":
    main()
