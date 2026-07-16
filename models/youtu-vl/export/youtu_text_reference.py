#!/usr/bin/env python3
"""Dump text-only HF/PyTorch reference tensors for Youtu-VL P2.

This script is the P2B ruler for splitting the text path into:
  - embed_token: input_ids -> inputs_embeds
  - decoder: inputs_embeds + cache/mask/positions -> hidden states + cache
  - lm_head: hidden states -> logits

It intentionally does not process images.  Use youtu_vl_reference.py for the
full VLM path.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "assets" / "youtu_text_test" / "text_manifest.json"
DEFAULT_DUMP_DIR = ROOT / "assets" / "youtu_text_test" / "dumps"


def die(message: str, code: int = 2) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def import_runtime_deps():
    try:
        import numpy as np
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoProcessor
    except ModuleNotFoundError as exc:
        missing = exc.name or "a required module"
        die(
            "missing Python dependency "
            f"`{missing}`. Activate the project venv or install runtime deps:\n"
            "  pip install -U torch transformers accelerate pillow numpy safetensors",
            code=3,
        )
    return np, torch, transformers, AutoModelForCausalLM, AutoProcessor


def install_pydensecrf_import_stub() -> bool:
    """Install a tiny pydensecrf stub for importing Youtu-VL remote code."""
    try:
        import pydensecrf.densecrf  # type: ignore  # noqa: F401
        import pydensecrf.utils  # type: ignore  # noqa: F401
        return False
    except ModuleNotFoundError:
        pass

    def unavailable(*_args, **_kwargs):
        raise RuntimeError(
            "pydensecrf is not installed. It is not needed for text-only "
            "reference dumps, but is required for dense CRF post-processing."
        )

    package = types.ModuleType("pydensecrf")
    densecrf = types.ModuleType("pydensecrf.densecrf")
    utils = types.ModuleType("pydensecrf.utils")

    class DenseCRF2D:  # pragma: no cover - should not be used in P2B
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


def resolve_project_path(path_like: str | Path, base: Path = ROOT) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def parse_dtype(torch, name: str):
    normalized = name.lower()
    if normalized == "auto":
        return "auto"
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32", "float"}:
        return torch.float32
    die(f"unsupported dtype: {name}")


def resolve_torch_device(torch, device_name: str):
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def tensor_to_numpy(np, torch, value: Any):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        x = value.detach().cpu()
        if x.dtype is torch.bfloat16:
            x = x.float()
        return x.numpy()
    return np.asarray(value)


def tensor_shape(value: Any) -> Optional[List[int]]:
    if value is None:
        return None
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(x) for x in shape]


def tensor_dtype(value: Any) -> Optional[str]:
    dtype = getattr(value, "dtype", None)
    return str(dtype) if dtype is not None else None


def to_model_device(inputs: Any, device):
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}


def first_parameter_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return getattr(model, "device", "cpu")


def past_key_value_shapes(past_key_values: Any) -> List[Dict[str, Any]]:
    if past_key_values is None:
        return []

    layers: List[Dict[str, Any]] = []
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        for i, (k, v) in enumerate(zip(past_key_values.key_cache, past_key_values.value_cache)):
            layers.append({
                "layer": i,
                "key": tensor_shape(k),
                "value": tensor_shape(v),
                "key_dtype": tensor_dtype(k),
                "value_dtype": tensor_dtype(v),
            })
        return layers

    try:
        iterable = list(past_key_values)
    except TypeError:
        return [{"repr": repr(type(past_key_values))}]

    for i, layer in enumerate(iterable):
        if isinstance(layer, (tuple, list)) and len(layer) >= 2:
            layers.append({
                "layer": i,
                "key": tensor_shape(layer[0]),
                "value": tensor_shape(layer[1]),
                "key_dtype": tensor_dtype(layer[0]),
                "value_dtype": tensor_dtype(layer[1]),
            })
        else:
            layers.append({"layer": i, "repr": repr(type(layer))})
    return layers


def add_cache_arrays(np, torch, arrays: Dict[str, Any], prefix: str, past_key_values: Any) -> int:
    """Flatten HF cache tensors into npz entries for wrapper parity tests."""
    if past_key_values is None:
        return 0

    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        key_cache = list(past_key_values.key_cache)
        value_cache = list(past_key_values.value_cache)
    else:
        try:
            cache_layers = list(past_key_values)
        except TypeError:
            return 0
        key_cache = [layer[0] for layer in cache_layers if isinstance(layer, (tuple, list)) and len(layer) >= 2]
        value_cache = [layer[1] for layer in cache_layers if isinstance(layer, (tuple, list)) and len(layer) >= 2]

    count = min(len(key_cache), len(value_cache))
    for layer_idx in range(count):
        arrays[f"{prefix}_layer{layer_idx:02d}_key"] = tensor_to_numpy(np, torch, key_cache[layer_idx])
        arrays[f"{prefix}_layer{layer_idx:02d}_value"] = tensor_to_numpy(np, torch, value_cache[layer_idx])
    return count


def topk_arrays(torch, logits, k: int) -> Tuple[Any, Any]:
    k = min(int(k), int(logits.shape[-1]))
    values, indices = torch.topk(logits, k=k, dim=-1)
    return indices.detach().cpu(), values.detach().float().cpu()


def iter_samples(manifest: Dict[str, Any], only: Optional[Sequence[str]]) -> Iterable[Dict[str, Any]]:
    wanted = set(only or [])
    for sample in manifest.get("samples", []):
        sample_id = sample["id"]
        if wanted and sample_id not in wanted:
            continue
        yield sample


def build_messages(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "messages" in sample:
        messages = sample["messages"]
        if not isinstance(messages, list):
            die(f"sample {sample.get('id')} messages must be a list")
        return messages

    text = sample.get("text")
    if not isinstance(text, str):
        die(f"sample {sample.get('id')} must contain string field `text` or list field `messages`")
    return [{"role": "user", "content": text}]


def prompt_text_from_messages(processor, messages: List[Dict[str, Any]]) -> str:
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def load_model_and_processor(args, manifest: Dict[str, Any]):
    np, torch, transformers, AutoModelForCausalLM, AutoProcessor = import_runtime_deps()
    pydensecrf_stubbed = install_pydensecrf_import_stub()

    model_cfg = manifest.get("model", {})
    model_arg = args.model or model_cfg.get("local_dir") or model_cfg.get("hf_model_id")
    if not model_arg:
        die("no model path/id provided")
    model_path = resolve_project_path(model_arg) if not model_arg.startswith("tencent/") else model_arg

    dtype = parse_dtype(torch, args.dtype)
    kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": dtype,
    }
    if args.attn_implementation != "none":
        kwargs["attn_implementation"] = args.attn_implementation
    if args.device_map != "none":
        kwargs["device_map"] = args.device_map

    print(f"[load] processor: {model_path}")
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        use_fast=True,
        trust_remote_code=True,
    )

    print(f"[load] model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(str(model_path), **kwargs).eval()
    if args.device_map == "none":
        device = resolve_torch_device(torch, args.device)
        print(f"[load] moving model to {device}")
        model = model.to(device)
    else:
        device = first_parameter_device(model)

    return np, torch, transformers, model_path, model, processor, device, pydensecrf_stubbed


def collect_sample_dump(
    *,
    np,
    torch,
    model,
    processor,
    sample: Dict[str, Any],
    args,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    messages = build_messages(sample)
    prompt_text = prompt_text_from_messages(processor, messages)

    inputs = processor(
        text=prompt_text,
        return_tensors="pt",
    )
    input_device = first_parameter_device(model)
    inputs = to_model_device(inputs, input_device)
    model_inputs = dict(inputs)

    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs.get("attention_mask", None)
    prompt_len = int(input_ids.shape[1])
    eos_token_id = getattr(processor.tokenizer, "eos_token_id", None)

    with torch.inference_mode():
        inputs_embeds = model.model.embed_tokens(input_ids)
        outputs = model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
            output_attentions=False,
        )

        prefill_last_hidden_state = outputs.hidden_states[-1].detach().cpu()
        prefill_logits = outputs.logits[:, -1, :].detach().float().cpu()
        prefill_top_ids, prefill_top_values = topk_arrays(torch, prefill_logits, args.topk)
        next_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        past = outputs.past_key_values

        arrays: Dict[str, Any] = {
            "input_ids": tensor_to_numpy(np, torch, input_ids),
            "inputs_embeds": tensor_to_numpy(np, torch, inputs_embeds),
            "prefill_last_hidden_state": tensor_to_numpy(np, torch, prefill_last_hidden_state),
            "prefill_logits": tensor_to_numpy(np, torch, prefill_logits),
            "prefill_top_ids": tensor_to_numpy(np, torch, prefill_top_ids),
            "prefill_top_values": tensor_to_numpy(np, torch, prefill_top_values),
            "prefill_next_id": np.asarray([int(next_id.item())], dtype=np.int64),
        }
        if attention_mask is not None:
            arrays["attention_mask"] = tensor_to_numpy(np, torch, attention_mask)

        if args.capture_all_hidden_states:
            all_hidden = torch.stack([x.detach().cpu() for x in outputs.hidden_states], dim=0)
            arrays["prefill_all_hidden_states"] = tensor_to_numpy(np, torch, all_hidden)

        decoder_io_meta: Dict[str, Any] = {}
        if args.capture_decoder_io:
            prefill_cache_position = torch.arange(prompt_len, dtype=torch.long, device=input_device)
            arrays["decoder_prefill_inputs_embeds"] = tensor_to_numpy(np, torch, inputs_embeds)
            arrays["decoder_prefill_attention_mask"] = tensor_to_numpy(np, torch, attention_mask)
            arrays["decoder_prefill_cache_position"] = tensor_to_numpy(np, torch, prefill_cache_position)
            arrays["decoder_prefill_last_hidden_state"] = tensor_to_numpy(np, torch, prefill_last_hidden_state)
            decoder_io_meta["prefill_cache_layers"] = add_cache_arrays(
                np, torch, arrays, "decoder_prefill_output_cache", past
            )

        generated_new_ids: List[int] = [int(next_id.item())]
        decode_input_ids: List[int] = []
        decode_next_ids: List[int] = []
        decode_logits: List[Any] = []
        decode_top_ids: List[Any] = []
        decode_top_values: List[Any] = []
        decode_last_hidden: List[Any] = []

        running_attention_mask = attention_mask
        for _step in range(max(0, args.max_new_tokens - 1)):
            if args.stop_at_eos and eos_token_id is not None and int(next_id.item()) == int(eos_token_id):
                break

            cur_input = next_id.view(1, 1).to(input_device)
            decode_input_ids.append(int(cur_input.item()))

            if running_attention_mask is not None:
                one = torch.ones(
                    (running_attention_mask.shape[0], 1),
                    dtype=running_attention_mask.dtype,
                    device=running_attention_mask.device,
                )
                running_attention_mask = torch.cat([running_attention_mask, one], dim=1)

            if args.capture_decoder_io and not decoder_io_meta.get("decode_step0_captured"):
                decode_inputs_embeds = model.model.embed_tokens(cur_input)
                decode_cache_position = torch.arange(
                    prompt_len,
                    prompt_len + cur_input.shape[1],
                    dtype=torch.long,
                    device=input_device,
                )
                arrays["decoder_decode_step0_input_ids"] = tensor_to_numpy(np, torch, cur_input)
                arrays["decoder_decode_step0_inputs_embeds"] = tensor_to_numpy(np, torch, decode_inputs_embeds)
                arrays["decoder_decode_step0_attention_mask"] = tensor_to_numpy(np, torch, running_attention_mask)
                arrays["decoder_decode_step0_cache_position"] = tensor_to_numpy(np, torch, decode_cache_position)
                decoder_io_meta["decode_step0_input_cache_layers"] = add_cache_arrays(
                    np, torch, arrays, "decoder_decode_step0_input_cache", past
                )

            step_outputs = model(
                input_ids=cur_input,
                attention_mask=running_attention_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=args.capture_decode_hidden_states or args.capture_decoder_io,
                output_attentions=False,
            )
            step_logits = step_outputs.logits[:, -1, :].detach().float().cpu()
            step_top_ids, step_top_values = topk_arrays(torch, step_logits, args.topk)
            next_id = torch.argmax(step_outputs.logits[:, -1, :], dim=-1)
            past = step_outputs.past_key_values

            decode_logits.append(step_logits)
            decode_top_ids.append(step_top_ids)
            decode_top_values.append(step_top_values)
            decode_next_ids.append(int(next_id.item()))
            generated_new_ids.append(int(next_id.item()))

            if args.capture_decode_hidden_states:
                decode_last_hidden.append(step_outputs.hidden_states[-1].detach().cpu())
            if args.capture_decoder_io and not decoder_io_meta.get("decode_step0_captured"):
                if step_outputs.hidden_states is not None:
                    arrays["decoder_decode_step0_last_hidden_state"] = tensor_to_numpy(
                        np, torch, step_outputs.hidden_states[-1]
                    )
                decoder_io_meta["decode_step0_output_cache_layers"] = add_cache_arrays(
                    np, torch, arrays, "decoder_decode_step0_output_cache", past
                )
                decoder_io_meta["decode_step0_captured"] = True

    generated_tensor = torch.tensor([generated_new_ids], dtype=torch.long)
    generated_text = processor.batch_decode(
        generated_tensor,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    generated_ids_full = torch.cat([input_ids.detach().cpu(), generated_tensor], dim=1)

    arrays["decode_input_ids"] = np.asarray(decode_input_ids, dtype=np.int64)
    arrays["decode_next_ids"] = np.asarray(decode_next_ids, dtype=np.int64)
    arrays["generated_new_ids"] = np.asarray(generated_new_ids, dtype=np.int64)
    arrays["generated_ids"] = tensor_to_numpy(np, torch, generated_ids_full)

    if decode_logits:
        arrays["decode_step_logits"] = tensor_to_numpy(np, torch, torch.cat(decode_logits, dim=0))
        arrays["decode_step_top_ids"] = tensor_to_numpy(np, torch, torch.cat(decode_top_ids, dim=0))
        arrays["decode_step_top_values"] = tensor_to_numpy(np, torch, torch.cat(decode_top_values, dim=0))
    else:
        arrays["decode_step_logits"] = np.empty((0, int(prefill_logits.shape[-1])), dtype=np.float32)
        arrays["decode_step_top_ids"] = np.empty((0, args.topk), dtype=np.int64)
        arrays["decode_step_top_values"] = np.empty((0, args.topk), dtype=np.float32)

    if decode_last_hidden:
        arrays["decode_last_hidden_states"] = tensor_to_numpy(np, torch, torch.cat(decode_last_hidden, dim=0))

    sample_meta = {
        "sample_id": sample["id"],
        "purpose": sample.get("purpose", ""),
        "source_text": sample.get("text", ""),
        "messages": messages,
        "prompt_text": prompt_text,
        "prompt_len": prompt_len,
        "max_new_tokens_requested": args.max_new_tokens,
        "generated_new_ids": generated_new_ids,
        "generated_text": generated_text,
        "eos_token_id": int(eos_token_id) if eos_token_id is not None else None,
        "past_key_value_shapes": past_key_value_shapes(past),
        "array_shapes": {k: list(v.shape) for k, v in arrays.items() if hasattr(v, "shape")},
        "array_dtypes": {k: str(v.dtype) for k, v in arrays.items() if hasattr(v, "dtype")},
    }
    if args.capture_decoder_io:
        sample_meta["decoder_io"] = decoder_io_meta
    return arrays, sample_meta


def save_npz(np, path: Path, arrays: Dict[str, Any], compress: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump text-only HF/PyTorch reference tensors for Youtu-VL P2."
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Path to text_manifest.json.")
    parser.add_argument("--model", default="", help="Local HF model dir or model id. Defaults to manifest model.local_dir.")
    parser.add_argument("--output-dir", default=str(DEFAULT_DUMP_DIR), help="Directory for .npz and metadata.json.")
    parser.add_argument("--only", action="append", default=[], help="Run one sample id. Can be repeated.")
    parser.add_argument("--max-new-tokens", type=int, default=8, help="Greedy new tokens to dump.")
    parser.add_argument("--topk", type=int, default=10, help="Store top-k ids/values for each dumped logits vector.")
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--device", default="auto", help="Used when --device-map none. Use 'auto', 'cuda:0', or 'cpu'.")
    parser.add_argument(
        "--device-map",
        default="none",
        help="Passed to from_pretrained. Default 'none' loads normally then model.to(--device).",
    )
    parser.add_argument(
        "--attn-implementation",
        default="eager",
        help="Transformers attention backend: eager, sdpa, flash_attention_2, or none.",
    )
    parser.add_argument("--capture-all-hidden-states", action="store_true")
    parser.add_argument("--capture-decode-hidden-states", action="store_true")
    parser.add_argument(
        "--capture-decoder-io",
        action="store_true",
        help="Store prefill/decode decoder inputs, outputs, and KV cache tensors for wrapper parity.",
    )
    parser.add_argument("--no-compress", action="store_true", help="Use np.savez instead of np.savez_compressed.")
    parser.add_argument("--no-stop-at-eos", dest="stop_at_eos", action="store_false", default=True)
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

    (
        np,
        torch,
        transformers,
        model_path,
        model,
        processor,
        model_device,
        pydensecrf_stubbed,
    ) = load_model_and_processor(args, manifest)

    dump_dir = resolve_project_path(args.output_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)

    metadata: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "manifest": str(manifest_path),
        "model": {
            "path_or_id": str(model_path),
            "hf_model_id": manifest.get("model", {}).get("hf_model_id"),
            "revision": manifest.get("model", {}).get("revision", ""),
            "config_model_type": getattr(model.config, "model_type", None),
            "vocab_size": int(getattr(model.config, "vocab_size", 0)),
            "hidden_size": int(getattr(model.config, "hidden_size", 0)),
            "num_hidden_layers": int(getattr(model.config, "num_hidden_layers", 0)),
            "num_attention_heads": int(getattr(model.config, "num_attention_heads", 0)),
            "qk_head_dim": int(getattr(model.config, "qk_head_dim", 0)),
            "v_head_dim": int(getattr(model.config, "v_head_dim", 0)),
            "rope_interleave": bool(getattr(model.config, "rope_interleave", False)),
        },
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "device_map": args.device_map,
            "model_device": str(model_device),
            "dtype_arg": args.dtype,
            "attn_implementation": args.attn_implementation,
            "pydensecrf_stubbed": pydensecrf_stubbed,
        },
        "args": {
            "max_new_tokens": args.max_new_tokens,
            "topk": args.topk,
            "capture_all_hidden_states": args.capture_all_hidden_states,
            "capture_decode_hidden_states": args.capture_decode_hidden_states,
            "capture_decoder_io": args.capture_decoder_io,
            "stop_at_eos": args.stop_at_eos,
        },
        "samples": [],
    }

    for sample in samples:
        sample_name = sample["id"]
        print(f"[sample] {sample_name}")
        arrays, sample_meta = collect_sample_dump(
            np=np,
            torch=torch,
            model=model,
            processor=processor,
            sample=sample,
            args=args,
        )

        npz_path = dump_dir / f"{sample_name}.npz"
        save_npz(np, npz_path, arrays, compress=not args.no_compress)
        sample_meta["npz_path"] = str(npz_path)
        metadata["samples"].append(sample_meta)
        print(f"[write] {npz_path}")
        print(f"[text] {sample_meta['generated_text']!r}")

    metadata_path = dump_dir / "metadata.json"
    write_json(metadata_path, metadata)
    print(f"[write] {metadata_path}")


if __name__ == "__main__":
    main()
