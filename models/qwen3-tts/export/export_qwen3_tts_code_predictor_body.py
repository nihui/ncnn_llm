#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from qwen_tts import Qwen3TTSModel


class CodePredictorBody(torch.nn.Module):
    def __init__(self, talker):
        super().__init__()
        self.predictor = talker.code_predictor

    def forward(self, inputs_embeds, attention_mask, position_ids, cache_position):
        return self.predictor.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            cache_position=cache_position,
        ).last_hidden_state


def causal_mask(seq_len: int, device):
    mask = torch.full((seq_len, seq_len), torch.finfo(torch.float32).min, dtype=torch.float32, device=device)
    return torch.triu(mask, diagonal=1).view(1, 1, seq_len, seq_len)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--talker-hidden", required=True)
    ap.add_argument("--first-code", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--all-lengths", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tts = Qwen3TTSModel.from_pretrained(args.model, device_map=args.device, dtype=torch.float32, attn_implementation="eager")
    talker = tts.model.talker
    predictor = talker.code_predictor
    body = CodePredictorBody(talker).eval()

    h_flat = np.fromfile(args.talker_hidden, dtype=np.float32)
    h = h_flat.reshape(1, h_flat.size // 1024, 1024)
    past_hidden = torch.from_numpy(h[:, -1:, :]).to(talker.device)
    first_code = torch.tensor([[args.first_code]], dtype=torch.long, device=talker.device)
    first_emb = talker.get_input_embeddings()(first_code)
    inputs = torch.cat((past_hidden, first_emb), dim=1).detach().float()
    seq_len = inputs.shape[1]
    mask = causal_mask(seq_len, talker.device)
    # The ncnn runtime supplies RoPE positions as float32.  The Qwen rotary
    # implementation immediately casts them to float, so tracing this boundary
    # as float avoids leaving an unsupported Tensor.to layer in the ncnn graph.
    position_ids = torch.arange(seq_len, dtype=torch.float32, device=talker.device).view(1, seq_len)
    cache_position = torch.arange(seq_len, dtype=torch.long, device=talker.device)

    with torch.inference_mode():
        hidden = body(inputs, mask, position_ids, cache_position)
        logits0 = predictor.lm_head[0](hidden[:, -1:, :])
        token0 = torch.argmax(logits0, dim=-1)

    inputs.cpu().numpy().astype(np.float32).tofile(out_dir / "code_predictor_body_input_f32.bin")
    mask.cpu().numpy().astype(np.float32).tofile(out_dir / "code_predictor_body_mask_f32.bin")
    position_ids.cpu().numpy().astype(np.float32).tofile(out_dir / "code_predictor_body_position_f32.bin")
    cache_position.cpu().numpy().astype(np.int64).tofile(out_dir / "code_predictor_body_cache_position_i64.bin")
    hidden.cpu().numpy().astype(np.float32).tofile(out_dir / "code_predictor_body_hidden_ref_f32.bin")

    # Raw C-order weights for the C++ validation loop.
    weights_dir = out_dir / "code_predictor_weights"
    weights_dir.mkdir(exist_ok=True)
    talker.get_input_embeddings().weight.detach().cpu().numpy().astype(np.float32).tofile(
        weights_dir / "talker_codec_embedding.f32"
    )
    for i, emb in enumerate(predictor.get_input_embeddings()):
        emb.weight.detach().cpu().numpy().astype(np.float32).tofile(weights_dir / f"codec_embedding_{i}.f32")
    for i, head in enumerate(predictor.get_output_embeddings()):
        head.weight.detach().cpu().numpy().astype(np.float32).tofile(weights_dir / f"lm_head_{i}.f32")

    traced = torch.jit.trace(body, (inputs.clone(), mask.clone(), position_ids.clone(), cache_position.clone()), strict=False)
    traced.save(str(out_dir / "code_predictor_body.pt"))

    if args.all_lengths:
        seq = inputs.clone()
        per_len = out_dir / "code_predictor_body_by_len"
        per_len.mkdir(exist_ok=True)
        tokens = []
        for step in range(15):
            cur_len = seq.shape[1]
            cur_mask = causal_mask(cur_len, talker.device)
            cur_pos = torch.arange(cur_len, dtype=torch.float32, device=talker.device).view(1, cur_len)
            cur_cache = torch.arange(cur_len, dtype=torch.long, device=talker.device)
            traced_len = torch.jit.trace(
                body,
                (seq.clone(), cur_mask.clone(), cur_pos.clone(), cur_cache.clone()),
                strict=False,
            )
            traced_len.save(str(per_len / f"code_predictor_body_s{cur_len:02d}.pt"))
            with torch.no_grad():
                cur_hidden = body(seq, cur_mask, cur_pos, cur_cache)
                logits = predictor.lm_head[step](cur_hidden[:, -1:, :])
                token = torch.argmax(logits, dim=-1)
            tokens.append(int(token.item()))
            cur_hidden.cpu().numpy().astype(np.float32).tofile(per_len / f"hidden_s{cur_len:02d}_ref_f32.bin")
            if step < 14:
                seq = torch.cat((seq, predictor.get_input_embeddings()[step](token)), dim=1)
        (per_len / "tokens_ref_i32.bin").write_bytes(np.asarray(tokens, dtype=np.int32).tobytes())
    meta = {
        "input_shape": list(inputs.shape),
        "mask_shape": list(mask.shape),
        "position_ids_shape": list(position_ids.shape),
        "cache_position_shape": list(cache_position.shape),
        "hidden_shape": list(hidden.shape),
        "token0": int(token0.item()),
        "all_lengths": args.all_lengths,
        "weight_layout": {
            "talker_codec_embedding": "shape [3072, 1024], C-order float32",
            "codec_embedding": "14 files, shape [2048, 1024], C-order float32",
            "lm_head": "15 files, shape [2048, 1024], C-order float32",
        },
    }
    (out_dir / "code_predictor_body_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
