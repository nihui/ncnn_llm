#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

from qwen_tts import Qwen3TTSModel
from qwen_tts.core.models.modeling_qwen3_tts import rotate_half
from export_qwen3_tts_talker_next import build_nonstream_prefill


def apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class TalkerSimpleRope(torch.nn.Module):
    def __init__(self, talker):
        super().__init__()
        self.layers = talker.model.layers
        self.norm = talker.model.norm
        self.codec_head = talker.codec_head

    def forward(self, inputs_embeds, attention_mask, cos_cache, sin_cache):
        hidden = inputs_embeds
        for layer in self.layers:
            residual = hidden
            x = layer.input_layernorm(hidden)
            attn = layer.self_attn
            bsz, seqlen, _ = x.shape
            q = attn.q_norm(attn.q_proj(x).view(bsz, seqlen, attn.config.num_attention_heads, attn.head_dim)).transpose(1, 2)
            k = attn.k_norm(attn.k_proj(x).view(bsz, seqlen, attn.config.num_key_value_heads, attn.head_dim)).transpose(1, 2)
            v = attn.v_proj(x).view(bsz, seqlen, attn.config.num_key_value_heads, attn.head_dim).transpose(1, 2)
            q, k = apply_rope(q, k, cos_cache, sin_cache)
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=0.0, scale=attn.scaling, enable_gqa=True)
            x = x.transpose(1, 2).contiguous().reshape(bsz, seqlen, -1)
            hidden = residual + attn.o_proj(x)
            residual = hidden
            hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))
        hidden = self.norm(hidden)
        logits = self.codec_head(hidden)
        return hidden, logits


def causal_mask(seq_len: int, device):
    mask = torch.full((seq_len, seq_len), torch.finfo(torch.float32).min, dtype=torch.float32, device=device)
    return torch.triu(mask, diagonal=1).view(1, 1, seq_len, seq_len)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--text", default="ä½ å¥½ï¼æ¬¢è¿ä½¿ç¨éä¹åé®è¯­é³åæã")
    ap.add_argument("--language", default="Chinese")
    ap.add_argument("--speaker", default="Vivian")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--first-frame-codes", default="")
    ap.add_argument("--trace-decode", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tts = Qwen3TTSModel.from_pretrained(args.model, device_map=args.device, dtype=torch.float32, attn_implementation="eager")
    talker = tts.model.talker
    embeds, tts_pad_embed = build_nonstream_prefill(tts, args.text, args.language, args.speaker)
    embeds = embeds.detach().float()
    seq_len = embeds.shape[1]
    mask = causal_mask(seq_len, embeds.device)
    pos = torch.arange(seq_len, device=embeds.device).view(1, seq_len)
    pos3 = pos.unsqueeze(0).expand(3, -1, -1)
    with torch.no_grad():
        cos3, sin3 = talker.model.rotary_emb(embeds, pos3)
        cos = cos3[0]
        sin = sin3[0]
    wrapper = TalkerSimpleRope(talker).eval()
    with torch.inference_mode():
        hidden, logits = wrapper(embeds, mask, cos, sin)

    embeds.cpu().numpy().astype("float32").tofile(out_dir / "talker_simple_input_f32.bin")
    mask.cpu().numpy().astype("float32").tofile(out_dir / "talker_simple_mask_f32.bin")
    cos.cpu().numpy().astype("float32").tofile(out_dir / "talker_simple_cos_f32.bin")
    sin.cpu().numpy().astype("float32").tofile(out_dir / "talker_simple_sin_f32.bin")
    hidden.cpu().numpy().astype("float32").tofile(out_dir / "talker_simple_hidden_ref_f32.bin")
    logits.cpu().numpy().astype("float32").tofile(out_dir / "talker_simple_logits_ref_f32.bin")
    tts_pad_embed.detach().cpu().numpy().astype("float32").tofile(out_dir / "tts_pad_embed_f32.bin")

    if args.first_frame_codes:
        codes = np.fromfile(args.first_frame_codes, dtype=np.int32).reshape(1, 16)
        code_t = torch.from_numpy(codes).to(talker.device).long()
        pieces = [talker.get_input_embeddings()(code_t[:, :1])]
        for i in range(15):
            pieces.append(talker.code_predictor.get_input_embeddings()[i](code_t[:, i + 1:i + 2]))
        # In non-streaming mode trailing_text_hidden is tts_pad_embed for every generated frame.
        next_embed = torch.cat(pieces, dim=1).sum(1, keepdim=True) + tts_pad_embed
        decode_pos = seq_len
        decode_mask = torch.zeros((1, seq_len + 1), dtype=torch.float32, device=embeds.device)
        pos3_next = torch.full((3, 1, 1), decode_pos, dtype=torch.long, device=embeds.device)
        with torch.no_grad():
            cos3_next, sin3_next = talker.model.rotary_emb(next_embed, pos3_next)
            cos_next = cos3_next[0]
            sin_next = sin3_next[0]
        full23 = torch.cat((embeds, next_embed.detach().float()), dim=1)
        full23_mask = causal_mask(seq_len + 1, embeds.device)
        full23_pos = torch.arange(seq_len + 1, device=embeds.device).view(1, seq_len + 1)
        full23_pos3 = full23_pos.unsqueeze(0).expand(3, -1, -1)
        with torch.no_grad():
            full23_cos3, full23_sin3 = talker.model.rotary_emb(full23, full23_pos3)
        with torch.inference_mode():
            full23_hidden, full23_logits = wrapper(full23, full23_mask, full23_cos3[0], full23_sin3[0])
        next_embed.detach().cpu().numpy().astype("float32").tofile(out_dir / "talker_simple_decode_embed_f32.bin")
        decode_mask.cpu().numpy().astype("float32").tofile(out_dir / "talker_simple_decode_mask_f32.bin")
        cos_next.cpu().numpy().astype("float32").tofile(out_dir / "talker_simple_decode_cos_f32.bin")
        sin_next.cpu().numpy().astype("float32").tofile(out_dir / "talker_simple_decode_sin_f32.bin")
        full23_hidden.cpu().numpy().astype("float32").tofile(out_dir / "talker_simple_full23_hidden_ref_f32.bin")
        full23_logits.cpu().numpy().astype("float32").tofile(out_dir / "talker_simple_full23_logits_ref_f32.bin")
        if args.trace_decode:
            trace_decode_mask = torch.zeros((1, 1, 1, 1), dtype=torch.float32, device=embeds.device)
            decode_traced = torch.jit.trace(
                wrapper,
                (next_embed.detach().float().clone(), trace_decode_mask, cos_next.clone(), sin_next.clone()),
                strict=False,
            )
            decode_traced.save(str(out_dir / "talker_simple_rope_decode_s1.pt"))
    traced = torch.jit.trace(wrapper, (embeds.clone(), mask.clone(), cos.clone(), sin.clone()), strict=False)
    traced.save(str(out_dir / "talker_simple_rope_s22.pt"))
    meta = {
        "input_shape": list(embeds.shape),
        "mask_shape": list(mask.shape),
        "cos_shape": list(cos.shape),
        "sin_shape": list(sin.shape),
        "hidden_shape": list(hidden.shape),
        "logits_shape": list(logits.shape),
        "argmax_last": int(torch.argmax(logits[:, -1, :], dim=-1).item()),
    }
    if args.first_frame_codes:
        meta["decode_next_argmax_full23"] = int(torch.argmax(full23_logits[:, -1, :], dim=-1).item())
    (out_dir / "talker_simple_rope_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
