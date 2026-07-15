#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from qwen_tts import Qwen3TTSModel


class TalkerFull(torch.nn.Module):
    def __init__(self, talker):
        super().__init__()
        self.model = talker.model
        self.codec_head = talker.codec_head

    def forward(self, inputs_embeds):
        seq_len = inputs_embeds.shape[1]
        mask = torch.full((seq_len, seq_len), torch.finfo(inputs_embeds.dtype).min, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        mask = torch.triu(mask, diagonal=1).view(1, 1, seq_len, seq_len)
        position_ids = torch.arange(seq_len, device=inputs_embeds.device).view(1, seq_len)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        hidden = self.model(
            input_ids=None,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            cache_position=torch.arange(seq_len, device=inputs_embeds.device),
        ).last_hidden_state
        logits = self.codec_head(hidden)
        return hidden, logits


def build_nonstream_prefill(tts, text, language, speaker):
    model = tts.model
    talker = model.talker
    input_ids = tts._tokenize_texts([tts._build_assistant_text(text)])
    input_id = input_ids[0]
    language_key = language.lower()
    speaker_key = speaker.lower()
    if language_key == "auto":
        raise NotImplementedError("language=Auto is not supported by this non-streaming ncnn parity helper yet")
    if speaker_key not in model.config.talker_config.spk_id:
        raise NotImplementedError(f"Speaker {speaker} not implemented")
    if language_key not in model.config.talker_config.codec_language_id:
        raise NotImplementedError(f"Language {language} not implemented")
    if (
        language_key == "chinese"
        and model.config.talker_config.spk_is_dialect.get(speaker_key, False) is not False
    ):
        language_key = model.config.talker_config.spk_is_dialect[speaker_key]
    language_id = model.config.talker_config.codec_language_id[language_key]
    spk_id = model.config.talker_config.spk_id[speaker_key]
    speaker_embed = talker.get_input_embeddings()(torch.tensor(spk_id, device=input_id.device, dtype=input_id.dtype))
    tts_bos_embed, tts_eos_embed, tts_pad_embed = talker.text_projection(
        talker.get_text_embeddings()(torch.tensor(
            [[model.config.tts_bos_token_id, model.config.tts_eos_token_id, model.config.tts_pad_token_id]],
            device=talker.device, dtype=input_id.dtype,
        ))
    ).chunk(3, dim=1)
    codec0 = talker.get_input_embeddings()(torch.tensor([[
        model.config.talker_config.codec_think_id,
        model.config.talker_config.codec_think_bos_id,
        language_id,
        model.config.talker_config.codec_think_eos_id,
    ]], device=talker.device, dtype=input_id.dtype))
    codec1 = talker.get_input_embeddings()(torch.tensor([[
        model.config.talker_config.codec_pad_id,
        model.config.talker_config.codec_bos_id,
    ]], device=talker.device, dtype=input_id.dtype))
    codec_input = torch.cat([codec0, speaker_embed.view(1, 1, -1), codec1], dim=1)
    role = talker.text_projection(talker.get_text_embeddings()(input_id[:, :3]))
    body = torch.cat((tts_pad_embed.expand(-1, codec_input.shape[1] - 2, -1), tts_bos_embed), dim=1) + codec_input[:, :-1]
    embeds = torch.cat((role, body), dim=1)
    text_part = torch.cat((
        talker.text_projection(talker.get_text_embeddings()(input_id[:, 3:-5])),
        tts_eos_embed,
    ), dim=1) + talker.get_input_embeddings()(torch.tensor(
        [[model.config.talker_config.codec_pad_id] * (input_id[:, 3:-5].shape[1] + 1)],
        device=talker.device, dtype=input_id.dtype,
    ))
    codec_bos = tts_pad_embed + talker.get_input_embeddings()(torch.tensor(
        [[model.config.talker_config.codec_bos_id]], device=talker.device, dtype=input_id.dtype,
    ))
    return torch.cat((embeds, text_part, codec_bos), dim=1), tts_pad_embed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--codes", required=True, help="int32 file containing the first 16 generated codebooks")
    ap.add_argument("--text", default="ä½ å¥½ï¼æ¬¢è¿ä½¿ç¨éä¹åé®è¯­é³åæã")
    ap.add_argument("--language", default="Chinese")
    ap.add_argument("--speaker", default="Vivian")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tts = Qwen3TTSModel.from_pretrained(args.model, device_map=args.device, dtype=torch.float32, attn_implementation="eager")
    talker = tts.model.talker
    prefill, tts_pad_embed = build_nonstream_prefill(tts, args.text, args.language, args.speaker)
    codes = np.fromfile(args.codes, dtype=np.int32).reshape(-1)
    if codes.size != 16:
        raise ValueError(f"expected 16 codes, got {codes.size}")
    code_t = torch.from_numpy(codes.reshape(1, 16)).to(talker.device).long()
    pieces = [talker.get_input_embeddings()(code_t[:, :1])]
    for i in range(15):
        pieces.append(talker.code_predictor.get_input_embeddings()[i](code_t[:, i + 1:i + 2]))
    next_embed = torch.cat(pieces, dim=1).sum(1, keepdim=True) + tts_pad_embed
    seq = torch.cat((prefill, next_embed), dim=1).detach().float()
    wrapper = TalkerFull(talker).eval()
    with torch.inference_mode():
        hidden, logits = wrapper(seq)
    seq.cpu().numpy().astype(np.float32).tofile(out_dir / "talker_next_input_f32.bin")
    hidden.cpu().numpy().astype(np.float32).tofile(out_dir / "talker_next_hidden_ref_f32.bin")
    logits.cpu().numpy().astype(np.float32).tofile(out_dir / "talker_next_logits_ref_f32.bin")
    np.asarray(codes, dtype=np.int32).tofile(out_dir / "talker_next_prev_codes_i32.bin")
    traced = torch.jit.trace(wrapper, seq.clone(), strict=False)
    traced.save(str(out_dir / "talker_next_s23.pt"))
    meta = {
        "input_shape": list(seq.shape),
        "hidden_shape": list(hidden.shape),
        "logits_shape": list(logits.shape),
        "prev_codes": codes.astype(int).tolist(),
        "next_main_argmax": int(torch.argmax(logits[:, -1, :], dim=-1).item()),
    }
    (out_dir / "talker_next_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
