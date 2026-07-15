#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from qwen_tts import Qwen3TTSModel
from export_qwen3_tts_talker_next import build_nonstream_prefill


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--frames", type=int, required=True)
    ap.add_argument("--text", default="")
    ap.add_argument("--text-file", default="")
    ap.add_argument("--language", default="Chinese")
    ap.add_argument("--speaker", default="Vivian")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--prefix", default="")
    args = ap.parse_args()

    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8")
    else:
        text = args.text
    if not text:
        raise ValueError("--text or --text-file is required")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix or f"talker_loop_{args.frames}f"

    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)

    tts = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    talker = tts.model.talker
    prefill, tts_pad_embed = build_nonstream_prefill(tts, text, args.language, args.speaker)

    seq = prefill.detach().float()
    all_codes = []

    with torch.inference_mode():
        for frame in range(args.frames):
            hidden = talker.model(inputs_embeds=seq)[0]
            logits = talker.codec_head(hidden)
            main = torch.argmax(logits[:, -1, :], dim=-1)

            pred = talker.code_predictor.generate(
                inputs_embeds=torch.cat((hidden[:, -1:, :], talker.get_input_embeddings()(main[:, None])), dim=1),
                max_new_tokens=talker.config.num_code_groups - 1,
                do_sample=False,
                top_p=0.00001,
                top_k=1,
                temperature=0.0,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            frame_codes = torch.cat((main[:, None], pred.sequences), dim=-1)
            all_codes.append(frame_codes.detach().cpu().numpy().astype(np.int32).reshape(-1))

            if (frame + 1) % 10 == 0 or frame + 1 == args.frames:
                print(f"generated_codes {frame + 1}", flush=True)

            if frame + 1 < args.frames:
                pieces = [talker.get_input_embeddings()(frame_codes[:, :1])]
                for i in range(talker.config.num_code_groups - 1):
                    pieces.append(talker.code_predictor.get_input_embeddings()[i](frame_codes[:, i + 1:i + 2]))
                next_embed = torch.cat(pieces, dim=1).sum(1, keepdim=True) + tts_pad_embed
                seq = torch.cat((seq, next_embed), dim=1).detach().float()

        codes_np = np.stack(all_codes, axis=0).astype(np.int32)
        codes_np.tofile(out_dir / f"{prefix}_codes_ref_i32.bin")
        np.save(out_dir / f"{prefix}_codes_ref.npy", codes_np)

        group_min = codes_np.min(axis=0).tolist()
        group_max = codes_np.max(axis=0).tolist()
        print("code_group_min", group_min, flush=True)
        print("code_group_max", group_max, flush=True)

        codes_tensor = torch.from_numpy(codes_np).to(talker.device)
        wavs, sr = tts.model.speech_tokenizer.decode([{"audio_codes": codes_tensor}])

    wav = np.asarray(wavs[0], dtype=np.float32)
    wav.tofile(out_dir / f"{prefix}_wav_ref_f32.bin")

    meta = {
        "frames": args.frames,
        "codes_shape": list(codes_np.shape),
        "wav_shape": list(wav.shape),
        "sample_rate": int(sr),
        "text": text,
        "language": args.language,
        "speaker": args.speaker,
        "device": args.device,
        "prefix": prefix,
    }
    (out_dir / f"{prefix}_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
