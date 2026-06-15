"""Generate BLAP captions for the thesis test split (Ubuntu, RTX 5090, BLAP venv).

BLAP (Lanzendoerfer et al., ICASSP 2025) is a BLIP-2 style music captioner:
a CLAP/HTS-AT audio encoder + Q-Former + frozen Flan-T5-XL. It bypasses the
CLAP->projection->LM pipeline, so it is a reference row, not a controlled
ablation. Weights are loaded offline from the transferred blap_bundle/ (see
scripts/blap_fetch.sh); nothing is fetched at run time.

Setup (once, from the repo root with blap_bundle/ in place):
    uv venv --python 3.12 .venv-blap
    uv pip install --python .venv-blap torch torchaudio \
        --index-url https://download.pytorch.org/whl/cu128
    uv pip install --python .venv-blap blap_bundle/blap-0.1.10-py3-none-any.whl --no-deps
    uv pip install --python .venv-blap transformers pytorch-lightning \
        torchlibrosa soundfile pandas "numpy>=2"
    export HF_HOME=$PWD/blap_bundle/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

Run:
    .venv-blap/bin/python scripts/blap_generate.py \
        --ckpt blap_bundle/model/checkpoint.ckpt \
        --model-config blap_bundle/model/config.json

BLAP expects 48 kHz mono input; data/segments/ clips are 16 kHz, so they are
upsampled here. The config's atRandom=true means the Flan-T5-XL weights load
from the checkpoint, not from Hugging Face. Predictions carry is_audioset_eval
so scoring can split the seed-42 test into the in-MusicCaps-train subset (where
a leaked checkpoint inflates scores) versus the held-out subset.
"""
import argparse
import json
from pathlib import Path

import soundfile as sf
import torch
import torchaudio.functional as AF
from tqdm import tqdm

from blap.model.BLAP2.BLAP2_Pretrain import BLAP2_Stage2

TARGET_SR = 48000
CLIP_SAMPLES = 480000  # 10 s at 48 kHz (config.audio_encoder.audio_cfg.clip_samples)


def load_audio(path):
    """Load a clip, downmix to mono, resample to 48 kHz, pad/trim to 10 s."""
    wav, sr = sf.read(path, dtype="float32", always_2d=True)  # (T, C)
    x = torch.from_numpy(wav).mean(dim=1)  # mono (T,)
    if sr != TARGET_SR:
        x = AF.resample(x, sr, TARGET_SR)
    if x.shape[0] < CLIP_SAMPLES:
        x = torch.nn.functional.pad(x, (0, CLIP_SAMPLES - x.shape[0]))
    else:
        x = x[:CLIP_SAMPLES]
    return x.unsqueeze(0)  # (1, CLIP_SAMPLES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",
                    default="results/test_split.json")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--prompt",
                    default="Provide a music caption for this audio clip. "
                            "Do not mention audio quality")
    ap.add_argument("--num-beams", type=int, default=5)
    ap.add_argument("--max-len", type=int, default=40)
    ap.add_argument("--min-len", type=int, default=30)
    ap.add_argument("--out",
                    default="results/reference/blap/predictions/blap_predictions_blap.json")
    ap.add_argument("--limit", type=int, default=None, help="Cap rows (debug).")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = json.loads(Path(args.manifest).read_text())
    if args.limit:
        rows = rows[: args.limit]
    print(f"Loaded {len(rows)} rows. Loading BLAP on {device} ...")
    model = BLAP2_Stage2.from_checkpoint(checkpoint_path=args.ckpt,
                                         modelConfig=args.model_config)
    model = model.eval().to(device)

    preds, failed = [], 0
    for r in tqdm(rows, desc="BLAP"):
        try:
            audio = load_audio(r["wav_path"]).to(device)
            with torch.no_grad():
                out = model.predict_answers(audio, args.prompt,
                                            num_beams=args.num_beams,
                                            max_len=args.max_len,
                                            min_len=args.min_len,
                                            device=device)
            preds.append({"reference": r["caption"],
                          "hypothesis": out[0].strip(),
                          "ytid": r.get("ytid"),
                          "is_audioset_eval": r.get("is_audioset_eval")})
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  FAIL {r.get('ytid')}: {e}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": "blap/stage2",
        "stage": "reference",
        "framework": "blap",
        "num_samples": len(preds),
        "num_failed": failed,
        "gen_kwargs": {"num_beams": args.num_beams, "max_len": args.max_len,
                       "min_len": args.min_len, "prompt": args.prompt},
        "predictions": preds,
    }, indent=2))
    print(f"{len(preds)} predictions, {failed} failed -> {out}")


if __name__ == "__main__":
    main()
