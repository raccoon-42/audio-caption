"""Generate LP-MusicCaps captions for the thesis test split (CPU, separate venv).

RUN WITH THE LP-MUSICCAPS VENV, not `uv run`:
    cd ~/dev/audio-caption
    ~/dev/lp-music-caps/.venv/bin/python scripts/lp_musiccaps_generate.py

LP-MusicCaps pins torch==1.13.1 (CUDA 11.7), which cannot drive the RTX 5090
(Blackwell), so this runs on CPU. It imports the `lpmc` package (installed
editable in that venv) for the model + audio loader, reads the manifest written
by lp_musiccaps_manifest.py, and writes predictions in evaluate.py's schema so
the main env can score them with evaluate.compute_metrics.

Only the `pretrain` checkpoint is leakage-free (MSD pseudo-captions only, never
MusicCaps); supervised/transfer trained on MusicCaps-train and overlap ~half the
random seed-42 test split.

The checkpoint loader is inlined (not lpmc.utils.eval_utils.load_pretrained) to
avoid that module's pandas/sklearn imports and its hardcoded .cuda() calls.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from lpmc.music_captioning.model.bart import BartCaptionModel
from lpmc.utils.audio_utils import load_audio, STR_CH_FIRST

TARGET_SR = 16000
DURATION = 10
N_SAMPLES = TARGET_SR * DURATION


def get_audio(path):
    """Mirror lpmc's get_audio: 16 kHz mono, pad/truncate to 10 s chunks.
    MusicCaps clips are 10 s -> one [1, 160000] chunk -> one caption."""
    audio, _ = load_audio(path=path, ch_format=STR_CH_FIRST,
                          sample_rate=TARGET_SR, downmix_to_mono=True)
    if audio.ndim == 2:
        audio = audio.mean(0)
    if audio.shape[-1] < N_SAMPLES:
        pad = np.zeros(N_SAMPLES)
        pad[: audio.shape[-1]] = audio
        audio = pad
    ceil = int(audio.shape[-1] // N_SAMPLES)
    audio = np.stack(np.split(audio[: ceil * N_SAMPLES], ceil))
    return torch.from_numpy(audio.astype("float32"))


def load_model(exp_dir):
    config = OmegaConf.load(os.path.join(exp_dir, "hparams.yaml"))
    model = BartCaptionModel(max_length=config.max_length)
    ckpt = torch.load(os.path.join(exp_dir, "last.pth"), map_location="cpu")
    state_dict = ckpt["state_dict"]
    if any(k.startswith("module.") for k in state_dict):  # mdp checkpoints
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",
                    default="results/reference/lp_music_caps/test_split.json")
    ap.add_argument("--exp-dir",
                    default=str(Path.home() / "dev/lp-music-caps/lpmc/music_captioning/exp/pretrain/lp_music_caps"),
                    help="Dir with hparams.yaml + last.pth for the chosen checkpoint.")
    ap.add_argument("--framework", default="pretrain",
                    help="Label only (pretrain = leakage-free).")
    ap.add_argument("--num-beams", type=int, default=5)
    ap.add_argument("--out",
                    default="results/reference/lp_music_caps/predictions/lp_music_caps_predictions_pretrain.json")
    ap.add_argument("--limit", type=int, default=None, help="Cap rows (debug).")
    args = ap.parse_args()

    rows = json.loads(Path(args.manifest).read_text())
    if args.limit:
        rows = rows[: args.limit]
    print(f"Loaded {len(rows)} rows. Loading model from {args.exp_dir} (CPU) ...")
    model = load_model(args.exp_dir)

    preds, failed = [], 0
    for r in tqdm(rows, desc="LP-MusicCaps"):
        try:
            audio = get_audio(r["wav_path"])
            with torch.no_grad():
                out = model.generate(samples=audio, num_beams=args.num_beams)
            preds.append({"reference": r["caption"], "hypothesis": out[0].strip()})
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  FAIL {r['ytid']}: {e}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": f"lp-music-caps/{args.framework}",
        "stage": "reference",
        "framework": args.framework,
        "num_samples": len(preds),
        "num_failed": failed,
        "gen_kwargs": {"num_beams": args.num_beams, "max_length": model.max_length},
        "predictions": preds,
    }, indent=2))
    print(f"{len(preds)} predictions, {failed} failed -> {out}")


if __name__ == "__main__":
    main()
