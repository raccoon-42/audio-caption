"""Dump the seed-42 test-split manifest for LP-MusicCaps generation.

LP-MusicCaps runs in a separate torch-1.13 venv (CPU-only; its pin can't drive
the RTX 5090). To guarantee it scores the IDENTICAL rows as the pipeline without
trusting that venv's datasets.train_test_split to shuffle the same way, we dump
the split here (canonical env) as a flat manifest the gen script reads directly.

Reproduces evaluate.py's split exactly: ds.train_test_split(test_size=0.1,
seed=cfg["seed"])["test"]. Each row carries the per-clip wav already written by
prepare_study under data/segments/, so the gen script needs no dataset/HF deps.
"""
import argparse
import json
from pathlib import Path

import yaml
from datasets import load_from_disk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gpt2.yaml",
                    help="Only used for split seed + data_dir (must match evaluate.py).")
    ap.add_argument("--segments", default="data/segments")
    ap.add_argument("--out", default="results/reference/lp_music_caps/test_split.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    seed = cfg["seed"]
    ds = load_from_disk(str(Path(cfg["data_dir"])))
    test = ds.train_test_split(test_size=0.1, seed=seed)["test"]

    seg = Path(args.segments)
    rows = []
    missing = 0
    for ytid, s, e, cap in zip(test["ytid"], test["start_s"],
                               test["end_s"], test["caption"]):
        wav = seg / f"{ytid}_{s:.2f}_{e:.2f}.wav"
        if not wav.exists():
            missing += 1
        rows.append({"ytid": ytid, "start_s": s, "end_s": e,
                     "caption": cap, "wav_path": str(wav)})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"{len(rows)} test rows (seed {seed}) -> {out}")
    if missing:
        print(f"WARNING: {missing} wavs missing under {seg} (gen will skip them)")


if __name__ == "__main__":
    main()
