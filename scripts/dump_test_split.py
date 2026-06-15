"""Dump the seed-42 test split as a flat manifest (results/test_split.json).

Reference captioners run in their own venvs (LP-MusicCaps on torch 1.13 CPU,
BLAP on torch cu128) and shouldn't depend on datasets/HF. This dumps the shared
test split once, in the canonical env, so every generator scores the IDENTICAL
rows without trusting another venv's datasets.train_test_split to shuffle the
same way.

Reproduces evaluate.py's split exactly: ds.train_test_split(test_size=0.1,
seed=cfg["seed"])["test"]. Each row carries the per-clip wav already written by
prepare_study under data/segments/, plus is_audioset_eval for leakage splits.
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
    ap.add_argument("--out", default="results/test_split.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    seed = cfg["seed"]
    ds = load_from_disk(str(Path(cfg["data_dir"])))
    test = ds.train_test_split(test_size=0.1, seed=seed)["test"]

    seg = Path(args.segments)
    # is_audioset_eval marks the official MusicCaps eval half; carried so a
    # reference model fine-tuned on the official train split can be checked for
    # leakage on this random split (in-train subset vs held-out subset).
    evals = (test["is_audioset_eval"] if "is_audioset_eval" in test.column_names
             else [None] * len(test["ytid"]))
    rows = []
    missing = 0
    for ytid, s, e, cap, ev in zip(test["ytid"], test["start_s"],
                                   test["end_s"], test["caption"], evals):
        wav = seg / f"{ytid}_{s:.2f}_{e:.2f}.wav"
        if not wav.exists():
            missing += 1
        rows.append({"ytid": ytid, "start_s": s, "end_s": e,
                     "caption": cap, "wav_path": str(wav),
                     "is_audioset_eval": ev})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"{len(rows)} test rows (seed {seed}) -> {out}")
    if missing:
        print(f"WARNING: {missing} wavs missing under {seg} (gen will skip them)")


if __name__ == "__main__":
    main()
