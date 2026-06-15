#!/usr/bin/env python3
"""Generate the encoder-ablation noise-floor figure (vector PDF) from results/.

The encoder ablation fixes GPT-2 and swaps the frozen audio encoder, reporting
stage-1 (frozen-LM) numbers. Each encoder is trained over seeds {42, 43, 44};
this plots per-encoder FENSE as mean +/- std across seeds, with the baseline
CLAP shown as a +/- std band so an encoder counts as a real move only if it
clears the baseline's own run-to-run variance. Read-only on results/.

Figure:
  fig_encoder_noisefloor.pdf
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 7,
    "figure.facecolor": "white",
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})

SEEDS = [42, 43, 44]
METRIC = "FENSE"
# dir slug -> display name (matches the README encoder table)
ENCODERS = {
    "clap_music": "clap-music",
    "clap_music_speech": "clap-music-speech",
    "mert_95m": "mert-95m",
    "mert_330m": "mert-330m",
    "musicfm": "musicfm",
    "muq": "muq",
    "muq_mulan": "muq-mulan",
}
RESULTS = Path("results")
OUT = Path("reports/figures")


def _seed_tag(base, seed):
    return base if seed == 42 else f"{base}_seed{seed}"


def seed_scores(arch_dir, base):
    """FENSE across seeds for a stage-1 arch-ablation run; None per missing seed."""
    out = []
    for seed in SEEDS:
        p = arch_dir / f"gpt2_ablation_{_seed_tag(base, seed)}_s1.json"
        out.append(json.load(open(p))["metrics"][METRIC] if p.exists() else None)
    return [v for v in out if v is not None]


def baseline_scores():
    return seed_scores(RESULTS / "gpt2" / "ablations" / "arch", "gpt2")


def encoder_scores(slug):
    arch_dir = RESULTS / "encoder_swap" / slug / "gpt2" / "ablations" / "arch"
    return seed_scores(arch_dir, f"gpt2_{slug}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = baseline_scores()
    if not base:
        print("skip: no baseline CLAP seed runs")
        return
    b_mean, b_std = np.mean(base), np.std(base)

    names, means, stds = [], [], []
    for slug, name in ENCODERS.items():
        sc = encoder_scores(slug)
        if not sc:
            print(f"skip {name}: no seed runs")
            continue
        names.append(name)
        means.append(np.mean(sc))
        stds.append(np.std(sc))

    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, capsize=4, color="#4C72B0", alpha=0.85,
           label=f"encoder (mean ± std, {len(SEEDS)} seeds)")
    ax.axhline(b_mean, color="#C44E52", linestyle="--",
               label=f"baseline CLAP ({b_mean:.3f})")
    ax.axhspan(b_mean - b_std, b_mean + b_std, color="#C44E52", alpha=0.15,
               label="baseline ± std (noise floor)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel(f"{METRIC} (stage 1)")
    ax.set_title("Encoder ablation vs baseline-CLAP noise floor (GPT-2 fixed)")
    ax.legend()
    fig.tight_layout()
    out = OUT / "fig_encoder_noisefloor.pdf"
    fig.savefig(out)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
