#!/usr/bin/env python3
"""Generate the LR-search figures (vector PDF) from results/.../lr_search/.

Shows where the selected learning rate sits within the Optuna search, for both
the language-model ablation (GPT-2, T5) and the encoder ablation (GPT-2 fixed,
one search per swapped encoder). Read-only on results/.

Figures:
  fig_lr_search.pdf -- 2x2 grid, one row per language model:
    col 1 (stage 1): val-loss vs the 1-D stage-1 LR sweep, chosen LR starred.
    col 2 (stage 2): the 2-D (projection LR x LM LR) trials as a scatter
                     coloured by val-loss, chosen point starred.
  fig_lr_search_encoders.pdf -- small multiples, one panel per encoder, each
    the 1-D stage-1 LR sweep with the chosen LR starred (the encoder ablation
    reports stage 1, so the stage-1 search is the relevant axis).
"""
import json
from pathlib import Path

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

MODELS = ["gpt2", "t5"]
NAMES = {"gpt2": "GPT-2", "t5": "T5"}
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


def load(model):
    p = RESULTS / model / "lr_search" / f"{model}_lr_search.json"
    return json.load(open(p)) if p.exists() else None


def load_encoder(slug):
    p = RESULTS / "encoder_swap" / slug / "gpt2" / "lr_search" / "gpt2_lr_search.json"
    return json.load(open(p)) if p.exists() else None


def stage1_panel(ax, data, title):
    s1 = data["stage1"]
    trials = [t for t in s1["all_trials"] if t.get("value") is not None]
    lrs = [t["params"]["stage1_lr"] for t in trials]
    vals = [t["value"] for t in trials]
    ax.scatter(lrs, vals, s=20, color="#4C72B0", alpha=0.8, label="trials")
    ax.scatter([s1["best_lr"]], [s1["best_val_loss"]], marker="*", s=220,
               color="#C44E52", zorder=5, label=f"chosen ({s1['best_lr']:.2e})")
    ax.set_xscale("log")
    ax.set_xlabel("stage-1 LR")
    ax.set_ylabel("val loss")
    ax.set_title(title)
    ax.legend()


def stage2_panel(ax, fig, model, data):
    s2 = data["stage2"]
    trials = [t for t in s2["all_trials"] if t.get("value") is not None]
    proj = [t["params"]["stage2_proj_lr"] for t in trials]
    lm = [t["params"]["stage2_lm_lr"] for t in trials]
    vals = [t["value"] for t in trials]
    sc = ax.scatter(proj, lm, c=vals, cmap="viridis_r", s=40, alpha=0.9)
    bp = s2["best_params"]
    ax.scatter([bp["stage2_proj_lr"]], [bp["stage2_lm_lr"]], marker="*", s=220,
               edgecolor="#C44E52", facecolor="none", linewidth=2, zorder=5,
               label="chosen")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("projection LR")
    ax.set_ylabel("LM LR")
    ax.set_title(f"{NAMES[model]} — stage 2 (projection + LM)")
    ax.legend()
    fig.colorbar(sc, ax=ax, label="val loss")


def models_figure():
    fig, axes = plt.subplots(len(MODELS), 2, figsize=(11, 4.4 * len(MODELS)))
    for row, model in enumerate(MODELS):
        data = load(model)
        if not data:
            print(f"skip {model}: no lr_search JSON")
            continue
        stage1_panel(axes[row, 0], data, f"{NAMES[model]} — stage 1 (projection)")
        stage2_panel(axes[row, 1], fig, model, data)
    fig.tight_layout()
    out = OUT / "fig_lr_search.pdf"
    fig.savefig(out)
    print(f"Saved -> {out}")


def encoders_figure():
    loaded = [(name, load_encoder(slug)) for slug, name in ENCODERS.items()]
    loaded = [(n, d) for n, d in loaded if d]
    if not loaded:
        print("skip encoders: no lr_search JSONs")
        return
    ncols = 3
    nrows = (len(loaded) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows))
    axes = axes.ravel()
    for ax, (name, data) in zip(axes, loaded):
        stage1_panel(ax, data, f"{name} — stage 1")
    for ax in axes[len(loaded):]:
        ax.axis("off")
    fig.tight_layout()
    out = OUT / "fig_lr_search_encoders.pdf"
    fig.savefig(out)
    print(f"Saved -> {out}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    models_figure()
    encoders_figure()


if __name__ == "__main__":
    main()
