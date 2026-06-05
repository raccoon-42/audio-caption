#!/usr/bin/env python3
"""Generate paper figures (vector PDF) from results/ JSONs.

Mirrors notebooks/comparison.ipynb styling but emits standalone PDFs for the
paper, restricted to the in-scope models (GPT-2, T5). Read-only on results/.

Figures:
  1. fig_training_curves.pdf  -- S1 vs S2 val-loss per epoch, all configs
                                 (S1 spread collapses after S2 fine-tuning).
  2. fig_arch_noisefloor.pdf  -- arch-ablation FENSE per config with the
                                 baseline +/- noise-floor band (significance).
  3. fig_decoding_fense_fer.pdf -- FER vs FENSE over decoding configs
                                 (trimming inflates FENSE via the fluency term).
  4. fig_baseline_groups.pdf  -- baseline metrics as grouped bars, one panel
                                 per metric group (legacy / AAC) so each group
                                 is read at its own scale; noise-floor error bars.
  5. fig_decoding_groups.pdf  -- decoding ablations as grouped bars, one panel
                                 per model x metric group.
"""
import json
import glob
import os
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
    "pdf.fonttype": 42,  # editable/embeddable fonts
})

MODELS = ["gpt2", "t5"]
NAMES = {"gpt2": "GPT-2", "t5": "T5"}
ARCH_CONFIGS = ["depth1", "depth3", "drop0.1", "drop0.5", "prefix4", "prefix16"]
LEGACY_METRICS = ["BLEU-1", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr-D", "SPICE", "SPIDEr"]
AAC_METRICS = ["SBERT-sim", "FER", "FENSE"]
METRIC_GROUPS = [("Legacy metrics", LEGACY_METRICS), ("AAC metrics", AAC_METRICS)]
RESULTS = Path("results")
OUT = Path("reports/figures")


def load(p):
    p = Path(p)
    return json.load(open(p)) if p.exists() else None


def training_runs(model, stage):
    """Return {label: history} for baseline + arch configs at a given stage."""
    runs = {}
    base = load(RESULTS / model / "training" / f"{model}_stage{stage}.json")
    if base:
        runs["baseline"] = base["history"]
    for cfg in ARCH_CONFIGS:
        d = load(RESULTS / model / "training" / f"{model}_{cfg}_stage{stage}.json")
        if d:
            runs[cfg] = d["history"]
    return runs


# --------------------------------------------------------------------------- #
# Figure 1: training curves (S1 spread -> S2 collapse)
# --------------------------------------------------------------------------- #
def fig_training_curves():
    fig, axes = plt.subplots(len(MODELS), 2, figsize=(7.0, 4.6), squeeze=False)
    for r, model in enumerate(MODELS):
        for c, stage in enumerate([1, 2]):
            ax = axes[r][c]
            runs = training_runs(model, stage)
            for label, hist in runs.items():
                ep = [h["epoch"] for h in hist]
                vl = [h["val_loss"] for h in hist]
                lw, z = (2.2, 5) if label == "baseline" else (1.0, 2)
                ax.plot(ep, vl, marker="", linewidth=lw, zorder=z,
                        label=label, color="black" if label == "baseline" else None)
            ax.set_title(f"{NAMES[model]} -- Stage {stage}")
            ax.grid(True, alpha=0.3)
            if c == 0:
                ax.set_ylabel("Val loss")
            if r == len(MODELS) - 1:
                ax.set_xlabel("Epoch")
        # shared legend per row (configs)
        axes[r][1].legend(ncol=2, fontsize=6, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_training_curves.pdf")
    plt.close(fig)
    print("wrote fig_training_curves.pdf")


# --------------------------------------------------------------------------- #
# Figure 2: arch-ablation FENSE vs noise floor
# --------------------------------------------------------------------------- #
def fig_arch_noisefloor():
    fig, axes = plt.subplots(1, len(MODELS), figsize=(7.0, 2.8), squeeze=False)
    for c, model in enumerate(MODELS):
        ax = axes[0][c]
        nf = load(RESULTS / model / f"{model}_noise_floor.json")
        std = nf["noise_floor"]["FENSE"]["std"] if nf else 0.0
        base = load(RESULTS / model / "ablations" / "arch" / f"{model}_ablation_{model}.json")
        base_f = base["metrics"]["FENSE"]
        labels, vals = ["baseline"], [base_f]
        for cfg in ARCH_CONFIGS:
            d = load(RESULTS / model / "ablations" / "arch" / f"{model}_ablation_{model}_{cfg}.json")
            if d:
                labels.append(cfg)
                vals.append(d["metrics"]["FENSE"])
        x = range(len(labels))
        # shaded +/- noise-floor band around baseline
        ax.axhspan(base_f - std, base_f + std, color="tab:blue", alpha=0.15,
                   label=r"baseline $\pm$ noise floor")
        ax.axhline(base_f, color="tab:blue", linewidth=1, linestyle="--")
        ax.bar(x, vals, color="0.6", width=0.6)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
        ax.set_title(f"{NAMES[model]} architecture ablations")
        ax.set_ylim(min(vals) - 2 * std - 0.01, max(vals) + 2 * std + 0.01)
        if c == 0:
            ax.set_ylabel("FENSE")
        ax.legend(fontsize=6, loc="upper left")
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_arch_noisefloor.pdf")
    plt.close(fig)
    print("wrote fig_arch_noisefloor.pdf")


# --------------------------------------------------------------------------- #
# Figure 3: FENSE vs FER across decoding configs (the metric artifact)
# --------------------------------------------------------------------------- #
def fig_decoding_fense_fer():
    fig, axes = plt.subplots(1, len(MODELS), figsize=(7.0, 3.0), squeeze=False)
    for c, model in enumerate(MODELS):
        ax = axes[0][c]
        files = glob.glob(str(RESULTS / model / "ablations" / "decoding" / f"{model}_ablation_*.json"))
        trim_x, trim_y, plain_x, plain_y = [], [], [], []
        for f in files:
            d = load(f)
            m = d.get("metrics", {})
            if "FER" not in m or "FENSE" not in m:
                continue
            is_trim = "trim" in os.path.basename(f)
            (trim_x if is_trim else plain_x).append(m["FER"])
            (trim_y if is_trim else plain_y).append(m["FENSE"])
        # baseline point
        base = load(RESULTS / model / "ablations" / "arch" / f"{model}_ablation_{model}.json")
        bm = base["metrics"]
        ax.scatter(plain_x, plain_y, s=18, color="tab:orange", alpha=0.8, label="no trim")
        ax.scatter(trim_x, trim_y, s=18, color="tab:blue", alpha=0.8, label="trimmed")
        ax.scatter([bm["FER"]], [bm["FENSE"]], s=70, marker="*", color="black",
                   zorder=5, label="greedy baseline")
        ax.set_title(f"{NAMES[model]} decoding configs")
        ax.set_xlabel("FER (fluency-error rate)")
        if c == 0:
            ax.set_ylabel("FENSE")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_decoding_fense_fer.pdf")
    plt.close(fig)
    print("wrote fig_decoding_fense_fer.pdf")


# --------------------------------------------------------------------------- #
# Figure 4: baseline metrics as grouped bars, one panel per metric group
# --------------------------------------------------------------------------- #
def fig_baseline_groups():
    base, floor = {}, {}
    for model in MODELS:
        d = load(RESULTS / model / "ablations" / "arch" / f"{model}_ablation_{model}.json")
        base[model] = d["metrics"]
        nf = load(RESULTS / model / f"{model}_noise_floor.json")
        floor[model] = nf["noise_floor"] if nf else {}

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9),
                             gridspec_kw={"width_ratios": [len(LEGACY_METRICS),
                                                           len(AAC_METRICS)]})
    width = 0.38
    colors = {"gpt2": "tab:blue", "t5": "tab:orange"}
    for ax, (gtitle, gmetrics) in zip(axes, METRIC_GROUPS):
        x = range(len(gmetrics))
        for i, model in enumerate(MODELS):
            vals = [base[model][m] for m in gmetrics]
            errs = [floor[model][m]["std"] if m in floor[model] else 0 for m in gmetrics]
            ax.bar([xi + (i - 0.5) * width for xi in x], vals, width,
                   yerr=errs, capsize=2, color=colors[model], label=NAMES[model])
        ax.set_xticks(list(x))
        ax.set_xticklabels(gmetrics, rotation=45, ha="right", fontsize=7)
        ax.set_title(gtitle, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT / "fig_baseline_groups.pdf")
    plt.close(fig)
    print("wrote fig_baseline_groups.pdf")


# --------------------------------------------------------------------------- #
# Figure 5: decoding ablations as grouped bars, model x metric-group panels
# --------------------------------------------------------------------------- #
def fig_decoding_groups(top_n=8):
    cmap = plt.get_cmap("tab10")
    # FER gets its own panel: its near-zero values (trimmed configs) would
    # otherwise share an axis with the 0.5-0.6 similarity scores and crush
    # their resolution.
    panels = [("Legacy metrics", LEGACY_METRICS),
              ("AAC metrics", ["SBERT-sim", "FENSE"]),
              (r"FER $\downarrow$", ["FER"])]
    fig, axes = plt.subplots(len(MODELS), 3, figsize=(7.0, 6.0), squeeze=False,
                             gridspec_kw={"width_ratios": [7, 2.4, 1.6]})
    for r, model in enumerate(MODELS):
        base = load(RESULTS / model / "ablations" / "arch" / f"{model}_ablation_{model}.json")["metrics"]
        configs = []
        for f in glob.glob(str(RESULTS / model / "ablations" / "decoding" / f"{model}_ablation_*.json")):
            d = load(f)
            m = d.get("metrics", {})
            if "FENSE" in m:
                tag = os.path.basename(f)[len(f"{model}_ablation_"):-len(".json")]
                configs.append((tag, m))
        configs.sort(key=lambda kv: kv[1]["FENSE"], reverse=True)
        rows = [("greedy", base)] + configs[:top_n]
        for c, (gtitle, gmetrics) in enumerate(panels):
            ax = axes[r][c]
            n_m = len(gmetrics)
            if gtitle == "AAC metrics":
                # point plot on a zoomed (non-zero) axis: similarity scores
                # cluster in a narrow band, and truncated bars would distort
                # proportions while markers do not.
                lo, hi = 1.0, 0.0
                for j, met in enumerate(gmetrics):
                    vals = [m.get(met) for _, m in rows]
                    lo, hi = min(lo, min(vals)), max(hi, max(vals))
                    ax.plot(range(len(rows)), vals, marker="o" if j == 0 else "s",
                            markersize=4, linewidth=0.8, linestyle="--",
                            color=cmap(j % 10), label=met)
                pad = (hi - lo) * 0.18 + 0.005
                ax.set_ylim(lo - pad, hi + pad)
            else:
                width = 0.8 / n_m
                top = 0.0
                for j, met in enumerate(gmetrics):
                    vals = [m.get(met, 0) for _, m in rows]
                    top = max(top, max(vals))
                    xs = [i + (j - (n_m - 1) / 2) * width for i in range(len(rows))]
                    color = "tab:red" if met == "FER" else cmap(j % 10)
                    ax.bar(xs, vals, width, color=color, label=met)
                ax.set_ylim(0, top * 1.3)   # headroom so the legend clears the bars
            ax.set_xticks(range(len(rows)))
            if len(gmetrics) == 1:
                # config order matches the adjacent panel; labels would not fit
                ax.set_xticklabels(["" for _ in rows])
            else:
                ax.set_xticklabels([t for t, _ in rows], rotation=55, ha="right", fontsize=5)
            ax.set_title(f"{NAMES[model]} -- {gtitle}", fontweight="bold", fontsize=8)
            ax.grid(True, axis="y", alpha=0.3)
            if len(gmetrics) > 1:
                ax.legend(fontsize=5, ncol=4 if n_m > 4 else 2,
                          loc="upper center" if gtitle != "AAC metrics" else "lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_decoding_groups.pdf")
    plt.close(fig)
    print("wrote fig_decoding_groups.pdf")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_training_curves()
    fig_arch_noisefloor()
    fig_decoding_fense_fer()
    fig_baseline_groups()
    fig_decoding_groups()
    print(f"figures written to {OUT}/")


if __name__ == "__main__":
    main()
