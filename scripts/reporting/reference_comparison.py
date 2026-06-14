#!/usr/bin/env python3
"""Compare the trained GPT-2 pipeline against zero-shot reference captioners
(Audio Flamingo, Qwen2-Audio) on the shared MusicCaps test split (read-only on
results/). Any reference under results/reference/<name>/ present is included;
missing ones are skipped with a note.

These captioners bypass the CLAP->projection->LM pipeline, so they sit in
separate reference rows, not controlled ablations. The fair pipeline comparator
is GPT-2 stage-1 (frozen LM, same as the encoder ablation); stage-2 is shown as
the fully fine-tuned upper end. GPT-2 error bars are the across-seed std
(42/43/44) for that stage; the captioners are single zero-shot passes (no band).

Outputs:
  reports/figures/fig_reference_comparison.pdf  -- grouped bars, one panel per
                                                   metric group (Legacy / AAC).
  reports/tables/reference_comparison.tex       -- booktabs LaTeX fragment.
"""
import json
from pathlib import Path
from statistics import mean, pstdev

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

RESULTS = Path("results")
FIG_OUT = Path("reports/figures")
TAB_OUT = Path("reports/tables")

LEGACY_METRICS = ["BLEU-1", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr-D", "SPICE", "SPIDEr"]
AAC_METRICS = ["SBERT-sim", "FER", "FENSE"]
ALL_METRICS = LEGACY_METRICS + AAC_METRICS
# (title, metrics, ymin): Legacy values are small so a 0 baseline is honest;
# AAC values all sit above ~0.24, so zoom that panel to make differences legible
# (axis truncation is flagged in the panel title).
METRIC_GROUPS = [("Legacy metrics", LEGACY_METRICS, 0.0),
                 ("AAC metrics", AAC_METRICS, None)]
LOWER_BETTER = {"FER"}

ARCH = RESULTS / "gpt2" / "ablations" / "arch"
# Each pipeline row: label -> seed-tagged eval files (seed42 first = point est.).
GPT2_S1_FILES = [ARCH / "gpt2_ablation_gpt2_s1.json",
                 ARCH / "gpt2_ablation_gpt2_seed43_s1.json",
                 ARCH / "gpt2_ablation_gpt2_seed44_s1.json"]
GPT2_S2_FILES = [ARCH / "gpt2_ablation_gpt2.json",
                 ARCH / "gpt2_ablation_gpt2_seed43.json",
                 ARCH / "gpt2_ablation_gpt2_seed44.json"]
# Reference captioners: (display label, results/reference/<name>/ dir). The
# metrics file is <name>_metrics_<name>.json (tag defaults to name).
REFERENCES = [
    ("Audio Flamingo", "audio_flamingo"),
    ("Qwen2-Audio", "qwen2_audio"),
    ("Qwen3-Omni Captioner", "qwen3_omni_captioner"),
]


def ref_file(name):
    return RESULTS / "reference" / name / f"{name}_metrics_{name}.json"


def load(p):
    p = Path(p)
    return json.load(open(p)) if p.exists() else None


def seed_stats(files):
    """(point-estimate metrics from seed42 file, per-metric std over all seeds)."""
    runs = [load(f) for f in files]
    runs = [r for r in runs if r]
    if not runs:
        return None, {}
    point = runs[0]["metrics"]
    std = {}
    for m in ALL_METRICS:
        vals = [r["metrics"][m] for r in runs if m in r["metrics"]]
        std[m] = pstdev(vals) if len(vals) > 1 else 0.0
    return point, std


def main():
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    TAB_OUT.mkdir(parents=True, exist_ok=True)

    s1_metrics, s1_std = seed_stats(GPT2_S1_FILES)
    s2_metrics, s2_std = seed_stats(GPT2_S2_FILES)
    if s1_metrics is None:
        raise SystemExit("Missing GPT-2 S1 baseline eval JSONs.")

    # rows: (label, metrics dict, per-metric std dict, n, is_reference)
    rows = [("GPT-2 (S1)", s1_metrics, s1_std, load(GPT2_S1_FILES[0])["num_samples"], False)]
    if s2_metrics:
        rows.append(("GPT-2 (S2)", s2_metrics, s2_std, load(GPT2_S2_FILES[0])["num_samples"], False))

    for label, name in REFERENCES:
        ref = load(ref_file(name))
        if ref:
            rows.append((label, ref["metrics"], {}, ref["num_samples"], True))
        else:
            print(f"skip {label}: missing {ref_file(name)}")

    if not any(is_ref for *_, is_ref in rows):
        raise SystemExit("No reference captioner results found under results/reference/.")

    _figure(rows)
    _table(rows)


def _figure(rows):
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974"]
    fig, axes = plt.subplots(1, len(METRIC_GROUPS), figsize=(11, 3.6),
                             gridspec_kw={"width_ratios": [len(g[1]) for g in METRIC_GROUPS]})
    for ax, (gname, metrics, ymin) in zip(axes, METRIC_GROUPS):
        x = range(len(metrics))
        n_rows = len(rows)
        w = 0.8 / n_rows
        for ri, (label, mvals, mstd, _, _) in enumerate(rows):
            offs = [xi + (ri - (n_rows - 1) / 2) * w for xi in x]
            heights = [mvals.get(m, 0.0) for m in metrics]
            errs = [mstd.get(m, 0.0) for m in metrics]
            ax.bar(offs, heights, width=w, color=colors[ri % len(colors)],
                   label=label, yerr=errs if any(errs) else None, capsize=2)
        # arrows mark metric direction; FER is lower-better.
        ticks = [f"{m}\n{'↓' if m in LOWER_BETTER else '↑'}" for m in metrics]
        ax.set_xticks(list(x))
        ax.set_xticklabels(ticks)
        vals = [mv.get(m, 0.0) for _, mv, _, _, _ in rows for m in metrics]
        top = max(0.7, max(vals) * 1.12)
        if ymin is None:  # auto-zoom: floor just below the smallest bar
            lo = min(vals)
            ymin = max(0.0, (int(lo * 10) / 10) - 0.05)
        title = gname if ymin == 0 else f"{gname} (y-axis from {ymin:g})"
        ax.set_title(title)
        ax.set_ylim(ymin, top)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("score")
    # Figure-level legend below the panels: an in-axes legend collides with the
    # tall SBERT/FENSE bars in the AAC panel.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=len(rows),
               loc="upper center", bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("Trained GPT-2 pipeline vs zero-shot reference captioners (MusicCaps test)",
                 fontweight="bold")
    out = FIG_OUT / "fig_reference_comparison.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def _best_per_metric(rows):
    """metric -> label of the best row (direction-aware)."""
    best = {}
    for m in ALL_METRICS:
        vals = [(lbl, mv[m]) for lbl, mv, _, _, _ in rows if m in mv]
        if not vals:
            continue
        pick = min if m in LOWER_BETTER else max
        best[m] = pick(vals, key=lambda kv: kv[1])[0]
    return best


def _table(rows):
    best = _best_per_metric(rows)
    header = " & ".join(
        f"{m} $\\{'downarrow' if m in LOWER_BETTER else 'uparrow'}$" for m in ALL_METRICS)

    lines = []
    for label, mvals, _, _, is_ref in rows:
        cells = []
        for m in ALL_METRICS:
            v = mvals.get(m)
            s = "--" if v is None else f"{v:.3f}"
            if v is not None and best.get(m) == label:
                s = f"\\textbf{{{s}}}"
            cells.append(s)
        rowlabel = label + (r"$^{\dagger}$" if is_ref else "")
        lines.append(f"{rowlabel} & " + " & ".join(cells) + r" \\")

    ns = ", ".join(f"{lbl} $n{{=}}{n}$" for lbl, _, _, n, _ in rows)
    tex = r"""\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\caption{Trained GPT-2 pipeline (stage-1 frozen LM, stage-2 fine-tuned) versus the
zero-shot reference captioners on the shared MusicCaps test split. GPT-2 values are
the seed-42 point estimate. $\uparrow$/$\downarrow$ mark metric direction; best per
column in bold. Decoding differs (pipeline greedy \texttt{max\_new\_tokens}=64,
captioners greedy 128) to reflect each model's natural output length.
$^{\dagger}$reference captioners may have seen MusicCaps in pre-training (treat as
reference upper bounds, not clean held-out scores). Sample counts: """ + ns + r""".}
\label{tab:reference-comparison}
\begin{tabular}{l""" + "c" * len(ALL_METRICS) + r"""}
\toprule
Model & """ + header + r""" \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    out = TAB_OUT / "reference_comparison.tex"
    out.write_text(tex)
    print(f"wrote {out}\n")
    print(tex)


if __name__ == "__main__":
    main()
