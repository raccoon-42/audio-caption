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

Outputs (figures split by reference domain for poster use; pipeline rows anchor both):
  reports/figures/fig_reference_comparison_music.pdf    -- vs music-domain captioners.
  reports/figures/fig_reference_comparison_general.pdf  -- vs general-audio captioners.
  reports/tables/reference_comparison.tex               -- booktabs LaTeX fragment (all rows).
"""
import json
from collections import defaultdict, namedtuple
from pathlib import Path
from statistics import mean, pstdev

# One table/figure row. in_fig: show in the figure (False = table-only, keeps the
# figure legible). bold: eligible for best-in-column bolding (only full-test,
# non-leaked rows -- daggered upper bounds and held-out n=284 subsets are excluded).
Row = namedtuple("Row", "label metrics std n is_ref leaks domain in_fig bold")

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
# Best-decoding rows: the top-FENSE decoding config per model from the decoding
# ablation (single seed-42 eval, no across-seed band). The fair "what our pipeline
# does at its best", vs the greedy S1/S2 controlled comparators.
GPT2_BEST_FILE = RESULTS / "gpt2" / "ablations" / "decoding" / "gpt2_ablation_trim_ngram2.json"
T5_BEST_FILE = RESULTS / "t5" / "ablations" / "decoding" / "t5_ablation_trim_rep1.2.json"
# Reference captioners: (display label, results/reference/<name>/ dir, leaks).
# The metrics file is <name>_metrics_<name>.json (tag defaults to name). `leaks`
# marks rows that may have seen MusicCaps in training -> daggered upper bounds.
# LP-MusicCaps (pretrain) trained ONLY on MSD pseudo-captions -> leakage-free.
# (display label, results/reference/<name>/ dir, leaks, domain, in_fig). domain splits
# the output into two poster figures: "music" = music-specific captioners (+ our
# pipeline, trained for music), "general" = general-purpose audio captioners. Leaking
# domain models (LP-transfer, BLAP) are shown in the music figure alongside their
# held-out clean twins, so the leakage inflation is visible (advisor's ask). Held-out
# rows (name ends with _clean) are not bold-eligible (n=284, not comparable).
REFERENCES = [
    ("Audio Flamingo", "audio_flamingo", True, "general", True),
    ("Qwen2-Audio", "qwen2_audio", True, "general", True),
    ("Qwen3-Omni Captioner", "qwen3_omni_captioner", True, "general", True),
    ("LP-MusicCaps (pretrain)", "lp_music_caps", False, "music", True),
    ("LP-MusicCaps (transfer)", "lp_music_caps_transfer", True, "music", True),
    ("LP-MusicCaps (transfer, held-out)", "lp_music_caps_transfer_clean", False, "music", True),
    ("BLAP", "blap", True, "music", True),
    ("BLAP (held-out)", "blap_clean", False, "music", True),
]
DOMAIN_RANK = {"music": 0, "general": 1}  # music-domain left, general right
# Per-domain colour families: cool/green = music, warm = general; shades within a
# family distinguish individual models. Hatch (not colour) marks leakage.
DOMAIN_COLORS = {
    "music": ["#08306B", "#3182BD", "#9ECAE1", "#5BBFA5", "#006D2C",
              "#41AB5D", "#78C679", "#ADDD8E", "#D9F0A3"],
    "general": ["#C0392B", "#E67E22", "#D4A017"],
}


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

    # Our GPT-2/T5 pipeline is trained for music captioning -> "music" domain. Greedy
    # S1/S2 are controlled comparators (S2 table-only, superseded in the figure by the
    # best-decoding row); all pipeline rows are full-test non-leaked -> bold-eligible.
    rows = [Row("GPT-2 (S1)", s1_metrics, s1_std, load(GPT2_S1_FILES[0])["num_samples"],
                False, False, "music", True, True)]
    if s2_metrics:
        rows.append(Row("GPT-2 (S2)", s2_metrics, s2_std, load(GPT2_S2_FILES[0])["num_samples"],
                        False, False, "music", False, True))
    gb = load(GPT2_BEST_FILE)
    if gb:
        rows.append(Row("GPT-2 (best dec.)", gb["metrics"], {}, gb["num_samples"],
                        False, False, "music", True, True))
    tb = load(T5_BEST_FILE)
    if tb:
        rows.append(Row("T5 (best dec.)", tb["metrics"], {}, tb["num_samples"],
                        False, False, "music", True, True))

    for label, name, leaks, domain, in_fig in REFERENCES:
        ref = load(ref_file(name))
        if ref:
            bold = (not leaks) and (not name.endswith("_clean"))
            rows.append(Row(label, ref["metrics"], {}, ref["num_samples"],
                            True, leaks, domain, in_fig, bold))
        else:
            print(f"skip {label}: missing {ref_file(name)}")

    if not any(r.is_ref for r in rows):
        raise SystemExit("No reference captioner results found under results/reference/.")

    # Group by domain (music left, general right); within a domain, clean rows before
    # leaked upper bounds. Stable so the within-group source order is preserved.
    rows.sort(key=lambda r: (DOMAIN_RANK[r.domain], r.leaks))

    # Split into one poster figure per reference domain; the pipeline (non-reference)
    # rows appear in both as the comparison anchor. Table keeps all rows together.
    pipeline = [r for r in rows if r.in_fig and not r.is_ref]
    music = pipeline + [r for r in rows if r.in_fig and r.is_ref and r.domain == "music"]
    general = pipeline + [r for r in rows if r.in_fig and r.is_ref and r.domain == "general"]
    hatch_note = (r"   $\dagger$ hatched: may have seen MusicCaps in training "
                  "(leakage upper bound, not a clean held-out score).")
    _make_figure(music, "fig_reference_comparison_music.pdf",
                 "Our pipeline vs music-domain reference captioners (MusicCaps test)",
                 r"Solid = clean; held-out rows re-score a leaky model on the "
                 r"official-eval subset ($n{=}284$)." + hatch_note)
    _make_figure(general, "fig_reference_comparison_general.pdf",
                 "Our pipeline vs general-audio reference captioners (MusicCaps test)",
                 "Blue = our pipeline (comparison anchor); warm = general audio models."
                 + hatch_note)
    # Table rows ranked by descending FENSE; the figures above keep the domain grouping.
    _table(sorted(rows, key=lambda r: r.metrics.get("FENSE", float("-inf")), reverse=True))


def _row_colors(rows):
    """One colour per row, drawn from its domain's family (shade = within-domain
    index), so adjacent same-domain bars read as a group."""
    seen = defaultdict(int)
    out = []
    for r in rows:
        fam = DOMAIN_COLORS[r.domain]
        out.append(fam[seen[r.domain] % len(fam)])
        seen[r.domain] += 1
    return out


def _make_figure(rows, out_name, suptitle, footnote):
    colors = _row_colors(rows)
    fig, axes = plt.subplots(1, len(METRIC_GROUPS), figsize=(11, 3.6),
                             gridspec_kw={"width_ratios": [len(g[1]) for g in METRIC_GROUPS]})
    for ax, (gname, metrics, ymin) in zip(axes, METRIC_GROUPS):
        x = range(len(metrics))
        n_rows = len(rows)
        w = 0.8 / n_rows
        vals = [r.metrics.get(m, 0.0) for r in rows for m in metrics]
        # Cap the panel from the <=1 bars only: CIDEr-D/SPIDEr are unbounded and a
        # leaked memorizing row (LP-MusicCaps transfer, CIDEr-D ~3.7) would crush
        # every other bar. Bars above `top` are clipped and labeled with the value.
        capped = [v for v in vals if v <= 1.0] or vals
        top = max(0.7, max(capped) * 1.12)
        if ymin is None:  # auto-zoom: floor just below the smallest bar
            lo = min(vals)
            ymin = max(0.0, (int(lo * 10) / 10) - 0.05)
        overflowed = False
        for ri, r in enumerate(rows):
            offs = [xi + (ri - (n_rows - 1) / 2) * w for xi in x]
            heights = [r.metrics.get(m, 0.0) for m in metrics]
            errs = [r.std.get(m, 0.0) for m in metrics]
            # Hatch + dagger leaked rows so the FIGURE (not just the table) flags
            # them as leakage upper bounds, never clean held-out scores.
            leg = r.label + r" $\dagger$" if r.leaks else r.label
            ax.bar(offs, heights, width=w, color=colors[ri],
                   label=leg, hatch="//" if r.leaks else None,
                   edgecolor="white" if r.leaks else "none", linewidth=0,
                   yerr=errs if any(errs) else None, capsize=2)
            for off, h in zip(offs, heights):
                if h > top:  # annotate true value at the clipped bar top
                    overflowed = True
                    ax.annotate(f"{h:.2f}", xy=(off, top), xytext=(0, -1),
                                textcoords="offset points", ha="center", va="top",
                                fontsize=5, rotation=90)
        # arrows mark metric direction; FER is lower-better.
        ticks = [f"{m}\n{'↓' if m in LOWER_BETTER else '↑'}" for m in metrics]
        ax.set_xticks(list(x))
        ax.set_xticklabels(ticks)
        title = gname if ymin == 0 else f"{gname} (y-axis from {ymin:g})"
        if overflowed:
            title += f" (bars >{top:.2g} clipped, value shown)"
        ax.set_title(title)
        ax.set_ylim(ymin, top)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("score")
    # Figure-level legend below the panels: an in-axes legend collides with the
    # tall SBERT/FENSE bars in the AAC panel.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=len(rows),
               loc="upper center", bbox_to_anchor=(0.5, 0.02))
    fig.suptitle(suptitle, fontweight="bold")
    # Spell out the visual channels so the figure stands alone without the caption.
    fig.text(0.5, -0.04, footnote, ha="center", va="top", fontsize=7, style="italic")
    out = FIG_OUT / out_name
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def _best_per_metric(rows):
    """metric -> label of the best bold-eligible row (direction-aware). Only
    full-test, non-leaked rows are eligible (r.bold): leaked daggered upper bounds
    and held-out n=284 subsets are excluded, so bolding never rewards leakage nor
    compares across different sample sizes."""
    best = {}
    for m in ALL_METRICS:
        vals = [(r.label, r.metrics[m]) for r in rows if m in r.metrics and r.bold]
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
    for r in rows:
        cells = []
        for m in ALL_METRICS:
            v = r.metrics.get(m)
            s = "--" if v is None else f"{v:.3f}"
            if v is not None and best.get(m) == r.label:
                s = f"\\textbf{{{s}}}"
            cells.append(s)
        rowlabel = r.label + (r"$^{\dagger}$" if r.leaks else "")
        lines.append(f"{rowlabel} & " + " & ".join(cells) + r" \\")

    ns = ", ".join(f"{r.label} $n{{=}}{r.n}$" for r in rows)
    tex = r"""\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\caption{Our pipeline (GPT-2/T5) versus the zero-shot reference captioners on the
shared MusicCaps test split. GPT-2 (S1) is the frozen-LM stage-1 baseline and (S2)
the fine-tuned stage-2, both greedy (\texttt{max\_new\_tokens}=64) as the controlled
comparators; \emph{best dec.} rows are the top-FENSE decoding config per model from
the decoding ablation (single seed-42 eval) -- the pipeline at its best. GPT-2
S1/S2 values are the seed-42 point estimate. $\uparrow$/$\downarrow$ mark metric
direction; best per column in bold among \emph{full-test, non-leaked} rows only
(daggered upper bounds and held-out $n{=}284$ subsets are not bold-eligible, so
bolding rewards neither leakage nor a smaller sample). $^{\dagger}$may have seen MusicCaps in pre-training (treat as
reference upper bounds, not clean held-out scores); LP-MusicCaps (pretrain) trained
only on MSD pseudo-captions, so it is leakage-free on this split, whereas LP-MusicCaps
(transfer) was fine-tuned on MusicCaps official-train and overlaps ${\sim}48\%$ of
this test split -- its inflated CIDEr-D/SPIDEr are a train-on-test signature, not a
quality result. The \emph{held-out} rows re-score a leaky model on only the
MusicCaps-official-eval subset ($n{=}284$), its leakage-free number. Rows are ordered by
descending FENSE. Sample counts:
""" + ns + r""".}
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
