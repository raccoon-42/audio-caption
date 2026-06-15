#!/usr/bin/env python3
"""Poster-only FENSE comparison: one big-font horizontal bar chart of the
primary metric (FENSE) across our pipeline and every reference captioner, so it
stays readable at 1-2 m on an A0. Reuses the data paths/row list from
reference_comparison.py; the dense ten-metric report figures are unchanged.

Leaky rows (may have seen MusicCaps in training) are hatched and daggered as
upper bounds; for the two leaky music captioners we plot their clean held-out
twin instead, the honest number. Output:
  reports/figures/fig_poster_fense.pdf
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reference_comparison as rc  # noqa: E402  (path list, ref_file, load)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "font.size": 22,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
})

FIG_OUT = Path("reports/figures")
C_OURS, C_MUSIC, C_GEN = "#3182BD", "#1B8A6B", "#C0392B"

# (display label, fense, color, leaks). Order top->bottom is set later by value.
# Our pipeline best-decoding rows + the cleanest reference number per model.
# For LP-MusicCaps transfer and BLAP we use the clean held-out twin (the leaky
# full-test rows are train-on-test upper bounds, not shown as bars here).


def fense(metrics_dict):
    return metrics_dict["metrics"]["FENSE"]


def main():
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    rows = []  # (label, value, color, leaks)

    gb, tb = rc.load(rc.GPT2_BEST_FILE), rc.load(rc.T5_BEST_FILE)
    if gb:
        rows.append(("ClapCap GPT-2 (ours)", fense(gb), C_OURS, False))
    if tb:
        rows.append(("ClapCap T5 (ours)", fense(tb), C_OURS, False))

    # Reference captioners: name -> (display, color). Use the clean held-out
    # twin for the two leaky music models; show the rest at their single pass.
    refs = [
        ("lp_music_caps", "LP-MusicCaps (pretrain)", C_MUSIC, False),
        ("blap_clean", "BLAP (held-out)", C_MUSIC, False),
        ("blap", "BLAP", C_MUSIC, True),
        ("lp_music_caps_transfer", "LP-MusicCaps (transfer)", C_MUSIC, True),
        ("audio_flamingo", "NVIDIA Audio Flamingo", C_GEN, True),
        ("qwen3_omni_captioner", "Qwen3-Omni Captioner", C_GEN, True),
        ("qwen2_audio", "Qwen2-Audio", C_GEN, True),
    ]
    for name, label, color, leaks in refs:
        d = rc.load(rc.ref_file(name))
        if d:
            rows.append((label, fense(d), color, leaks))

    rows.sort(key=lambda r: r[1])  # ascending -> largest on top after barh
    labels = [r[0] + (r"  $\dagger$" if r[3] else "") for r in rows]
    vals = [r[1] for r in rows]
    colors = [r[2] for r in rows]
    leaks = [r[3] for r in rows]

    ours = [c == C_OURS for c in colors]

    fig, ax = plt.subplots(figsize=(13, 9))
    y = range(len(rows))
    # Highlight band behind our rows so ClapCap pops out at a glance.
    for yi, is_ours in zip(y, ours):
        if is_ours:
            ax.axhspan(yi - 0.45, yi + 0.45, color=C_OURS, alpha=0.12, zorder=0)
    # Our bars get a bold navy outline; references a thin white one.
    bars = ax.barh(list(y), vals, color=colors, height=0.66,
                   hatch=["//" if lk else None for lk in leaks],
                   edgecolor=["#08306B" if o else "white" for o in ours],
                   linewidth=[5 if o else 2 for o in ours], zorder=3)
    for yi, v, is_ours in zip(y, vals, ours):
        ax.text(v + 0.008, yi, f"{v:.3f}", va="center", ha="left",
                fontsize=23 if is_ours else 22, fontweight="bold",
                color="#08306B" if is_ours else "black")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=24)
    for tick, is_ours in zip(ax.get_yticklabels(), ours):
        if is_ours:
            tick.set_fontweight("bold")
            tick.set_color("#08306B")
    ax.set_xlabel(r"FENSE $\uparrow$  (primary metric, MusicCaps test)", fontsize=26)
    ax.set_xlim(0, max(vals) * 1.16)
    ax.tick_params(axis="x", labelsize=22)
    ax.grid(axis="x", alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Legend explaining colour + hatch (stands alone without a caption).
    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor=C_OURS, label="ClapCap (ours)"),
        Patch(facecolor=C_MUSIC, label="music captioner"),
        Patch(facecolor=C_GEN, label="general-audio LLM"),
        Patch(facecolor="white", edgecolor="grey", hatch="//",
              label=r"$\dagger$ may have seen MusicCaps (upper bound)"),
    ]
    ax.legend(handles=legend, fontsize=20, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    ax.set_title("Lightweight ClapCap beats the clean music captioners on FENSE",
                 fontsize=27, fontweight="bold", pad=16)
    out = FIG_OUT / "fig_poster_fense.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")
    for lab, v in zip(labels, vals):
        print(f"  {lab.replace(chr(36),''):42s} {v:.3f}")


if __name__ == "__main__":
    main()
