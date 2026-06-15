#!/usr/bin/env python3
"""Poster figure for the pairwise preference study: stacked win-rate bars per
comparison and judge panel, with each panel's Fleiss' kappa annotated.

Reads the LLM-judge pools (results/llm_judge/{audio,text}) and, if present, a
human pool (--human-csv / --human-json-dir), reusing human_eval/compute_kappa.py
so the numbers match the analysis script exactly. One stacked bar per
(comparison, pool): share of votes to GPT-2 / its opponent / tie, on Q1
(accuracy). The message: GPT-2-best is preferred over T5-best by every panel,
yet the human reference still wins -- the lightweight LM is competitive among
models, with the human caption as the honest ceiling.

Output:
  reports/figures/fig_eval_winrate.pdf
"""
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "human_eval"))
import json
from compute_kappa import CATS, fleiss_kappa, load_pool  # noqa: E402

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 21,
    "axes.labelsize": 19,
    "legend.fontsize": 17,
    "figure.facecolor": "white",
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})

KEY = Path("human_eval/key.json")
LLM = Path("results/llm_judge")
FIG_OUT = Path("reports/figures")

# Audio palette consistent with 416A: GPT-2 = signature blue, opponent = warm,
# tie = light grey. "opponent" share collapses reference (ref_best) and T5
# (gpt2_t5) into one slot -- each bar is labelled with who the opponent is.
C_GPT2, C_OPP, C_TIE = "#3182BD", "#C0392B", "#BDBDBD"

# (comparison pair_type, pool dir, pool label, opponent display name). Each is one
# stacked bar. Audio = judges that heard the clip; text = judges given the human
# reference as ground truth (caption-vs-caption only -> gpt2_t5 alone).
BARS = [
    ("ref_best", "audio", "audio judges", "human reference"),
    ("gpt2_t5", "audio", "audio judges", "T5-best"),
    ("gpt2_t5", "text", "text judges", "T5-best"),
]


def win_shares(dedup, key, pair_type, q_idx=0):
    """Vote shares to {gpt2, opp, tie} for one comparison, on question q_idx.
    gpt2 collects the GPT-2 system's wins; opp collects the other system's."""
    ids = [i for i, k in key.items() if k["pair_type"] == pair_type]
    tally = defaultdict(int)
    n = 0
    for pid in ids:
        ka = key[pid]
        for (rater, p), ans in dedup.items():
            if p != pid or ans[q_idx] not in CATS:
                continue
            n += 1
            if ans[q_idx] == "tie":
                tally["tie"] += 1
                continue
            sysname = ka["systemA" if ans[q_idx] == "A" else "systemB"]
            tally["gpt2" if sysname == "gpt2_best" else "opp"] += 1
    if not n:
        return None
    return {k: tally[k] / n for k in ("gpt2", "opp", "tie")}, n


def panel_kappa(dedup, key, pair_types, q_idx=0):
    """Fleiss' kappa over the given comparison types' pairs, question q_idx."""
    item = defaultdict(list)
    for (rater, pid), ans in dedup.items():
        if pid not in key or key[pid]["pair_type"] not in pair_types:
            continue
        if ans[q_idx] in CATS:
            item[pid].append(ans[q_idx])
    k, _ = fleiss_kappa(item)
    return k


def main():
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    key = {k["id"]: k for k in json.loads(KEY.read_text())}

    pools = {d: load_pool(None, LLM / d) for d in ("audio", "text")}
    kappa = {d: panel_kappa(pools[d], key,
                            {"ref_best", "gpt2_t5"} if d == "audio" else {"gpt2_t5"})
             for d in pools}

    labels, shares, ns = [], [], []
    for pair_type, pool, pool_label, opp in BARS:
        res = win_shares(pools[pool], key, pair_type)
        if res is None:
            continue
        sh, n = res
        labels.append(f"GPT-2-best vs\n{opp}\n({pool_label})")
        shares.append(sh)
        ns.append(n)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    y = range(len(labels))
    left = [0.0] * len(labels)
    segs = [("gpt2", C_GPT2, "GPT-2-best wins"),
            ("opp", C_OPP, "opponent wins"),
            ("tie", C_TIE, "tie")]
    for kk, color, leg in segs:
        widths = [s[kk] for s in shares]
        ax.barh(list(y), widths, left=left, color=color, label=leg,
                edgecolor="white", height=0.62)
        for yi, (w, l) in enumerate(zip(widths, left)):
            if w > 0.07:
                ax.text(l + w / 2, yi, f"{w*100:.0f}%", ha="center", va="center",
                        color="white", fontweight="bold", fontsize=19)
        left = [l + w for l, w in zip(left, widths)]

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of judge votes  (Q1: which caption is more accurate)")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    # Per-bar n on the right edge; per-panel Fleiss kappa in the title.
    for yi, n in enumerate(ns):
        ax.text(1.005, yi, f"n={n}", va="center", ha="left", fontsize=13, color="#555")
    ax.set_title("LLM-judge win-rates (Fleiss' "
                 fr"$\kappa$: audio {kappa['audio']:.2f}, text {kappa['text']:.2f})",
                 fontweight="bold")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32),
              ncol=3, frameon=False)
    ax.grid(axis="x", alpha=0.3)
    out = FIG_OUT / "fig_eval_winrate.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")
    for lab, sh, n in zip(labels, shares, ns):
        print(f"  {lab.replace(chr(10), ' ')}: "
              f"gpt2 {sh['gpt2']:.0%} / opp {sh['opp']:.0%} / tie {sh['tie']:.0%}  (n={n})")


if __name__ == "__main__":
    main()
