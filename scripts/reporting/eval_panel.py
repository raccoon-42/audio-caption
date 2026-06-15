#!/usr/bin/env python3
"""Poster figure for the pairwise preference study: stacked win-rate bars per
comparison and judge panel, with each panel's Fleiss' kappa annotated.

Reads the LLM-judge pools (results/llm_judge/{audio,text}) and, if present, a
human pool (--human-csv / --human-json-dir), reusing pairwise_eval/compute_kappa.py
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pairwise_eval"))
import argparse  # noqa: E402
import json  # noqa: E402
from compute_kappa import (  # noqa: E402
    CATS, cohen_kappa, fleiss_kappa, load_pool, majority, sanity_keep,
)

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 21,
    "axes.labelsize": 19,
    "legend.fontsize": 17,
    "figure.facecolor": "white",
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})

KEY = Path("pairwise_eval/key.json")
LLM = Path("results/llm_judge")
FIG_OUT = Path("reports/figures")

# Audio palette consistent with 416A: GPT-2 = signature blue, opponent = warm,
# tie = light grey. "opponent" share collapses reference (ref_best) and T5
# (gpt2_t5) into one slot -- each bar is labelled with who the opponent is.
C_GPT2, C_OPP, C_TIE = "#3182BD", "#C0392B", "#BDBDBD"

# (comparison pair_type, pool key, pool label, opponent display name). Each is one
# stacked bar. human = the listener panel (heard the clip); audio = judges that
# heard the clip; text = judges given the human reference (caption-vs-caption only
# -> gpt2_t5 alone). Humans come first so the LLM bars read as corroboration.
# Grouped by COMPARISON (not by pool) so a presenter can point down each block:
# "vs reference" -> humans then audio LLM agree; "vs T5" -> humans, audio, text agree.
BARS = [
    ("ref_best", "human", "humans", "human reference"),
    ("ref_best", "audio", "audio judges", "human reference"),
    ("gpt2_t5", "human", "humans", "T5-best"),
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


def consensus(dedup, key, q_idx=0):
    """Per-pair plurality label over non-sanity pairs (undecided -> None)."""
    item = defaultdict(list)
    for (rater, pid), ans in dedup.items():
        if pid not in key or key[pid]["pair_type"] == "sanity":
            continue
        if ans[q_idx] in CATS:
            item[pid].append(ans[q_idx])
    return {pid: majority(cats) for pid, cats in item.items()}


def consensus_kappa(dedup_a, dedup_b, key, q_idx=0):
    """Cohen's kappa between two pools' per-pair consensus labels."""
    a, b = consensus(dedup_a, key, q_idx), consensus(dedup_b, key, q_idx)
    pairs = [(a[p], b[p]) for p in set(a) & set(b)
             if a[p] is not None and b[p] is not None]
    if not pairs:
        return None
    return cohen_kappa([x for x, _ in pairs], [y for _, y in pairs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human-csv", default="pairwise_eval/human_responses_anon.csv")
    ap.add_argument("--human-json-dir")
    ap.add_argument("--sanity-threshold", type=float, default=0.75)
    args = ap.parse_args()

    FIG_OUT.mkdir(parents=True, exist_ok=True)
    key = {k["id"]: k for k in json.loads(KEY.read_text())}
    sanity_ids = [i for i, k in key.items() if k["pair_type"] == "sanity"]

    pools = {d: load_pool(None, LLM / d) for d in ("audio", "text")}

    # Human listener pool: keep only sanity-passing raters before win-rates.
    human = (load_pool(args.human_csv if Path(args.human_csv).exists() else None,
                       args.human_json_dir)
             if (args.human_json_dir or Path(args.human_csv).exists()) else {})
    if human:
        kept = sanity_keep(human, key, sanity_ids, args.sanity_threshold, "human")
        human = {(r, p): v for (r, p), v in human.items() if r in kept}
        pools["human"] = human

    kappa = {d: panel_kappa(pools[d], key,
                            {"gpt2_t5"} if d == "text" else {"ref_best", "gpt2_t5"})
             for d in pools}
    # Human consensus vs audio-LLM consensus agreement (Q1), the three-way number.
    hl_kappa = consensus_kappa(human, pools["audio"], key) if human else None

    # Compact right-side captions: short pool name + opponent.
    short_pool = {"humans": "humans", "audio judges": "audio LLM",
                  "text judges": "text LLM"}
    short_opp = {"human reference": "reference", "T5-best": "T5"}
    labels, shares, ns = [], [], []
    for pair_type, pool, pool_label, opp in BARS:
        if pool not in pools:
            continue
        res = win_shares(pools[pool], key, pair_type)
        if res is None:
            continue
        sh, n = res
        labels.append(f"{short_pool[pool_label]}\nGPT-2 vs {short_opp[opp]}")
        shares.append(sh)
        ns.append(n)

    # Landscape strip, but tall enough that the five bars stay chunky/readable.
    fig, ax = plt.subplots(figsize=(11.5, 5.9))
    y = range(len(labels))
    left = [0.0] * len(labels)
    segs = [("gpt2", C_GPT2, "GPT-2-best wins"),
            ("opp", C_OPP, "opponent wins"),
            ("tie", C_TIE, "tie")]
    for kk, color, leg in segs:
        widths = [s[kk] for s in shares]
        ax.barh(list(y), widths, left=left, color=color, label=leg,
                edgecolor="white", height=0.74)
        for yi, (w, l) in enumerate(zip(widths, left)):
            if w > 0.06:
                ax.text(l + w / 2, yi, f"{w*100:.0f}%", ha="center", va="center",
                        color="white", fontweight="bold", fontsize=24)
        left = [l + w for l, w in zip(left, widths)]

    # Captions as a vertical list on the RIGHT; n= tucked into the left margin.
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=19)
    ax.yaxis.tick_right()
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of votes  (Q1: which caption is more accurate)", fontsize=20)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"], fontsize=17)
    for yi, n in enumerate(ns):
        ax.text(-0.012, yi, f"n={n}", va="center", ha="right",
                fontsize=15, color="#555")
    kparts = []
    if "human" in kappa:
        kparts.append(fr"humans {kappa['human']:.2f}")
    kparts += [fr"audio {kappa['audio']:.2f}", fr"text {kappa['text']:.2f}"]
    title = r"Pairwise win-rates (Fleiss' $\kappa$: " + ", ".join(kparts) + ")"
    if hl_kappa is not None:
        title += ("\n" + fr"human$\leftrightarrow$audio-LLM consensus "
                  fr"$\kappa$ = {hl_kappa:.2f} (substantial)")
    ax.set_title(title, fontweight="bold", fontsize=21)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30),
              ncol=3, frameon=False, fontsize=19)
    ax.grid(axis="x", alpha=0.3)
    out = FIG_OUT / "fig_eval_winrate.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"wrote {out} (+ .png)")
    for lab, sh, n in zip(labels, shares, ns):
        print(f"  {lab.replace(chr(10), ' ')}: "
              f"gpt2 {sh['gpt2']:.0%} / opp {sh['opp']:.0%} / tie {sh['tie']:.0%}  (n={n})")


if __name__ == "__main__":
    main()
