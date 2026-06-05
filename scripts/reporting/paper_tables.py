#!/usr/bin/env python3
"""Lean paper tables for the two-column ISMIR draft (read-only on results/).

Emits exactly two metric tables as full-width table* floats:
  reports/tables/paper_baseline.tex   -- both models, all 10 metrics, +/- noise floor
  reports/tables/paper_decoding.tex   -- top decoding configs per model, key metrics

Val-loss, per-seed noise, and arch ablations are shown as FIGURES instead.
"""
import json
import glob
import os
from pathlib import Path

MODELS = ["gpt2", "t5"]
NAMES = {"gpt2": "GPT-2", "t5": "T5"}
ALL_METRICS = ["BLEU-1", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr-D",
               "SPICE", "SPIDEr", "SBERT-sim", "FER", "FENSE"]
DEC_METRICS = ALL_METRICS
LOWER_BETTER = {"FER"}
DIRECTION_NOTE = r"$\uparrow$: higher is better; $\downarrow$: lower is better."
RESULTS = Path("results")
OUT = Path("reports/tables")


def load(p):
    p = Path(p)
    return json.load(open(p)) if p.exists() else None


def esc(s):
    return s.replace("_", r"\_")


def with_arrow(met):
    return rf"{met} $\downarrow$" if met in LOWER_BETTER else rf"{met} $\uparrow$"


def table_star(body_lines, col_format, header, caption, label, note=None):
    L = [r"\begin{table*}[t]", r"\centering", rf"\caption{{{caption}}}",
         rf"\label{{{label}}}", r"\footnotesize",
         r"\setlength{\tabcolsep}{4pt}",
         rf"\begin{{tabular}}{{{col_format}}}", r"\toprule",
         " & ".join(header) + r" \\", r"\midrule"]
    L += body_lines
    L += [r"\bottomrule", r"\end{tabular}"]
    if note:
        L.append(rf"\\[2pt]{{\footnotesize\emph{{{note}}}}}")
    L.append(r"\end{table*}")
    OUT.joinpath(f"paper_{label.split(':')[1]}.tex").write_text("\n".join(L) + "\n")


def baseline_table():
    data, nf = {}, {}
    for m in MODELS:
        d = load(RESULTS / m / "ablations" / "arch" / f"{m}_ablation_{m}.json")
        data[m] = d["metrics"]
        n = load(RESULTS / m / f"{m}_noise_floor.json")
        nf[m] = n["noise_floor"] if n else {}
    best = {met: (min if met in LOWER_BETTER else max)(
        MODELS, key=lambda m: data[m].get(met, float("inf") if met in LOWER_BETTER else -1))
        for met in ALL_METRICS}
    rows = []
    for m in MODELS:
        cells = [NAMES[m]]
        for met in ALL_METRICS:
            v = data[m].get(met)
            s = f"{v:.4f}"
            # keep +/- noise floor only on the primary metric to stay within width
            if met == "FENSE" and met in nf[m]:
                s += rf" {{\scriptsize$\pm${nf[m][met]['std']:.4f}}}"
            if best[met] == m:
                s = rf"\textbf{{{s}}}"
            cells.append(s)
        rows.append(" & ".join(cells) + r" \\")
    header = ["Model"] + [with_arrow(met) for met in ALL_METRICS]
    table_star(
        rows, "l" + "r" * len(ALL_METRICS), header,
        r"Baseline (greedy) test-set metrics for both language models, with "
        r"per-model run-to-run standard deviation (noise floor) from three seeds. "
        r"\textsc{Fense} is the primary metric; best per column in bold. "
        + DIRECTION_NOTE,
        "tab:baseline",
        note="The GPT-2/T5 \\textsc{Fense} gap (0.004) is smaller than either "
             "model's noise floor, i.e. not significant.")
    print("wrote paper_baseline.tex")


def decoding_table(top_n=6):
    rows = []
    for m in MODELS:
        # baseline
        base = load(RESULTS / m / "ablations" / "arch" / f"{m}_ablation_{m}.json")["metrics"]
        # all decoding configs
        configs = []
        for f in glob.glob(str(RESULTS / m / "ablations" / "decoding" / f"{m}_ablation_*.json")):
            d = load(f)
            tag = os.path.basename(f)[len(f"{m}_ablation_"):-len(".json")]
            if "FENSE" in d.get("metrics", {}):
                configs.append((tag, d["metrics"]))
        configs.sort(key=lambda kv: kv[1]["FENSE"], reverse=True)
        top = configs[:top_n]
        rows.append(rf"\multicolumn{{{1+len(DEC_METRICS)}}}{{l}}{{\textbf{{{NAMES[m]}}}}} \\")
        # baseline row
        rows.append("greedy (baseline) & " +
                    " & ".join(f"{base[x]:.4f}" for x in DEC_METRICS) + r" \\")
        best_f = max(c[1]["FENSE"] for c in top)
        for tag, met in top:
            cells = [esc(tag)]
            for x in DEC_METRICS:
                s = f"{met[x]:.4f}"
                if x == "FENSE" and met[x] == best_f:
                    s = rf"\textbf{{{s}}}"
                cells.append(s)
            rows.append(" & ".join(cells) + r" \\")
        if m != MODELS[-1]:
            rows.append(r"\midrule")
    header = ["Config"] + [with_arrow(met) for met in DEC_METRICS]
    table_star(
        rows, "l" + "r" * len(DEC_METRICS), header,
        r"Top decoding configurations per model (eval-only on the baseline "
        r"checkpoint), ranked by \textsc{Fense}. Note that high \textsc{Fense} "
        r"coincides with near-zero FER while the lexical metrics "
        r"barely move -- see Figure~\ref{fig:fensefer}. " + DIRECTION_NOTE,
        "tab:decoding")
    print("wrote paper_decoding.tex")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    baseline_table()
    decoding_table()


if __name__ == "__main__":
    main()
