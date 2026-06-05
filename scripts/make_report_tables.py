"""Generate LaTeX table fragments from the results/ tree.

Reads eval ablation JSONs, training JSONs, and noise-floor JSONs, then writes
booktabs LaTeX fragments to reports/tables/ and prints each to stdout.

Tables produced per model:
  - noise floor: per-seed metric values plus std.
  - stage-1 and stage-2 best validation loss per config (separate tables),
    with the value spread in the caption.
  - architecture ablation test-set metrics.
  - decoding ablation test-set metrics (top-N by FENSE, baseline prepended).
A baseline table lists all models together.

Metric columns are split into two tables following the aac-metrics taxonomy:
  - legacy : BLEU-1/4, METEOR, ROUGE-L, CIDEr-D, SPICE, SPIDEr
  - aac    : SBERT-sim, FER, FENSE

Usage:
    uv run python scripts/make_report_tables.py
    uv run python scripts/make_report_tables.py --models gpt2 t5 --top-decoding 15
"""

import argparse
import json
from pathlib import Path


# Metric groups (aac-metrics taxonomy). Filtered to whatever is present.
LEGACY_METRICS = ["BLEU-1", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr-D", "SPICE", "SPIDEr"]
AAC_METRICS = ["SBERT-sim", "FER", "FENSE"]
ALL_METRICS = LEGACY_METRICS + AAC_METRICS
PRIMARY = "FENSE"

GROUP_TITLES = {
    "legacy": "Legacy metrics",
    "aac": "AAC-specific metrics",
}
METRIC_GROUPS = [("legacy", LEGACY_METRICS), ("aac", AAC_METRICS)]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def collect_eval_results(results_dir, model, ablation_type):
    """Return {tag: metrics_dict} for one model + ablation_type."""
    out = {}
    ablation_dir = results_dir / model / "ablations" / ablation_type
    if not ablation_dir.exists():
        return out
    for f in sorted(ablation_dir.glob(f"{model}_ablation_*.json")):
        d = load_json(f)
        if d is None:
            continue
        tag = f.stem.replace(f"{model}_ablation_", "")
        out[tag] = d["metrics"]
    return out


def collect_training_results(results_dir, model):
    """Return list of {tag, stage, best_val_loss, stopped_epoch}."""
    rows = []
    training_dir = results_dir / model / "training"
    if not training_dir.exists():
        return rows
    for f in sorted(training_dir.glob("*_stage*.json")):
        d = load_json(f)
        if d is None:
            continue
        tag = f.stem.replace(f"_stage{d['stage']}", "")
        rows.append({
            "tag": tag,
            "stage": d["stage"],
            "best_val_loss": d["best_val_loss"],
            "stopped_epoch": d["stopped_epoch"],
        })
    return rows


def load_noise_floor(results_dir, model):
    """Return (seeds, {metric: {'values': {seed: v}, 'std': s}}) or (None, None)."""
    d = load_json(results_dir / model / f"{model}_noise_floor.json")
    if d is None:
        return None, None
    return d["seeds"], d["noise_floor"]


# --------------------------------------------------------------------------- #
# LaTeX helpers
# --------------------------------------------------------------------------- #
def tex_escape(s):
    return str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def write_table(path, header, rows, caption, label, col_format,
                resize=True, note=None):
    """Write a booktabs table fragment.

    header: list[str] of already-escaped column headers (first is the row label).
    rows:   list[list[str]] of already-formatted/escaped cells.
    """
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\footnotesize")
    open_resize = r"\resizebox{\textwidth}{!}{%" if resize else ""
    if open_resize:
        lines.append(open_resize)
    lines.append(rf"\begin{{tabular}}{{{col_format}}}")
    lines.append(r"\toprule")
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")
    for r in rows:
        lines.append(" & ".join(r) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if open_resize:
        lines.append(r"}")
    if note:
        lines.append(rf"\par\vspace{{2pt}}\footnotesize\emph{{{note}}}")
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


def fmt_metric_cell(value, is_best, baseline_std=None):
    s = f"{value:.4f}"
    if baseline_std is not None:
        s = f"{s} $\\pm${baseline_std:.4f}"
    if is_best:
        s = rf"\textbf{{{s}}}"
    return s


def metric_table(path, data, metrics, row_order, baseline_tag,
                 noise_floor, caption, label, row_label="Config"):
    """data: {tag: metrics_dict}. Bold best per metric column; baseline row
    gets +/- noise-floor std where available."""
    present = [m for m in metrics if any(m in data[t] for t in row_order)]
    # best (max) per metric across rows
    best = {}
    for m in present:
        vals = {t: data[t][m] for t in row_order if m in data[t]}
        if vals:
            best[m] = max(vals, key=vals.get)
    header = [row_label] + present
    rows = []
    for t in row_order:
        cells = [tex_escape(t)]
        for m in present:
            if m not in data[t]:
                cells.append("--")
                continue
            std = None
            if t == baseline_tag and noise_floor and m in noise_floor:
                std = noise_floor[m]["std"]
            cells.append(fmt_metric_cell(data[t][m], best.get(m) == t, std))
        rows.append(cells)
    col_format = "l" + "r" * len(present)
    return write_table(path, header, rows, caption, label, col_format)


def valloss_table(path, train_rows, stage, model, caption, label):
    """One stage's per-config best val loss + stopped epoch. Bold the min."""
    rows_data = [r for r in train_rows if r["stage"] == stage]
    if not rows_data:
        return None
    # order: baseline first, then the rest alphabetically
    rows_data.sort(key=lambda r: (r["tag"] != model, r["tag"]))
    vals = [r["best_val_loss"] for r in rows_data]
    spread = max(vals) - min(vals)
    best_tag = min(rows_data, key=lambda r: r["best_val_loss"])["tag"]
    header = ["Config", "Best val loss", "Stopped epoch"]
    rows = []
    for r in rows_data:
        vl = f"{r['best_val_loss']:.4f}"
        if r["tag"] == best_tag:
            vl = rf"\textbf{{{vl}}}"
        rows.append([tex_escape(r["tag"]), vl, str(r["stopped_epoch"])])
    cap = f"{caption} (spread = {spread:.4f})."
    write_table(path, header, rows, cap, label, "lrr", resize=False)
    return spread


def noise_floor_table(path, seeds, nf, metrics, model, caption, label):
    present = [m for m in metrics if m in nf]
    header = ["Seed"] + present
    rows = []
    for seed in seeds:
        cells = [str(seed)]
        for m in present:
            cells.append(f"{nf[m]['values'][str(seed)]:.4f}")
        rows.append(cells)
    std_cells = [r"\textbf{std}"] + [rf"\textbf{{{nf[m]['std']:.4f}}}" for m in present]
    rows.append(std_cells)
    col_format = "l" + "r" * len(present)
    return write_table(path, header, rows, caption, label, col_format)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gpt2", "t5"])
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="reports/tables")
    ap.add_argument("--top-decoding", type=int, default=15,
                    help="Top-N decoding configs by FENSE per model")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = METRIC_GROUPS

    written = []

    def emit(tex, header_line):
        print(f"\n% ===== {header_line} =====")
        print(tex)

    # ---- baseline metrics (both models in one table per group) ----
    baselines = {}      # model -> metrics
    arch_data = {}      # model -> {tag: metrics}
    decode_data = {}    # model -> {tag: metrics}
    for model in args.models:
        arch_data[model] = collect_eval_results(results_dir, model, "arch")
        decode_data[model] = collect_eval_results(results_dir, model, "decoding")
        if model in arch_data[model]:                     # tag == model == greedy baseline
            baselines[model] = arch_data[model][model]

    for gname, gmetrics in groups:
        present = [m for m in gmetrics if any(m in baselines.get(mo, {}) for mo in args.models)]
        if not present:
            continue
        best = {}
        for m in present:
            vals = {mo: baselines[mo][m] for mo in args.models
                    if mo in baselines and m in baselines[mo]}
            if vals:
                best[m] = max(vals, key=vals.get)
        header = ["Model"] + present
        rows = []
        for mo in args.models:
            if mo not in baselines:
                continue
            cells = [tex_escape(mo)]
            for m in present:
                cells.append(fmt_metric_cell(baselines[mo][m], best.get(m) == mo))
            rows.append(cells)
        p = out_dir / f"baseline_{gname}.tex"
        tex = write_table(
            p, header, rows,
            f"Baseline (greedy) test-set metrics --- {GROUP_TITLES[gname]}.",
            f"tab:baseline_{gname}", "l" + "r" * len(present))
        written.append(p); emit(tex, f"baseline_{gname}")

    # ---- per-model tables ----
    for model in args.models:
        # noise floor
        seeds, nf = load_noise_floor(results_dir, model)
        if nf:
            p = out_dir / f"noise_floor_{model}.tex"
            tex = noise_floor_table(
                p, seeds, nf, ALL_METRICS, model,
                f"{model.upper()} run-to-run variance over seeds {seeds} "
                f"(std = noise floor).",
                f"tab:noise_{model}")
            written.append(p); emit(tex, f"noise_floor_{model}")
        else:
            print(f"% {model}: no noise floor JSON")

        # S1 / S2 val loss (separate tables)
        train_rows = collect_training_results(results_dir, model)
        # exclude noise-floor seed reruns; keep arch configs only
        train_rows = [r for r in train_rows
                      if "seed" not in r["tag"] or r["tag"] == model]
        for stage in (1, 2):
            p = out_dir / f"s{stage}_valloss_{model}.tex"
            spread = valloss_table(
                p, train_rows, stage, model,
                f"{model.upper()} stage-{stage} best validation loss per config",
                f"tab:s{stage}_valloss_{model}")
            if spread is not None:
                written.append(p)
                emit(p.read_text(), f"s{stage}_valloss_{model}")

        # arch + decoding metric tables (per group)
        for gname, gmetrics in groups:
            # arch: baseline first, then variants, drop seed reruns
            arch = arch_data[model]
            order = [t for t in sorted(arch) if "seed" not in t]
            order.sort(key=lambda t: (t != model, t))
            if order:
                p = out_dir / f"arch_{model}_{gname}.tex"
                tex = metric_table(
                    p, arch, gmetrics, order, model, nf,
                    f"{model.upper()} architecture ablations (greedy) --- "
                    f"{GROUP_TITLES[gname]}. Best per column in bold; "
                    f"baseline row shows $\\pm$ noise floor.",
                    f"tab:arch_{model}_{gname}")
                written.append(p); emit(tex, f"arch_{model}_{gname}")

            # decoding: top-N by FENSE, baseline prepended for reference
            dec = dict(decode_data[model])
            ranked = sorted(dec, key=lambda t: dec[t].get(PRIMARY, -1), reverse=True)
            top = ranked[:args.top_decoding]
            order = []
            base_row = {}
            if model in arch:                   # greedy baseline lives in arch dir
                base_row = {model: arch[model]}
                order.append(model)
            order += [t for t in top if t != model]
            merged = {**base_row, **{t: dec[t] for t in top}}
            if order:
                p = out_dir / f"decoding_{model}_{gname}.tex"
                tex = metric_table(
                    p, merged, gmetrics, order, model, nf,
                    f"{model.upper()} decoding ablations --- top {args.top_decoding} "
                    f"by {PRIMARY}, {GROUP_TITLES[gname]}. First row = greedy baseline.",
                    f"tab:decoding_{model}_{gname}")
                written.append(p); emit(tex, f"decoding_{model}_{gname}")

    # ---- index file that \inputs everything ----
    index = out_dir / "_all_tables.tex"
    index.write_text("\n".join(rf"\input{{tables/{p.name}}}" for p in written) + "\n")
    print(f"\n% Wrote {len(written)} fragments + {index.name} to {out_dir}/")
    print("% Thesis preamble needs: \\usepackage{booktabs,graphicx}")


if __name__ == "__main__":
    main()
