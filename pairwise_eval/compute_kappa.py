"""Analyse the pairwise human-eval responses and the LLM-judge panels.

Reads two independent rater pools and reports agreement within and across them:
  - human pool from a Sheet CSV export (--csv) and/or per-rater JSON (--json-dir)
  - LLM-judge panel from a directory of judge JSON files (--llm-dir), e.g.
    results/llm_judge/audio or results/llm_judge/text (one condition per run)

Both file shapes share the {rater, responses: {pid: {q1, q2}}} schema, so the
same loader ingests either. For each present pool it reports, per question:
  - per-rater sanity-check accuracy (data-quality filter; raters below
    --sanity-threshold are dropped before everything else)
  - Fleiss' kappa + mean pairwise Cohen's kappa among the kept raters
  - 'can't tell' abstention rate (excluded from kappa)
  - win-rates per comparison type

When both pools are present it also reports llm-vs-human consensus: each pool is
reduced to a per-pair plurality label (a within-pool tie is "undecided" and
excluded, reported separately), then Cohen's kappa + raw agreement on the pairs
where both pools are decisive.

Categories are {A, B, tie}. Humans heard the clip, so the audio LLM panel is the
natural comparator; run --llm-dir results/llm_judge/text separately for the
text-only panel.

Two extra analyses gate on flags:
  --heatmap-out PATH  writes a per-question entity x entity Cohen-kappa heatmap
    (one human-consensus 'rater' + each kept audio judge); needs --csv + --llm-dir.
  --text-dir DIR  with --llm-dir (audio) computes each judge's audio-vs-text
    grounding self-contrast (same model, two conditions; differing models flagged).
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

CATS = ["A", "B", "tie"]
QUESTIONS = [(0, "Q1 (more accurate)"), (1, "Q2 (less wrong)")]


def load_csv(path, rows):
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append((r["rater"], r["pair_id"], r.get("q1"), r.get("q2")))


def load_json_dir(d, rows):
    for f in sorted(Path(d).glob("*.json")):
        data = json.loads(f.read_text())
        for pid, r in data.get("responses", {}).items():
            rows.append((data["rater"], pid, r.get("q1"), r.get("q2")))


def fleiss_kappa(item_to_ratings):
    """item_to_ratings: {item: [category, ...]} with varying raters per item."""
    counts = []  # per item: [n_A, n_B, n_tie]
    for ratings in item_to_ratings.values():
        c = [sum(1 for x in ratings if x == cat) for cat in CATS]
        if sum(c) >= 2:
            counts.append(c)
    if not counts:
        return None, 0
    totals = [sum(c) for c in counts]
    p_item = []
    for c, n in zip(counts, totals):
        p_item.append((sum(x * x for x in c) - n) / (n * (n - 1)))
    p_bar = sum(p_item) / len(p_item)
    grand = sum(totals)
    p_cat = [sum(c[j] for c in counts) / grand for j in range(len(CATS))]
    p_e = sum(p * p for p in p_cat)
    if p_e == 1:
        return 1.0, len(counts)
    return (p_bar - p_e) / (1 - p_e), len(counts)


def cohen_kappa(a, b):
    """a, b: equal-length lists of categories for shared items."""
    n = len(a)
    if n == 0:
        return None
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = 0.0
    for cat in CATS:
        pe += (a.count(cat) / n) * (b.count(cat) / n)
    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)


def mean_pairwise_cohen(rater_item_cat):
    """rater_item_cat: {rater: {item: cat}}. Mean Cohen over rater pairs."""
    raters = list(rater_item_cat)
    vals = []
    for ra, rb in combinations(raters, 2):
        shared = set(rater_item_cat[ra]) & set(rater_item_cat[rb])
        if not shared:
            continue
        a = [rater_item_cat[ra][i] for i in shared]
        b = [rater_item_cat[rb][i] for i in shared]
        k = cohen_kappa(a, b)
        if k is not None:
            vals.append(k)
    return sum(vals) / len(vals) if vals else None


def majority(cats):
    """Plurality label over a list of categories. None on an empty list or a
    strict tie for first place (undecided)."""
    counts = Counter(cats).most_common()
    if not counts:
        return None
    if len(counts) > 1 and counts[0][1] == counts[1][1]:
        return None
    return counts[0][0]


def sanity_keep(dedup, key, sanity_ids, threshold, label):
    """Print per-rater sanity accuracy for a pool, return the kept-rater set."""
    rater_ids = sorted({r for r, _ in dedup})
    print(f"=== {label}: sanity-check accuracy (Q1) ===")
    kept = []
    for rater in rater_ids:
        hits = tot = 0
        for sid in sanity_ids:
            if (rater, sid) in dedup:
                tot += 1
                if dedup[(rater, sid)][0] == key[sid].get("correct_side"):
                    hits += 1
        if tot == 0:
            # No sanity pairs shown to this rater (e.g. the text-only LLM panel,
            # where sanity needs the audio). Can't assess, so keep.
            print(f"  {rater:20s} 0/0 = n/a (no sanity pairs; kept)")
            kept.append(rater)
            continue
        acc = hits / tot
        flag = "" if acc >= threshold else "  <-- DROPPED"
        print(f"  {rater:20s} {hits}/{tot} = {acc:.2f}{flag}")
        if acc >= threshold:
            kept.append(rater)
    print(f"kept {len(kept)}/{len(rater_ids)} raters\n")
    return set(kept)


def analyze_pool(dedup, kept_set, key, label):
    """Print within-pool Fleiss/Cohen per question and return the per-pair
    plurality consensus: {q_idx: {pid: cat}} over kept, decisive pairs only."""
    consensus = {}
    for q_idx, q_name in QUESTIONS:
        item_ratings = defaultdict(list)
        rater_item_cat = defaultdict(dict)
        n_total = n_abstain = 0
        for (rater, pid), ans in dedup.items():
            if rater not in kept_set or pid not in key:
                continue
            if key[pid]["pair_type"] == "sanity":
                continue
            cat = ans[q_idx]
            n_total += 1
            if cat == "cant_tell":
                n_abstain += 1          # excluded from kappa, reported below
            elif cat in CATS:
                item_ratings[pid].append(cat)
                rater_item_cat[rater][pid] = cat
        fk, n_items = fleiss_kappa(item_ratings)
        ck = mean_pairwise_cohen(rater_item_cat)
        print(f"=== {label}: {q_name} ===")
        if n_total:
            print(f"  'can't tell' abstentions: {n_abstain}/{n_total} "
                  f"({n_abstain / n_total:.0%})  [excluded from kappa]")
        print(f"  Fleiss' kappa = {fk:.3f} over {n_items} items"
              if fk is not None else "  no data")
        if ck is not None:
            print(f"  mean pairwise Cohen's kappa = {ck:.3f}")
        print()
        consensus[q_idx] = {pid: majority(cats)
                            for pid, cats in item_ratings.items()}
    return consensus


def win_rates(dedup, kept, key, label):
    print(f"=== {label}: win-rates (kept raters) ===")
    for ptype in ["ref_best", "gpt2_t5"]:
        ids = [i for i, k in key.items() if k["pair_type"] == ptype]
        for q_idx, q_name in [(0, "Q1"), (1, "Q2")]:
            tally = defaultdict(int)
            n = 0
            for pid in ids:
                ka = key[pid]
                for rater in kept:
                    ans = dedup.get((rater, pid))
                    if not ans or ans[q_idx] not in CATS:
                        continue
                    n += 1
                    choice = ans[q_idx]
                    if choice == "tie":
                        tally["tie"] += 1
                    else:
                        tally[ka["systemA" if choice == "A" else "systemB"]] += 1
            if n:
                parts = ", ".join(f"{k} {v}/{n} ({v / n:.0%})"
                                  for k, v in sorted(tally.items()))
                print(f"  {ptype} {q_name}: {parts}")
    print()


def cross_consensus(cons_a, label_a, cons_b, label_b):
    """Cohen's kappa + raw agreement between two pools' consensus labels."""
    print(f"=== {label_a} vs {label_b}: consensus agreement ===")
    for q_idx, q_name in QUESTIONS:
        a, b = cons_a.get(q_idx, {}), cons_b.get(q_idx, {})
        shared = sorted(set(a) & set(b))
        decisive = [(a[p], b[p]) for p in shared
                    if a[p] is not None and b[p] is not None]
        undecided = len(shared) - len(decisive)
        if not decisive:
            print(f"  {q_name}: no pairs decisive in both pools "
                  f"({undecided} undecided)")
            continue
        xs = [x for x, _ in decisive]
        ys = [y for _, y in decisive]
        ck = cohen_kappa(xs, ys)
        raw = sum(1 for x, y in decisive if x == y) / len(decisive)
        print(f"  {q_name}: Cohen's kappa = {ck:.3f}, raw agreement = "
              f"{raw:.0%} over {len(decisive)} pairs "
              f"({undecided} undecided, excluded)")
    print()


def base_judge(rater):
    """Strip the '__audio'/'__text' condition suffix from a judge rater id."""
    return rater.rsplit("__", 1)[0]


def rater_labels(dedup, kept, key, q_idx):
    """{rater: {pid: cat}} over kept raters, non-sanity pairs, CATS-only
    (cant_tell / missing dropped). Used to build the judge-vs-judge matrix."""
    out = defaultdict(dict)
    for (rater, pid), ans in dedup.items():
        if rater not in kept or pid not in key:
            continue
        if key[pid]["pair_type"] == "sanity":
            continue
        if ans[q_idx] in CATS:
            out[rater][pid] = ans[q_idx]
    return out


def heatmap(human_cons, llm_dedup, llm_kept, key, out_path):
    """Render a per-question entity x entity Cohen-kappa heatmap. Entities are
    one human-consensus 'rater' (per-pair plurality of the kept humans) plus each
    kept audio judge. Diagonal forced to 1.0; off-diagonal = Cohen's kappa over
    pairs both label decisively."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    judges = sorted({base_judge(r) for r in llm_kept})
    fig, axes = plt.subplots(1, len(QUESTIONS), figsize=(5.2 * len(QUESTIONS), 4.6))
    for ax, (q_idx, q_name) in zip(np.atleast_1d(axes), QUESTIONS):
        # Per-entity {pid: cat}: human consensus (drop undecided), then judges.
        labels = {"human (consensus)":
                  {p: c for p, c in human_cons.get(q_idx, {}).items() if c}}
        jl = rater_labels(llm_dedup, llm_kept, key, q_idx)
        per_judge = defaultdict(dict)
        for rater, d in jl.items():
            per_judge[base_judge(rater)].update(d)
        for j in judges:
            labels[j] = per_judge[j]
        names = list(labels)
        ticks = [{"gemini_2.5": "gemini-2.5-pro"}.get(nm, nm) for nm in names]
        n = len(names)
        mat = np.full((n, n), np.nan)
        for i in range(n):
            for k in range(n):
                if i == k:
                    mat[i, k] = 1.0
                    continue
                shared = sorted(set(labels[names[i]]) & set(labels[names[k]]))
                if not shared:
                    continue
                a = [labels[names[i]][p] for p in shared]
                b = [labels[names[k]][p] for p in shared]
                ck = cohen_kappa(a, b)
                if ck is not None:
                    mat[i, k] = ck
        im = ax.imshow(mat, cmap="RdYlGn", vmin=-0.3, vmax=1.0)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(ticks, rotation=40, ha="right", fontsize=8)
        ax.set_yticklabels(ticks, fontsize=8)
        ax.set_title(q_name, fontsize=10)
        for i in range(n):
            for k in range(n):
                if not np.isnan(mat[i, k]):
                    ax.text(k, i, f"{mat[i, k]:.2f}", ha="center", va="center",
                            fontsize=8, color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Cohen's kappa")
    fig.suptitle("Audio condition: rater x rater agreement", fontsize=11)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"=== heatmap written to {out} (+ .png) ===\n")


def self_contrast(audio_dedup, audio_kept, text_dedup, text_kept, key):
    """Per-judge audio-vs-text grounding self-contrast: for judges present in
    BOTH conditions (same model both times), Cohen's kappa + raw agreement on the
    pairs each labels decisively in both. Judges that differ across conditions
    (e.g. gpt_audio vs gpt_text) are not a true self-contrast -> flagged."""
    print("=== audio-vs-text grounding self-contrast (per judge) ===")
    a_by = defaultdict(set)
    for r in audio_kept:
        a_by[base_judge(r)].add(r)
    t_by = defaultdict(set)
    for r in text_kept:
        t_by[base_judge(r)].add(r)
    shared_judges = sorted(set(a_by) & set(t_by))
    if not shared_judges:
        print("  no judge present in both conditions\n")
        return
    for judge in shared_judges:
        ar = next(iter(a_by[judge]))
        tr = next(iter(t_by[judge]))
        for q_idx, q_name in QUESTIONS:
            al = {p: c for (rr, p), ans in audio_dedup.items()
                  if rr == ar and p in key
                  and key[p]["pair_type"] != "sanity"
                  and (c := ans[q_idx]) in CATS}
            tl = {p: c for (rr, p), ans in text_dedup.items()
                  if rr == tr and p in key
                  and key[p]["pair_type"] != "sanity"
                  and (c := ans[q_idx]) in CATS}
            shared = sorted(set(al) & set(tl))
            if not shared:
                print(f"  {judge} {q_name}: no shared decisive pairs")
                continue
            a = [al[p] for p in shared]
            b = [tl[p] for p in shared]
            ck = cohen_kappa(a, b)
            raw = sum(1 for x, y in zip(a, b) if x == y) / len(shared)
            print(f"  {judge} {q_name}: Cohen's kappa = {ck:.3f}, raw = {raw:.0%} "
                  f"over {len(shared)} pairs")
    print()


def load_pool(csv_path, json_dir):
    raw = []
    if csv_path:
        load_csv(csv_path, raw)
    if json_dir:
        load_json_dir(json_dir, raw)
    # Dedup by (rater, pair_id); last write wins.
    return {(rater, pid): (q1, q2) for rater, pid, q1, q2 in raw}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="human Sheet export")
    ap.add_argument("--json-dir", help="dir of human results_*.json")
    ap.add_argument("--llm-dir", help="dir of LLM-judge JSON (one condition; the "
                    "audio panel when paired with --heatmap-out / --text-dir)")
    ap.add_argument("--text-dir", help="dir of the TEXT-condition LLM-judge JSON; "
                    "with --llm-dir (audio) triggers the audio-vs-text self-contrast")
    ap.add_argument("--heatmap-out", help="write the human-consensus x audio-judge "
                    "Cohen-kappa heatmap PDF here (needs --csv and --llm-dir audio)")
    ap.add_argument("--key", default="pairwise_eval/key.json")
    ap.add_argument("--sanity-threshold", type=float, default=0.75)
    args = ap.parse_args()

    if not args.csv and not args.json_dir and not args.llm_dir:
        raise SystemExit("provide at least one of --csv / --json-dir / --llm-dir")

    key = {k["id"]: k for k in json.loads(Path(args.key).read_text())}
    sanity_ids = [i for i, k in key.items() if k["pair_type"] == "sanity"]

    human = load_pool(args.csv, args.json_dir) if (args.csv or args.json_dir) else {}
    llm = load_pool(None, args.llm_dir) if args.llm_dir else {}

    human_cons = llm_cons = None
    llm_kept = set()
    if human:
        kept = sanity_keep(human, key, sanity_ids, args.sanity_threshold, "human")
        human_cons = analyze_pool(human, kept, key, "human")
        win_rates(human, kept, key, "human")
    if llm:
        llm_kept = sanity_keep(llm, key, sanity_ids, args.sanity_threshold, "llm")
        llm_cons = analyze_pool(llm, llm_kept, key, "llm")
        win_rates(llm, llm_kept, key, "llm")

    if human_cons and llm_cons:
        cross_consensus(llm_cons, "llm", human_cons, "human")

    if args.heatmap_out:
        if not (human_cons and llm):
            raise SystemExit("--heatmap-out needs both --csv and --llm-dir")
        heatmap(human_cons, llm, llm_kept, key, args.heatmap_out)

    if args.text_dir:
        if not llm:
            raise SystemExit("--text-dir needs --llm-dir (the audio panel)")
        text = load_pool(None, args.text_dir)
        text_kept = sanity_keep(text, key, sanity_ids, args.sanity_threshold, "text")
        self_contrast(llm, llm_kept, text, text_kept, key)


if __name__ == "__main__":
    main()
