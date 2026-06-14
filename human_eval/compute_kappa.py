"""Analyse the pairwise human-eval responses.

Reads responses from a Sheet CSV export (--csv) and/or downloaded per-rater
JSON files (--json-dir), joins them to key.json, and reports:
  - Fleiss' kappa among raters for Q1 and Q2 (protocol reliability)
  - per-rater sanity-check accuracy (data-quality filter)
  - win-rates per comparison (the result)

Categories are {A, B, tie}. A rater below --sanity-threshold on the sanity
pairs is dropped before computing kappa and win-rates.
"""
import argparse
import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

CATS = ["A", "B", "tie"]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="Sheet export")
    ap.add_argument("--json-dir", help="dir of downloaded results_*.json")
    ap.add_argument("--key", default="human_eval/key.json")
    ap.add_argument("--sanity-threshold", type=float, default=0.75)
    args = ap.parse_args()

    if not args.csv and not args.json_dir:
        raise SystemExit("provide --csv and/or --json-dir")

    raw = []
    if args.csv:
        load_csv(args.csv, raw)
    if args.json_dir:
        load_json_dir(args.json_dir, raw)

    # Dedup by (rater, pair_id); last write wins.
    dedup = {(rater, pid): (q1, q2) for rater, pid, q1, q2 in raw}
    key = {k["id"]: k for k in json.loads(Path(args.key).read_text())}

    # Sanity accuracy per rater (Q1 picking the matching caption).
    sanity_ids = [i for i, k in key.items() if k["pair_type"] == "sanity"]
    rater_ids = sorted({r for r, _ in dedup})
    print("=== sanity-check accuracy (Q1) ===")
    kept = []
    for rater in rater_ids:
        hits = tot = 0
        for sid in sanity_ids:
            if (rater, sid) in dedup:
                tot += 1
                if dedup[(rater, sid)][0] == key[sid].get("correct_side"):
                    hits += 1
        acc = hits / tot if tot else 0.0
        flag = "" if acc >= args.sanity_threshold else "  <-- DROPPED"
        print(f"  {rater:12s} {hits}/{tot} = {acc:.2f}{flag}")
        if acc >= args.sanity_threshold:
            kept.append(rater)
    print(f"kept {len(kept)}/{len(rater_ids)} raters\n")

    kept_set = set(kept)

    # Fleiss + Cohen on the non-sanity pairs, per question.
    for q_idx, q_name in [(0, "Q1 (more accurate)"), (1, "Q2 (less wrong)")]:
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
        print(f"=== {q_name} ===")
        if n_total:
            print(f"  'can't tell' abstentions: {n_abstain}/{n_total} "
                  f"({n_abstain / n_total:.0%})  [excluded from kappa]")
        print(f"  Fleiss' kappa = {fk:.3f} over {n_items} items"
              if fk is not None else "  no data")
        if ck is not None:
            print(f"  mean pairwise Cohen's kappa = {ck:.3f}")
        print()

    # Win-rates per comparison (kept raters).
    print("=== win-rates (kept raters) ===")
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


if __name__ == "__main__":
    main()
