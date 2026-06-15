"""Score BLAP predictions and run the MusicCaps-leakage split (main env).

Two things:
1. Overall metrics via evaluate.compute_metrics (same 10 metrics + FENSE as every
   reference row), written to the path reference_comparison.py expects
   (results/reference/<name>/<name>_metrics_<name>.json) so BLAP drops into the
   table like the other rows.
2. A leakage split: the seed-42 test set is divided by is_audioset_eval into the
   in-MusicCaps-train subset (==0) and the held-out subset (==1). A checkpoint
   fine-tuned on the MusicCaps official-train half inflates scores on the ==0
   subset of this random split; a clean model scores evenly across both. The
   CIDEr-D in/out ratio is the verdict (same signature that exposed LP transfer).

Works for any predictions file that carries is_audioset_eval (e.g. a leaky
LP-MusicCaps transfer rerun), not just BLAP.
"""
import argparse
import json
from pathlib import Path

from evaluate import compute_metrics

WATCH = ["CIDEr-D", "SPIDEr", "FENSE", "SBERT-sim"]  # leakage-sensitive metrics
LEAK_RATIO = 1.5  # CIDEr-D in/out above this flags train-on-test


def _metrics(items):
    return compute_metrics([p["reference"] for p in items],
                           [p["hypothesis"] for p in items])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred",
                    default="results/reference/blap/predictions/blap_predictions_blap.json")
    ap.add_argument("--name", default="blap")
    ap.add_argument("--out", default=None,
                    help="Defaults to results/reference/<name>/<name>_metrics_<name>.json")
    args = ap.parse_args()

    pred = json.loads(Path(args.pred).read_text())
    items = pred["predictions"]
    print(f"Scoring {len(items)} predictions ({pred.get('framework')}) ...")
    metrics = _metrics(items)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    out = Path(args.out) if args.out else (
        Path("results/reference") / args.name / f"{args.name}_metrics_{args.name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": pred.get("model", args.name),
        "stage": "reference",
        "framework": pred.get("framework", args.name),
        "num_samples": len(items),
        "num_failed": pred.get("num_failed", 0),
        "gen_kwargs": pred.get("gen_kwargs", {}),
        "metrics": metrics,
        "predictions": items,
    }, indent=2))
    print(f"Saved -> {out}")

    # --- leakage split -------------------------------------------------------
    in_train = [p for p in items if p.get("is_audioset_eval") in (0, 0.0)]
    held_out = [p for p in items if p.get("is_audioset_eval") in (1, 1.0)]
    if not in_train or not held_out:
        n_tagged = sum(1 for p in items if p.get("is_audioset_eval") is not None)
        print(f"\nLeakage split skipped: need both subsets "
              f"(tagged={n_tagged}/{len(items)}, in_train={len(in_train)}, "
              f"held_out={len(held_out)}). Regenerate the manifest with "
              f"is_audioset_eval, then re-run generation.")
        return

    print(f"\n=== Leakage split: in-MusicCaps-train (n={len(in_train)}) vs "
          f"held-out (n={len(held_out)}) ===")
    m_in, m_out = _metrics(in_train), _metrics(held_out)
    print(f"{'metric':10} {'in_train':>10} {'held_out':>10} {'ratio':>8}")
    for m in WATCH:
        a, b = m_in.get(m, float("nan")), m_out.get(m, float("nan"))
        ratio = a / b if b else float("inf")
        print(f"{m:10} {a:10.4f} {b:10.4f} {ratio:8.2f}")

    cd_in, cd_out = m_in.get("CIDEr-D", 0.0), m_out.get("CIDEr-D", 0.0)
    ratio = cd_in / cd_out if cd_out else float("inf")
    if ratio > LEAK_RATIO:
        verdict = ("LEAKY -> released checkpoint was fine-tuned on MusicCaps "
                   "train; report with a leakage dagger (leaks=True)")
    else:
        verdict = ("CLEAN -> even across subsets, consistent with the "
                   "ShutterStock-only checkpoint; no dagger (leaks=False)")
    print(f"\nVerdict: CIDEr-D in/out ratio = {ratio:.2f} -> {verdict}")


if __name__ == "__main__":
    main()
