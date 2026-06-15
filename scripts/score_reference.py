"""Score a reference captioner's predictions and run the MusicCaps-leakage split
(main env). Model-agnostic: works for any predictions file in the evaluate.py
schema (BLAP, LP-MusicCaps transfer, etc.) -- only blap_generate.py is BLAP-specific.

Three things:
1. Overall metrics via evaluate.compute_metrics (same 10 metrics + FENSE as every
   reference row), written to the path reference_comparison.py expects
   (results/reference/<name>/<name>_metrics_<name>.json) so the model drops into
   the table like the other rows.
2. A leakage split: the seed-42 test set is divided by is_audioset_eval into the
   in-MusicCaps-train subset (==0) and the held-out subset (==1). A checkpoint
   fine-tuned on the MusicCaps official-train half inflates scores on the ==0
   subset of this random split; a clean model scores evenly across both. The
   CIDEr-D in/out ratio is the verdict.
3. With --held-out-name, also writes the held-out subset (==1) -- the leakage-free
   clean number -- as its own reference row.

Predictions must carry is_audioset_eval for steps 2-3 (carry it through at
generation, or join it on by reference text from results/test_split.json).
"""
import argparse
import json
import re
from pathlib import Path

from evaluate import compute_metrics

WATCH = ["CIDEr-D", "SPIDEr", "FENSE", "SBERT-sim"]  # leakage-sensitive metrics
LEAK_RATIO = 1.5  # CIDEr-D in/out above this flags train-on-test

_SENT_END = re.compile(r"[.!?](?=[\"')\]\s]|$)")


def trim_to_last_sentence(text):
    """Cut a caption at its last sentence-terminating mark so a hypothesis that hit
    the generator's token cap mid-sentence is not penalised by FENSE's incomplete-
    sentence fluency detector. Returns the text unchanged when no terminator is
    present (a single unfinished fragment -- nothing safe to trim to)."""
    ends = [m.end() for m in _SENT_END.finditer(text)]
    return text[:ends[-1]].strip() if ends else text.strip()


def _metrics(items):
    return compute_metrics([p["reference"] for p in items],
                           [p["hypothesis"] for p in items])


def _write_ref(name, pred, items, metrics, out=None):
    """Write a reference-schema metrics file at the path reference_comparison.py
    expects: results/reference/<name>/<name>_metrics_<name>.json."""
    out = Path(out) if out else (
        Path("results/reference") / name / f"{name}_metrics_{name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": pred.get("model", name),
        "stage": "reference",
        "framework": pred.get("framework", name),
        "num_samples": len(items),
        "num_failed": pred.get("num_failed", 0),
        "gen_kwargs": pred.get("gen_kwargs", {}),
        "metrics": metrics,
        "predictions": items,
    }, indent=2))
    print(f"Saved -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred",
                    default="results/reference/blap/predictions/blap_predictions_blap.json")
    ap.add_argument("--name", default="blap")
    ap.add_argument("--out", default=None,
                    help="Defaults to results/reference/<name>/<name>_metrics_<name>.json")
    ap.add_argument("--held-out-name", default=None,
                    help="If set, also write the held-out subset (is_audioset_eval==1, "
                         "the leakage-free clean number) as a reference row under this "
                         "name, e.g. blap_clean.")
    ap.add_argument("--manifest", default="results/test_split.json",
                    help="Used to backfill is_audioset_eval by reference text when the "
                         "predictions file lacks it (older generators).")
    ap.add_argument("--trim-sentences", action="store_true",
                    help="Trim each hypothesis to its last complete sentence before "
                         "scoring, so captions truncated mid-sentence by the generator's "
                         "token cap are not penalised by FENSE's incomplete-sentence "
                         "fluency detector.")
    args = ap.parse_args()

    pred = json.loads(Path(args.pred).read_text())
    items = pred["predictions"]

    if args.trim_sentences:
        n_trim = 0
        for p in items:
            trimmed = trim_to_last_sentence(p["hypothesis"])
            if trimmed != p["hypothesis"]:
                p["hypothesis"] = trimmed
                n_trim += 1
        print(f"Trimmed {n_trim}/{len(items)} hypotheses to last complete sentence.")

    # Backfill the leakage tag for older predictions (reference/hypothesis only) by
    # joining on reference caption text (unique per clip); a no-op if already tagged.
    if any(p.get("is_audioset_eval") is None for p in items):
        manifest = Path(args.manifest)
        if manifest.exists():
            tag = {r["caption"]: r["is_audioset_eval"] for r in json.loads(manifest.read_text())}
            filled = 0
            for p in items:
                if p.get("is_audioset_eval") is None and p["reference"] in tag:
                    p["is_audioset_eval"] = tag[p["reference"]]
                    filled += 1
            if filled:
                print(f"Backfilled is_audioset_eval for {filled} predictions from {manifest}")

    print(f"Scoring {len(items)} predictions ({pred.get('framework')}) ...")
    metrics = _metrics(items)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    _write_ref(args.name, pred, items, metrics, args.out)

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

    if args.held_out_name:
        # The held-out subset (never in any MusicCaps finetune) is the leakage-free
        # reportable score regardless of whether the released ckpt leaks.
        print(f"\nWriting held-out clean row '{args.held_out_name}' "
              f"(n={len(held_out)}):")
        _write_ref(args.held_out_name, pred, held_out, m_out)


if __name__ == "__main__":
    main()
