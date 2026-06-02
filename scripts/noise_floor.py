"""Compute noise floor statistics from seed evaluation results."""

import argparse
import json
from pathlib import Path

import numpy as np


SEED_CONFIGS = {
    "gpt2": [42, 43, 44],
    "t5": [42, 43],
}


def load_seed_metrics(results_dir, model, seeds):
    metrics = {}
    for seed in seeds:
        tag = model if seed == 42 else f"{model}_seed{seed}"
        path = results_dir / model / "ablations" / "arch" / f"{model}_ablation_{tag}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")
        with open(path) as f:
            data = json.load(f)
        metrics[seed] = data["metrics"]
    return metrics


def compute_noise_floor(metrics_by_seed):
    keys = list(next(iter(metrics_by_seed.values())).keys())
    stats = {}
    for key in keys:
        values = [metrics_by_seed[s][key] for s in metrics_by_seed]
        stats[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "range": float(np.max(values) - np.min(values)),
            "values": {str(s): v for s, v in zip(metrics_by_seed.keys(), values)},
        }
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(SEED_CONFIGS.keys()))
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    seeds = SEED_CONFIGS[args.model]

    print(f"Computing noise floor for {args.model} (seeds: {seeds})")
    metrics_by_seed = load_seed_metrics(results_dir, args.model, seeds)
    stats = compute_noise_floor(metrics_by_seed)

    print(f"\n{'Metric':<12} {'Mean':>8} {'Std':>8} {'Range':>8}")
    print("-" * 40)
    for key, s in stats.items():
        print(f"{key:<12} {s['mean']:>8.4f} {s['std']:>8.4f} {s['range']:>8.4f}")

    out = {
        "model": args.model,
        "seeds": seeds,
        "noise_floor": stats,
    }
    out_path = results_dir / args.model / f"{args.model}_noise_floor.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
