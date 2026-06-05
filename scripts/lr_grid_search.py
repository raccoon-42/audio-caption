"""Exhaustive grid search over a fixed set of S2 LR pairs (vs. Optuna's sampled search)."""

import argparse
import json
from pathlib import Path

import optuna
import torch
import yaml

from dataset import load_dataloaders
from lr_search import CONFIGS, EVAL_FNS, TRAIN_FNS, load_model, run_stage
from projection import Projection
from utils import set_seed

# Narrow S2 grid around the suspected-problematic region; lm_lr=0 keeps the LM frozen
GRID = {
    "stage2_proj_lr": [1e-5, 5e-5, 1e-4, 2e-4],
    "stage2_lm_lr": [0.0, 5e-7, 1e-6],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama", choices=["gpt2", "t5", "opt", "llama"])
    parser.add_argument("--epochs-per-stage", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--proj-depth", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the grid combinations and exit")
    args = parser.parse_args()

    combos = [(p, l) for p in GRID["stage2_proj_lr"] for l in GRID["stage2_lm_lr"]]
    n_combos = len(combos)
    if args.dry_run:
        for i, (p, l) in enumerate(combos):
            print(f"{i:2d}: stage2_proj_lr={p:.0e}  stage2_lm_lr={l:.0e}")
        print(f"{n_combos} combinations")
        return

    proj_depth = args.proj_depth
    tag = args.model if proj_depth == 2 else f"{args.model}_depth{proj_depth}"

    config_path = CONFIGS[args.model]
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    db_dir = Path(cfg.get("results_dir", "results")) / args.model / "lr_search"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / f"{tag}_s2_grid.db"
    storage = f"sqlite:///{db_path}"
    print(f"Optuna DB: {db_path}")

    s1_proj_path = db_dir / f"{tag}_s1_best_proj.pt"
    if not s1_proj_path.exists():
        print(f"ERROR: best S1 projection not found at {s1_proj_path}. Run lr_search --stage s1 first.")
        return

    seed = cfg["seed"]
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading models for {args.model}...")
    lm, tokenizer, lm_dim = load_model(args.model, cfg, device)

    # do_3way=True so search validates on the val split, never the held-out test rows
    train_loader, val_loader, _test_loader, audio_dim = load_dataloaders(cfg, tokenizer, seed=seed, do_3way=True)

    prefix_len = cfg["prefix_len"]
    dropout = cfg["dropout"]
    use_layernorm = cfg.get("use_layernorm", False)
    lm_init_state = {k: v.cpu().clone() for k, v in lm.state_dict().items()}
    best_s1_projection_state = torch.load(s1_proj_path, weights_only=True, map_location=device)

    train_fn = TRAIN_FNS[args.model]
    eval_fn = EVAL_FNS[args.model]

    def objective(trial):
        s2_proj_lr = trial.suggest_categorical("stage2_proj_lr", GRID["stage2_proj_lr"])
        s2_lm_lr = trial.suggest_categorical("stage2_lm_lr", GRID["stage2_lm_lr"])

        print(f"\n--- S2 grid trial {trial.number} ---")
        print(f"  stage2_proj_lr={s2_proj_lr:.6g}  stage2_lm_lr={s2_lm_lr:.6g}")

        set_seed(seed)
        projection = Projection(audio_dim, lm_dim, prefix_len, dropout=dropout,
                                depth=proj_depth, use_layernorm=use_layernorm).to(device)
        projection.load_state_dict(best_s1_projection_state)

        lm.load_state_dict(lm_init_state)
        if s2_lm_lr > 0:
            for p in lm.parameters():
                p.requires_grad = True
            lm.train()
            param_groups = [
                {"params": projection.parameters(), "lr": s2_proj_lr},
                {"params": lm.parameters(), "lr": s2_lm_lr},
            ]
        else:
            # lm_lr=0: identical updates to a zero-LR param group, but no LM grads/optimizer state
            for p in lm.parameters():
                p.requires_grad = False
            lm.eval()
            param_groups = [{"params": projection.parameters(), "lr": s2_proj_lr}]

        optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg["weight_decay"])

        s2_val, s2_best_epoch = run_stage(
            train_fn, eval_fn, projection, lm, train_loader, val_loader,
            optimizer, prefix_len, device, args.epochs_per_stage, args.patience,
            trial, epoch_offset=0, t_max=args.epochs_per_stage,
        )
        print(f"  Trial {trial.number}: val={s2_val:.4f} (best epoch {s2_best_epoch})")
        trial.set_user_attr("best_epoch", s2_best_epoch)
        return s2_val

    sampler = optuna.samplers.GridSampler(GRID)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=optuna.pruners.NopPruner(),  # exhaustive grid: never prune
        study_name=f"lr_grid_{tag}_s2",
        storage=storage,
        load_if_exists=True,
    )
    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, n_combos - completed)
    if remaining > 0:
        print(f"Grid: {completed} done, {remaining} remaining")
        study.optimize(objective, n_trials=remaining)
    else:
        print(f"Grid already complete ({completed} trials)")

    done = sorted(
        (t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE),
        key=lambda t: t.value,
    )
    print(f"\n{'='*50}")
    print("Grid results (best to worst):")
    for t in done:
        print(f"  proj_lr={t.params['stage2_proj_lr']:.0e}  lm_lr={t.params['stage2_lm_lr']:.0e}"
              f"  val={t.value:.4f}  best_epoch={t.user_attrs.get('best_epoch')}")

    results = {
        "model": args.model,
        "proj_depth": proj_depth,
        "use_layernorm": use_layernorm,
        "epochs_per_stage": args.epochs_per_stage,
        "grid": GRID,
        "best": {"params": done[0].params, "val_loss": done[0].value} if done else None,
        "all_trials": [
            {"number": t.number, "value": t.value, "params": t.params,
             "best_epoch": t.user_attrs.get("best_epoch"), "state": str(t.state)}
            for t in study.trials
        ],
    }
    out_path = db_dir / f"{tag}_s2_grid.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")
    print("NOTE: config not auto-updated; apply the best pair manually after inspection.")


if __name__ == "__main__":
    main()
