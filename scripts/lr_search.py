"""Optuna-based learning rate search for all models."""

import argparse
import json
from pathlib import Path

import optuna
import torch
import yaml
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OPTForCausalLM,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

from dataset import load_dataloaders
from projection import Projection
from train_gpt2 import train_one_epoch as gpt2_train, evaluate as gpt2_eval
from train_t5 import train_one_epoch as t5_train, evaluate as t5_eval
from train_opt import train_one_epoch as opt_train, evaluate as opt_eval
from train_llama import train_one_epoch as llama_train, evaluate as llama_eval
from utils import set_seed


CONFIGS = {
    "gpt2": "configs/gpt2.yaml", "t5": "configs/t5.yaml", "opt": "configs/opt.yaml",
    "llama": "configs/llama.yaml",
}

TRAIN_FNS = {"gpt2": gpt2_train, "t5": t5_train, "opt": opt_train, "llama": llama_train}
EVAL_FNS = {"gpt2": gpt2_eval, "t5": t5_eval, "opt": opt_eval, "llama": llama_eval}

SEARCH_RANGES = {
    "gpt2":  {"s1_lr": (1e-4, 5e-2), "s2_proj_lr": (1e-5, 1e-3), "s2_lm_lr": (5e-6, 1e-2)},
    "t5":    {"s1_lr": (1e-2, 1e-1), "s2_proj_lr": (1e-5, 1e-3), "s2_lm_lr": (5e-6, 1e-2)},
    "opt":   {"s1_lr": (1e-5, 5e-3), "s2_proj_lr": (1e-5, 1e-3), "s2_lm_lr": (5e-6, 1e-2)},
    "llama": {"s1_lr": (1e-5, 5e-2), "s2_proj_lr": (1e-5, 1e-2), "s2_lm_lr": (1e-7, 1e-4)},
}


def load_model(model_key, cfg, device):
    if model_key == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        lm = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        lm_dim = lm.config.n_embd
    elif model_key == "t5":
        model_name = cfg["model_name"]
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        lm = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
        lm_dim = lm.config.d_model
    elif model_key == "opt":
        model_name = cfg["model_name"]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        lm = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
        lm_dim = lm.config.word_embed_proj_dim
    elif model_key == "llama":
        model_name = cfg["model_name"]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        lm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
        lm_dim = lm.config.hidden_size

    lm.eval()
    for p in lm.parameters():
        p.requires_grad = False

    return lm, tokenizer, lm_dim


def run_stage(train_fn, eval_fn, projection, lm, train_loader, val_loader,
              optimizer, prefix_len, device, max_epochs, patience, trial, epoch_offset,
              t_max=None):
    best_val = float("inf")
    best_epoch = 0
    patience_counter = 0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max or max_epochs)

    stage = "S1" if epoch_offset == 0 else "S2"

    for epoch in range(1, max_epochs + 1):
        train_loss = train_fn(projection, lm, train_loader, optimizer, prefix_len, device)
        val_loss = eval_fn(projection, lm, val_loader, prefix_len, device)
        scheduler.step()

        lr_str = ", ".join(f"{pg['lr']:.2e}" for pg in optimizer.param_groups)
        print(f"  [{stage}] Epoch {epoch}/{max_epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  lr=[{lr_str}]")

        trial.report(val_loss, epoch_offset + epoch)
        if trial.should_prune():
            print(f"  [{stage}] Pruned at epoch {epoch}")
            raise optuna.TrialPruned()

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [{stage}] Early stopping at epoch {epoch}")
                break

    return best_val, best_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["gpt2", "t5", "opt", "llama"])
    parser.add_argument("--stage", choices=["s1", "s2", "both"], default="both")
    parser.add_argument("--n-trials", type=int, default=16)
    parser.add_argument("--epochs-per-stage", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--use-layernorm", action="store_true")
    parser.add_argument("--proj-depth", type=int, default=2)
    args = parser.parse_args()

    proj_depth = args.proj_depth
    tag = args.model if proj_depth == 2 else f"{args.model}_depth{proj_depth}"

    config_path = CONFIGS[args.model]
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    db_dir = Path(cfg.get("results_dir", "results")) / args.model / "lr_search"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / f"{tag}_lr_search.db"
    storage = f"sqlite:///{db_path}"
    print(f"Optuna DB: {db_path}")

    seed = cfg["seed"]
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading models for {args.model}...")
    lm, tokenizer, lm_dim = load_model(args.model, cfg, device)

    # do_3way=True so search validates on the val split, never the held-out test rows
    train_loader, val_loader, _test_loader, audio_dim = load_dataloaders(cfg, tokenizer, seed=seed, do_3way=True)

    prefix_len = cfg["prefix_len"]
    dropout = cfg["dropout"]
    use_layernorm = args.use_layernorm or cfg.get("use_layernorm", False)
    lm_init_state = {k: v.cpu().clone() for k, v in lm.state_dict().items()}

    train_fn = TRAIN_FNS[args.model]
    eval_fn = EVAL_FNS[args.model]

    # ---- Stage 1 search: find best S1 LR ----
    s1_proj_path = db_dir / f"{tag}_s1_best_proj.pt"
    best_s1_lr = None

    if args.stage in ("s1", "both"):
        torch.cuda.empty_cache()
        print(f"\n{'='*50}")
        print("Stage 1 LR search")
        print(f"{'='*50}")

        s1_range = SEARCH_RANGES[args.model]["s1_lr"]

        def s1_objective(trial):
            lo, hi = s1_range
            s1_lr = trial.suggest_float("stage1_lr", lo, hi, log=True)

            print(f"\n--- S1 Trial {trial.number} ---")
            print(f"  stage1_lr={s1_lr:.6g}  (range: [{lo:.1e}, {hi:.1e}])")

            set_seed(seed)
            projection = Projection(audio_dim, lm_dim, prefix_len, dropout=dropout, depth=proj_depth, use_layernorm=use_layernorm).to(device)

            lm.load_state_dict(lm_init_state)
            lm.eval()
            for p in lm.parameters():
                p.requires_grad = False

            optimizer = torch.optim.AdamW(
                projection.parameters(), lr=s1_lr, weight_decay=cfg["weight_decay"]
            )
            s1_val, s1_best_epoch = run_stage(
                train_fn, eval_fn, projection, lm, train_loader, val_loader,
                optimizer, prefix_len, device, args.epochs_per_stage, args.patience,
                trial, epoch_offset=0, t_max=args.epochs_per_stage,
            )

            try:
                study_best = trial.study.best_value
                study_best_trial = trial.study.best_trial.number
                is_best = s1_val <= study_best
            except ValueError:
                study_best = None
                study_best_trial = None
                is_best = True

            print(f"  Trial {trial.number}: val={s1_val:.4f} (best epoch {s1_best_epoch})"
                  f"  | Study best: trial {study_best_trial} val={study_best:.4f}"
                  if study_best is not None else
                  f"  Trial {trial.number}: val={s1_val:.4f} (best epoch {s1_best_epoch})"
                  f"  | First trial")

            if is_best:
                torch.save(projection.state_dict(), s1_proj_path)
                print(f"  New best S1 -- projection saved to {s1_proj_path}")

            return s1_val

        s1_study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5),
            study_name=f"lr_search_{tag}_s1",
            storage=storage,
            load_if_exists=True,
        )
        completed_s1 = len([t for t in s1_study.trials if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)])
        remaining_s1 = max(0, args.n_trials - completed_s1)
        if remaining_s1 > 0:
            print(f"Resuming S1: {completed_s1} done, {remaining_s1} remaining")
            s1_study.optimize(s1_objective, n_trials=remaining_s1)
        else:
            print(f"S1 already complete ({completed_s1} trials)")

        best_s1 = s1_study.best_trial
        best_s1_lr = best_s1.params["stage1_lr"]
        print(f"\nBest S1 trial #{best_s1.number} (val_loss={best_s1.value:.4f}):")
        print(f"  stage1_lr: {best_s1_lr:.6g}")

        if not s1_proj_path.exists():
            print("Best S1 projection not on disk -- rerunning best trial to regenerate")
            set_seed(seed)
            projection = Projection(audio_dim, lm_dim, prefix_len, dropout=dropout, depth=proj_depth, use_layernorm=use_layernorm).to(device)
            lm.load_state_dict(lm_init_state)
            lm.eval()
            for p in lm.parameters():
                p.requires_grad = False
            optimizer = torch.optim.AdamW(
                projection.parameters(), lr=best_s1_lr, weight_decay=cfg["weight_decay"]
            )
            dummy_trial = s1_study.ask()
            run_stage(train_fn, eval_fn, projection, lm, train_loader, val_loader,
                      optimizer, prefix_len, device, args.epochs_per_stage, args.patience,
                      dummy_trial, epoch_offset=0, t_max=args.epochs_per_stage)
            torch.save(projection.state_dict(), s1_proj_path)

    # ---- Stage 2 search: find best S2 LRs using best S1 projection ----
    if args.stage in ("s2", "both"):
        torch.cuda.empty_cache()
        if not s1_proj_path.exists():
            print("ERROR: S1 projection not found. Run --stage s1 first.")
            return

        print(f"\n{'='*50}")
        print("Stage 2 LR search (using best S1 projection)")
        print(f"{'='*50}")

        best_s1_projection_state = torch.load(s1_proj_path, weights_only=True, map_location=device)

        s2_proj_range = SEARCH_RANGES[args.model]["s2_proj_lr"]
        s2_lm_range = SEARCH_RANGES[args.model]["s2_lm_lr"]

        def s2_objective(trial):
            lo, hi = s2_proj_range
            s2_proj_lr = trial.suggest_float("stage2_proj_lr", lo, hi, log=True)
            lo, hi = s2_lm_range
            s2_lm_lr = trial.suggest_float("stage2_lm_lr", lo, hi, log=True)

            print(f"\n--- S2 Trial {trial.number} ---")
            print(f"  stage2_proj_lr={s2_proj_lr:.6g}  (range: [{s2_proj_range[0]:.1e}, {s2_proj_range[1]:.1e}])")
            print(f"  stage2_lm_lr={s2_lm_lr:.6g}    (range: [{s2_lm_range[0]:.1e}, {s2_lm_range[1]:.1e}])")

            set_seed(seed)
            projection = Projection(audio_dim, lm_dim, prefix_len, dropout=dropout, depth=proj_depth, use_layernorm=use_layernorm).to(device)
            projection.load_state_dict(best_s1_projection_state)

            lm.load_state_dict(lm_init_state)
            for p in lm.parameters():
                p.requires_grad = True
            lm.train()

            optimizer = torch.optim.AdamW([
                {"params": projection.parameters(), "lr": s2_proj_lr},
                {"params": lm.parameters(), "lr": s2_lm_lr},
            ], weight_decay=cfg["weight_decay"])

            s2_val, s2_best_epoch = run_stage(
                train_fn, eval_fn, projection, lm, train_loader, val_loader,
                optimizer, prefix_len, device, args.epochs_per_stage, args.patience,
                trial, epoch_offset=0, t_max=args.epochs_per_stage,
            )

            try:
                study_best = trial.study.best_value
                study_best_trial = trial.study.best_trial.number
            except ValueError:
                study_best = None
                study_best_trial = None

            print(f"  Trial {trial.number}: val={s2_val:.4f} (best epoch {s2_best_epoch})"
                  f"  | Study best: trial {study_best_trial} val={study_best:.4f}"
                  if study_best is not None else
                  f"  Trial {trial.number}: val={s2_val:.4f} (best epoch {s2_best_epoch})"
                  f"  | First trial")

            return s2_val

        s2_study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5),
            study_name=f"lr_search_{tag}_s2",
            storage=storage,
            load_if_exists=True,
        )
        completed_s2 = len([t for t in s2_study.trials if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)])
        remaining_s2 = max(0, args.n_trials - completed_s2)
        if remaining_s2 > 0:
            print(f"Resuming S2: {completed_s2} done, {remaining_s2} remaining")
            s2_study.optimize(s2_objective, n_trials=remaining_s2)
        else:
            print(f"S2 already complete ({completed_s2} trials)")

        best_s2 = s2_study.best_trial
        print(f"\nBest S2 trial #{best_s2.number} (val_loss={best_s2.value:.4f}):")
        print(f"  stage2_proj_lr: {best_s2.params['stage2_proj_lr']:.6g}")
        print(f"  stage2_lm_lr:  {best_s2.params['stage2_lm_lr']:.6g}")

    # ---- Save results ----
    results_dir = Path(cfg["results_dir"]) / args.model / "lr_search"
    results_dir.mkdir(parents=True, exist_ok=True)

    if best_s1_lr is None:
        s1_study = optuna.load_study(
            study_name=f"lr_search_{tag}_s1",
            storage=storage,
        )
        best_s1 = s1_study.best_trial
        best_s1_lr = best_s1.params["stage1_lr"]

    if args.stage == "s1":
        if proj_depth == 2:
            cfg["stage1"]["lr"] = float(best_s1_lr)
            with open(config_path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            print(f"Updated {config_path} with best S1 LR")
        else:
            print(f"depth={proj_depth}: best S1 LR = {best_s1_lr:.6g} (config not modified)")
        return

    print(f"\n{'='*50}")
    print("Final best LRs:")
    print(f"  stage1_lr:      {best_s1_lr:.6g}")
    print(f"  stage2_proj_lr: {best_s2.params['stage2_proj_lr']:.6g}")
    print(f"  stage2_lm_lr:   {best_s2.params['stage2_lm_lr']:.6g}")

    results = {
        "model": args.model,
        "proj_depth": proj_depth,
        "use_layernorm": use_layernorm,
        "n_trials": args.n_trials,
        "epochs_per_stage": args.epochs_per_stage,
        "stage1": {
            "best_trial": best_s1.number,
            "best_val_loss": best_s1.value,
            "best_lr": best_s1_lr,
            "all_trials": [
                {"number": t.number, "value": t.value, "params": t.params, "state": str(t.state)}
                for t in s1_study.trials
            ],
        },
        "stage2": {
            "best_trial": best_s2.number,
            "best_val_loss": best_s2.value,
            "best_params": best_s2.params,
            "all_trials": [
                {"number": t.number, "value": t.value, "params": t.params, "state": str(t.state)}
                for t in s2_study.trials
            ],
        },
    }
    out_path = results_dir / f"{tag}_lr_search.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")

    if proj_depth == 2:
        cfg["stage1"]["lr"] = float(best_s1_lr)
        cfg["stage2"]["projection_lr"] = float(best_s2.params["stage2_proj_lr"])
        cfg["stage2"]["lm_lr"] = float(best_s2.params["stage2_lm_lr"])
        with open(config_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f"Updated {config_path} with best LRs")
    else:
        print(f"depth={proj_depth}: config not modified (best LRs saved to {out_path})")


if __name__ == "__main__":
    main()
