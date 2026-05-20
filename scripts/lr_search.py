"""Optuna-based learning rate search for all models."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import optuna
import torch
import yaml
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    ClapModel,
    ClapProcessor,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OPTForCausalLM,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

from dataset import MusicCapsDataset
from projection import Projection
from train_gpt2 import train_one_epoch as gpt2_train, evaluate as gpt2_eval
from train_t5 import train_one_epoch as t5_train, evaluate as t5_eval
from train_opt import train_one_epoch as opt_train, evaluate as opt_eval
from utils import set_seed


CONFIGS = {"gpt2": "configs/gpt2.yaml", "t5": "configs/t5.yaml", "opt": "configs/opt.yaml"}

TRAIN_FNS = {"gpt2": gpt2_train, "t5": t5_train, "opt": opt_train}
EVAL_FNS = {"gpt2": gpt2_eval, "t5": t5_eval, "opt": opt_eval}


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

    lm.eval()
    for p in lm.parameters():
        p.requires_grad = False

    return lm, tokenizer, lm_dim


def run_stage(train_fn, eval_fn, projection, lm, train_loader, val_loader,
              optimizer, prefix_len, device, max_epochs, patience, trial, epoch_offset,
              t_max=None):
    best_val = float("inf")
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
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [{stage}] Early stopping at epoch {epoch}")
                break

    return best_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["gpt2", "t5", "opt"])
    parser.add_argument("--n-trials", type=int, default=8)
    parser.add_argument("--epochs-per-stage", type=int, default=5)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    config_path = CONFIGS[args.model]
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    db_dir = Path(cfg.get("results_dir", "results")) / args.model
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / f"{args.model}_lr_search.db"
    storage = f"sqlite:///{db_path}"
    print(f"Optuna DB: {db_path}")

    seed = cfg["seed"]
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading models for {args.model}...")
    lm, tokenizer, lm_dim = load_model(args.model, cfg, device)

    clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(device).eval()
    for p in clap_model.parameters():
        p.requires_grad = False
    audio_dim = clap_model.config.projection_dim

    ds = load_from_disk(cfg["data_dir"])
    split = ds.train_test_split(test_size=cfg["test_size"], seed=seed)
    train_ds = MusicCapsDataset(split["train"], tokenizer, clap_processor, clap_model, device)
    val_ds = MusicCapsDataset(split["test"], tokenizer, clap_processor, clap_model, device)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"])

    prefix_len = cfg["prefix_len"]
    dropout = cfg["dropout"]
    lm_init_state = deepcopy(lm.state_dict())

    train_fn = TRAIN_FNS[args.model]
    eval_fn = EVAL_FNS[args.model]

    # ---- Stage 1 search: find best S1 LR ----
    print(f"\n{'='*50}")
    print("Stage 1 LR search")
    print(f"{'='*50}")

    s1_proj_path = db_dir / f"{args.model}_s1_best_proj.pt"

    def s1_objective(trial):
        s1_lr = trial.suggest_float("stage1_lr", 1e-5, 1e-2, log=True)

        print(f"\n--- S1 Trial {trial.number} ---")
        print(f"  stage1_lr={s1_lr:.6g}")

        set_seed(seed)
        projection = Projection(audio_dim, lm_dim, prefix_len, dropout=dropout).to(device)

        lm.load_state_dict(lm_init_state)
        lm.eval()
        for p in lm.parameters():
            p.requires_grad = False

        optimizer = torch.optim.AdamW(
            projection.parameters(), lr=s1_lr, weight_decay=cfg["weight_decay"]
        )
        s1_val = run_stage(
            train_fn, eval_fn, projection, lm, train_loader, val_loader,
            optimizer, prefix_len, device, args.epochs_per_stage, args.patience,
            trial, epoch_offset=0, t_max=30,
        )

        if s1_val <= trial.study.best_value if len(trial.study.trials) > 1 else True:
            torch.save(projection.state_dict(), s1_proj_path)
            print(f"  New best S1 -- projection saved to {s1_proj_path}")

        return s1_val

    s1_study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5),
        study_name=f"lr_search_{args.model}_s1",
        storage=storage,
        load_if_exists=True,
    )
    completed_s1 = len([t for t in s1_study.trials if t.state == optuna.trial.TrialState.COMPLETE])
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
        projection = Projection(audio_dim, lm_dim, prefix_len, dropout=dropout).to(device)
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
                  dummy_trial, epoch_offset=0, t_max=30)
        torch.save(projection.state_dict(), s1_proj_path)

    # ---- Stage 2 search: find best S2 LRs using best S1 projection ----
    print(f"\n{'='*50}")
    print("Stage 2 LR search (using best S1 projection)")
    print(f"{'='*50}")

    best_s1_projection_state = torch.load(s1_proj_path, weights_only=True, map_location=device)

    def s2_objective(trial):
        s2_proj_lr = trial.suggest_float("stage2_proj_lr", 1e-5, 1e-3, log=True)
        s2_lm_lr = trial.suggest_float("stage2_lm_lr", 1e-7, 1e-4, log=True)

        print(f"\n--- S2 Trial {trial.number} ---")
        print(f"  stage2_proj_lr={s2_proj_lr:.6g}  stage2_lm_lr={s2_lm_lr:.6g}")

        set_seed(seed)
        projection = Projection(audio_dim, lm_dim, prefix_len, dropout=dropout).to(device)
        projection.load_state_dict(best_s1_projection_state)

        lm.load_state_dict(lm_init_state)
        for p in lm.parameters():
            p.requires_grad = True
        lm.train()

        optimizer = torch.optim.AdamW([
            {"params": projection.parameters(), "lr": s2_proj_lr},
            {"params": lm.parameters(), "lr": s2_lm_lr},
        ], weight_decay=cfg["weight_decay"])

        s2_val = run_stage(
            train_fn, eval_fn, projection, lm, train_loader, val_loader,
            optimizer, prefix_len, device, args.epochs_per_stage, args.patience,
            trial, epoch_offset=0, t_max=20,
        )

        return s2_val

    s2_study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5),
        study_name=f"lr_search_{args.model}_s2",
        storage=storage,
        load_if_exists=True,
    )
    completed_s2 = len([t for t in s2_study.trials if t.state == optuna.trial.TrialState.COMPLETE])
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
    print(f"\n{'='*50}")
    print("Final best LRs:")
    print(f"  stage1_lr:      {best_s1_lr:.6g}")
    print(f"  stage2_proj_lr: {best_s2.params['stage2_proj_lr']:.6g}")
    print(f"  stage2_lm_lr:   {best_s2.params['stage2_lm_lr']:.6g}")

    results_dir = Path(cfg["results_dir"]) / args.model
    results_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "model": args.model,
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
    out_path = results_dir / f"{args.model}_lr_search.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")

    cfg["stage1"]["lr"] = float(best_s1_lr)
    cfg["stage2"]["projection_lr"] = float(best_s2.params["stage2_proj_lr"])
    cfg["stage2"]["lm_lr"] = float(best_s2.params["stage2_lm_lr"])
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Updated {config_path} with best LRs")


if __name__ == "__main__":
    main()
