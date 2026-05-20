"""Shared training loop with early stopping for all models."""

import json
from pathlib import Path

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR


def train_loop(train_fn, eval_fn, optimizer, max_epochs, patience, stage_label,
               ckpt_paths, device, t_max=None):
    best_val = float("inf")
    patience_counter = 0
    history = []
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max or max_epochs)

    for epoch in range(1, max_epochs + 1):
        train_loss = train_fn()
        val_loss = eval_fn()
        scheduler.step()

        lr_str = ", ".join(f"{pg['lr']:.2e}" for pg in optimizer.param_groups)
        print(f"[{stage_label}] Epoch {epoch}/{max_epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  lr=[{lr_str}]")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            for name, (state_dict, path) in ckpt_paths.items():
                torch.save(state_dict(), path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[{stage_label}] Early stopping at epoch {epoch} "
                      f"(patience={patience})")
                break

    return best_val, history


def save_results(path, model_name, stage, seed, best_val, history, hyperparams):
    results = {
        "model": model_name,
        "stage": stage,
        "seed": seed,
        "best_val_loss": best_val,
        "stopped_epoch": len(history),
        "history": history,
        "hyperparams": hyperparams,
    }
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
