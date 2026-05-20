"""Shared training loop with early stopping for all models."""

import json
from pathlib import Path

import torch


def train_loop(train_fn, eval_fn, optimizer, max_epochs, patience, stage_label,
               ckpt_paths, device):
    """Run a training loop with early stopping.

    Args:
        train_fn: callable() -> train_loss
        eval_fn: callable() -> val_loss
        optimizer: torch optimizer
        max_epochs: maximum number of epochs
        patience: early stopping patience
        stage_label: e.g. "S1" or "S2" for logging
        ckpt_paths: dict of name -> path for best checkpoints,
                    e.g. {"proj": path} or {"proj": path, "lm": path}
        device: torch device

    Returns:
        (best_val_loss, history, stopped_epoch)
    """
    best_val = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        train_loss = train_fn()
        val_loss = eval_fn()
        print(f"[{stage_label}] Epoch {epoch}/{max_epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
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
