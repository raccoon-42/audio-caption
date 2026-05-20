"""Self-contained T5 audio captioning training script (stage 1 + stage 2)."""

import argparse
import json
from pathlib import Path

import torch
import yaml
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ClapModel, ClapProcessor, T5ForConditionalGeneration, T5Tokenizer

from dataset import MusicCapsDataset
from projection import Projection
from utils import set_seed


def train_one_epoch(projection, t5, dataloader, optimizer, prefix_len, device):
    projection.train()
    total_loss = 0.0
    for audio_emb, input_ids, attention_mask in tqdm(dataloader, leave=False):
        audio_emb = audio_emb.to(device)
        input_ids = input_ids.to(device)

        prefix = projection(audio_emb)
        encoder_attention_mask = torch.ones(
            prefix.size(0), prefix_len, device=device
        )

        labels = input_ids.clone()
        labels[labels == t5.config.pad_token_id] = -100

        outputs = t5(
            inputs_embeds=prefix,
            attention_mask=encoder_attention_mask,
            labels=labels,
        )
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(projection, t5, dataloader, prefix_len, device):
    projection.eval()
    total_loss = 0.0
    for audio_emb, input_ids, attention_mask in dataloader:
        audio_emb = audio_emb.to(device)
        input_ids = input_ids.to(device)

        prefix = projection(audio_emb)
        encoder_attention_mask = torch.ones(
            prefix.size(0), prefix_len, device=device
        )

        labels = input_ids.clone()
        labels[labels == t5.config.pad_token_id] = -100

        outputs = t5(
            inputs_embeds=prefix,
            attention_mask=encoder_attention_mask,
            labels=labels,
        )
        total_loss += outputs.loss.item()

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/t5.yaml")
    parser.add_argument("--prefix-len", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--proj-depth", type=int, default=2)
    parser.add_argument("--ablation-tag", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    prefix_len = args.prefix_len or cfg["prefix_len"]
    dropout = args.dropout if args.dropout is not None else cfg["dropout"]

    seed = cfg["seed"]
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- models --
    model_name = cfg["model_name"]
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    t5 = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    t5.eval()
    for p in t5.parameters():
        p.requires_grad = False

    clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(device).eval()
    for p in clap_model.parameters():
        p.requires_grad = False

    audio_dim = clap_model.config.projection_dim
    lm_dim = t5.config.d_model

    projection = Projection(
        audio_dim, lm_dim, prefix_len, dropout=dropout, depth=args.proj_depth
    ).to(device)

    # -- dataset --
    ds = load_from_disk(cfg["data_dir"])
    split = ds.train_test_split(test_size=cfg["test_size"], seed=seed)
    train_ds = MusicCapsDataset(
        split["train"], tokenizer, clap_processor, clap_model, device
    )
    val_ds = MusicCapsDataset(
        split["test"], tokenizer, clap_processor, clap_model, device
    )
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"])

    tag = args.ablation_tag or "t5"
    ckpt_dir = Path(cfg["checkpoint_dir"]) / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # ========== STAGE 1: projection only ==========
    print(f"=== Stage 1: training projection ({model_name} frozen) ===")
    s1 = cfg["stage1"]
    optimizer = torch.optim.AdamW(
        projection.parameters(), lr=s1["lr"], weight_decay=cfg["weight_decay"]
    )

    best_val = float("inf")
    history_s1 = []
    for epoch in range(1, s1["epochs"] + 1):
        train_loss = train_one_epoch(
            projection, t5, train_loader, optimizer, prefix_len, device
        )
        val_loss = evaluate(projection, t5, val_loader, prefix_len, device)
        print(f"[S1] Epoch {epoch}/{s1['epochs']}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        history_s1.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            torch.save(projection.state_dict(), ckpt_dir / "stage1_best.pt")

    torch.save(projection.state_dict(), ckpt_dir / "stage1_last.pt")

    stage1_results = {
        "model": model_name, "stage": 1, "seed": seed,
        "best_val_loss": best_val,
        "history": history_s1,
        "hyperparams": {
            "prefix_len": prefix_len, "lr": s1["lr"],
            "epochs": s1["epochs"], "batch_size": cfg["batch_size"],
            "weight_decay": cfg["weight_decay"], "dropout": dropout,
            "proj_depth": args.proj_depth,
        },
    }
    with open(results_dir / f"{tag}_stage1.json", "w") as f:
        json.dump(stage1_results, f, indent=2)

    # ========== STAGE 2: projection + T5 fine-tune ==========
    print(f"=== Stage 2: fine-tuning projection + {model_name} ===")
    projection.load_state_dict(torch.load(ckpt_dir / "stage1_best.pt", weights_only=True))

    for p in t5.parameters():
        p.requires_grad = True
    t5.train()

    s2 = cfg["stage2"]
    optimizer = torch.optim.AdamW([
        {"params": projection.parameters(), "lr": s2["projection_lr"]},
        {"params": t5.parameters(), "lr": s2["lm_lr"]},
    ], weight_decay=cfg["weight_decay"])

    best_val = float("inf")
    history_s2 = []
    for epoch in range(1, s2["epochs"] + 1):
        train_loss = train_one_epoch(
            projection, t5, train_loader, optimizer, prefix_len, device
        )
        val_loss = evaluate(projection, t5, val_loader, prefix_len, device)
        print(f"[S2] Epoch {epoch}/{s2['epochs']}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        history_s2.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            torch.save(projection.state_dict(), ckpt_dir / "stage2_proj_best.pt")
            torch.save(t5.state_dict(), ckpt_dir / "stage2_t5_best.pt")

    torch.save(projection.state_dict(), ckpt_dir / "stage2_proj_last.pt")
    torch.save(t5.state_dict(), ckpt_dir / "stage2_t5_last.pt")

    stage2_results = {
        "model": model_name, "stage": 2, "seed": seed,
        "best_val_loss": best_val,
        "history": history_s2,
        "hyperparams": {
            "prefix_len": prefix_len,
            "projection_lr": s2["projection_lr"], "lm_lr": s2["lm_lr"],
            "epochs": s2["epochs"], "batch_size": cfg["batch_size"],
            "weight_decay": cfg["weight_decay"], "dropout": dropout,
            "proj_depth": args.proj_depth,
        },
    }
    with open(results_dir / f"{tag}_stage2.json", "w") as f:
        json.dump(stage2_results, f, indent=2)

    print("Done. Results saved to", results_dir)


if __name__ == "__main__":
    main()
