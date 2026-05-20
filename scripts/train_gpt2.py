"""Self-contained GPT-2 audio captioning training script (stage 1 + stage 2)."""

import argparse
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from dataset import load_dataloaders
from projection import Projection
from trainer import train_loop, save_results
from utils import set_seed


def train_one_epoch(projection, gpt2, dataloader, optimizer, prefix_len, device):
    projection.train()
    total_loss = 0.0
    for audio_emb, input_ids, attention_mask in tqdm(dataloader, leave=False):
        audio_emb = audio_emb.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        prefix = projection(audio_emb)
        text_emb = gpt2.transformer.wte(input_ids)
        inputs_embeds = torch.cat([prefix, text_emb], dim=1)

        prefix_labels = torch.full(
            (input_ids.size(0), prefix_len), -100, device=device
        )
        labels = torch.cat([prefix_labels, input_ids], dim=1)
        labels[labels == gpt2.config.eos_token_id] = -100

        # mask padding in labels
        pad_mask = torch.cat(
            [torch.ones(input_ids.size(0), prefix_len, device=device), attention_mask],
            dim=1,
        )
        labels[pad_mask == 0] = -100

        outputs = gpt2(inputs_embeds=inputs_embeds, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(projection, gpt2, dataloader, prefix_len, device):
    projection.eval()
    total_loss = 0.0
    for audio_emb, input_ids, attention_mask in dataloader:
        audio_emb = audio_emb.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        prefix = projection(audio_emb)
        text_emb = gpt2.transformer.wte(input_ids)
        inputs_embeds = torch.cat([prefix, text_emb], dim=1)

        prefix_labels = torch.full(
            (input_ids.size(0), prefix_len), -100, device=device
        )
        labels = torch.cat([prefix_labels, input_ids], dim=1)
        labels[labels == gpt2.config.eos_token_id] = -100

        pad_mask = torch.cat(
            [torch.ones(input_ids.size(0), prefix_len, device=device), attention_mask],
            dim=1,
        )
        labels[pad_mask == 0] = -100

        outputs = gpt2(inputs_embeds=inputs_embeds, labels=labels)
        total_loss += outputs.loss.item()

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/gpt2.yaml")
    parser.add_argument("--prefix-len", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--proj-depth", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5)
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
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    gpt2 = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    gpt2.eval()
    for p in gpt2.parameters():
        p.requires_grad = False

    train_loader, val_loader, audio_dim = load_dataloaders(cfg, tokenizer, seed=seed)
    lm_dim = gpt2.config.n_embd

    projection = Projection(
        audio_dim, lm_dim, prefix_len, dropout=dropout, depth=args.proj_depth
    ).to(device)

    tag = args.ablation_tag or "gpt2"
    ckpt_dir = Path(cfg["checkpoint_dir"]) / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(cfg["results_dir"]) / "gpt2"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ========== STAGE 1: projection only ==========
    print(f"=== Stage 1: training projection ({tag} / GPT-2 frozen) ===")
    s1 = cfg["stage1"]
    optimizer = torch.optim.AdamW(
        projection.parameters(), lr=s1["lr"], weight_decay=cfg["weight_decay"]
    )

    best_val, history_s1 = train_loop(
        train_fn=lambda: train_one_epoch(projection, gpt2, train_loader, optimizer, prefix_len, device),
        eval_fn=lambda: evaluate(projection, gpt2, val_loader, prefix_len, device),
        optimizer=optimizer,
        max_epochs=s1["epochs"],
        patience=args.patience,
        stage_label="S1",
        ckpt_paths={"proj": (projection.state_dict, ckpt_dir / "stage1_best.pt")},
        device=device,
        t_max=s1["epochs"],
    )
    torch.save(projection.state_dict(), ckpt_dir / "stage1_last.pt")

    save_results(
        results_dir / f"{tag}_stage1.json",
        model_name="gpt2", stage=1, seed=seed,
        best_val=best_val, history=history_s1,
        hyperparams={
            "prefix_len": prefix_len, "lr": s1["lr"],
            "max_epochs": s1["epochs"], "patience": args.patience,
            "batch_size": cfg["batch_size"],
            "weight_decay": cfg["weight_decay"], "dropout": dropout,
            "proj_depth": args.proj_depth,
        },
    )

    # ========== STAGE 2: projection + GPT-2 fine-tune ==========
    print(f"=== Stage 2: fine-tuning projection + {tag} / GPT-2 ===")
    projection.load_state_dict(torch.load(ckpt_dir / "stage1_best.pt", weights_only=True))

    for p in gpt2.parameters():
        p.requires_grad = True
    gpt2.train()

    s2 = cfg["stage2"]
    optimizer = torch.optim.AdamW([
        {"params": projection.parameters(), "lr": s2["projection_lr"]},
        {"params": gpt2.parameters(), "lr": s2["lm_lr"]},
    ], weight_decay=cfg["weight_decay"])

    best_val, history_s2 = train_loop(
        train_fn=lambda: train_one_epoch(projection, gpt2, train_loader, optimizer, prefix_len, device),
        eval_fn=lambda: evaluate(projection, gpt2, val_loader, prefix_len, device),
        optimizer=optimizer,
        max_epochs=s2["epochs"],
        patience=args.patience,
        stage_label="S2",
        ckpt_paths={
            "proj": (projection.state_dict, ckpt_dir / "stage2_proj_best.pt"),
            "lm": (gpt2.state_dict, ckpt_dir / "stage2_gpt2_best.pt"),
        },
        device=device,
        t_max=s2["epochs"],
    )
    torch.save(projection.state_dict(), ckpt_dir / "stage2_proj_last.pt")
    torch.save(gpt2.state_dict(), ckpt_dir / "stage2_gpt2_last.pt")

    save_results(
        results_dir / f"{tag}_stage2.json",
        model_name="gpt2", stage=2, seed=seed,
        best_val=best_val, history=history_s2,
        hyperparams={
            "prefix_len": prefix_len,
            "projection_lr": s2["projection_lr"], "lm_lr": s2["lm_lr"],
            "max_epochs": s2["epochs"], "patience": args.patience,
            "batch_size": cfg["batch_size"],
            "weight_decay": cfg["weight_decay"], "dropout": dropout,
            "proj_depth": args.proj_depth,
        },
    )

    print("Done. Results saved to", results_dir)


if __name__ == "__main__":
    main()
