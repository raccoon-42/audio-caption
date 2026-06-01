"""Self-contained T5 audio captioning training script (stage 1 + stage 2)."""

import argparse
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

from dataset import load_dataloaders
from projection import Projection
from trainer import train_loop, save_results
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
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--ablation-tag", type=str, default=None)
    parser.add_argument("--use-layernorm", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    prefix_len = args.prefix_len or cfg["prefix_len"]
    dropout = args.dropout if args.dropout is not None else cfg["dropout"]
    use_layernorm = args.use_layernorm or cfg.get("use_layernorm", False)

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

    train_loader, val_loader, test_loader, audio_dim = load_dataloaders(cfg, tokenizer, seed=seed)
    lm_dim = t5.config.d_model

    projection = Projection(
        audio_dim, lm_dim, prefix_len, dropout=dropout, depth=args.proj_depth,
        use_layernorm=use_layernorm,
    ).to(device)

    tag = args.ablation_tag or "t5"
    ckpt_dir = Path(cfg["checkpoint_dir"]) / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(cfg["results_dir"]) / "t5"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ========== STAGE 1: projection only ==========
    print(f"=== Stage 1: training projection ({tag} / {model_name} frozen) ===")
    s1 = cfg["stage1"]
    optimizer = torch.optim.AdamW(
        projection.parameters(), lr=s1["lr"], weight_decay=cfg["weight_decay"]
    )

    best_val, history_s1 = train_loop(
        train_fn=lambda: train_one_epoch(projection, t5, train_loader, optimizer, prefix_len, device),
        eval_fn=lambda: evaluate(projection, t5, val_loader, prefix_len, device),
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
        model_name=model_name, stage=1, seed=seed,
        best_val=best_val, history=history_s1,
        hyperparams={
            "prefix_len": prefix_len, "lr": s1["lr"],
            "max_epochs": s1["epochs"], "patience": args.patience,
            "batch_size": cfg["batch_size"],
            "weight_decay": cfg["weight_decay"], "dropout": dropout,
            "proj_depth": args.proj_depth, "use_layernorm": use_layernorm,
        },
    )

    # ========== STAGE 2: projection + T5 fine-tune ==========
    print(f"=== Stage 2: fine-tuning projection + {tag} / {model_name} ===")
    projection.load_state_dict(torch.load(ckpt_dir / "stage1_best.pt", weights_only=True))

    for p in t5.parameters():
        p.requires_grad = True
    t5.train()

    s2 = cfg["stage2"]
    optimizer = torch.optim.AdamW([
        {"params": projection.parameters(), "lr": s2["projection_lr"]},
        {"params": t5.parameters(), "lr": s2["lm_lr"]},
    ], weight_decay=cfg["weight_decay"])

    best_val, history_s2 = train_loop(
        train_fn=lambda: train_one_epoch(projection, t5, train_loader, optimizer, prefix_len, device),
        eval_fn=lambda: evaluate(projection, t5, val_loader, prefix_len, device),
        optimizer=optimizer,
        max_epochs=s2["epochs"],
        patience=args.patience,
        stage_label="S2",
        ckpt_paths={
            "proj": (projection.state_dict, ckpt_dir / "stage2_proj_best.pt"),
            "lm": (t5.state_dict, ckpt_dir / "stage2_t5_best.pt"),
        },
        device=device,
        t_max=s2["epochs"],
    )
    torch.save(projection.state_dict(), ckpt_dir / "stage2_proj_last.pt")
    torch.save(t5.state_dict(), ckpt_dir / "stage2_t5_last.pt")

    save_results(
        results_dir / f"{tag}_stage2.json",
        model_name=model_name, stage=2, seed=seed,
        best_val=best_val, history=history_s2,
        hyperparams={
            "prefix_len": prefix_len,
            "projection_lr": s2["projection_lr"], "lm_lr": s2["lm_lr"],
            "max_epochs": s2["epochs"], "patience": args.patience,
            "batch_size": cfg["batch_size"],
            "weight_decay": cfg["weight_decay"], "dropout": dropout,
            "proj_depth": args.proj_depth, "use_layernorm": use_layernorm,
        },
    )

    print("Done. Results saved to", results_dir)


if __name__ == "__main__":
    main()
