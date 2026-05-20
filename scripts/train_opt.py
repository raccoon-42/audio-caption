"""Self-contained OPT audio captioning training script (stage 1 + stage 2)."""

import argparse
from pathlib import Path

import torch
import yaml
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ClapModel, ClapProcessor, OPTForCausalLM, AutoTokenizer

from dataset import MusicCapsDataset
from projection import Projection
from trainer import train_loop, save_results
from utils import set_seed


def train_one_epoch(projection, model, dataloader, optimizer, prefix_len, device):
    projection.train()
    total_loss = 0.0
    for audio_emb, input_ids, attention_mask in tqdm(dataloader, leave=False):
        audio_emb = audio_emb.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        prefix = projection(audio_emb)
        text_emb = model.model.decoder.embed_tokens(input_ids)
        inputs_embeds = torch.cat([prefix, text_emb], dim=1)

        prefix_labels = torch.full(
            (input_ids.size(0), prefix_len), -100, device=device
        )
        labels = torch.cat([prefix_labels, input_ids], dim=1)

        pad_mask = torch.cat(
            [torch.ones(input_ids.size(0), prefix_len, device=device), attention_mask],
            dim=1,
        )
        labels[pad_mask == 0] = -100

        outputs = model(inputs_embeds=inputs_embeds, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(projection, model, dataloader, prefix_len, device):
    projection.eval()
    total_loss = 0.0
    for audio_emb, input_ids, attention_mask in dataloader:
        audio_emb = audio_emb.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        prefix = projection(audio_emb)
        text_emb = model.model.decoder.embed_tokens(input_ids)
        inputs_embeds = torch.cat([prefix, text_emb], dim=1)

        prefix_labels = torch.full(
            (input_ids.size(0), prefix_len), -100, device=device
        )
        labels = torch.cat([prefix_labels, input_ids], dim=1)

        pad_mask = torch.cat(
            [torch.ones(input_ids.size(0), prefix_len, device=device), attention_mask],
            dim=1,
        )
        labels[pad_mask == 0] = -100

        outputs = model(inputs_embeds=inputs_embeds, labels=labels)
        total_loss += outputs.loss.item()

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/opt.yaml")
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
    model_name = cfg["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(device).eval()
    for p in clap_model.parameters():
        p.requires_grad = False

    audio_dim = clap_model.config.projection_dim
    lm_dim = model.config.word_embed_proj_dim

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

    tag = args.ablation_tag or "opt"
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

    best_val, history_s1 = train_loop(
        train_fn=lambda: train_one_epoch(projection, model, train_loader, optimizer, prefix_len, device),
        eval_fn=lambda: evaluate(projection, model, val_loader, prefix_len, device),
        optimizer=optimizer,
        max_epochs=s1["epochs"],
        patience=args.patience,
        stage_label="S1",
        ckpt_paths={"proj": (projection.state_dict, ckpt_dir / "stage1_best.pt")},
        device=device,
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
            "proj_depth": args.proj_depth,
        },
    )

    # ========== STAGE 2: projection + OPT fine-tune ==========
    print(f"=== Stage 2: fine-tuning projection + {model_name} ===")
    projection.load_state_dict(torch.load(ckpt_dir / "stage1_best.pt", weights_only=True))

    for p in model.parameters():
        p.requires_grad = True
    model.train()

    s2 = cfg["stage2"]
    optimizer = torch.optim.AdamW([
        {"params": projection.parameters(), "lr": s2["projection_lr"]},
        {"params": model.parameters(), "lr": s2["lm_lr"]},
    ], weight_decay=cfg["weight_decay"])

    best_val, history_s2 = train_loop(
        train_fn=lambda: train_one_epoch(projection, model, train_loader, optimizer, prefix_len, device),
        eval_fn=lambda: evaluate(projection, model, val_loader, prefix_len, device),
        optimizer=optimizer,
        max_epochs=s2["epochs"],
        patience=args.patience,
        stage_label="S2",
        ckpt_paths={
            "proj": (projection.state_dict, ckpt_dir / "stage2_proj_best.pt"),
            "lm": (model.state_dict, ckpt_dir / "stage2_opt_best.pt"),
        },
        device=device,
    )
    torch.save(projection.state_dict(), ckpt_dir / "stage2_proj_last.pt")
    torch.save(model.state_dict(), ckpt_dir / "stage2_opt_last.pt")

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
            "proj_depth": args.proj_depth,
        },
    )

    print("Done. Results saved to", results_dir)


if __name__ == "__main__":
    main()
