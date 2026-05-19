"""Self-contained GPT-2 audio captioning training script (stage 1 + stage 2)."""

import argparse
import json
import random
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
import yaml
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import ClapModel, ClapProcessor, GPT2LMHeadModel, GPT2Tokenizer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MusicCapsDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, clap_processor, clap_model, device,
                 max_len=64):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.clap_processor = clap_processor
        self.clap_model = clap_model
        self.device = device
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        audio_array = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]

        if audio_array.ndim == 2:
            audio_array = audio_array.mean(axis=1)
        if sr != 48000:
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=48000)

        inputs = self.clap_processor(
            audios=audio_array, sampling_rate=48000, return_tensors="pt"
        )
        with torch.no_grad():
            audio_emb = self.clap_model.get_audio_features(
                **{k: v.to(self.device) for k, v in inputs.items()}
            )
        audio_emb = audio_emb.squeeze(0).cpu()

        tokens = self.tokenizer(
            sample["caption"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.squeeze(0)
        attention_mask = tokens.attention_mask.squeeze(0)

        return audio_emb, input_ids, attention_mask


class Projection(nn.Module):
    def __init__(self, audio_dim, lm_dim, prefix_len, dropout=0.3):
        super().__init__()
        self.prefix_len = prefix_len
        self.lm_dim = lm_dim
        self.net = nn.Sequential(
            nn.Linear(audio_dim, lm_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lm_dim * 2, prefix_len * lm_dim),
        )

    def forward(self, x):
        return self.net(x).view(-1, self.prefix_len, self.lm_dim)


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
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

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

    clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(device).eval()
    for p in clap_model.parameters():
        p.requires_grad = False

    audio_dim = clap_model.config.projection_dim
    lm_dim = gpt2.config.n_embd
    prefix_len = cfg["prefix_len"]

    projection = Projection(
        audio_dim, lm_dim, prefix_len, dropout=cfg["dropout"]
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

    ckpt_dir = Path(cfg["checkpoint_dir"]) / "gpt2"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # ========== STAGE 1: projection only ==========
    print("=== Stage 1: training projection (GPT-2 frozen) ===")
    s1 = cfg["stage1"]
    optimizer = torch.optim.AdamW(
        projection.parameters(), lr=s1["lr"], weight_decay=cfg["weight_decay"]
    )

    best_val = float("inf")
    for epoch in range(1, s1["epochs"] + 1):
        train_loss = train_one_epoch(
            projection, gpt2, train_loader, optimizer, prefix_len, device
        )
        val_loss = evaluate(projection, gpt2, val_loader, prefix_len, device)
        print(f"[S1] Epoch {epoch}/{s1['epochs']}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(projection.state_dict(), ckpt_dir / "stage1_best.pt")

    torch.save(projection.state_dict(), ckpt_dir / "stage1_last.pt")

    stage1_results = {
        "model": "gpt2", "stage": 1, "seed": seed,
        "best_val_loss": best_val,
        "hyperparams": {
            "prefix_len": prefix_len, "lr": s1["lr"],
            "epochs": s1["epochs"], "batch_size": cfg["batch_size"],
            "weight_decay": cfg["weight_decay"], "dropout": cfg["dropout"],
        },
    }
    with open(results_dir / "gpt2_stage1.json", "w") as f:
        json.dump(stage1_results, f, indent=2)

    # ========== STAGE 2: projection + GPT-2 fine-tune ==========
    print("=== Stage 2: fine-tuning projection + GPT-2 ===")
    projection.load_state_dict(torch.load(ckpt_dir / "stage1_best.pt", weights_only=True))

    for p in gpt2.parameters():
        p.requires_grad = True
    gpt2.train()

    s2 = cfg["stage2"]
    optimizer = torch.optim.AdamW([
        {"params": projection.parameters(), "lr": s2["projection_lr"]},
        {"params": gpt2.parameters(), "lr": s2["lm_lr"]},
    ], weight_decay=cfg["weight_decay"])

    best_val = float("inf")
    for epoch in range(1, s2["epochs"] + 1):
        train_loss = train_one_epoch(
            projection, gpt2, train_loader, optimizer, prefix_len, device
        )
        val_loss = evaluate(projection, gpt2, val_loader, prefix_len, device)
        print(f"[S2] Epoch {epoch}/{s2['epochs']}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(projection.state_dict(), ckpt_dir / "stage2_proj_best.pt")
            torch.save(gpt2.state_dict(), ckpt_dir / "stage2_gpt2_best.pt")

    torch.save(projection.state_dict(), ckpt_dir / "stage2_proj_last.pt")
    torch.save(gpt2.state_dict(), ckpt_dir / "stage2_gpt2_last.pt")

    stage2_results = {
        "model": "gpt2", "stage": 2, "seed": seed,
        "best_val_loss": best_val,
        "hyperparams": {
            "prefix_len": prefix_len,
            "projection_lr": s2["projection_lr"], "lm_lr": s2["lm_lr"],
            "epochs": s2["epochs"], "batch_size": cfg["batch_size"],
            "weight_decay": cfg["weight_decay"], "dropout": cfg["dropout"],
        },
    }
    with open(results_dir / "gpt2_stage2.json", "w") as f:
        json.dump(stage2_results, f, indent=2)

    print("Done. Results saved to", results_dir)


if __name__ == "__main__":
    main()
