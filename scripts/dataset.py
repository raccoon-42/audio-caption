from pathlib import Path

import torch
from datasets import load_from_disk
from torch.utils.data import Dataset, DataLoader


class MusicCapsDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, clap_embeddings, max_len=64):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.clap_embeddings = clap_embeddings
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_emb = self.clap_embeddings[idx]

        tokens = self.tokenizer(
            self.data[idx]["caption"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.squeeze(0)
        attention_mask = tokens.attention_mask.squeeze(0)

        return audio_emb, input_ids, attention_mask


def load_dataloaders(cfg, tokenizer, batch_size=None, seed=None):
    data_dir = Path(cfg["data_dir"])
    all_clap = torch.load(data_dir.parent / "clap_embeddings.pt", weights_only=True)
    audio_dim = all_clap.shape[1]

    ds = load_from_disk(str(data_dir))
    seed = seed or cfg["seed"]
    split = ds.train_test_split(test_size=cfg["test_size"], seed=seed)

    train_indices = split["train"]._indices.column("indices").to_pylist()
    val_indices = split["test"]._indices.column("indices").to_pylist()

    bs = batch_size or cfg["batch_size"]
    train_loader = DataLoader(
        MusicCapsDataset(split["train"], tokenizer, all_clap[train_indices]),
        batch_size=bs, shuffle=True,
    )
    val_loader = DataLoader(
        MusicCapsDataset(split["test"], tokenizer, all_clap[val_indices]),
        batch_size=bs,
    )

    return train_loader, val_loader, audio_dim
