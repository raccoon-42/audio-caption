from pathlib import Path

import torch
from datasets import load_from_disk
from torch.utils.data import Dataset, DataLoader


class MusicCapsDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, clap_embeddings, max_len=64):
        self.captions = hf_dataset["caption"]
        self.tokenizer = tokenizer
        self.clap_embeddings = clap_embeddings
        self.max_len = max_len

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        audio_emb = self.clap_embeddings[idx]

        tokens = self.tokenizer(
            self.captions[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.squeeze(0)
        attention_mask = tokens.attention_mask.squeeze(0)

        return audio_emb, input_ids, attention_mask


def load_dataloaders(cfg, tokenizer, batch_size=None, seed=None, do_3way=True):
    data_dir = Path(cfg["data_dir"])
    all_clap = torch.load(data_dir.parent / "clap_embeddings.pt", weights_only=True)
    audio_dim = all_clap.shape[1]

    ds = load_from_disk(str(data_dir))
    seed = seed or cfg["seed"]

    if not do_3way:
        # Original 2-way split for LR search (10% val)
        split = ds.train_test_split(test_size=0.1, seed=seed)
        train_ds = split["train"]
        val_ds = split["test"]
        
        train_indices = train_ds._indices.column("indices").to_pylist()
        val_indices = val_ds._indices.column("indices").to_pylist()
        
        bs = batch_size or cfg["batch_size"]
        train_loader = DataLoader(
            MusicCapsDataset(train_ds, tokenizer, all_clap[train_indices]),
            batch_size=bs, shuffle=True,
        )
        val_loader = DataLoader(
            MusicCapsDataset(val_ds, tokenizer, all_clap[val_indices]),
            batch_size=bs,
        )
        return train_loader, val_loader, audio_dim

    # 1. Split off test set (10%)
    split1 = ds.train_test_split(test_size=0.1, seed=seed)
    test_ds = split1["test"]
    temp_ds = split1["train"]

    # 2. Split remainder into train/val (10% of remainder for val)
    split2 = temp_ds.train_test_split(test_size=0.1, seed=seed)
    train_ds = split2["train"]
    val_ds = split2["test"]

    def get_indices(dataset):
        return dataset._indices.column("indices").to_pylist()

    train_indices = get_indices(train_ds)
    val_indices = get_indices(val_ds)
    test_indices = get_indices(test_ds)

    bs = batch_size or cfg["batch_size"]
    train_loader = DataLoader(
        MusicCapsDataset(train_ds, tokenizer, all_clap[train_indices]),
        batch_size=bs, shuffle=True,
    )
    val_loader = DataLoader(
        MusicCapsDataset(val_ds, tokenizer, all_clap[val_indices]),
        batch_size=bs,
    )
    test_loader = DataLoader(
        MusicCapsDataset(test_ds, tokenizer, all_clap[test_indices]),
        batch_size=bs,
    )

    return train_loader, val_loader, test_loader, audio_dim
