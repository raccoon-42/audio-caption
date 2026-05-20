import librosa
import torch
from torch.utils.data import Dataset


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
        for i in range(idx, idx + 10):
            try:
                return self._load(i % len(self.data))
            except Exception:
                continue
        return self._load(0)

    def _load(self, idx):
        sample = self.data[idx]
        audio_array = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]

        if audio_array.ndim == 2:
            audio_array = audio_array.mean(axis=1)
        if sr != 48000:
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=48000)

        inputs = self.clap_processor(
            audio=audio_array, sampling_rate=48000, return_tensors="pt"
        )
        with torch.no_grad():
            out = self.clap_model.get_audio_features(
                **{k: v.to(self.device) for k, v in inputs.items()}
            )
            audio_emb = out if isinstance(out, torch.Tensor) else out.pooler_output
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
