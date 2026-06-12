"""Precompute clip-level audio embeddings for the dataset with a chosen encoder.

Saves one (N, dim) tensor to <out-dir>/<encoder>_embeddings.pt, row-aligned with
the dataset on disk so the seeded train/test split stays consistent across encoders.

Pooled encoders (clap, muq-mulan) return a clip vector directly. Frame-level
encoders (mert, musicfm, muq) return per-frame features that are mean-pooled over
time; --layer selects which transformer layer to read.

Examples:
    uv run python scripts/precompute_embeddings.py --encoder clap-music
    uv run python scripts/precompute_embeddings.py --encoder mert-330m
    uv run python scripts/precompute_embeddings.py --encoder musicfm --musicfm-repo ~/musicfm
"""

import argparse
import sys
from pathlib import Path

import librosa
import torch
from datasets import load_from_disk
from tqdm import tqdm

ENCODER_REPOS = {
    "clap": "laion/clap-htsat-unfused",  # general-audio baseline (current frontend)
    "clap-music": "laion/larger_clap_music",
    "clap-music-speech": "laion/larger_clap_music_and_speech",
    "mert-95m": "m-a-p/MERT-v1-95M",
    "mert-330m": "m-a-p/MERT-v1-330M",
    "musicfm": "minzwon/MusicFM",
    "muq": "OpenMuQ/MuQ-large-msd-iter",
    "muq-mulan": "OpenMuQ/MuQ-MuLan-large",
}

# Sample rate each encoder expects.
TARGET_SR = {
    "clap": 48000,
    "clap-music": 48000,
    "clap-music-speech": 48000,
    "mert-95m": 24000,
    "mert-330m": 24000,
    "musicfm": 24000,
    "muq": 24000,
    "muq-mulan": 24000,
}

# Default transformer layer for frame-level encoders (None = last hidden state).
DEFAULT_LAYER = {"musicfm": 7}


def to_mono(audio):
    return audio.mean(axis=1) if audio.ndim == 2 else audio


def build_clap(repo_id, device):
    from transformers import ClapModel, ClapProcessor

    processor = ClapProcessor.from_pretrained(repo_id)
    model = ClapModel.from_pretrained(repo_id).to(device).eval()

    @torch.no_grad()
    def extract(audio, target_sr):
        inputs = processor(audio=audio, sampling_rate=target_sr, return_tensors="pt")
        feats = model.get_audio_features(**{k: v.to(device) for k, v in inputs.items()})
        return feats.squeeze(0).cpu()

    return extract


def build_mert(repo_id, device, layer):
    from transformers import AutoModel, Wav2Vec2FeatureExtractor

    processor = Wav2Vec2FeatureExtractor.from_pretrained(repo_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(repo_id, trust_remote_code=True).to(device).eval()

    @torch.no_grad()
    def extract(audio, target_sr):
        inputs = processor(audio, sampling_rate=target_sr, return_tensors="pt")
        out = model(**{k: v.to(device) for k, v in inputs.items()}, output_hidden_states=True)
        hidden = out.hidden_states[layer] if layer is not None else out.last_hidden_state
        return hidden.mean(dim=1).squeeze(0).cpu()  # mean over time

    return extract


def build_muq(repo_id, device, layer):
    from muq import MuQ

    model = MuQ.from_pretrained(repo_id).to(device).eval()

    @torch.no_grad()
    def extract(audio, target_sr):
        wavs = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
        out = model(wavs, output_hidden_states=True)
        hidden = out.hidden_states[layer] if layer is not None else out.last_hidden_state
        return hidden.mean(dim=1).squeeze(0).cpu()  # mean over time

    return extract


def build_mulan(repo_id, device):
    from muq import MuQMuLan

    model = MuQMuLan.from_pretrained(repo_id).to(device).eval()

    @torch.no_grad()
    def extract(audio, target_sr):
        wavs = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
        return model(wavs=wavs).squeeze(0).cpu()

    return extract


def build_musicfm(repo_id, device, layer, repo_dir):
    from huggingface_hub import hf_hub_download

    repo = Path(repo_dir).expanduser().resolve()
    if repo.name != "musicfm":
        raise SystemExit(
            f"--musicfm-repo must point to the cloned dir named 'musicfm' (got '{repo.name}'); "
            "run: git clone https://github.com/minzwon/musicfm"
        )
    sys.path.append(str(repo.parent))  # import is `musicfm.model...`, so the package's parent goes on the path
    from musicfm.model.musicfm_25hz import MusicFM25Hz

    stat_path = hf_hub_download(repo_id, "msd_stats.json")
    model_path = hf_hub_download(repo_id, "pretrained_msd.pt")
    model = MusicFM25Hz(is_flash=False, stat_path=stat_path, model_path=model_path)
    model = model.to(device).eval()

    @torch.no_grad()
    def extract(audio, target_sr):
        wav = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
        frames = model.get_latent(wav, layer_ix=layer)
        return frames.mean(dim=1).squeeze(0).cpu()  # mean over time

    return extract


def build_extractor(encoder, device, layer, musicfm_repo):
    repo_id = ENCODER_REPOS[encoder]
    if encoder.startswith("clap"):
        return build_clap(repo_id, device)
    if encoder.startswith("mert"):
        return build_mert(repo_id, device, layer)
    if encoder == "muq":
        return build_muq(repo_id, device, layer)
    if encoder == "muq-mulan":
        return build_mulan(repo_id, device)
    if encoder == "musicfm":
        if not musicfm_repo:
            raise SystemExit("musicfm needs --musicfm-repo <path to cloned minzwon/musicfm>")
        return build_musicfm(repo_id, device, layer, musicfm_repo)
    raise ValueError(encoder)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--encoder", required=True, choices=sorted(ENCODER_REPOS))
    p.add_argument("--data-dir", default="data/dataset", help="HF dataset on disk")
    p.add_argument("--out-dir", default="data/embeddings")
    p.add_argument("--layer", type=int, default=None,
                   help="frame-level encoders: transformer layer to read (default per-encoder)")
    p.add_argument("--musicfm-repo", default=None,
                   help="path to the cloned 'musicfm' dir (git clone minzwon/musicfm)")
    args = p.parse_args()

    layer = args.layer if args.layer is not None else DEFAULT_LAYER.get(args.encoder)
    target_sr = TARGET_SR[args.encoder]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = load_from_disk(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.encoder.replace('-', '_')}_embeddings.pt"

    print(f"Encoder {args.encoder} ({ENCODER_REPOS[args.encoder]}), "
          f"target_sr={target_sr}, layer={layer}, device={device}")
    extract = build_extractor(args.encoder, device, layer, args.musicfm_repo)

    print(f"Precomputing embeddings for {len(ds)} samples...")
    embeddings = []
    failed = 0
    for i in tqdm(range(len(ds)), desc=args.encoder):
        try:
            sample = ds[i]
            audio = to_mono(sample["audio"]["array"])
            sr = sample["audio"]["sampling_rate"]
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            embeddings.append(extract(audio, target_sr))
        except Exception as e:
            failed += 1
            embeddings.append(None)
            if failed <= 5:
                print(f"  Sample {i} failed: {e}")

    dim = next(e.shape[0] for e in embeddings if e is not None)
    embeddings = [e if e is not None else torch.zeros(dim) for e in embeddings]
    embeddings = torch.stack(embeddings)
    torch.save(embeddings, out_path)
    print(f"Saved {tuple(embeddings.shape)} to {out_path} ({failed} failed)")


if __name__ == "__main__":
    main()
