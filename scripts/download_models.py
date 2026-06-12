#!/usr/bin/env python3
"""Fetch audio-encoder and reference-captioner checkpoints into the local Hugging Face cache.

Encoders output frame- or clip-level embeddings for the projection pipeline.
Captioners generate text directly and are used as zero-shot references.
This script only downloads checkpoint files; loading some entries also needs the
external package or source repo listed in REQUIRES.

Examples:
    uv run python scripts/download_models.py                       # all encoders
    uv run python scripts/download_models.py --models clap-music muq
    uv run python scripts/download_models.py --group captioners    # reference models
    uv run python scripts/download_models.py --group all
"""

import argparse
from fnmatch import fnmatch

from huggingface_hub import HfApi, snapshot_download

# name -> HF repo id
ENCODERS = {
    "clap-music": "laion/larger_clap_music",                 # joint audio-text, pooled clip embedding
    "clap-music-speech": "laion/larger_clap_music_and_speech",
    "mert-95m": "m-a-p/MERT-v1-95M",                          # SSL, frame-level hidden states (768)
    "mert-330m": "m-a-p/MERT-v1-330M",                        # SSL, frame-level hidden states (1024)
    "musicfm": "minzwon/MusicFM",                             # BEST-RQ conformer, frame-level (1024)
    "muq": "OpenMuQ/MuQ-large-msd-iter",                     # SSL, frame-level hidden states
    "muq-mulan": "OpenMuQ/MuQ-MuLan-large",                  # joint audio-text, pooled clip embedding
}

CAPTIONERS = {
    "audio-flamingo-captioner": "nvidia/audio-flamingo-next-captioner-hf",  # 8B audio-LM, 16 kHz mono
}

# Restrict which files are pulled for repos that ship more than one variant.
ALLOW_PATTERNS = {
    # MusicFM ships MSD- and FMA-pretrained weights; MSD scores higher on MIR tasks.
    "musicfm": ["pretrained_msd.pt", "msd_stats.json"],
}

# External dependency required to load (not download) the model.
REQUIRES = {
    "musicfm": "model code: git clone https://github.com/minzwon/musicfm",
    "muq": "package: uv add muq (loads via MuQ.from_pretrained)",
    "muq-mulan": "package: uv add muq (loads via MuQMuLan.from_pretrained)",
    "mert-95m": "transformers AutoModel(trust_remote_code=True)",
    "mert-330m": "transformers AutoModel(trust_remote_code=True)",
    "audio-flamingo-captioner": "transformers AutoModelForMultimodalLM",
}


def repo_size(name, repo_id):
    """Return (total_bytes, file_count) of the files that will be fetched, or (None, None)."""
    patterns = ALLOW_PATTERNS.get(name)
    info = HfApi().repo_info(repo_id, files_metadata=True)
    total, count = 0, 0
    for s in info.siblings:
        if patterns and not any(fnmatch(s.rfilename, pat) for pat in patterns):
            continue
        if s.size is None:
            return None, None  # incomplete metadata; don't report a misleading total
        total += s.size
        count += 1
    return total, count


def gb(num_bytes):
    return "?" if num_bytes is None else f"{num_bytes / 1e9:.2f} GB"


def download(name, repo_id):
    path = snapshot_download(repo_id=repo_id, allow_patterns=ALLOW_PATTERNS.get(name))
    print(f"[ok] {name:26s} {repo_id}")
    print(f"     -> {path}")
    if name in REQUIRES:
        print(f"     load requires: {REQUIRES[name]}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--group", choices=["encoders", "captioners", "all"], default="encoders")
    p.add_argument("--models", nargs="*", help="subset of names to fetch (default: whole group)")
    args = p.parse_args()

    if args.group == "encoders":
        selected = dict(ENCODERS)
    elif args.group == "captioners":
        selected = dict(CAPTIONERS)
    else:
        selected = {**ENCODERS, **CAPTIONERS}

    if args.models:
        unknown = [m for m in args.models if m not in selected]
        if unknown:
            p.error(
                f"unknown model(s) for group '{args.group}': {unknown}. "
                f"available: {sorted(selected)}"
            )
        selected = {k: selected[k] for k in args.models}

    # Preview pass: report per-model download size and the batch total before fetching.
    print(f"Resolving sizes for {len(selected)} model(s):\n")
    sizes, grand_total = {}, 0
    for name, repo_id in selected.items():
        try:
            total, count = repo_size(name, repo_id)
        except Exception as e:
            total, count = None, None
            print(f"  {name:26s} size query failed: {e}")
        sizes[name] = total
        files = "?" if count is None else f"{count} files"
        print(f"  {name:26s} {gb(total):>10s}  ({files})  {repo_id}")
        if total:
            grand_total += total
    print(f"  {'TOTAL':26s} {gb(grand_total):>10s}\n")

    print("Downloading into the Hugging Face cache (per-file progress/ETA shown by hf_hub):\n")
    failed = {}
    for name, repo_id in selected.items():
        try:
            download(name, repo_id)
        except Exception as e:  # keep going so one gated/missing repo doesn't abort the rest
            failed[name] = repr(e)
            print(f"[FAIL] {name:26s} {repo_id}\n     {e}")

    print(f"\nDone: {len(selected) - len(failed)} ok, {len(failed)} failed.")
    if failed:
        for name, err in failed.items():
            print(f"  - {name}: {err}")


if __name__ == "__main__":
    main()
