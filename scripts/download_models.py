"""Download Phi-2 and LLaMA-3.2-1B for offline transfer."""

from huggingface_hub import snapshot_download
from pathlib import Path

TARGET_DIR = Path.home() / "hf_models"

MODELS = {
    "microsoft/phi-2": "phi-2",
    "meta-llama/Llama-3.2-1B": "llama-3.2-1b",
}

for repo_id, folder_name in MODELS.items():
    local_dir = TARGET_DIR / folder_name
    print(f"Downloading {repo_id} -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        ignore_patterns=["*.gguf", "*.bin.index.json"],
    )
    print(f"Done: {repo_id}\n")

print(f"All models saved to {TARGET_DIR}")
