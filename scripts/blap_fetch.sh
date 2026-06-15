#!/usr/bin/env bash
# Fetch and stage BLAP for offline transfer to the Ubuntu GPU box.
#
# Run on the Mac (the machine with internet). Produces ./blap_bundle/ which is
# copied to the Ubuntu box via USB; nothing is downloaded at run time there.
#
# snapshot_download is avoided (it fails on this connection): the 12.9 GB
# checkpoint is pulled with resumable curl, and the small tokenizer/config/BERT
# assets go through per-file hf_hub_download into a portable HF cache. The
# Flan-T5-XL weights are NOT downloaded -- config.json has atRandom=true, so the
# T5 weights are restored from the checkpoint, and only its tokenizer/config are
# needed.
set -euo pipefail

BUNDLE="${1:-blap_bundle}"
mkdir -p "$BUNDLE/model" "$BUNDLE/hf_cache"

BASE="https://huggingface.co/Tino3141/blap/resolve/main"
echo "[1/2] BLAP checkpoint + config + wheel -> $BUNDLE"
curl -fL -C - -o "$BUNDLE/model/checkpoint.ckpt"            "$BASE/checkpoint.ckpt"
curl -fL -C - -o "$BUNDLE/model/config.json"               "$BASE/config.json"
curl -fL -C - -o "$BUNDLE/blap-0.1.10-py3-none-any.whl"     "$BASE/blap-0.1.10-py3-none-any.whl"

echo "[2/2] BERT (Q-Former, hardcoded id) + Flan-T5-XL tokenizer/config -> HF cache"
HF_HOME="$PWD/$BUNDLE/hf_cache" uv run --with huggingface_hub python - <<'PY'
from huggingface_hub import hf_hub_download
jobs = {
    "bert-base-uncased": [
        "config.json", "tokenizer_config.json", "vocab.txt",
        "tokenizer.json", "model.safetensors",
    ],
    "google/flan-t5-xl": [  # tokenizer/config only; weights ride in the ckpt
        "config.json", "generation_config.json", "special_tokens_map.json",
        "spiece.model", "tokenizer.json", "tokenizer_config.json",
    ],
}
for repo, files in jobs.items():
    for f in files:
        hf_hub_download(repo_id=repo, filename=f)
        print("  ok", repo, f)
PY

echo
echo "Bundle ready:"
du -sh "$BUNDLE"
cat <<EOF

Copy '$BUNDLE/' to the Ubuntu box, then (in the repo, BLAP venv):

  uv venv --python 3.12 .venv-blap
  uv pip install --python .venv-blap torch torchaudio \\
      --index-url https://download.pytorch.org/whl/cu128
  uv pip install --python .venv-blap $BUNDLE/blap-0.1.10-py3-none-any.whl --no-deps
  uv pip install --python .venv-blap transformers pytorch-lightning \\
      torchlibrosa soundfile pandas "numpy>=2"
  export HF_HOME=\$PWD/$BUNDLE/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  .venv-blap/bin/python scripts/blap_generate.py \\
      --ckpt $BUNDLE/model/checkpoint.ckpt \\
      --model-config $BUNDLE/model/config.json
EOF
