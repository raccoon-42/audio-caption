#!/bin/bash
# Usage: ./scripts/run_1_lr_search.sh gpt2 t5 opt llama

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <model1> [model2] ..."
    echo "Models: gpt2, t5, opt, llama"
    exit 1
fi

for model in "$@"; do
    echo "===== LR search: $model ====="
    if ! uv run python scripts/lr_search.py --model "$model"; then
        echo "LR SEARCH FAILED: $model"
        exit 1
    fi
    echo ""
done

echo "Done. Check results/<model>/<model>_lr_search.db for ceiling/flooring before proceeding."
