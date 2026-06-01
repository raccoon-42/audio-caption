#!/bin/bash
# Usage: ./scripts/run_1_lr_search.sh gpt2 t5 opt llama
#        ./scripts/run_1_lr_search.sh gpt2 --proj-depth 3

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
set -e

models=()
extra_args=()

while [ $# -gt 0 ]; do
    case "$1" in
        --*) extra_args+=("$1" "$2"); shift 2 ;;
        *)   models+=("$1"); shift ;;
    esac
done

if [ ${#models[@]} -eq 0 ]; then
    echo "Usage: $0 <model1> [model2] ... [--proj-depth N]"
    echo "Models: gpt2, t5, opt, llama"
    exit 1
fi

for model in "${models[@]}"; do
    echo "===== LR search: $model ${extra_args[*]} ====="
    if ! uv run python scripts/lr_search.py --model "$model" "${extra_args[@]}"; then
        echo "LR SEARCH FAILED: $model"
        exit 1
    fi
    echo ""
done

echo "Done. Check results/<model>/<model>_lr_search.db for ceiling/flooring before proceeding."
