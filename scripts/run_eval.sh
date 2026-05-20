#!/bin/bash
set -e

for model in gpt2 t5 opt; do
    echo "=== Evaluating $model stage 2 ==="
    uv run python scripts/evaluate.py --model $model --stage 2
    echo ""
done

echo "All evaluations complete."
