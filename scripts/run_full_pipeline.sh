#!/bin/bash

echo "===== LEARNING RATE SEARCH ====="
for model in gpt2 t5 opt; do
    echo "--- LR search: $model ---"
    if ! uv run python scripts/lr_search.py --model "$model"; then
        echo "LR SEARCH FAILED: $model"
        exit 1
    fi
    echo ""
done

echo ""
echo "===== ARCHITECTURAL ABLATIONS (with baselines) ====="
./scripts/run_arch_ablations.sh
ARCH_EXIT=$?

echo ""
if [ $ARCH_EXIT -ne 0 ]; then
    echo "Some ablations failed. Check output above for details."
    exit 1
else
    echo "Full pipeline complete."
fi
