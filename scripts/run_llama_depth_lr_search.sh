#!/bin/bash
# Re-run LLaMA LR search for proj_depth 1 and 3 with the widened ranges.
# Deletes the stale (boundary-capped) depth studies first so Optuna starts fresh
# instead of resuming the old DB.
set -e

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODEL=llama
RUN="uv run python"
LR_DIR=results/${MODEL}/lr_search

for DEPTH in 1 3; do
    echo "===== Cleaning stale depth-${DEPTH} LR search artifacts ====="
    rm -f ${LR_DIR}/${MODEL}_depth${DEPTH}_lr_search.db
    rm -f ${LR_DIR}/${MODEL}_depth${DEPTH}_lr_search.json
    rm -f ${LR_DIR}/${MODEL}_depth${DEPTH}_s1_best_proj.pt

    echo "===== LR search: ${MODEL} proj_depth ${DEPTH} ====="
    $RUN scripts/lr_search.py --model $MODEL --proj-depth $DEPTH
done

echo "===== LLaMA depth-1/3 LR re-search complete ====="
