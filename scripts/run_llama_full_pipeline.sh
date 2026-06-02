#!/bin/bash
# Full LLaMA pipeline: clean slate -> LR search -> train -> eval -> ablations
# S1=40 epochs, S2=30 epochs, batch_size=4. No noise floor (not seed-replicated).
set -e

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODEL=llama
RUN="uv run python"

echo "===== Cleaning old LLaMA results and checkpoints ====="
rm -rf results/${MODEL}/
rm -rf checkpoints/llama*/

echo "===== Phase 1: LR search (depth 1, 2, 3) ====="
$RUN scripts/lr_search.py --model $MODEL
$RUN scripts/lr_search.py --model $MODEL --proj-depth 1
$RUN scripts/lr_search.py --model $MODEL --proj-depth 3

echo "===== Phase 2: Baseline training (seed 42) ====="
$RUN scripts/train_llama.py

echo "===== Phase 3: Evaluate baseline ====="
$RUN scripts/evaluate.py --stage 2 --model $MODEL --ckpt-tag llama --ablation-type arch

echo "===== Phase 4: Arch ablations (6 configs) ====="
./scripts/run_arch_ablations.sh $MODEL

echo "===== Phase 5: Decoding ablations ====="
./scripts/run_decoding_ablations.sh $MODEL

echo "===== LLaMA pipeline complete ====="
