#!/bin/bash
# Full T5 pipeline: clean slate -> LR search -> train -> eval -> ablations
set -e

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODEL=t5
RUN="uv run python"

echo "===== Cleaning old T5 results and checkpoints ====="
rm -rf results/${MODEL}/
rm -rf checkpoints/t5*/

echo "===== Phase 1: LR search (depth 1, 2, 3) ====="
$RUN scripts/lr_search.py --model $MODEL
$RUN scripts/lr_search.py --model $MODEL --proj-depth 1
$RUN scripts/lr_search.py --model $MODEL --proj-depth 3

echo "===== Phase 2: Baseline training (seed 42) ====="
$RUN scripts/train_t5.py

echo "===== Phase 3: Noise floor training (seed 43) ====="
$RUN scripts/train_t5.py --seed 43

echo "===== Phase 4: Evaluate baseline + noise floor ====="
$RUN scripts/evaluate.py --stage 2 --model $MODEL --ckpt-tag t5
$RUN scripts/evaluate.py --stage 2 --model $MODEL --ckpt-tag t5_seed43

echo "===== Phase 5: Noise floor stats ====="
$RUN scripts/noise_floor.py --model $MODEL

echo "===== Phase 6: Arch ablations (6 configs) ====="
./scripts/run_arch_ablations.sh $MODEL

echo "===== Phase 7: Decoding ablations (~23 configs) ====="
./scripts/run_decoding_ablations.sh $MODEL

echo "===== T5 pipeline complete ====="
