#!/bin/bash
# Retrain 4 T5 configs that maxed S1 at 50 epochs (now 100)
set -e

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

RUN="uv run python"

echo "===== Retrain: t5_prefix4 ====="
$RUN scripts/train_t5.py --ablation-tag t5_prefix4 --prefix-len 4

echo "===== Retrain: t5_drop0.5 ====="
$RUN scripts/train_t5.py --ablation-tag t5_drop0.5 --dropout 0.5

echo "===== Retrain: t5_depth3 ====="
$RUN scripts/train_t5.py --ablation-tag t5_depth3 --proj-depth 3

echo "===== Retrain: t5_seed43 ====="
$RUN scripts/train_t5.py --seed 43

echo "===== Re-eval all 4 ====="
$RUN scripts/evaluate.py --stage 2 --model t5 --ckpt-tag t5_prefix4 --ablation-tag t5_prefix4 --ablation-type arch
$RUN scripts/evaluate.py --stage 2 --model t5 --ckpt-tag t5_drop0.5 --ablation-tag t5_drop0.5 --ablation-type arch
$RUN scripts/evaluate.py --stage 2 --model t5 --ckpt-tag t5_depth3 --ablation-tag t5_depth3 --ablation-type arch
$RUN scripts/evaluate.py --stage 2 --model t5 --ckpt-tag t5_seed43 --ablation-tag t5_seed43 --ablation-type arch

echo "===== Recompute noise floor ====="
$RUN scripts/noise_floor.py --model t5

echo "===== New decoding ablations ====="
./scripts/run_decoding_ablations.sh t5

echo "===== Done ====="
