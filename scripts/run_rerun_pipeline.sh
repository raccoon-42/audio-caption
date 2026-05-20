#!/bin/bash

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

set -e

echo "===== STEP 1: RERUN LR SEARCHES ====="

echo "--- GPT-2: S2 only (S1 was fine) ---"
python3 -c "import optuna; optuna.delete_study(study_name='lr_search_gpt2_s2', storage='sqlite:///results/gpt2/gpt2_lr_search.db')" 2>/dev/null || true
uv run python scripts/lr_search.py --model gpt2 --stage s2

echo ""
echo "--- T5: S1 + S2 (both hit ceilings) ---"
rm -f results/t5/t5_lr_search.db results/t5/t5_s1_best_proj.pt
uv run python scripts/lr_search.py --model t5 --stage both

echo ""
echo "===== STEP 2: BASELINES + ARCH ABLATIONS ====="
./scripts/run_arch_ablations.sh

echo ""
echo "Pipeline complete."
echo "NEXT: Inspect results, pick best arch per model, then run decoding ablations."
