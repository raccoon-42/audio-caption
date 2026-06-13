#!/usr/bin/env bash
# Extra noise-floor seeds (43, 44) for the GPT-2 encoder-swap comparison.
# Mirrors run_train_eval_encoders.sh but varies the training RNG seed so each
# encoder gets multiple stage-1 runs to estimate its own noise floor.
#
# Per seed:
#   - swap encoders (configs/gpt2_*.yaml): train --stage 1 --seed N + eval --stage 1
#   - baseline CLAP (configs/gpt2.yaml): EVAL-ONLY, reuses the existing
#     checkpoints/gpt2_seedN checkpoint from the baseline noise-floor runs.
#
# train_gpt2.py --seed N suffixes the checkpoint tag to gpt2_seedN (N != split
# seed 42), so evaluate.py is pointed at it with --ckpt-tag gpt2_seedN. Eval
# outputs are tagged <enc>_seedN_s1 so they never collide with the seed-42 run.
#
# NOT resume-safe for training: rerunning retrains from scratch and overwrites
# the gpt2_seedN checkpoints. Continues past a failure and reports a summary.
set -uo pipefail

cd "$(dirname "$0")/.."

seeds=(43 44)
failed=()

for seed in "${seeds[@]}"; do
    # --- baseline CLAP: eval-only (reuses checkpoints/gpt2_seed${seed}) ---
    echo "=============================================================="
    echo ">>> seed ${seed}: baseline (configs/gpt2.yaml) -- eval only, stage 1"
    echo "=============================================================="
    if [ ! -f "checkpoints/gpt2_seed${seed}/stage1_best.pt" ]; then
        echo ">>> FAILED: checkpoints/gpt2_seed${seed}/stage1_best.pt missing -- train baseline seed first"
        failed+=("configs/gpt2.yaml @seed${seed} (no checkpoint)")
    elif HF_HUB_OFFLINE=1 uv run python scripts/evaluate.py --model gpt2 \
            --config configs/gpt2.yaml --stage 1 \
            --ckpt-tag "gpt2_seed${seed}" --ablation-tag "gpt2_seed${seed}_s1"; then
        echo ">>> OK: baseline @seed${seed}"
    else
        echo ">>> FAILED: baseline @seed${seed}"
        failed+=("configs/gpt2.yaml @seed${seed}")
    fi

    # --- swap encoders: train stage 1 + eval stage 1 ---
    for cfg in configs/gpt2_*.yaml; do
        enc=$(basename "${cfg}" .yaml)   # e.g. gpt2_clap_music
        echo "=============================================================="
        echo ">>> seed ${seed}: ${cfg}"
        echo "=============================================================="
        if HF_HUB_OFFLINE=1 uv run python scripts/train_gpt2.py --config "${cfg}" --stage 1 --seed "${seed}" \
           && HF_HUB_OFFLINE=1 uv run python scripts/evaluate.py --model gpt2 \
                  --config "${cfg}" --stage 1 \
                  --ckpt-tag "gpt2_seed${seed}" --ablation-tag "${enc}_seed${seed}_s1"; then
            echo ">>> OK: ${cfg} @seed${seed}"
        else
            echo ">>> FAILED: ${cfg} @seed${seed}"
            failed+=("${cfg} @seed${seed}")
        fi
    done
done

echo "=============================================================="
if [ ${#failed[@]} -eq 0 ]; then
    echo "Stage-1 done: seeds ${seeds[*]} for baseline + all swap encoders."
else
    echo "${#failed[@]} failed:"
    printf '  %s\n' "${failed[@]}"
    exit 1
fi
