#!/usr/bin/env bash
# Train + evaluate GPT-2 stage 1 for every encoder-swap config (run after
# lr_search). Stage 1 (frozen LM) is the discriminative encoder-comparison
# number; stage 2 can be run later from the saved stage1_best.pt with
# `train_gpt2.py --stage 2`.
# NOT resume-safe: rerunning retrains from scratch and overwrites checkpoints.
# Continues past a failed encoder and reports a summary.
set -uo pipefail

cd "$(dirname "$0")/.."

configs=(configs/gpt2_*.yaml)

failed=()
for cfg in "${configs[@]}"; do
    echo "=============================================================="
    echo ">>> ${cfg}"
    echo "=============================================================="
    if HF_HUB_OFFLINE=1 uv run python scripts/train_gpt2.py --config "${cfg}" --stage 1 \
       && HF_HUB_OFFLINE=1 uv run python scripts/evaluate.py --model gpt2 --config "${cfg}" --stage 1; then
        echo ">>> OK: ${cfg}"
    else
        echo ">>> FAILED: ${cfg}"
        failed+=("${cfg}")
    fi
done

echo "=============================================================="
if [ ${#failed[@]} -eq 0 ]; then
    echo "All ${#configs[@]} encoders trained + evaluated (stage 1)."
else
    echo "${#failed[@]}/${#configs[@]} failed:"
    printf '  %s\n' "${failed[@]}"
    exit 1
fi
