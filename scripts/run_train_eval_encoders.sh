#!/usr/bin/env bash
# Produce the stage-1 encoder-comparison numbers (run after lr_search).
# Stage 1 (frozen LM) is the discriminative comparison; stage 2 can be run later
# from the saved stage1_best.pt with `train_gpt2.py --stage 2`.
#
# Baseline CLAP (configs/gpt2.yaml) is the comparison anchor: it is already
# trained, so it is EVAL-ONLY here (no retrain). The 7 swap encoders are
# train --stage 1 + eval --stage 1.
#
# Eval outputs are tagged `<config>_s1` so stage-1 predictions/metrics never
# collide with (or get reused from) a later stage-2 eval.
#
# NOT resume-safe for training: rerunning retrains swaps from scratch and
# overwrites checkpoints. Continues past a failure and reports a summary.
set -uo pipefail

cd "$(dirname "$0")/.."

failed=()

# --- baseline CLAP: eval-only (uses existing checkpoints/gpt2/stage1_best.pt) ---
echo "=============================================================="
echo ">>> baseline (configs/gpt2.yaml) -- eval only, stage 1"
echo "=============================================================="
if [ ! -f checkpoints/gpt2/stage1_best.pt ]; then
    echo ">>> FAILED: baseline stage1_best.pt missing -- train baseline first"
    failed+=("configs/gpt2.yaml (no stage1_best.pt)")
elif HF_HUB_OFFLINE=1 uv run python scripts/evaluate.py --model gpt2 \
        --config configs/gpt2.yaml --stage 1 --ablation-tag gpt2_s1; then
    echo ">>> OK: baseline"
else
    echo ">>> FAILED: baseline"
    failed+=("configs/gpt2.yaml")
fi

# --- swap encoders: train stage 1 + eval stage 1 ---
for cfg in configs/gpt2_*.yaml; do
    enc=$(basename "${cfg}" .yaml)   # e.g. gpt2_clap_music
    echo "=============================================================="
    echo ">>> ${cfg}"
    echo "=============================================================="
    if HF_HUB_OFFLINE=1 uv run python scripts/train_gpt2.py --config "${cfg}" --stage 1 \
       && HF_HUB_OFFLINE=1 uv run python scripts/evaluate.py --model gpt2 \
              --config "${cfg}" --stage 1 --ablation-tag "${enc}_s1"; then
        echo ">>> OK: ${cfg}"
    else
        echo ">>> FAILED: ${cfg}"
        failed+=("${cfg}")
    fi
done

echo "=============================================================="
if [ ${#failed[@]} -eq 0 ]; then
    echo "Stage-1 done: baseline + all swap encoders evaluated."
else
    echo "${#failed[@]} failed:"
    printf '  %s\n' "${failed[@]}"
    exit 1
fi
