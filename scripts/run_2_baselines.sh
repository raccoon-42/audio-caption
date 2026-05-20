#!/bin/bash
# Usage: ./scripts/run_2_baselines.sh gpt2 t5 opt llama

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

EVAL="uv run python scripts/evaluate.py --stage 2"
FAILED=""

if [ $# -eq 0 ]; then
    echo "Usage: $0 <model1> [model2] ..."
    echo "Models: gpt2, t5, opt, llama"
    exit 1
fi

for model in "$@"; do
    result_file="results/${model}/${model}_ablation_${model}.json"

    if [ -f "$result_file" ]; then
        echo "SKIP: $model baseline (already done: $result_file)"
        continue
    fi

    echo "===== Training baseline: $model ====="
    if ! uv run python "scripts/train_${model}.py" --ablation-tag "$model"; then
        echo "TRAIN FAILED: $model"
        FAILED="$FAILED $model(train)"
        continue
    fi

    echo "--- Evaluating $model baseline ---"
    if ! $EVAL --model "$model" --ckpt-tag "$model" --ablation-tag "$model"; then
        echo "EVAL FAILED: $model"
        FAILED="$FAILED $model(eval)"
    fi
    echo ""
done

if [ -n "$FAILED" ]; then
    echo "FAILED:$FAILED"
    exit 1
else
    echo "All baselines complete."
fi
