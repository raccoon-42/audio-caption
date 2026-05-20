#!/bin/bash
# Usage: ./scripts/run_3_arch_ablations.sh gpt2 t5 opt llama

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

EVAL="uv run python scripts/evaluate.py --stage 2"
FAILED=""

if [ $# -eq 0 ]; then
    echo "Usage: $0 <model1> [model2] ..."
    echo "Models: gpt2, t5, opt, llama"
    exit 1
fi

train_and_eval() {
    local model=$1 tag=$2
    shift 2

    local result_file="results/${model}/${model}_ablation_${tag}.json"

    if [ -f "$result_file" ]; then
        echo "SKIP: $tag (already done: $result_file)"
        return
    fi

    echo "=========================================="
    echo "TRAINING: $tag"
    echo "=========================================="
    if ! uv run python "scripts/train_${model}.py" --ablation-tag "$tag" "$@"; then
        echo "TRAIN FAILED: $tag"
        FAILED="$FAILED $tag(train)"
        return
    fi

    echo "--- Evaluating $tag ---"
    if ! $EVAL --model "$model" --ckpt-tag "$tag" --ablation-tag "$tag" "$@"; then
        echo "EVAL FAILED: $tag"
        FAILED="$FAILED $tag(eval)"
    fi
    echo ""
}

for model in "$@"; do
    echo "===== Arch ablations: $model ====="

    train_and_eval $model ${model}_prefix4 --prefix-len 4
    train_and_eval $model ${model}_prefix16 --prefix-len 16

    train_and_eval $model ${model}_drop0.1 --dropout 0.1
    train_and_eval $model ${model}_drop0.5 --dropout 0.5

    train_and_eval $model ${model}_depth1 --proj-depth 1
    train_and_eval $model ${model}_depth3 --proj-depth 3
done

if [ -n "$FAILED" ]; then
    echo "FAILED:$FAILED"
    exit 1
else
    echo "All arch ablations complete."
fi
