#!/bin/bash

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

EVAL="uv run python scripts/evaluate.py --stage 2"
FAILED=""

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

echo "===== PRECOMPUTE CLAP EMBEDDINGS ====="
if [ -f "data/clap_embeddings.pt" ]; then
    echo "SKIP: clap_embeddings.pt already exists"
else
    if ! uv run python scripts/prepare_data.py --precompute-clap; then
        echo "CLAP precomputation failed"
        exit 1
    fi
fi
echo ""

echo "===== LEARNING RATE SEARCH ====="
for model in llama; do
    echo "--- LR search: $model ---"
    if ! uv run python scripts/lr_search.py --model "$model"; then
        echo "LR SEARCH FAILED: $model"
        exit 1
    fi
    echo ""
done

echo "===== BASELINES ====="
for model in llama; do
    train_and_eval $model $model
done

echo "===== ARCHITECTURAL ABLATIONS ====="
for model in llama; do
    train_and_eval $model ${model}_prefix4 --prefix-len 4
    train_and_eval $model ${model}_prefix16 --prefix-len 16

    train_and_eval $model ${model}_drop0.1 --dropout 0.1
    train_and_eval $model ${model}_drop0.5 --dropout 0.5

    train_and_eval $model ${model}_depth1 --proj-depth 1
    train_and_eval $model ${model}_depth3 --proj-depth 3
done

if [ -n "$FAILED" ]; then
    echo "FAILED RUNS:$FAILED"
    exit 1
else
    echo ""
    echo "LLaMA pipeline complete."
    echo ""
    echo "NEXT STEP: Inspect results, pick best architecture per model, then run decoding ablations."
fi
