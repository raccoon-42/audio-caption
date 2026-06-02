#!/bin/bash
# Usage: ./scripts/run_arch_ablations.sh [--force-train] [--force-eval] gpt2 [t5] ...

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

EVAL="uv run python scripts/evaluate.py --stage 2"
FAILED=""
FORCE_TRAIN=0
FORCE_EVAL=0

MODELS=()
for arg in "$@"; do
    case "$arg" in
        --force-train) FORCE_TRAIN=1 ;;
        --force-eval)  FORCE_EVAL=1 ;;
        *)             MODELS+=("$arg") ;;
    esac
done

if [ ${#MODELS[@]} -eq 0 ]; then
    echo "Usage: $0 [--force-train] [--force-eval] <model1> [model2] ..."
    echo "Models: gpt2, t5, opt, llama"
    exit 1
fi

train_and_eval() {
    local model=$1 tag=$2
    shift 2

    local ckpt_file="checkpoints/${tag}/stage2_proj_best.pt"
    local result_file="results/${model}/ablations/arch/${model}_ablation_${tag}.json"

    if [ "$FORCE_TRAIN" -eq 0 ] && [ -f "$ckpt_file" ]; then
        echo "SKIP TRAIN: $tag (checkpoint exists: $ckpt_file)"
    else
        echo "=========================================="
        echo "TRAINING: $tag"
        echo "=========================================="
        if ! uv run python "scripts/train_${model}.py" --ablation-tag "$tag" "$@"; then
            echo "TRAIN FAILED: $tag"
            FAILED="$FAILED $tag(train)"
            return
        fi
    fi

    if [ "$FORCE_EVAL" -eq 0 ] && [ -f "$result_file" ]; then
        echo "SKIP EVAL: $tag (already done: $result_file)"
    else
        echo "--- Evaluating $tag ---"
        if ! $EVAL --model "$model" --ckpt-tag "$tag" --ablation-tag "$tag" --ablation-type arch "$@"; then
            echo "EVAL FAILED: $tag"
            FAILED="$FAILED $tag(eval)"
        fi
    fi
    echo ""
}

for model in "${MODELS[@]}"; do
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
