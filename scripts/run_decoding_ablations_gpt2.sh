#!/bin/bash
# GPT-2 specific decoding ablations (run after run_decoding_ablations.sh gpt2)
# Usage: ./scripts/run_decoding_ablations_gpt2.sh [--force-eval]
# Override checkpoint: CKPT_gpt2=gpt2_prefix4 ./scripts/run_decoding_ablations_gpt2.sh

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

EVAL="uv run python scripts/evaluate.py --stage 2"
FAILED=""
FORCE_EVAL=0
m=gpt2

for arg in "$@"; do
    case "$arg" in
        --force-eval) FORCE_EVAL=1 ;;
    esac
done

get_ckpt_tag() {
    local var="CKPT_$1"
    echo "${!var:-$1}"
}

run() {
    local model=$1 tag=$2
    shift 2
    local ckpt_tag=$(get_ckpt_tag "$model")

    local result_file="results/${model}/ablations/decoding/${model}_ablation_${tag}.json"
    if [ "$FORCE_EVAL" -eq 0 ] && [ -f "$result_file" ]; then
        echo "SKIP: $model | $tag (already done)"
        return
    fi

    echo "=== $model (ckpt=$ckpt_tag) | $tag ==="
    if ! $EVAL --model "$model" --ckpt-tag "$ckpt_tag" --ablation-tag "$tag" --ablation-type decoding "$@"; then
        echo "FAILED: $model | $tag"
        FAILED="$FAILED $model|$tag"
    fi
    echo ""
}

echo "===== GPT-2 specific decoding ablations ====="

# Multi-variable: trimming + best single-var winners
run $m trim_rep1.2 --trim-incomplete --repetition-penalty 1.2
run $m trim_rep1.3 --trim-incomplete --repetition-penalty 1.3
run $m trim_rep1.2_ngram2 --trim-incomplete --repetition-penalty 1.2 --no-repeat-ngram-size 2
run $m trim_rep1.2_ngram3 --trim-incomplete --repetition-penalty 1.2 --no-repeat-ngram-size 3
run $m trim_rep1.8_ngram2_t0.3 --trim-incomplete --do-sample --repetition-penalty 1.8 --no-repeat-ngram-size 2 --temperature 0.3 --top-p 0.8

# All-best combo
run $m trim_rep1.2_ngram2_t48 --trim-incomplete --repetition-penalty 1.2 --no-repeat-ngram-size 2 --max-new-tokens 48

# Experimental
run $m trim_t48 --trim-incomplete --max-new-tokens 48
run $m trim_rep1.2_ngram2_t0.3 --trim-incomplete --do-sample --repetition-penalty 1.2 --no-repeat-ngram-size 2 --temperature 0.3 --top-p 0.8
run $m trim_ngram2 --trim-incomplete --no-repeat-ngram-size 2

if [ -n "$FAILED" ]; then
    echo "FAILED:$FAILED"
    exit 1
else
    echo "All GPT-2 specific decoding ablations complete."
fi
