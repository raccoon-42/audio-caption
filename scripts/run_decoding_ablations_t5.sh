#!/bin/bash
# T5 specific decoding ablations (run after run_decoding_ablations.sh t5)
# Usage: ./scripts/run_decoding_ablations_t5.sh [--force-eval]
# Override checkpoint: CKPT_t5=t5_prefix4 ./scripts/run_decoding_ablations_t5.sh

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

EVAL="uv run python scripts/evaluate.py --stage 2"
FAILED=""
FORCE_EVAL=0
m=t5

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

echo "===== T5 specific decoding ablations ====="

# Multi-variable: trimming + T5 single-var winners
run $m trim_rep1.5 --trim-incomplete --repetition-penalty 1.5
run $m trim_rep1.5_ngram2 --trim-incomplete --repetition-penalty 1.5 --no-repeat-ngram-size 2
run $m trim_beam5 --trim-incomplete --num-beams 5
run $m trim_rep1.2_beam5 --trim-incomplete --repetition-penalty 1.2 --num-beams 5

# Beam + length penalty combos (T5 responds to beams unlike GPT-2)
run $m trim_beam5_lp0.6 --trim-incomplete --num-beams 5 --length-penalty 0.6
run $m trim_beam5_lp1.5 --trim-incomplete --num-beams 5 --length-penalty 1.5
run $m trim_rep1.2_beam5_lp0.6 --trim-incomplete --repetition-penalty 1.2 --num-beams 5 --length-penalty 0.6

# Beam + ngram blocking
run $m trim_beam5_ngram2 --trim-incomplete --num-beams 5 --no-repeat-ngram-size 2

# rep1.3 + beam (T5 #2 FENSE winner with beam)
run $m trim_rep1.3_beam5 --trim-incomplete --repetition-penalty 1.3 --num-beams 5

# Three-way combo: beam + ngram + rep
run $m trim_rep1.2_beam5_ngram2 --trim-incomplete --repetition-penalty 1.2 --num-beams 5 --no-repeat-ngram-size 2

# rep1.3 + ngram2 (filling the gap between 1.2 and 1.5)
run $m trim_rep1.3_ngram2 --trim-incomplete --repetition-penalty 1.3 --no-repeat-ngram-size 2

if [ -n "$FAILED" ]; then
    echo "FAILED:$FAILED"
    exit 1
else
    echo "All T5 specific decoding ablations complete."
fi
