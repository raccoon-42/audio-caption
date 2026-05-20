#!/bin/bash
# Usage: ./scripts/run_4_decoding_ablations.sh gpt2 t5 opt llama
# Override checkpoint: CKPT_gpt2=gpt2_prefix4 ./scripts/run_4_decoding_ablations.sh gpt2

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

EVAL="uv run python scripts/evaluate.py --stage 2"
FAILED=""

if [ $# -eq 0 ]; then
    echo "Usage: $0 <model1> [model2] ..."
    echo "Override checkpoint per model: CKPT_<model>=<tag>"
    exit 1
fi

get_ckpt_tag() {
    local var="CKPT_$1"
    echo "${!var:-$1}"
}

run() {
    local model=$1 tag=$2
    shift 2
    local ckpt_tag=$(get_ckpt_tag "$model")

    local result_file="results/${model}/${model}_ablation_${tag}.json"
    if [ -f "$result_file" ]; then
        echo "SKIP: $model | $tag (already done)"
        return
    fi

    echo "=== $model (ckpt=$ckpt_tag) | $tag ==="
    if ! $EVAL --model "$model" --ckpt-tag "$ckpt_tag" --ablation-tag "$tag" "$@"; then
        echo "FAILED: $model | $tag"
        FAILED="$FAILED $model|$tag"
    fi
    echo ""
}

for m in "$@"; do
    echo "===== Decoding ablations: $m ====="

    run $m rep_1.1 --repetition-penalty 1.1
    run $m rep_1.2 --repetition-penalty 1.2
    run $m rep_1.3 --repetition-penalty 1.3

    run $m ngram_2 --no-repeat-ngram-size 2
    run $m ngram_3 --no-repeat-ngram-size 3
    run $m ngram_4 --no-repeat-ngram-size 4

    run $m tokens_32 --max-new-tokens 32
    run $m tokens_48 --max-new-tokens 48
    run $m tokens_96 --max-new-tokens 96

    run $m beam_2 --num-beams 2
    run $m beam_4 --num-beams 4
    run $m beam_5 --num-beams 5

    run $m rep1.2_ngram3 --repetition-penalty 1.2 --no-repeat-ngram-size 3

    run $m beam4_lp0.6 --num-beams 4 --length-penalty 0.6
    run $m beam4_lp1.5 --num-beams 4 --length-penalty 1.5

    run $m sample_t0.7_p0.9 --do-sample --temperature 0.7 --top-p 0.9
    run $m sample_t0.9_p0.95 --do-sample --temperature 0.9 --top-p 0.95
    run $m sample_t1.0_p0.95 --do-sample --temperature 1.0 --top-p 0.95

    # Ablation: sentence trimming
    run $m trim_incomplete --trim-incomplete
done

if [ -n "$FAILED" ]; then
    echo "FAILED:$FAILED"
    exit 1
else
    echo "All decoding ablations complete."
fi
