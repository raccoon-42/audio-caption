#!/bin/bash

MODELS="gpt2 t5 opt"
EVAL="uv run python scripts/evaluate.py --stage 2"
FAILED=""

run() {
    local model=$1 tag=$2
    shift 2
    echo "=== $model | $tag ==="
    if ! $EVAL --model "$model" --ablation-tag "$tag" "$@"; then
        echo "FAILED: $model | $tag"
        FAILED="$FAILED $model|$tag"
    fi
    echo ""
}

for m in $MODELS; do
    # Baseline: pure greedy
    run $m baseline

    # Ablation 1: repetition_penalty
    run $m rep_1.1 --repetition-penalty 1.1
    run $m rep_1.2 --repetition-penalty 1.2
    run $m rep_1.3 --repetition-penalty 1.3

    # Ablation 2: no_repeat_ngram_size
    run $m ngram_2 --no-repeat-ngram-size 2
    run $m ngram_3 --no-repeat-ngram-size 3
    run $m ngram_4 --no-repeat-ngram-size 4

    # Ablation 3: max_new_tokens
    run $m tokens_32 --max-new-tokens 32
    run $m tokens_48 --max-new-tokens 48
    run $m tokens_96 --max-new-tokens 96

    # Ablation 4: beam search
    run $m beam_2 --num-beams 2
    run $m beam_4 --num-beams 4
    run $m beam_5 --num-beams 5

    # Ablation 5: rep_penalty + no_repeat_ngram combined
    run $m rep1.2_ngram3 --repetition-penalty 1.2 --no-repeat-ngram-size 3

    # Ablation 6: beam search + length_penalty (beam4_lp1.0 skipped, same as beam_4)
    run $m beam4_lp0.6 --num-beams 4 --length-penalty 0.6
    run $m beam4_lp1.5 --num-beams 4 --length-penalty 1.5

    # Ablation 7: sampling
    run $m sample_t0.7_p0.9 --do-sample --temperature 0.7 --top-p 0.9
    run $m sample_t0.9_p0.95 --do-sample --temperature 0.9 --top-p 0.95
    run $m sample_t1.0_p0.95 --do-sample --temperature 1.0 --top-p 0.95
done

if [ -n "$FAILED" ]; then
    echo "FAILED RUNS:$FAILED"
    exit 1
else
    echo "All ablations complete."
fi
