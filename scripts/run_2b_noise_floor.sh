#!/bin/bash
# Usage: ./scripts/run_2b_noise_floor.sh [gpt2] [t5]
# Runs AFTER run_2_baselines.sh (reuses its seed-42 run) and BEFORE the ablations.
#
# Measures run-to-run variance (the "noise floor") by retraining the baseline with
# extra seeds, so single-run ablation deltas can be judged real vs luck. Only the
# training RNG (init / shuffle / dropout) varies; the data split stays fixed at
# cfg['seed'], so every seed scores on one identical test set. Reuses the tuned LR
# (no Optuna). Seed 42 is NOT retrained here -- it is the run_2 baseline, reused as
# one of the points. Per the plan: GPT-2 x3 (42,43,44) primary; T5 x2 (42,43) cross-check.
#
# Override the extra-seed list per model: SEEDS_gpt2="43 44 45" ./scripts/run_2b_noise_floor.sh gpt2

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

EVAL="uv run python scripts/evaluate.py --stage 2"
FAILED=""

# Default to the planned noise-floor models if none given.
if [ $# -eq 0 ]; then
    set -- gpt2 t5
fi

# Extra seeds beyond the seed-42 baseline (which run_2 already produced).
get_seeds() {
    local override="SEEDS_$1"
    if [ -n "${!override}" ]; then
        echo "${!override}"
        return
    fi
    case "$1" in
        gpt2) echo "43 44" ;;   # 3 seeds total -> primary floor
        t5)   echo "43" ;;      # 2 seeds total -> transferability cross-check
        *)    echo "" ;;
    esac
}

for model in "$@"; do
    seeds=$(get_seeds "$model")
    if [ -z "$seeds" ]; then
        echo "SKIP: no noise-floor seed list for '$model' (set SEEDS_$model=\"43 44\" to add)"
        continue
    fi

    baseline="results/${model}/${model}_ablation_${model}.json"
    if [ ! -f "$baseline" ]; then
        echo "WARN: $model seed-42 baseline missing ($baseline). Run run_2_baselines.sh $model first;"
        echo "      the floor needs it as the third/second point."
    fi

    echo "===== Noise floor: $model (extra seeds: $seeds; seed 42 = run_2 baseline) ====="
    for s in $seeds; do
        tag="${model}_seed${s}"
        result_file="results/${model}/${model}_ablation_${tag}.json"

        if [ -f "$result_file" ]; then
            echo "SKIP: $tag (already done: $result_file)"
            continue
        fi

        echo "--- Training $tag ---"
        if ! uv run python "scripts/train_${model}.py" --seed "$s"; then
            echo "TRAIN FAILED: $tag"
            FAILED="$FAILED $tag(train)"
            continue
        fi

        echo "--- Evaluating $tag ---"
        if ! $EVAL --model "$model" --ckpt-tag "$tag" --ablation-tag "$tag"; then
            echo "EVAL FAILED: $tag"
            FAILED="$FAILED $tag(eval)"
        fi
        echo ""
    done
done

if [ -n "$FAILED" ]; then
    echo "FAILED:$FAILED"
    exit 1
else
    echo "Noise-floor runs complete."
    echo "Floor = spread (std/range) of FENSE across seeds, e.g. for GPT-2 over:"
    echo "  results/gpt2/gpt2_ablation_gpt2.json  (seed 42)"
    echo "  results/gpt2/gpt2_ablation_gpt2_seed43.json"
    echo "  results/gpt2/gpt2_ablation_gpt2_seed44.json"
fi
