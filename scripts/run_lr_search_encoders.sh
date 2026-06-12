#!/usr/bin/env bash
# Run GPT-2 LR search for every encoder-swap config.
# Resume-safe: a config whose Optuna study is already full is skipped, so this
# can be rerun freely. Continues past a failed encoder and reports a summary.
set -uo pipefail

cd "$(dirname "$0")/.."

configs=(configs/gpt2_*.yaml)

failed=()
for cfg in "${configs[@]}"; do
    echo "=============================================================="
    echo ">>> lr_search: ${cfg}"
    echo "=============================================================="
    if HF_HUB_OFFLINE=1 uv run python scripts/lr_search.py --model gpt2 --config "${cfg}"; then
        echo ">>> OK: ${cfg}"
    else
        echo ">>> FAILED: ${cfg}"
        failed+=("${cfg}")
    fi
done

echo "=============================================================="
if [ ${#failed[@]} -eq 0 ]; then
    echo "All ${#configs[@]} encoder LR searches completed."
else
    echo "${#failed[@]}/${#configs[@]} failed:"
    printf '  %s\n' "${failed[@]}"
    exit 1
fi
