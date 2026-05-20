#!/bin/bash

# Prevent Hugging Face hub bug during network instability/offline use
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

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
for model in gpt2 t5 opt; do
    echo "--- LR search: $model ---"
    if ! uv run python scripts/lr_search.py --model "$model"; then
        echo "LR SEARCH FAILED: $model"
        exit 1
    fi
    echo ""
done

echo ""
echo "===== ARCHITECTURAL ABLATIONS (with baselines) ====="
./scripts/run_arch_ablations.sh
ARCH_EXIT=$?

echo ""
if [ $ARCH_EXIT -ne 0 ]; then
    echo "Some ablations failed. Check output above for details."
    exit 1
else
    echo "Full pipeline complete."
    echo ""
    echo "NEXT STEP: Inspect results, pick best architecture per model, then run:"
    echo "  CKPT_gpt2=<best_tag> CKPT_t5=<best_tag> CKPT_opt=<best_tag> ./scripts/run_decoding_ablations.sh"
    echo ""
    echo "Example:"
    echo "  CKPT_gpt2=gpt2_prefix4 CKPT_t5=t5 CKPT_opt=opt_depth3 ./scripts/run_decoding_ablations.sh"
fi
