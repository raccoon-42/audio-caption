#!/bin/bash

echo "===== DECODING HYPERPARAMETER ABLATIONS ====="
./scripts/run_decoding_ablations.sh
DECODING_EXIT=$?

echo ""
echo "===== ARCHITECTURAL ABLATIONS (RETRAINING) ====="
./scripts/run_arch_ablations.sh
ARCH_EXIT=$?

echo ""
if [ $DECODING_EXIT -ne 0 ] || [ $ARCH_EXIT -ne 0 ]; then
    echo "Some ablations failed. Check output above for details."
    exit 1
else
    echo "All ablations complete."
fi
