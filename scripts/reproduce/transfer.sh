#!/usr/bin/env bash
set -euo pipefail
source scripts/caf/common.sh
if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: bash $0 DATA_ROOT train|test [MODEL_ROOT]" >&2
    exit 1
fi
DATA=$1
ACTION=$2
MODEL_ROOT=${3:-pretrained}
if [[ "$ACTION" != train && "$ACTION" != test ]]; then
    echo "ACTION must be train or test" >&2; exit 1
fi
if [[ "$ACTION" == train && $# == 3 ]]; then
    echo "MODEL_ROOT is used only for testing" >&2; exit 1
fi

# Table 1 source-training hyperparameters.
PV=1.0
FT=40.0

for SEED in 1 2 3; do
    if [[ "$ACTION" == train ]]; then
        bash scripts/caf/transfer_train.sh \
            "$DATA" "$SEED" "$PV" "$FT" "$OUTPUT_ROOT/transfer/imagenet/seed$SEED"
    else
        bash scripts/caf/transfer_test.sh \
            "$DATA" "$SEED" "$MODEL_ROOT/transfer/imagenet/seed$SEED" all \
            "$OUTPUT_ROOT/evaluation/transfer"
    fi
done

if [[ "$ACTION" == test ]]; then
    "$PYTHON" -B tools/summarize.py --track transfer \
        --root "$OUTPUT_ROOT/evaluation/transfer" --output "$OUTPUT_ROOT/tables/transfer"
fi
