#!/usr/bin/env bash
set -euo pipefail
source scripts/caf/common.sh
if [[ $# -lt 4 || $# -gt 5 ]]; then
    echo "Usage: bash $0 DATA_ROOT SEED PV FT [OUTPUT_DIR]" >&2
    exit 1
fi
DATA=$1
SEED=$2
PV=$3
FT=$4
OUTPUT_DIR=${5:-$OUTPUT_ROOT/transfer/pv${PV}_ft${FT}/imagenet/seed$SEED}

"$PYTHON" -B train.py \
    --root "$DATA" \
    --seed "$SEED" \
    --trainer CAF \
    --dataset-config-file configs/datasets/imagenet.yaml \
    --config-file configs/trainers/CAF/transfer.yaml \
    --output-dir "$OUTPUT_DIR" \
    --lambda-margin "$PV" \
    --lambda-text "$FT" \
    DATASET.SUBSAMPLE_CLASSES all \
    DATASET.MANIFEST "$ASSETS/splits/imagenet/test_transfer.json" \
    DATASET.TRAIN_MANIFEST "$ASSETS/splits/imagenet/train_transfer.json" \
    TRAINER.CAF.INIT_PROMPT "$ASSETS/initialization/transfer.pt" \
    TRAINER.CAF.REFERENCE "$ASSETS/references/imagenet/all.pt"

