#!/usr/bin/env bash
set -euo pipefail
source scripts/caf/common.sh
if [[ $# -lt 4 || $# -gt 6 ]]; then
    echo "Usage: bash $0 DATA_ROOT DATASET SEED MODEL_DIR [OUTPUT_DIR] [BASE_READOUT]" >&2
    exit 1
fi
DATA=$1
DATASET=$2
SEED=$3
MODEL_DIR=$4
OUTPUT_DIR=${5:-$OUTPUT_ROOT/evaluation/base2new/$DATASET/seed$SEED}
BASE_READOUT=${6:-VT}
CONFIG=base2new
if [[ "$DATASET" == imagenet ]]; then CONFIG=base2new_imagenet; fi
if [[ "$BASE_READOUT" != VT && "$BASE_READOUT" != fusion ]]; then
    echo "BASE_READOUT must be VT or fusion" >&2; exit 1
fi

for SPLIT in base new; do
    READOUT=fusion
    BANK=all
    if [[ "$SPLIT" == base ]]; then
        BANK=base
        READOUT=$BASE_READOUT
    fi
    "$PYTHON" -B train.py \
        --root "$DATA" \
        --seed "$SEED" \
        --trainer CAF \
        --dataset-config-file "configs/datasets/${DATASET}.yaml" \
        --config-file "configs/trainers/CAF/${CONFIG}.yaml" \
        --output-dir "$OUTPUT_DIR/$SPLIT" \
        --model-dir "$MODEL_DIR" \
        --load-epoch 20 \
        --eval-only \
        DATASET.SUBSAMPLE_CLASSES "$SPLIT" \
        DATASET.MANIFEST "$ASSETS/splits/$DATASET/test.json" \
        TRAINER.CAF.REFERENCE "$ASSETS/references/$DATASET/$BANK.pt" \
        TEST.READOUT "$READOUT"
done

