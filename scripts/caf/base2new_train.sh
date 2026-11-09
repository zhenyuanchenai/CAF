#!/usr/bin/env bash
set -euo pipefail
source scripts/caf/common.sh
if [[ $# -lt 5 || $# -gt 6 ]]; then
    echo "Usage: bash $0 DATA_ROOT DATASET SEED PV FT [OUTPUT_DIR]" >&2
    exit 1
fi
DATA=$1
DATASET=$2
SEED=$3
PV=$4
FT=$5
OUTPUT_DIR=${6:-$OUTPUT_ROOT/base2new/pv${PV}_ft${FT}/$DATASET/seed$SEED}
CONFIG=base2new
if [[ "$DATASET" == imagenet ]]; then CONFIG=base2new_imagenet; fi

"$PYTHON" -B train.py \
    --root "$DATA" \
    --seed "$SEED" \
    --trainer CAF \
    --dataset-config-file "configs/datasets/${DATASET}.yaml" \
    --config-file "configs/trainers/CAF/${CONFIG}.yaml" \
    --output-dir "$OUTPUT_DIR" \
    --lambda-margin "$PV" \
    --lambda-text "$FT" \
    DATASET.SUBSAMPLE_CLASSES base \
    DATASET.MANIFEST "$ASSETS/splits/$DATASET/test.json" \
    DATASET.TRAIN_MANIFEST "$ASSETS/splits/$DATASET/train_seed$SEED.json" \
    TRAINER.CAF.INIT_PROMPT "$ASSETS/initialization/base2new/seed$SEED.pt" \
    TRAINER.CAF.REFERENCE "$ASSETS/references/$DATASET/base.pt"

