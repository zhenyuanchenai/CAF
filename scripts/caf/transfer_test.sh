#!/usr/bin/env bash
set -euo pipefail
source scripts/caf/common.sh
if [[ $# -lt 3 || $# -gt 5 ]]; then
    echo "Usage: bash $0 DATA_ROOT SEED MODEL_DIR [DATASET|all] [OUTPUT_DIR]" >&2
    exit 1
fi
DATA=$1
SEED=$2
MODEL_DIR=$3
DATASET=${4:-all}
OUTPUT_DIR=${5:-$OUTPUT_ROOT/evaluation/transfer}
DATASETS=(imagenet caltech101 oxford_pets stanford_cars oxford_flowers food101 fgvc_aircraft sun397 dtd eurosat ucf101 imagenetv2 imagenet_sketch imagenet_a imagenet_r)
if [[ "$DATASET" != all ]]; then
    if [[ " ${DATASETS[*]} " != *" $DATASET "* ]]; then
        echo "Unknown dataset: $DATASET" >&2; exit 1
    fi
    DATASETS=("$DATASET")
fi
for DATASET in "${DATASETS[@]}"; do
    "$PYTHON" -B train.py \
        --root "$DATA" \
        --seed "$SEED" \
        --trainer CAF \
        --dataset-config-file "configs/datasets/${DATASET}.yaml" \
        --config-file configs/trainers/CAF/transfer.yaml \
        --output-dir "$OUTPUT_DIR/$DATASET/seed$SEED" \
        --model-dir "$MODEL_DIR" \
        --load-epoch 5 \
        --eval-only \
        DATASET.SUBSAMPLE_CLASSES all \
        DATASET.MANIFEST "$ASSETS/splits/$DATASET/test_transfer.json" \
        TRAINER.CAF.REFERENCE "$ASSETS/references/$DATASET/all.pt" \
        TEST.READOUT fusion
done

