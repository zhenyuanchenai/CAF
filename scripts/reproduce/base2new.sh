#!/usr/bin/env bash
set -euo pipefail
source scripts/caf/common.sh
if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: bash $0 DATA_ROOT caf|caf_star train|test [MODEL_ROOT]" >&2
    exit 1
fi
DATA=$1
SCHEME=$2
ACTION=$3
MODEL_ROOT=${4:-pretrained}
if [[ "$ACTION" != train && "$ACTION" != test ]]; then
    echo "ACTION must be train or test" >&2; exit 1
fi
if [[ "$ACTION" == train && $# == 4 ]]; then
    echo "MODEL_ROOT is used only for testing" >&2; exit 1
fi

# Table 2 configurations: dataset, lambda_margin (PV), lambda_text (FT).
case "$SCHEME" in
    caf)
        SETTINGS=(
            "imagenet        4.0 2.5"
            "caltech101      4.0 2.5"
            "oxford_pets     4.0 2.5"
            "stanford_cars   4.0 2.5"
            "oxford_flowers  4.0 2.5"
            "food101         4.0 2.5"
            "fgvc_aircraft   4.0 2.5"
            "sun397          4.0 2.5"
            "dtd             4.0 2.5"
            "eurosat         4.0 2.5"
            "ucf101          4.0 2.5"
        ) ;;
    caf_star)
        SETTINGS=(
            "imagenet        1.0 2.5"
            "caltech101      0.5 0.0"
            "oxford_pets     2.0 20.0"
            "stanford_cars   0.0 2.5"
            "oxford_flowers  4.0 2.5"
            "food101         4.0 40.0"
            "fgvc_aircraft   2.0 0.0"
            "sun397          2.0 5.0"
            "dtd             4.0 5.0"
            "eurosat         1.0 2.5"
            "ucf101          0.0 2.5"
        ) ;;
    *) echo "SCHEME must be caf or caf_star" >&2; exit 1 ;;
esac

for SETTING in "${SETTINGS[@]}"; do
    read -r DATASET PV FT <<< "$SETTING"
    for SEED in 1 2 3; do
        RUN_DIR="$OUTPUT_ROOT/base2new/$SCHEME/$DATASET/seed$SEED"
        if [[ "$ACTION" == train ]]; then
            # Both schemes share the same Flowers102 setting and trained model.
            SHARED_DIR="$OUTPUT_ROOT/base2new/caf/oxford_flowers/seed$SEED"
            if [[ "$SCHEME" == caf_star && "$DATASET" == oxford_flowers &&
                  -f "$SHARED_DIR/CAF/model.pth.tar-20" && ! -e "$RUN_DIR" ]]; then
                mkdir -p "$(dirname "$RUN_DIR")"
                cp -a "$SHARED_DIR" "$RUN_DIR"
            else
                bash scripts/caf/base2new_train.sh \
                    "$DATA" "$DATASET" "$SEED" "$PV" "$FT" "$RUN_DIR"
            fi
        else
            BASE_READOUT=VT
            if [[ "$DATASET" == imagenet ]]; then BASE_READOUT=fusion; fi
            bash scripts/caf/base2new_test.sh \
                "$DATA" "$DATASET" "$SEED" \
                "$MODEL_ROOT/base2new/$SCHEME/$DATASET/seed$SEED" \
                "$OUTPUT_ROOT/evaluation/base2new/$SCHEME/$DATASET/seed$SEED" \
                "$BASE_READOUT"
        fi
    done
done

if [[ "$ACTION" == test ]]; then
    "$PYTHON" -B tools/summarize.py --track base2new \
        --root "$OUTPUT_ROOT/evaluation/base2new/$SCHEME" \
        --output "$OUTPUT_ROOT/tables/$SCHEME"
fi
