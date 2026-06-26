#!/usr/bin/env bash
# Corrector training-data generation (CSV-driven):
#   1. select keep=False & qa_status=yes images (first N) and THICK-degrade them
#      at steps 3/6/9 (drop step-9 if eff_res = spacing*9 > 10mm) -> data/images
#   2. predict the degraded images with the Dataset835 nnUNet     -> data/nnunet_pred
#   (CNISP predictions later land in data/cnisp_pred)
#
# Usage:
#   bash nnunet-c/run_corrector_data.sh            # build + 835 predict
#   bash nnunet-c/run_corrector_data.sh build      # degrade only
#   bash nnunet-c/run_corrector_data.sh predict    # 835 predict only
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${CONFIG:-$HERE/configs/corrector.yaml}"
WHICH="${1:-all}"

eval "$(python3 "$HERE/scripts/corrector_env.py" --config "$CONFIG" --control A)"

do_build() {
    echo "[run_corrector_data] (1) select + thick-degrade -> $DATA_IMAGES"
    python3 "$HERE/scripts/build_corrector_data.py" --config "$CONFIG"
}

do_predict() {
    : "${nnUNet_results:?export nnUNet_results}"
    # Resume by default: --continue_prediction skips cases whose output mask
    # already exists in $DATA_NNUNET_PRED. FORCE=1 re-predicts everything.
    local resume_flag="--continue_prediction"
    [[ "${FORCE:-0}" == "1" ]] && resume_flag=""
    echo "[run_corrector_data] (2) nnUNetv2_predict (Dataset$REF_DATASET_ID) -> $DATA_NNUNET_PRED ${resume_flag:+(resume)}"
    mkdir -p "$DATA_NNUNET_PRED"
    nnUNetv2_predict \
        -i "$DATA_IMAGES" \
        -o "$DATA_NNUNET_PRED" \
        -d "$REF_DATASET_ID" \
        -c "$CONFIGURATION" \
        -p "$REF_PLAN" \
        -tr "$TRAINER" \
        -f "$REF_FOLD" \
        $resume_flag
}

case "$WHICH" in
    build)   do_build ;;
    predict) do_predict ;;
    all)     do_build; do_predict ;;
    *) echo "usage: run_corrector_data.sh [all|build|predict]" >&2; exit 2 ;;
esac
echo "[run_corrector_data] done ($WHICH)"
