#!/usr/bin/env bash
# Corrector training-data generation (CSV-driven):
#   1. select keep=False & qa_status=yes images (first N) and THICK-degrade them
#      at steps 3/6/9/12 (drop step-9/12 if eff_res = spacing*step > 10mm) -> data/images
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
    # Worker counts: lower these if nnUNet reports "Background workers died /
    # RAM was full" or a multiprocessing manager error. NPP=1 NPS=1 is safest.
    local npp="${NPP:-2}" nps="${NPS:-2}"
    echo "[run_corrector_data] (2) nnUNetv2_predict (Dataset$REF_DATASET_ID) -> $DATA_NNUNET_PRED ${resume_flag:+(resume)} (npp=$npp nps=$nps)"
    mkdir -p "$DATA_NNUNET_PRED"
    nnUNetv2_predict \
        -i "$DATA_IMAGES" \
        -o "$DATA_NNUNET_PRED" \
        -d "$REF_DATASET_ID" \
        -c "$CONFIGURATION" \
        -p "$REF_PLAN" \
        -tr "$TRAINER" \
        -f "$REF_FOLD" \
        -npp "$npp" -nps "$nps" \
        $resume_flag

    # Folder check: how many samples have BOTH a degraded image and a prelabel.
    local n_img n_pre n_pair
    n_img=$(find "$DATA_IMAGES" -name '*_step*_0000.nii.gz' 2>/dev/null | wc -l | tr -d ' ')
    n_pre=$(find "$DATA_NNUNET_PRED" -name '*_step*.nii.gz' 2>/dev/null | wc -l | tr -d ' ')
    n_pair=0
    for f in "$DATA_IMAGES"/*_step*_0000.nii.gz; do
        [[ -e "$f" ]] || continue
        local b; b=$(basename "$f" _0000.nii.gz)
        [[ -e "$DATA_NNUNET_PRED/$b.nii.gz" ]] && n_pair=$((n_pair+1))
    done
    echo "[run_corrector_data] folder check: images=$n_img prelabels=$n_pre complete(image+prelabel)=$n_pair"
}

case "$WHICH" in
    build)   do_build ;;
    predict) do_predict ;;
    all)     do_build; do_predict ;;
    *) echo "usage: run_corrector_data.sh [all|build|predict]" >&2; exit 2 ;;
esac
echo "[run_corrector_data] done ($WHICH)"
