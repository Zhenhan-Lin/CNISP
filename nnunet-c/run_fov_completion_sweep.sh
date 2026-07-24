#!/usr/bin/env bash
# FOV-completion VALIDATION sweep (model selection) — revised-plan §6. For each
# periodic snapshot, run whole-volume prediction on the FIXED 7-condition VALIDATION
# fold, do strict FOV-validity-mask region-split eval -> the long metrics CSV, then
# SELECT the checkpoint. This is the validation/model-selection pipeline ONLY; the
# final held-out test is a SEPARATE script (run_fov_completion_test.sh, §7).
#
# Revised vs the first sweep:
#   * evalset is MANIFEST-DRIVEN (build_fov_completion_evalset.py): exact FOV case
#     ids from splits_final.json, no thickness --steps auto, no step inference (P0-1);
#   * a stored FOV VALIDITY MASK (fovMaskTs/) is resampled to the GT grid at eval,
#     never a source-grid visible_box on the GT array (P0-2);
#   * eval is STRICT — any missing/skipped/extra/geometry case raises (P0-3);
#   * per-snapshot prediction PROVENANCE gates --continue_prediction (P2-9).
#
# Output under FOV-completion-distinct roots (never collides with corrector/per-eye):
#   fov_completion_test_input/   imagesTs + prevsegTs + fovMaskTs + eval_cases_map.json
#   fov_completion_sweep/        pred_epoch_XXXX/ , metrics_long.csv , checkpoint_scores.csv
#
# Usage:
#   FOV_MANIFEST=/p/fov_completion_manifest.json SPLITS=/p/splits_final.json \
#       bash nnunet-c/run_fov_completion_sweep.sh C 0
#   FOVC_FINAL=1 FOVC_MIN_MISSING_MM3=3 FOVC_MIN_VISIBLE_MM3=3 ...   # strict paper selection
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CONTROL="${1:?usage: run_fov_completion_sweep.sh <C> <fold>}"
FOLD="${2:?usage: run_fov_completion_sweep.sh <C> <fold>}"
CONFIG="${CONFIG:-$HERE/configs/corrector_fov.yaml}"
PLAN_NAME="${PLAN_NAME:-nnUNetPlansFinetune}"
export CORRECTOR_TRAINER="${CORRECTOR_TRAINER:-nnUNetTrainer_OrbitalFOVCompletion}"
export nnUNet_compile="${nnUNet_compile:-f}"
FOV_MANIFEST="${FOV_MANIFEST:?export FOV_MANIFEST=/path/to/fov_completion_manifest.json}"
SPLITS="${SPLITS:?export SPLITS=/path/to/splits_final.json (revised-plan §6.1: no generic casefile)}"
export CORRECTOR_FOV_MANIFEST="$FOV_MANIFEST"

TEST_ROOT="${FOVC_TEST_ROOT:-$HERE/fov_completion_test_input}"
SWEEP_ROOT="${FOVC_SWEEP_ROOT:-$HERE/fov_completion_sweep}"
mkdir -p "$SWEEP_ROOT"

echo "================================================================"
echo "[fovc-sweep] VALIDATION control=$CONTROL fold=$FOLD trainer=$CORRECTOR_TRAINER"
echo "================================================================"

eval "$(python3 "$HERE/scripts/corrector_env.py" --config "$CONFIG" --control "$CONTROL")"
: "${nnUNet_results:?export nnUNet_results}"
: "${nnUNet_preprocessed:?export nnUNet_preprocessed}"
PLAN_JSON="$nnUNet_preprocessed/Dataset$(printf '%03d' "$CTRL_DATASET_ID")_${CTRL_DATASET_NAME}/${PLAN_NAME}.json"

# (0) install resampler + FOV modules so `-tr` + cascade resampler resolve at predict.
echo "[fovc-sweep] (0) install resampler + corrector/FOV modules into nnunetv2"
python3 - "$HERE/engine/corrector_resampling.py" <<'PY'
import sys, shutil, os, nnunetv2.preprocessing.resampling as r
shutil.copyfile(sys.argv[1], os.path.join(os.path.dirname(r.__file__), "corrector_resampling.py"))
PY
python3 - "$HERE/engine" <<'PY'
import sys, shutil, os
import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as m
pkg = os.path.dirname(m.__file__); eng = sys.argv[1]
required = ["nnUNetTrainer_corrector.py", "nnUNetTrainer_OrbitalCascade.py",
            "corrector_augment.py", "corrector_stratified_loader.py",
            "fov_completion_planner.py", "fov_completion_loader.py",
            "nnUNetTrainer_OrbitalFOVCompletion.py"]
missing = [n for n in required if not os.path.isfile(os.path.join(eng, n))]
if missing:
    raise FileNotFoundError(f"[fovc-sweep] required FOV runtime modules missing: {missing}")
for name in required:
    shutil.copyfile(os.path.join(eng, name), os.path.join(pkg, name)); print(f"[fovc-sweep] installed {name}")
PY

# (1) MANIFEST-DRIVEN validation evalset (exact FOV ids from splits_final; §6.2).
#     Writes imagesTs/ prevsegTs/ fovMaskTs/ eval_cases_map.json for the val fold.
echo "[fovc-sweep] (1) build_fov_completion_evalset (val fold $FOLD) -> $TEST_ROOT"
python3 "$HERE/scripts/build_fov_completion_evalset.py" \
    --completion-manifest "$FOV_MANIFEST" --splits-final "$SPLITS" --fold "$FOLD" --split val \
    --out "$TEST_ROOT" --control-name "$CTRL_DATASET_NAME" \
    ${FOVC_TRUNC_CT_DIR:+--truncated-ct-dir "$FOVC_TRUNC_CT_DIR"}
CDIR="$TEST_ROOT/$CTRL_DATASET_NAME"
IMAGES_TS="$CDIR/imagesTs"; PREVSEG_TS="$CDIR/prevsegTs"
FOVMASK_TS="$CDIR/fovMaskTs"; MAP_JSON="$CDIR/eval_cases_map.json"

# (2) snapshots (name-sorted -> epoch order).
MODEL_DIR="${nnUNet_results%/}/Dataset$(printf '%03d' "$CTRL_DATASET_ID")_${CTRL_DATASET_NAME}/${CORRECTOR_TRAINER}__${PLAN_NAME}__${CONFIGURATION}/fold_${FOLD}"
mapfile -t SNAPS < <(ls -1 "$MODEL_DIR"/checkpoint_epoch_*.pth 2>/dev/null | sort)
[[ "${#SNAPS[@]}" -gt 0 ]] || { echo "[fovc-sweep] ERROR: no checkpoint_epoch_*.pth in $MODEL_DIR" >&2; exit 1; }
echo "[fovc-sweep] (2) ${#SNAPS[@]} snapshot(s)"

# (3) per snapshot: provenance-gated predict -> strict FOV-mask eval -> append CSV.
LONG_CSV="$SWEEP_ROOT/metrics_long.csv"; rm -f "$LONG_CSV"
PROV="$HERE/diagnostics/fov_prediction_provenance.py"
for CKPT in "${SNAPS[@]}"; do
    BASE="$(basename "$CKPT")"; E="${BASE#checkpoint_epoch_}"; E="${E%.pth}"; EPOCH=$((10#$E))
    PRED_DIR="$SWEEP_ROOT/pred_epoch_$(printf '%04d' "$EPOCH")"; mkdir -p "$PRED_DIR"
    # provenance gate: only --continue_prediction when checkpoint+inputs match (P2-9).
    PREDICT_RESUME=""
    if [[ "${FORCE:-0}" != "1" ]] && python3 "$PROV" check --pred-dir "$PRED_DIR" \
            --checkpoint "$CKPT" --trainer "$CORRECTOR_TRAINER" --plans-file "$PLAN_JSON" \
            --fold "$FOLD" --case-map "$MAP_JSON" --completion-manifest "$FOV_MANIFEST" >/dev/null 2>&1; then
        PREDICT_RESUME="--continue_prediction"
    else
        rm -f "$PRED_DIR"/*.nii.gz 2>/dev/null || true      # stale/mismatched -> clean re-predict
    fi
    echo "[fovc-sweep] (3.$EPOCH) predict chk=$BASE ${PREDICT_RESUME:+(continue)}"
    nnUNetv2_predict -i "$IMAGES_TS" -o "$PRED_DIR" \
        -d "$CTRL_DATASET_ID" -c "$CONFIGURATION" -tr "$CORRECTOR_TRAINER" \
        -p "$PLAN_NAME" -f "$FOLD" -chk "$BASE" $PREDICT_RESUME \
        -prev_stage_predictions "$PREVSEG_TS"
    python3 "$PROV" write --pred-dir "$PRED_DIR" --checkpoint "$CKPT" \
        --trainer "$CORRECTOR_TRAINER" --plans-file "$PLAN_JSON" --fold "$FOLD" \
        --case-map "$MAP_JSON" --completion-manifest "$FOV_MANIFEST" >/dev/null
    python3 "$HERE/diagnostics/fov_completion_eval.py" \
        --map "$MAP_JSON" --pred-dir "$PRED_DIR" --fov-mask-dir "$FOVMASK_TS" \
        --completion-manifest "$FOV_MANIFEST" --epoch "$EPOCH" --out-csv "$LONG_CSV" --append
done
echo "[fovc-sweep] long metrics -> $LONG_CSV"

# (4) select the checkpoint (subject-level; hallucination + visible/full guardrails).
FINAL_ARG=""; [[ "${FOVC_FINAL:-0}" == "1" ]] && FINAL_ARG="--final"
echo "[fovc-sweep] (4) select checkpoint (final=${FOVC_FINAL:-0})"
python3 "$HERE/diagnostics/select_fov_checkpoint_driver.py" \
    --metrics-csv "$LONG_CSV" --plans-file "$PLAN_JSON" --configuration "$CONFIGURATION" \
    --min-missing-mm3 "${FOVC_MIN_MISSING_MM3:-0}" --min-visible-mm3 "${FOVC_MIN_VISIBLE_MM3:-0}" \
    --expect-structures "${FOVC_STRUCTURES:-ON,Recti,Globe,Fat}" $FINAL_ARG \
    --out-scores-csv "$SWEEP_ROOT/checkpoint_scores.csv"
echo "[fovc-sweep] done: scores -> $SWEEP_ROOT/checkpoint_scores.csv"
