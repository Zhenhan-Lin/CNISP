#!/usr/bin/env bash
# FOV-completion checkpoint SWEEP — the "test pipeline". For each periodic snapshot
# (checkpoint_epoch_XXXX.pth) of the FOV-completion corrector, it runs whole-volume
# prediction on the FIXED 7-condition VALIDATION set, does region-split eval
# (missing / visible / full-FOV) -> the long metrics CSV, then selects the
# checkpoint via diagnostics/select_fov_checkpoint_driver.py.
#
# It mirrors the corrector framework:
#   * predict   -> run_corrector_predict.sh  (corrector_env + build_corrector_testset
#                  + nnUNetv2_predict with the cascade -prev_stage_predictions prior)
#   * per-snap  -> diagnostics/select_checkpoint.py  (glob snapshots, predict+eval each)
#   * eval      -> diagnostics/fov_completion_eval.py (region-split Dice + gt-voxels)
#   * select    -> diagnostics/select_fov_checkpoint_driver.py
#
# Output lives under FOV-COMPLETION-DISTINCT roots so it never collides with the
# earlier corrector / per-eye truncation experiments (which use test_input/,
# predictions/, data_fov_pereye_test/):
#   fov_completion_test_input/   imagesTs + prevsegTs + test_cases_map.json
#   fov_completion_sweep/        pred_epoch_XXXX/ , eval per epoch, metrics_long.csv,
#                                checkpoint_scores.csv (the selection)
#
# Prereqs (same as run_train_fov_completion.sh): the FOV-completion dataset is built
# offline (build_fov_completion_data -> CNISP prior -> build_corrector_dataset
# --layout cascade -> preprocess) and the model has trained with periodic snapshots.
#
# Usage:
#   FOV_MANIFEST=/path/fov_completion_manifest.json \
#       bash nnunet-c/run_fov_completion_sweep.sh C 0
#   # paper selection (strict guardrail + physical floors + coverage):
#   FOVC_FINAL=1 FOVC_MIN_MISSING_MM3=3 FOVC_MIN_VISIBLE_MM3=3 \
#   FOV_MANIFEST=/path/fov_completion_manifest.json \
#       bash nnunet-c/run_fov_completion_sweep.sh C 0
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CONTROL="${1:?usage: run_fov_completion_sweep.sh <C> <fold>}"
FOLD="${2:?usage: run_fov_completion_sweep.sh <C> <fold>}"
CONFIG="${CONFIG:-$HERE/configs/corrector_fov.yaml}"
PLAN_NAME="${PLAN_NAME:-nnUNetPlansFinetune}"
export CORRECTOR_TRAINER="${CORRECTOR_TRAINER:-nnUNetTrainer_OrbitalFOVCompletion}"
export nnUNet_compile="${nnUNet_compile:-f}"
FOV_MANIFEST="${FOV_MANIFEST:?export FOV_MANIFEST=/path/to/fov_completion_manifest.json}"
export CORRECTOR_FOV_MANIFEST="$FOV_MANIFEST"

# FOV-completion-distinct output roots (the *_ folders; overridable).
TEST_ROOT="${FOVC_TEST_ROOT:-$HERE/fov_completion_test_input}"
SWEEP_ROOT="${FOVC_SWEEP_ROOT:-$HERE/fov_completion_sweep}"
mkdir -p "$SWEEP_ROOT"

echo "================================================================"
echo "[fovc-sweep] control=$CONTROL fold=$FOLD trainer=$CORRECTOR_TRAINER"
echo "[fovc-sweep] test_input=$TEST_ROOT  sweep=$SWEEP_ROOT"
echo "================================================================"

eval "$(python3 "$HERE/scripts/corrector_env.py" --config "$CONFIG" --control "$CONTROL")"
: "${nnUNet_results:?export nnUNet_results}"
: "${nnUNet_preprocessed:?export nnUNet_preprocessed}"

GRID="${GRID:-${PREDICT_GRID:-iso}}"
ISO_MM="${ISO_MM:-${PREDICT_ISO_MM:-0.5}}"

# (0) install the per-channel resampler + FOV runtime modules into nnunetv2 so
#     `-tr nnUNetTrainer_OrbitalFOVCompletion` and the cascade seg resampler resolve
#     at PREDICT time (identical to the train script's step 0 + 6b).
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

# (1) build the VALIDATION testset ONCE (cascade -> 1-ch CT imagesTs + prevsegTs/
#     prior + test_cases_map.json). FOVC_VAL_CASEFILE should list THIS fold's
#     validation cases (the 7 conditions of the held-out subjects); without it the
#     config default casefile is used and the eval scores whatever cases the manifest
#     also covers. (fov_completion_eval joins on case_id, so extra cases are ignored.)
CASCADE=1
LAYOUT_ARG="--layout cascade"
PREVSEG_TS="$TEST_ROOT/$CTRL_DATASET_NAME/prevsegTs"
SKIP_EXISTING_ARG="--skip-existing"
[[ "${REBUILD_TESTSET:-0}" == "1" ]] && SKIP_EXISTING_ARG=""
echo "[fovc-sweep] (1) build_corrector_testset -> $TEST_ROOT"
python3 "$HERE/scripts/build_corrector_testset.py" \
    --config "$CONFIG" --control "$CONTROL" --steps "${BUILD_STEPS:-auto}" \
    --prelabel-grid "$GRID" --iso-mm "$ISO_MM" --out "$TEST_ROOT" $SKIP_EXISTING_ARG $LAYOUT_ARG \
    ${FOVC_VAL_CASEFILE:+--casefile "$FOVC_VAL_CASEFILE"}

IMAGES_TS="$TEST_ROOT/$CTRL_DATASET_NAME/imagesTs"
MAP_JSON="$TEST_ROOT/$CTRL_DATASET_NAME/test_cases_map.json"

# (2) resolve the periodic snapshots (name-sorted -> epoch order; zero-padded).
MODEL_DIR="${nnUNet_results%/}/Dataset$(printf '%03d' "$CTRL_DATASET_ID")_${CTRL_DATASET_NAME}/${CORRECTOR_TRAINER}__${PLAN_NAME}__${CONFIGURATION}/fold_${FOLD}"
echo "[fovc-sweep] (2) snapshots in $MODEL_DIR"
mapfile -t SNAPS < <(ls -1 "$MODEL_DIR"/checkpoint_epoch_*.pth 2>/dev/null | sort)
if [[ "${#SNAPS[@]}" -eq 0 ]]; then
    echo "[fovc-sweep] ERROR: no checkpoint_epoch_*.pth in $MODEL_DIR" >&2
    echo "             (train with CORRECTOR_SAVE_EVERY set so snapshots are written)." >&2
    exit 1
fi
echo "[fovc-sweep] found ${#SNAPS[@]} snapshot(s)"

# (3) per snapshot: predict -> region-split eval -> append the long metrics CSV.
LONG_CSV="$SWEEP_ROOT/metrics_long.csv"
rm -f "$LONG_CSV"
for CKPT in "${SNAPS[@]}"; do
    BASE="$(basename "$CKPT")"                       # checkpoint_epoch_XXXX.pth
    E="${BASE#checkpoint_epoch_}"; E="${E%.pth}"     # XXXX (zero-padded)
    EPOCH=$((10#$E))                                 # force base-10 (strip leading zeros)
    PRED_DIR="$SWEEP_ROOT/pred_epoch_$(printf '%04d' "$EPOCH")"
    mkdir -p "$PRED_DIR"
    echo "[fovc-sweep] (3.$EPOCH) predict chk=$BASE -> $PRED_DIR"
    PREDICT_RESUME="--continue_prediction"; [[ "${FORCE:-0}" == "1" ]] && PREDICT_RESUME=""
    nnUNetv2_predict \
        -i "$IMAGES_TS" -o "$PRED_DIR" \
        -d "$CTRL_DATASET_ID" -c "$CONFIGURATION" -tr "$CORRECTOR_TRAINER" \
        -p "$PLAN_NAME" -f "$FOLD" -chk "$BASE" $PREDICT_RESUME \
        -prev_stage_predictions "$PREVSEG_TS"
    echo "[fovc-sweep] (3.$EPOCH) region-split eval -> $LONG_CSV"
    python3 "$HERE/diagnostics/fov_completion_eval.py" \
        --map "$MAP_JSON" --pred-dir "$PRED_DIR" \
        --completion-manifest "$FOV_MANIFEST" --epoch "$EPOCH" \
        --out-csv "$LONG_CSV" --append
done
echo "[fovc-sweep] long metrics table -> $LONG_CSV"

# (4) select the checkpoint (physical floors via the plan; strict when FOVC_FINAL=1).
PLAN_JSON="$nnUNet_preprocessed/Dataset$(printf '%03d' "$CTRL_DATASET_ID")_${CTRL_DATASET_NAME}/${PLAN_NAME}.json"
FINAL_ARG=""; [[ "${FOVC_FINAL:-0}" == "1" ]] && FINAL_ARG="--final"
echo "[fovc-sweep] (4) select checkpoint (final=${FOVC_FINAL:-0})"
python3 "$HERE/diagnostics/select_fov_checkpoint_driver.py" \
    --metrics-csv "$LONG_CSV" \
    --plans-file "$PLAN_JSON" --configuration "$CONFIGURATION" \
    --min-missing-mm3 "${FOVC_MIN_MISSING_MM3:-0}" --min-visible-mm3 "${FOVC_MIN_VISIBLE_MM3:-0}" \
    --expect-structures "${FOVC_STRUCTURES:-ON,Recti,Globe,Fat}" $FINAL_ARG \
    --out-scores-csv "$SWEEP_ROOT/checkpoint_scores.csv"
echo "[fovc-sweep] done: scores -> $SWEEP_ROOT/checkpoint_scores.csv"
