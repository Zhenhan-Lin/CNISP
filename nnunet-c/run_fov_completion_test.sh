#!/usr/bin/env bash
# FOV-completion FINAL HELD-OUT TEST (revised-plan §7) — SEPARATE from the validation
# sweep. It runs the ALREADY-SELECTED checkpoint ONCE on the held-out test subjects,
# performs NO checkpoint selection, changes NO thresholds, and alters NO crop
# generation. It then evaluates the fixed comparison arms and runs subject-level
# paired statistics.
#
# Split integrity (§7, enforced): train ∩ val = ∅, train ∩ test = ∅, val ∩ test = ∅
# (subject-level). The evalset builder asserts the test subjects are disjoint from
# train+val; this script refuses to run otherwise.
#
# Comparison arms (§9), each scored on the IDENTICAL test cases:
#   corrector  = the selected FOV corrector's prediction   (predicted here)
#   cnisp      = CNISP prior alone                          ($FOVC_CNISP_PRED_DIR)
#   stage1     = stage-1 nnU-Net prediction (optional)      ($FOVC_STAGE1_PRED_DIR)
#   full       = full-FOV condition rows (inside the same run)
#
# Output under a TEST-distinct root (never the validation sweep's):
#   fov_completion_final_test/  imagesTs/prevsegTs/fovMaskTs/eval map, pred/, metrics + stats
#
# Usage:
#   SELECTED_CKPT=/.../checkpoint_epoch_0125.pth \
#   FOV_MANIFEST=/p/fov_completion_manifest.json \
#   TEST_CASE_LIST=/p/test_cases.txt TRAINVAL_CASE_LIST=/p/trainval_cases.txt \
#   FOVC_CNISP_PRED_DIR=/p/cnisp_pred \
#       bash nnunet-c/run_fov_completion_test.sh C 0
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CONTROL="${1:?usage: run_fov_completion_test.sh <C> <fold>}"
FOLD="${2:?usage: run_fov_completion_test.sh <C> <fold>}"
CONFIG="${CONFIG:-$HERE/configs/corrector_fov.yaml}"
PLAN_NAME="${PLAN_NAME:-nnUNetPlansFinetune}"
export CORRECTOR_TRAINER="${CORRECTOR_TRAINER:-nnUNetTrainer_OrbitalFOVCompletion}"
export nnUNet_compile="${nnUNet_compile:-f}"
SELECTED_CKPT="${SELECTED_CKPT:?export SELECTED_CKPT=/path/to/checkpoint_epoch_XXXX.pth (from the sweep)}"
FOV_MANIFEST="${FOV_MANIFEST:?export FOV_MANIFEST=/path/to/fov_completion_manifest.json}"
TEST_CASE_LIST="${TEST_CASE_LIST:?export TEST_CASE_LIST=/path/to/held-out test case ids}"
TRAINVAL_CASE_LIST="${TRAINVAL_CASE_LIST:?export TRAINVAL_CASE_LIST=/path/to/train+val case ids (disjoint check)}"
[[ -f "$SELECTED_CKPT" ]] || { echo "[fovc-test] ERROR: checkpoint not found: $SELECTED_CKPT" >&2; exit 1; }

TEST_ROOT="${FOVC_TEST_ROOT:-$HERE/fov_completion_final_test}"
mkdir -p "$TEST_ROOT"

echo "================================================================"
echo "[fovc-test] FINAL held-out test control=$CONTROL fold=$FOLD"
echo "[fovc-test] selected checkpoint = $SELECTED_CKPT (run ONCE, no selection)"
echo "================================================================"

eval "$(python3 "$HERE/scripts/corrector_env.py" --config "$CONFIG" --control "$CONTROL")"
: "${nnUNet_results:?export nnUNet_results}"
: "${nnUNet_preprocessed:?export nnUNet_preprocessed}"
PLAN_JSON="$nnUNet_preprocessed/Dataset$(printf '%03d' "$CTRL_DATASET_ID")_${CTRL_DATASET_NAME}/${PLAN_NAME}.json"

# (0) install resampler + FOV modules (fail-fast).
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
    raise FileNotFoundError(f"[fovc-test] required FOV runtime modules missing: {missing}")
for name in required:
    shutil.copyfile(os.path.join(eng, name), os.path.join(pkg, name))
PY

# (1) manifest-driven TEST evalset from the explicit held-out case list, with the
#     SUBJECT-DISJOINT assertion against train+val (§7). Builder raises on overlap.
echo "[fovc-test] (1) build_fov_completion_evalset (held-out test; assert disjoint from train+val)"
python3 "$HERE/scripts/build_fov_completion_evalset.py" \
    --completion-manifest "$FOV_MANIFEST" --case-list "$TEST_CASE_LIST" \
    --assert-disjoint-with "$TRAINVAL_CASE_LIST" \
    --out "$TEST_ROOT" --control-name "$CTRL_DATASET_NAME" \
    ${FOVC_TRUNC_CT_DIR:+--truncated-ct-dir "$FOVC_TRUNC_CT_DIR"}
CDIR="$TEST_ROOT/$CTRL_DATASET_NAME"
IMAGES_TS="$CDIR/imagesTs"; PREVSEG_TS="$CDIR/prevsegTs"
FOVMASK_TS="$CDIR/fovMaskTs"; MAP_JSON="$CDIR/eval_cases_map.json"

# (2) predict ONCE with the selected checkpoint (+ provenance).
PRED_DIR="$TEST_ROOT/pred_corrector"; mkdir -p "$PRED_DIR"
CHK_NAME="$(basename "$SELECTED_CKPT")"
echo "[fovc-test] (2) predict (selected checkpoint only) -> $PRED_DIR"
nnUNetv2_predict -i "$IMAGES_TS" -o "$PRED_DIR" \
    -d "$CTRL_DATASET_ID" -c "$CONFIGURATION" -tr "$CORRECTOR_TRAINER" \
    -p "$PLAN_NAME" -f "$FOLD" -chk "$CHK_NAME" \
    -prev_stage_predictions "$PREVSEG_TS"
python3 "$HERE/diagnostics/fov_prediction_provenance.py" write --pred-dir "$PRED_DIR" \
    --checkpoint "$SELECTED_CKPT" --trainer "$CORRECTOR_TRAINER" --plans-file "$PLAN_JSON" \
    --fold "$FOLD" --case-map "$MAP_JSON" --completion-manifest "$FOV_MANIFEST" >/dev/null

# (3) STRICT eval of each comparison arm on the IDENTICAL cases (whole-volume surface
#     metrics ON for the final report). epoch column carries no meaning here (0).
echo "[fovc-test] (3) strict FOV eval — comparison arms"
CORR_CSV="$TEST_ROOT/metrics_corrector.csv"
python3 "$HERE/diagnostics/fov_completion_eval.py" \
    --map "$MAP_JSON" --pred-dir "$PRED_DIR" --fov-mask-dir "$FOVMASK_TS" \
    --completion-manifest "$FOV_MANIFEST" --epoch 0 --out-csv "$CORR_CSV" --whole-surface
ARMS=("corrector:$CORR_CSV")
if [[ -n "${FOVC_CNISP_PRED_DIR:-}" ]]; then
    CNISP_CSV="$TEST_ROOT/metrics_cnisp.csv"
    python3 "$HERE/diagnostics/fov_completion_eval.py" \
        --map "$MAP_JSON" --pred-dir "$FOVC_CNISP_PRED_DIR" --fov-mask-dir "$FOVMASK_TS" \
        --completion-manifest "$FOV_MANIFEST" --epoch 0 --out-csv "$CNISP_CSV" --whole-surface
    ARMS+=("cnisp:$CNISP_CSV")
fi
if [[ -n "${FOVC_STAGE1_PRED_DIR:-}" ]]; then
    STAGE1_CSV="$TEST_ROOT/metrics_stage1.csv"
    python3 "$HERE/diagnostics/fov_completion_eval.py" \
        --map "$MAP_JSON" --pred-dir "$FOVC_STAGE1_PRED_DIR" --fov-mask-dir "$FOVMASK_TS" \
        --completion-manifest "$FOV_MANIFEST" --epoch 0 --out-csv "$STAGE1_CSV" --whole-surface
    ARMS+=("stage1:$STAGE1_CSV")
fi
echo "[fovc-test] arms: ${ARMS[*]}"

# (4) subject-level paired statistics vs each baseline arm (§9/§10).
if [[ -n "${FOVC_CNISP_PRED_DIR:-}" ]]; then
    for METRIC in missing_dice visible_dice missing_fp_voxels whole_dice; do
        echo "[fovc-test] (4) paired corrector - cnisp on $METRIC (by subject)"
        python3 "$HERE/diagnostics/fov_completion_stats.py" \
            --corrector-csv "$CORR_CSV" --baseline-csv "$TEST_ROOT/metrics_cnisp.csv" \
            --baseline-name cnisp --metric "$METRIC" \
            --out-csv "$TEST_ROOT/delta_cnisp_${METRIC}.csv"
    done
fi
echo "[fovc-test] done: metrics + deltas under $TEST_ROOT"
