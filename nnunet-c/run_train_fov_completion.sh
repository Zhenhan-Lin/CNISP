#!/usr/bin/env bash
# Finetune the FOV-completion corrector (control C on the FOV-completion dataset)
# from the Dataset835 weights. Mirrors run_train.sh, adding:
#   * step 4b: write class_locations_fov into each preprocessed properties.pkl
#              (projects visible_box onto the preprocessed grid);
#   * step 6b: also install the FOV loader/planner/trainer into nnunetv2;
#   * FOV env: trainer, save cadence, FOV-safe prior dropout, region weights.
#
# Prereqs (run first, per the corrector RUNBOOK):
#   * build_fov_completion_data.py    -> truncated CTs + fov_completion_manifest.json
#   * 032 CNISP (control C) on those  -> CNISP prior
#   * build_corrector_dataset --layout cascade  -> Dataset8xx (main + prior) + relocate_prevseg
#   * nnUNetv2 preprocess with nnUNetPlansFinetune  (SKIP_PREPROCESS handles reuse)
#
# Usage:
#   FOV_MANIFEST=/path/fov_completion_manifest.json bash nnunet-c/run_train_fov_completion.sh C 0
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CONTROL="${1:?usage: run_train_fov_completion.sh <C> <fold>}"
FOLD="${2:?usage: run_train_fov_completion.sh <C> <fold>}"
CONFIG="${CONFIG:-$HERE/configs/corrector_fov.yaml}"
PLAN_NAME="${PLAN_NAME:-nnUNetPlansFinetune}"
MASK_INIT="${MASK_INIT:-zero}"
WORK_TMP="${WORK_TMP:-$HERE/staging/_finetune_fov}"
CASCADE=1                                  # FOV completion always uses Route A
export nnUNet_compile="${nnUNet_compile:-f}"

# ── FOV-completion trainer + sampler env ─────────────────────────────────────
export CORRECTOR_TRAINER="${CORRECTOR_TRAINER:-nnUNetTrainer_OrbitalFOVCompletion}"
export CORRECTOR_FOV_COMPLETION="${CORRECTOR_FOV_COMPLETION:-1}"
export CORRECTOR_SAVE_EVERY="${CORRECTOR_SAVE_EVERY:-25}"
export CORRECTOR_FOV_ANCHOR_FULL_PROB="${CORRECTOR_FOV_ANCHOR_FULL_PROB:-0.5}"
export CORRECTOR_DROP_ALL="${CORRECTOR_DROP_ALL:-0.0}"     # FOV-safe: no full-prior dropout
export CORRECTOR_DROP_EACH="${CORRECTOR_DROP_EACH:-0.10}"  # FOV-safe: reduced per-channel dropout
FOV_MANIFEST="${FOV_MANIFEST:?export FOV_MANIFEST=/path/to/fov_completion_manifest.json}"
export CORRECTOR_FOV_MANIFEST="$FOV_MANIFEST"

echo "================================================================"
echo "[fov-train] control=$CONTROL fold=$FOLD trainer=$CORRECTOR_TRAINER"
echo "            save_every=$CORRECTOR_SAVE_EVERY drop_all=$CORRECTOR_DROP_ALL drop_each=$CORRECTOR_DROP_EACH"
echo "================================================================"

eval "$(python3 "$HERE/scripts/corrector_env.py" --config "$CONFIG" --control "$CONTROL")"
: "${nnUNet_preprocessed:?export nnUNet_preprocessed}"
: "${nnUNet_results:?export nnUNet_results}"
[[ -f "$REF_CKPT" ]] || { echo "[fov-train] ERROR: 835 ckpt not found: $REF_CKPT" >&2; exit 1; }

# (0) per-channel resampler (idempotent) -- FOV cascade rides the seg resampler,
# but install unconditionally so validation's end-of-train predict can import it.
echo "[fov-train] (0) install per-channel resampler into nnunetv2"
python3 - "$HERE/engine/corrector_resampling.py" <<'PY'
import sys, shutil, os, nnunetv2.preprocessing.resampling as r
shutil.copyfile(sys.argv[1], os.path.join(os.path.dirname(r.__file__), "corrector_resampling.py"))
PY

# (1-4) fingerprint/plan/merge/preprocess -- reuse run_train.sh's logic; the FOV
# cascade prep (2 datasets + relocate_prevseg) is done offline, so default SKIP.
if [[ "${SKIP_PREPROCESS:-1}" == "1" ]]; then
    echo "[fov-train] SKIP_PREPROCESS=1 -> reuse existing preprocessed data"
else
    echo "[fov-train] ERROR: build the FOV cascade datasets offline, then re-run with SKIP_PREPROCESS=1" >&2
    exit 2
fi

# resolve the preprocessed data-identifier dir for the post-pass (control C dataset).
PPDIR="$nnUNet_preprocessed/Dataset$(printf '%03d' "$CTRL_DATASET_ID")_${CTRL_DATASET_NAME}/${PLAN_NAME}_${CONFIGURATION}"
PLAN_JSON="$nnUNet_preprocessed/Dataset$(printf '%03d' "$CTRL_DATASET_ID")_${CTRL_DATASET_NAME}/${PLAN_NAME}.json"

# (4b) POST-PASS: write class_locations_fov (audit ONE case first, then all).
echo "[fov-train] (4b) audit one case, then write class_locations_fov -> $PPDIR"
python3 "$HERE/scripts/write_class_locations_fov.py" --audit-one \
    --data-dir "$PPDIR" --completion-manifest "$FOV_MANIFEST" \
    --plans-file "$PLAN_JSON" --configuration "$CONFIGURATION"
if [[ "${FOV_AUDIT_ONLY:-0}" == "1" ]]; then
    echo "[fov-train] FOV_AUDIT_ONLY=1 -> stop after the audit (inspect the projection)."
    exit 0
fi
python3 "$HERE/scripts/write_class_locations_fov.py" \
    --data-dir "$PPDIR" --completion-manifest "$FOV_MANIFEST" \
    --plans-file "$PLAN_JSON" --configuration "$CONFIGURATION" \
    --seam-width-voxels "${CORRECTOR_FOV_SEAM_WIDTH:-12}"

# (5) POTHOLE-4 gate under the FOV trainer's seg_prev folder.
echo "[fov-train] (5) check_preprocessed"
python3 "$HERE/diagnostics/check_preprocessed.py" \
    --config "$CONFIG" --control "$CONTROL" --plan-name "$PLAN_NAME" \
    --cascade --trainer "$CORRECTOR_TRAINER"

# (6) first-conv surgery 1ch -> 5ch
mkdir -p "$WORK_TMP"
ADAPTED="$WORK_TMP/ckpt_${REF_DATASET_ID}_to${N_CHANNELS}ch_${CONTROL}.pth"
echo "[fov-train] (6) adapt_checkpoint 1ch->${N_CHANNELS}ch (mask_init=$MASK_INIT)"
python3 "$HERE/scripts/adapt_checkpoint.py" --in "$REF_CKPT" --out "$ADAPTED" \
    --channels "$N_CHANNELS" --mask-init "$MASK_INIT"

# (6b) install corrector + FOV runtime modules into nnunetv2 (import siblings).
echo "[fov-train] (6b) install corrector + FOV modules into nnunetv2"
python3 - "$HERE/engine" <<'PY'
import sys, shutil, os
import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as m
pkg = os.path.dirname(m.__file__); eng = sys.argv[1]
mods = ["nnUNetTrainer_corrector.py", "nnUNetTrainer_OrbitalCascade.py",
        "corrector_augment.py", "corrector_stratified_loader.py",
        "fov_completion_planner.py", "fov_completion_loader.py",
        "nnUNetTrainer_OrbitalFOVCompletion.py"]
for name in mods:
    src = os.path.join(eng, name)
    if os.path.isfile(src):
        shutil.copyfile(src, os.path.join(pkg, name)); print(f"[fov-train] installed {name}")
    else:
        print(f"[fov-train] (skip) {name} not present")
PY

export CORRECTOR_EPOCHS CORRECTOR_LR
# (7) train
if [[ "${RESUME:-0}" == "1" ]]; then
    echo "[fov-train] (7) RESUME nnUNetv2_train --c"
    nnUNetv2_train "$CTRL_DATASET_ID" "$CONFIGURATION" "$FOLD" -p "$PLAN_NAME" -tr "$CORRECTOR_TRAINER" --c
else
    echo "[fov-train] (7) nnUNetv2_train (trainer=$CORRECTOR_TRAINER epochs=$CORRECTOR_EPOCHS lr=$CORRECTOR_LR)"
    nnUNetv2_train "$CTRL_DATASET_ID" "$CONFIGURATION" "$FOLD" \
        -p "$PLAN_NAME" -tr "$CORRECTOR_TRAINER" -pretrained_weights "$ADAPTED"
fi
echo "[fov-train] done: Dataset${CTRL_DATASET_ID} fold $FOLD"
