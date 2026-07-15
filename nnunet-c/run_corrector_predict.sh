#!/usr/bin/env bash
# Predict a trained nnUNet-C corrector (B=855 or C=845) on the CNISP TEST set.
#
# Assumes the test cases already have degraded CTs + Dataset835 sparse preds +
# canonical-aligned patches from the earlier work_dir/run_pipeline sweep.
#
# Stages:
#   1. (control C, RUN_CNISP=1) CNISP test inference via the existing 032 launcher
#      -> <CNISP_TEST_DIR>/<gtstem>_step{XX}.nii.gz   ({1,2,3,4} native masks)
#   2. install the per-channel resampler into nnunetv2 (ch0 order3, ch1-4 order0)
#   3. build_corrector_testset.py -> nnunet-c/test_input/<name>/imagesTs (5-ch)
#   4. nnUNetv2_predict -d <id> -p nnUNetPlansFinetune -chk <CHK> -f <fold>
#
# Usage:
#   bash nnunet-c/run_corrector_predict.sh C 0
#   RUN_CNISP=0 bash nnunet-c/run_corrector_predict.sh B 0          # B has no CNISP step
#   GPUS="0 1" CHK=checkpoint_best.pth bash nnunet-c/run_corrector_predict.sh C 0
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CONTROL="${1:?usage: run_corrector_predict.sh <B|C> <fold>}"
FOLD="${2:?usage: run_corrector_predict.sh <B|C> <fold>}"
CONFIG="${CONFIG:-$HERE/configs/corrector.yaml}"
PLAN_NAME="${PLAN_NAME:-nnUNetPlansFinetune}"
# CASCADE=1 -> native-cascade (Route A) predict: build a 1-ch CT testset + a
# prevsegTs/ dir of {cid}.nii.gz CNISP prior masks, and pass that dir to
# nnUNetv2_predict via -prev_stage_predictions (nnUNet one-hots the prior itself).
# Set CORRECTOR_TRAINER=nnUNetTrainer_OrbitalCascade to match the trained model.
CASCADE="${CASCADE:-0}"
# Two DIFFERENT checkpoints (do not conflate):
#   CHK       = nnUNet-C predict checkpoint -> best (the finetuned corrector)
#   CNISP_CHK = CNISP test-inference checkpoint -> latest, to MATCH the training
#               prelabels (those were generated with CNISP 'latest').
CHK="${CHK:-checkpoint_best.pth}"     # nnUNet-C predict checkpoint (best)
CNISP_CHK="${CNISP_CHK:-latest}"      # CNISP test checkpoint (latest; matches train prelabels)
GPUS="${GPUS:-0 1}"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
CNISP_DIR="$REPO_ROOT/orbital_shape_prior_st1"
export nnUNet_compile="${nnUNet_compile:-f}"
ISO_PRELABEL_DIR="${ISO_PRELABEL_DIR:-$HERE/data/cnisp_pred_test_iso}"
# GRID / RUN_CNISP / EMIT_ISO / ISO_MM defaults come from corrector.yaml `predict:`
# (see corrector_env PREDICT_*); set AFTER the eval below. All env-overridable.

echo "================================================================"
echo "[predict] control=$CONTROL fold=$FOLD"
echo "[predict] nnUNet-C ckpt=$CHK   CNISP ckpt=$CNISP_CHK"
echo "================================================================"

eval "$(python3 "$HERE/scripts/corrector_env.py" --config "$CONFIG" --control "$CONTROL")"

# ── Prediction defaults from corrector.yaml `predict:` (overridable by env) ──
# GRID=iso -> assemble on CNISP's iso-0.5 DENSE head grid, so ch1..4 are never the
# degraded/native (sparse, zero-gapped) mask. RUN_CNISP/EMIT_ISO auto -> 1 for
# control C, 0 for B (B never runs CNISP).
GRID="${GRID:-${PREDICT_GRID:-iso}}"
RUN_CNISP="${RUN_CNISP:-${PREDICT_RUN_CNISP:-auto}}"
EMIT_ISO="${EMIT_ISO:-${PREDICT_EMIT_ISO:-auto}}"
# iso spacing for the CNISP emit + testset assembly. DEFAULT: the 835 iso plan
# spacing (resolve_target_spacing -> e.g. 0.4765625), so emit/test match the
# train builder + network plan. Falls back to 0.5 only if the plan can't be read.
# Override with ISO_MM=<mm> (or PREDICT_ISO_MM in the config predict block).
ISO_MM="${ISO_MM:-${PREDICT_ISO_MM:-}}"
if [[ -z "$ISO_MM" ]]; then
    ISO_MM="$(python3 - <<PY 2>/dev/null || true
import sys
sys.path.insert(0, "$HERE")
from lib.config import load_corrector_config
from lib.resample import resolve_target_spacing
cfg = load_corrector_config("$CONFIG", caller_file="$HERE/run_corrector_predict.sh")
print(f"{float(resolve_target_spacing(cfg)[0]):.7f}")
PY
)"
fi
ISO_MM="${ISO_MM:-0.4765625}"   # fallback = 835 iso plan (nnUNetPlans_iso05) spacing
echo "[predict] iso spacing (emit + testset assembly) = $ISO_MM mm"

if [[ "$EXTERNAL" == "1" ]]; then
    echo "[predict] control $CONTROL is external (Dataset$CTRL_DATASET_ID = pure nnUNet"
    echo "          on the degraded test CTs); predict it with the stock 835 model, e.g.:"
    echo "  nnUNetv2_predict -d $CTRL_DATASET_ID -c $CONFIGURATION -tr $TRAINER -p $REF_PLAN -f $REF_FOLD \\"
    echo "    -i <degraded test CTs> -o <out>"
    exit 0
fi
: "${nnUNet_results:?export nnUNet_results}"

if [[ "$RUN_CNISP" == "auto" ]]; then
    [[ "$PRELABEL_SOURCE" == "cnisp" ]] && RUN_CNISP=1 || RUN_CNISP=0
fi

# iso-0.5 prelabels are REQUIRED when the build consumes the iso grid for C.
if [[ "$EMIT_ISO" == "auto" ]]; then
    [[ "$PRELABEL_SOURCE" == "cnisp" && "$GRID" == "iso" ]] && EMIT_ISO=1 || EMIT_ISO=0
fi
ISO_ARGS=""
[[ "$EMIT_ISO" == "1" ]] && ISO_ARGS="--emit-iso-prelabel-dir $ISO_PRELABEL_DIR --emit-iso-mm $ISO_MM"
echo "[predict] GRID=$GRID EMIT_ISO=$EMIT_ISO  (iso dir: $ISO_PRELABEL_DIR)"

# RESUME_FROM_LATENT=1 -> the CNISP test run reuses cached preds/latents and
# re-runs ONLY the native/iso mapping (no latent optimization). Use to
# regenerate masks after the mapping-side fix + observed-metadata regeneration.
RESUME_FROM_LATENT="${RESUME_FROM_LATENT:-0}"
RESUME_ARGS=""
[[ "$RESUME_FROM_LATENT" == "1" ]] && RESUME_ARGS="--resume-from-latent"

# ── single-image debug mode ──────────────────────────────────────────
# SOURCE=<source_id> (or BUILD_CASEFILE=<path>) restricts the test build to ONE
# image (all its steps via --steps auto, or BUILD_STEPS), writing to ISOLATED
# test_input_single/ + predictions_single/ so the full build is never clobbered.
# Pair it with RUN_CNISP=0 (reuse the existing CNISP/native preds; a single image
# does not warrant re-running a full CNISP test sweep).
SOURCE="${SOURCE:-}"
BUILD_CASEFILE="${BUILD_CASEFILE:-}"
TEST_ROOT="$HERE/test_input"
PRED_ROOT="$HERE/predictions"
SINGLE_MODE=0
if [[ -n "$SOURCE" || -n "$BUILD_CASEFILE" ]]; then
    SINGLE_MODE=1
    TEST_ROOT="$HERE/test_input_single"
    PRED_ROOT="$HERE/predictions_single"
    if [[ -z "$BUILD_CASEFILE" ]]; then
        BUILD_CASEFILE="$(mktemp "${TMPDIR:-/tmp}/corr_one_XXXXXX.txt")"
        printf '%s\n' "$SOURCE" > "$BUILD_CASEFILE"
    fi
    echo "[predict] SINGLE-IMAGE mode: casefile=$BUILD_CASEFILE"
    echo "          -> isolated outputs under $TEST_ROOT and $PRED_ROOT"
fi

# ── 1. CNISP test inference = CNISP's OWN thick nnunet_pred deployment run ─
# (03_infer.py: test_cases.txt + adaptive sweep from the test yaml; outputs to
#  runs/$EXPERIMENT/$RUN_TAG/native_space_step_XX/.) nnUNet-C only consumes it.
if [[ "$RUN_CNISP" == "1" ]]; then
    echo "[predict] (1) CNISP thick nnunet_pred test (03_infer) -> runs/$EXPERIMENT/$RUN_TAG"
    PYTHONPATH="$CNISP_DIR:$REPO_ROOT:${PYTHONPATH:-}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPUS%% *}}" \
    python3 "$CNISP_DIR/scripts/03_infer.py" \
        -p "$CNISP_DIR/configs/paths.yaml" \
        -t "$CNISP_DIR/configs/$CNISP_TRAIN_YAML" \
        -c "$CNISP_DIR/configs/$CNISP_TEST_YAML" \
        -m "$CNISP_MODEL_NAME" --checkpoint "$CNISP_CHK" \
        --test-label-source nnunet_pred --run-tag "$RUN_TAG" --experiment "$EXPERIMENT" \
        $ISO_ARGS $RESUME_ARGS
else
    echo "[predict] (1) skip CNISP test inference (RUN_CNISP=0; using existing runs/$EXPERIMENT/$RUN_TAG)"
fi

# ── 2. install per-channel resampler (predict-time preprocess needs it) ─
echo "[predict] (2) install per-channel resampler into nnunetv2"
python3 - "$HERE/engine/corrector_resampling.py" <<'PY'
import sys, shutil, os
import nnunetv2.preprocessing.resampling as r
dst = os.path.join(os.path.dirname(r.__file__), "corrector_resampling.py")
shutil.copyfile(sys.argv[1], dst)
print(f"[predict] installed resampler -> {dst}")
PY

# The B/C model was trained with the corrector trainer, so its results dir is
# named <CORRECTOR_TRAINER>__nnUNetPlansFinetune__3d_fullres/. Install the trainer
# class (and, for the cascade model, its sibling imports) so nnUNetv2_predict can
# rebuild the network from it. Copying all four is cheap + idempotent; the stock
# corrector path just ignores the extra modules.
echo "[predict] (2b) install corrector runtime modules into nnunetv2 ($CORRECTOR_TRAINER)"
python3 - "$HERE/engine" <<'PY'
import sys, shutil, os
import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as m
pkg = os.path.dirname(m.__file__)
eng = sys.argv[1]
for name in ("nnUNetTrainer_corrector.py", "nnUNetTrainer_OrbitalCascade.py",
             "corrector_augment.py", "corrector_stratified_loader.py"):
    src = os.path.join(eng, name)
    if os.path.isfile(src):
        shutil.copyfile(src, os.path.join(pkg, name))
        print(f"[predict] installed {name} -> {pkg}")
PY

# ── 3. assemble 5-channel test inputs ────────────────────────────────
# Cache by default: a (source,step) whose 5ch imagesTs already exist is reused
# (the assembly is a deterministic resample, so re-running yields the same files).
# REBUILD_TESTSET=1 forces a full re-assembly (e.g. after the prelabels changed).
# Safety: if we just RE-MAPPED CNISP from latents (RESUME_FROM_LATENT=1), the
# ch1..4 prelabels changed, so the cached 5ch testset is stale -> force a rebuild
# unless the caller explicitly pinned REBUILD_TESTSET.
if [[ "$RESUME_FROM_LATENT" == "1" && -z "${REBUILD_TESTSET:-}" ]]; then
    echo "[predict] RESUME_FROM_LATENT=1 -> forcing REBUILD_TESTSET=1 (prelabels changed)"
    REBUILD_TESTSET=1
fi
SKIP_EXISTING_ARG="--skip-existing"
[[ "${REBUILD_TESTSET:-0}" == "1" ]] && SKIP_EXISTING_ARG=""
# Cascade (Route A): build a 1-ch CT testset + a prevsegTs/ dir of {cid}.nii.gz
# CNISP prior masks; the latter is fed to nnUNetv2_predict via -prev_stage_predictions.
LAYOUT_ARG=""; PREVSEG_ARG=""
if [[ "$CASCADE" == "1" ]]; then
    LAYOUT_ARG="--layout cascade"
    PREVSEG_TS="$TEST_ROOT/$CTRL_DATASET_NAME/prevsegTs"
    PREVSEG_ARG="-prev_stage_predictions $PREVSEG_TS"
fi
echo "[predict] (3) build_corrector_testset (layout=$([[ "$CASCADE" == 1 ]] && echo cascade || echo stacked)) -> $TEST_ROOT"
echo "          cache: ${SKIP_EXISTING_ARG:-off (REBUILD_TESTSET=1)}"
python3 "$HERE/scripts/build_corrector_testset.py" \
    --config "$CONFIG" --control "$CONTROL" --steps "${BUILD_STEPS:-auto}" \
    --prelabel-grid "$GRID" --iso-mm "$ISO_MM" --out "$TEST_ROOT" $SKIP_EXISTING_ARG $LAYOUT_ARG \
    ${BUILD_CASEFILE:+--casefile "$BUILD_CASEFILE"}

IMAGES_TS="$TEST_ROOT/$CTRL_DATASET_NAME/imagesTs"
OUT_DIR_PRED="${OUT_DIR_PRED:-$PRED_ROOT/$CTRL_DATASET_NAME/fold_${FOLD}}"
mkdir -p "$OUT_DIR_PRED"

# ── 4. nnUNetv2_predict with the finetuned corrector ─────────────────
# Prediction OVERWRITES (never a "resume that reuses stale masks") for a
# single-image run, so the Dice always reflects the CURRENT checkpoint. Only the
# 5ch INPUT is cached (step 3, --skip-existing); the masks are always re-predicted.
# Full-test runs resume by default (skip already-predicted cases); FORCE=1 there
# forces a full re-predict too.
if [[ "$SINGLE_MODE" == "1" ]]; then
    PREDICT_RESUME=""                       # single: always re-predict (overwrite)
else
    PREDICT_RESUME="--continue_prediction"  # full test: resume unless FORCE
    [[ "${FORCE:-0}" == "1" ]] && PREDICT_RESUME=""
fi
CKPT_PATH="${nnUNet_results%/}/Dataset$(printf '%03d' "$CTRL_DATASET_ID")_${CTRL_DATASET_NAME}/${CORRECTOR_TRAINER}__${PLAN_NAME}__${CONFIGURATION}/fold_${FOLD}/${CHK}"
echo "[predict] (4) nnUNetv2_predict d=$CTRL_DATASET_ID p=$PLAN_NAME chk=$CHK ${PREDICT_RESUME:+(resume)}"
echo "          checkpoint=$CKPT_PATH"
[[ -f "$CKPT_PATH" ]] || echo "          [warn] checkpoint file not found at that path (nnUNet may resolve it differently)"
echo "          in=$IMAGES_TS"
echo "          out=$OUT_DIR_PRED"
[[ -n "$PREVSEG_ARG" ]] && echo "          prev_stage=${PREVSEG_TS}"
nnUNetv2_predict \
    -i "$IMAGES_TS" -o "$OUT_DIR_PRED" \
    -d "$CTRL_DATASET_ID" -c "$CONFIGURATION" -tr "$CORRECTOR_TRAINER" \
    -p "$PLAN_NAME" -f "$FOLD" -chk "$CHK" $PREDICT_RESUME $PREVSEG_ARG

echo "[predict] done: predictions -> $OUT_DIR_PRED"

# ── 5. shared eval (same code/resample for A/B/C) ────────────────────
MAP_JSON="$TEST_ROOT/$CTRL_DATASET_NAME/test_cases_map.json"
EVAL_CSV="${EVAL_CSV:-$PRED_ROOT/$CTRL_DATASET_NAME/eval_${CONTROL}_fold${FOLD}.csv}"
if [[ "${RUN_EVAL:-1}" == "1" ]]; then
    echo "[predict] (5) eval (prediction -> native GT grid, order 0; Dice on GT grid)"
    python3 "$HERE/diagnostics/eval_corrector.py" \
        --map "$MAP_JSON" --pred-dir "$OUT_DIR_PRED" --out-csv "$EVAL_CSV"
else
    echo "[predict] (5) skip eval (RUN_EVAL=0). Map: $MAP_JSON"
fi
