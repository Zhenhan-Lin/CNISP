#!/usr/bin/env bash
# ============================================================
# nnUNet inference + CNISP retrain/infer + per-model
# visualization + nnUNet-vs-CNISP paired comparison.
#
# What each phase does:
#   cnisp-train           Train the orbital implicit shape prior.
#                         (orbital_shape_prior_st1/scripts/run_02_train.sh)
#                         Auto-skipped if best_checkpoint.pth already exists;
#                         pass --force-train to override.
#
#   nnunet-predict        Run nnUNetv2_predict on the staged native CT inputs
#                         under $work_dir/nnunet_input/. This is the
#                         step=1 dense baseline for the sweep.
#                         (nnunet/run_predict_native.sh)
#
#   cnisp-infer           CNISP test-time latent optimization + per-step
#                         native-space mapping (writes native_space_step_XX/
#                         + native_sweep_manifest.json + sweep_results.pkl
#                         that the next phase + compare phase consume).
#                         (orbital_shape_prior_st1/scripts/run_03_test.sh)
#
#   nnunet-predict-sweep  nnUNet on sparsified CTs, matched 1:1 to the
#                         (source_id, step_size) set CNISP just ran.
#                         Reads sweep_results.pkl, drops axial slices
#                         along each source's through-plane axis, runs
#                         nnUNetv2_predict per step, then NN-upsamples
#                         the predictions back to the native CT grid.
#                         Writes nnunet_pred_native_step_XX_upsampled/
#                         and nnunet_pred_native_sweep_manifest.json.
#                         Requires: nnunet-predict (step_01 baseline)
#                                   + cnisp-infer (sweep set).
#                         (nnunet/data_prep/sparsify_inputs.py
#                          + nnunet/run_predict_sparse_sweep.sh
#                          + nnunet/infer/upsample_sparse_preds.py)
#
#   nnunet-predict-smore  nnUNet on the SMORE-super-resolved CTs (produced
#                         out-of-band by
#                         nnunet/infer/build_smore_test_images.py; this
#                         phase only consumes them). Output is
#                         nnunet_pred_smore/<sid>.nii.gz on the SMORE
#                         grid -- mask only, no downstream comparison yet.
#                         (nnunet/data_prep/prepare_smore_inputs.py
#                          + nnunet/run_predict_smore.sh)
#
#   cnisp-viz             CNISP-side result summary: recon_summary.png,
#                         cross_resolution_analysis/, native_sweep_summary.json.
#                         (orbital_shape_prior_st1/scripts/run_04_visualization.sh)
#
#   compare               nnUNet vs CNISP paired Dice tables
#                         (paired_per_source.csv, paired_summary.csv|.txt).
#                         build_cnisp_native_sweep.py is a no-op when
#                         cnisp-infer already wrote native_space_step_XX/.
#                         compare_native.py now emits per-step nnUNet
#                         rows alongside CNISP's per-step rows.
#                         (nnunet/infer/build_cnisp_native_sweep.py
#                          + nnunet/compare_native.py)
#
# Dependency order (the order phases run when none are specified):
#   cnisp-train -> nnunet-predict -> cnisp-infer
#               -> nnunet-predict-sweep -> nnunet-predict-smore
#               -> cnisp-viz -> compare
#
# Idempotency / skip-if-done:
#   Each expensive phase auto-detects when its outputs are already complete
#   and short-circuits with a "[skip]" line. The checks are pure file-
#   existence tests so they are essentially free (~ms total) compared
#   to the GPU work they gate. Markers used:
#     cnisp-train           best_checkpoint.pth
#     nnunet-predict        nnunet_pred_native/ has 1 file per source
#     cnisp-infer           sweep_results.pkl + native_sweep_manifest.json
#     nnunet-predict-sweep  nnunet_pred_native_sweep_manifest.json
#     nnunet-predict-smore  nnunet_pred_smore/ has 1 file per source
#   cnisp-viz and compare are cheap (~minutes) so they always re-run.
#   Pass --force to ignore every check, or --force-train for just training.
#
# Usage:
#   bash run_pipeline.sh                                   # all phases
#   bash run_pipeline.sh cnisp-infer cnisp-viz             # subset
#   bash run_pipeline.sh --force                           # ignore every skip-if-done check
#   bash run_pipeline.sh --force-train                     # retrain even if checkpoint exists
#   bash run_pipeline.sh --test-config <path>              # override CNISP test yaml
#   bash run_pipeline.sh --config <path>                   # override nnunet/configs.yaml
#   bash run_pipeline.sh --gpu 0                           # forward to CUDA_VISIBLE_DEVICES
#   bash run_pipeline.sh -h
# ============================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ── Defaults ─────────────────────────────────────────────────
CONFIG="$REPO_ROOT/nnunet/configs.yaml"
TEST_CONFIG=""                       # passed through to CNISP run_03/run_04
FORCE_TRAIN=0                         # legacy: re-train even if checkpoint exists
FORCE=0                               # global: ignore every phase-level skip check
GPU_OVERRIDE="1"                      # CUDA_VISIBLE_DEVICES override
PHASES_DEFAULT=(cnisp-train nnunet-predict cnisp-infer nnunet-predict-sweep nnunet-predict-smore cnisp-viz compare)
PHASES=()

usage() {
    sed -n '2,/^# ====/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
}

# ── Arg parse ────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)         usage ;;
        --config)          CONFIG="$2"; shift 2 ;;
        --config=*)        CONFIG="${1#*=}"; shift ;;
        --test-config)     TEST_CONFIG="$2"; shift 2 ;;
        --test-config=*)   TEST_CONFIG="${1#*=}"; shift ;;
        --force-train)     FORCE_TRAIN=1; shift ;;
        --force)           FORCE=1; FORCE_TRAIN=1; shift ;;
        --gpu)             GPU_OVERRIDE="$2"; shift 2 ;;
        --gpu=*)           GPU_OVERRIDE="${1#*=}"; shift ;;
        --)                shift; while [[ $# -gt 0 ]]; do PHASES+=("$1"); shift; done ;;
        -*)
            echo "[run_pipeline] unknown option: $1" >&2
            usage
            ;;
        *)                 PHASES+=("$1"); shift ;;
    esac
done

if [[ ${#PHASES[@]} -eq 0 ]]; then
    PHASES=("${PHASES_DEFAULT[@]}")
fi

if [[ ! -f "$CONFIG" ]]; then
    echo "[run_pipeline] config not found: $CONFIG" >&2
    exit 2
fi

# ── Validate phase names early (no PyYAML needed) ────────────
VALID_PHASES=(cnisp-train nnunet-predict cnisp-infer nnunet-predict-sweep nnunet-predict-smore cnisp-viz compare)
for phase in "${PHASES[@]}"; do
    found=0
    for v in "${VALID_PHASES[@]}"; do [[ "$phase" == "$v" ]] && found=1; done
    if [[ $found -eq 0 ]]; then
        echo "[run_pipeline] unknown phase: '$phase'" >&2
        echo "  valid phases: ${VALID_PHASES[*]}" >&2
        exit 2
    fi
done

if [[ -n "$GPU_OVERRIDE" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU_OVERRIDE"
fi

# ── Resolve CNISP paths from yaml so we can do existence checks ──
read_yaml_field() {
    # $1 = yaml file, $2 = dotted field
    python3 - "$1" "$2" <<'PY'
import sys, yaml
path, field = sys.argv[1], sys.argv[2]
with open(path) as f:
    cfg = yaml.safe_load(f) or {}
cur = cfg
for k in field.split("."):
    if not isinstance(cur, dict):
        cur = None
        break
    cur = cur.get(k)
print("" if cur is None else cur)
PY
}

CNISP_PATHS_YAML_REL="$(read_yaml_field "$CONFIG" "cnisp_paths_yaml")"
if [[ -z "$CNISP_PATHS_YAML_REL" ]]; then
    echo "[run_pipeline] $CONFIG: missing 'cnisp_paths_yaml'" >&2
    exit 2
fi
# Resolve relative paths against the repo root
if [[ "$CNISP_PATHS_YAML_REL" = /* ]]; then
    CNISP_PATHS_YAML="$CNISP_PATHS_YAML_REL"
else
    CNISP_PATHS_YAML="$REPO_ROOT/$CNISP_PATHS_YAML_REL"
fi

CNISP_MODEL_NAME="$(read_yaml_field "$CONFIG" "cnisp_model_name")"
CNISP_MODEL_BASEDIR="$(read_yaml_field "$CNISP_PATHS_YAML" "model_basedir")"
CNISP_OUTPUT_BASEDIR="$(read_yaml_field "$CNISP_PATHS_YAML" "output_basedir")"
WORK_DIR="$(read_yaml_field "$CONFIG" "work_dir")"

echo "============================================================"
echo "CNISP <-> nnUNet pipeline"
echo "  repo_root:           $REPO_ROOT"
echo "  config:              $CONFIG"
echo "  cnisp_paths_yaml:    $CNISP_PATHS_YAML"
echo "  cnisp_model_name:    $CNISP_MODEL_NAME"
echo "  cnisp_model_dir:     $CNISP_MODEL_BASEDIR/$CNISP_MODEL_NAME"
echo "  cnisp_output_dir:    $CNISP_OUTPUT_BASEDIR/$CNISP_MODEL_NAME"
echo "  nnunet work_dir:     $WORK_DIR"
[[ -n "$TEST_CONFIG"   ]] && echo "  cnisp test yaml:     $TEST_CONFIG"
[[ -n "$GPU_OVERRIDE"  ]] && echo "  CUDA_VISIBLE_DEVICES=$GPU_OVERRIDE"
echo "  phases:              ${PHASES[*]}"
echo "============================================================"

# ── Skip-if-done helpers ─────────────────────────────────────
#
# Each expensive phase calls a tiny check at its top. All checks below
# are O(N_sources) file-existence tests -- microseconds of stat()'s --
# so the gating itself adds essentially no runtime compared to the
# GPU work it protects. Set --force (or delete the marker output) to
# re-run a phase whose outputs already look complete.

_count_sources_json() {
    # Count entries in $WORK_DIR/source_to_path.json. Returns "" if the
    # manifest doesn't exist yet (so the caller falls through to "not done").
    [[ -f "${WORK_DIR}/source_to_path.json" ]] || { echo ""; return; }
    python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' \
            "${WORK_DIR}/source_to_path.json"
}

_predict_dir_complete() {
    # Returns 0 (done) when $1 contains at least one *.nii.gz per source
    # listed in source_to_path.json. Cheap: one find + one tiny python.
    local pred_dir="$1"
    [[ -d "$pred_dir" ]] || return 1
    local n_src; n_src="$(_count_sources_json)"
    [[ -n "$n_src" && "$n_src" -gt 0 ]] || return 1
    local n_pred
    n_pred=$(find "$pred_dir" -maxdepth 1 -name '*.nii.gz' 2>/dev/null | wc -l | tr -d ' ')
    [[ "$n_pred" -ge "$n_src" ]]
}

# ── Phase implementations ────────────────────────────────────

phase_cnisp_train() {
    echo ""
    echo "[phase] cnisp-train -----------------------------------"
    local ckpt="$CNISP_MODEL_BASEDIR/$CNISP_MODEL_NAME/best_checkpoint.pth"
    if [[ -f "$ckpt" && $FORCE_TRAIN -eq 0 ]]; then
        echo "  best_checkpoint.pth already exists:"
        echo "    $ckpt"
        echo "  -> skipping training (pass --force-train or --force to override)."
        return 0
    fi
    bash "$REPO_ROOT/orbital_shape_prior_st1/scripts/run_02_train.sh"
}

phase_nnunet_predict() {
    echo ""
    echo "[phase] nnunet-predict --------------------------------"
    # Done iff every source in source_to_path.json has a prediction in
    # $WORK_DIR/nnunet_pred_native/<sid>.nii.gz.
    if [[ $FORCE -eq 0 ]] && _predict_dir_complete "${WORK_DIR}/nnunet_pred_native"; then
        echo "  ${WORK_DIR}/nnunet_pred_native/ already covers every source"
        echo "  -> skipping (pass --force to re-predict)."
        return 0
    fi
    CONFIG="$CONFIG" bash "$REPO_ROOT/nnunet/run_predict_native.sh"
}

phase_nnunet_predict_sweep() {
    echo ""
    echo "[phase] nnunet-predict-sweep --------------------------"
    # The upsample step writes this manifest only after every (sid, step)
    # has a sparse pred + an upsampled pred. Its presence is a complete-
    # success marker. The inner scripts also have per-file skip logic,
    # so partial re-runs are cheap even without this outer gate.
    local marker="${WORK_DIR}/nnunet_pred_native_sweep_manifest.json"
    if [[ $FORCE -eq 0 && -f "$marker" ]]; then
        echo "  sweep manifest already present:"
        echo "    $marker"
        echo "  -> skipping (pass --force or delete the manifest to rebuild)."
        return 0
    fi
    python3 "$REPO_ROOT/nnunet/data_prep/sparsify_inputs.py"   --config "$CONFIG"
    CONFIG="$CONFIG" bash "$REPO_ROOT/nnunet/run_predict_sparse_sweep.sh"
    python3 "$REPO_ROOT/nnunet/infer/upsample_sparse_preds.py" --config "$CONFIG"
}

phase_nnunet_predict_smore() {
    echo ""
    echo "[phase] nnunet-predict-smore --------------------------"
    # Done iff $WORK_DIR/nnunet_pred_smore/ covers every source.
    if [[ $FORCE -eq 0 ]] && _predict_dir_complete "${WORK_DIR}/nnunet_pred_smore"; then
        echo "  ${WORK_DIR}/nnunet_pred_smore/ already covers every source"
        echo "  -> skipping (pass --force to re-predict)."
        return 0
    fi
    python3 "$REPO_ROOT/nnunet/data_prep/prepare_smore_inputs.py" --config "$CONFIG"
    CONFIG="$CONFIG" bash "$REPO_ROOT/nnunet/run_predict_smore.sh"
}

phase_cnisp_infer() {
    echo ""
    echo "[phase] cnisp-infer -----------------------------------"
    # cnisp-infer is the single most expensive non-training phase (~hours
    # of latent optimisation). Treat it as done when both the sweep
    # results pickle AND the native-space manifest exist; the latter is
    # written at the very end of infer.py, so its presence implies the
    # whole eye x step grid completed successfully.
    local sweep_pkl="$CNISP_OUTPUT_BASEDIR/$CNISP_MODEL_NAME/sweep_results.pkl"
    local native_mf="$CNISP_OUTPUT_BASEDIR/$CNISP_MODEL_NAME/native_sweep_manifest.json"
    if [[ $FORCE -eq 0 && -f "$sweep_pkl" && -f "$native_mf" ]]; then
        echo "  sweep_results.pkl + native_sweep_manifest.json already exist:"
        echo "    $sweep_pkl"
        echo "    $native_mf"
        echo "  -> skipping cnisp-infer (pass --force or delete a marker to rerun)."
        return 0
    fi
    if [[ -n "$TEST_CONFIG" ]]; then
        bash "$REPO_ROOT/orbital_shape_prior_st1/scripts/run_03_test.sh" "$TEST_CONFIG"
    else
        bash "$REPO_ROOT/orbital_shape_prior_st1/scripts/run_03_test.sh"
    fi
}

phase_cnisp_viz() {
    echo ""
    echo "[phase] cnisp-viz -------------------------------------"
    if [[ -n "$TEST_CONFIG" ]]; then
        bash "$REPO_ROOT/orbital_shape_prior_st1/scripts/run_04_visualization.sh" "$TEST_CONFIG"
    else
        bash "$REPO_ROOT/orbital_shape_prior_st1/scripts/run_04_visualization.sh"
    fi
}

phase_compare() {
    echo ""
    echo "[phase] compare ---------------------------------------"
    python3 "$REPO_ROOT/nnunet/infer/build_cnisp_native_sweep.py" --config "$CONFIG"
    python3 "$REPO_ROOT/nnunet/compare_native.py"                 --config "$CONFIG"
}

# ── Dispatch ─────────────────────────────────────────────────
START_TS="$(date +%s)"
for phase in "${PHASES[@]}"; do
    case "$phase" in
        cnisp-train)          phase_cnisp_train ;;
        nnunet-predict)       phase_nnunet_predict ;;
        cnisp-infer)          phase_cnisp_infer ;;
        nnunet-predict-sweep) phase_nnunet_predict_sweep ;;
        nnunet-predict-smore) phase_nnunet_predict_smore ;;
        cnisp-viz)            phase_cnisp_viz ;;
        compare)              phase_compare ;;
    esac
done
END_TS="$(date +%s)"

echo ""
echo "============================================================"
printf "Pipeline complete in %ds. Phases run: %s\n" \
    "$((END_TS - START_TS))" "${PHASES[*]}"
echo ""
echo "Where to look for results:"
echo "  CNISP single-model summary:"
echo "    $CNISP_OUTPUT_BASEDIR/$CNISP_MODEL_NAME/recon_summary.png"
echo "    $CNISP_OUTPUT_BASEDIR/$CNISP_MODEL_NAME/cross_resolution_analysis/"
echo "    $CNISP_OUTPUT_BASEDIR/$CNISP_MODEL_NAME/native_sweep_summary.json"
echo "  nnUNet sparse-CT sweep (per-step preds on native CT grid):"
echo "    $WORK_DIR/nnunet_pred_native_step_XX_upsampled/"
echo "    $WORK_DIR/nnunet_pred_native_sweep_manifest.json"
echo "  nnUNet on SMORE'd CTs (mask only):"
echo "    $WORK_DIR/nnunet_pred_smore/"
echo "  nnUNet vs CNISP paired comparison (per-step rows):"
echo "    $WORK_DIR/paired_per_source.csv"
echo "    $WORK_DIR/paired_summary.csv"
echo "    $WORK_DIR/paired_summary.txt"
echo "============================================================"
