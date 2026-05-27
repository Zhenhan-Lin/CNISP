#!/usr/bin/env bash
# ============================================================
# nnUNet inference + CNISP retrain/infer + per-model
# visualization + nnUNet-vs-CNISP paired comparison.
#
# Option C: two CNISP runs are produced per pipeline invocation,
# both Diced against the same source set:
#   * run_tag=atlas_gt     latent-opt input  = sparsified canonical GT
#                          dense Dice target = canonical GT
#                          -> ceiling curve   (CNISP-atlasGT)
#   * run_tag=nnunet_pred  latent-opt input  = canonical-aligned
#                                              Dataset835 sparse-CT pred
#                          dense Dice target = canonical GT (atlas) /
#                                              Dataset835 dense pred
#                                              canonical-aligned (chk_*)
#                          -> deployment curve (CNISP-nnUNetPred)
#
# What each phase does:
#   cnisp-train           Train the orbital implicit shape prior.
#                         (orbital_shape_prior_st1/scripts/run_02_train.sh)
#                         Auto-skipped if best_checkpoint.pth already exists;
#                         pass --force-train to override.
#
#   nnunet-predict        Run nnUNetv2_predict on the staged native CT inputs
#                         under $work_dir/input/native/. This is the
#                         step=1 dense baseline for the sweep.
#                         (nnunet/run_predict_native.sh)
#
#   cnisp-infer           CNISP test-time latent optimization for the
#                         CEILING curve (run_tag=atlas_gt). Writes
#                         output_basedir/<model>/runs/atlas_gt/
#                         (sweep_results.pkl + native_sweep_manifest.json
#                         + native_space_step_XX/).
#                         (orbital_shape_prior_st1/scripts/run_03_test.sh)
#
#   nnunet-predict-sweep  nnUNet on sparsified CTs, matched 1:1 to the
#                         (source_id, step_size) set CNISP just ran.
#                         Reads sweep_results.pkl (from runs/atlas_gt/),
#                         drops axial slices along each source's
#                         through-plane axis, runs nnUNetv2_predict per
#                         step, then NN-upsamples back to the native
#                         CT grid. Writes
#                         prediction/sparse_step_XX_upsampled/ and
#                         prediction/sweep_manifest.json.
#                         Requires: nnunet-predict (step_01 baseline)
#                                   + cnisp-infer (sweep set).
#                         (nnunet/data_prep/sparsify_inputs.py
#                          + nnunet/run_predict_sparse_sweep.sh
#                          + nnunet/engine/upsample_sparse_preds.py)
#
#   nnunet-predict-smore  nnUNet on the SMORE-super-resolved CTs (produced
#                         out-of-band by
#                         nnunet/engine/build_smore_test_images.py; this
#                         phase only consumes them). Output is
#                         prediction/smore/<sid>.nii.gz on the SMORE
#                         grid -- mask only, no downstream comparison yet.
#                         (nnunet/data_prep/prepare_smore_inputs.py
#                          + nnunet/run_predict_smore.sh)
#
#   cnisp-prep-dataset835-gt
#                         Build the chk_* DENSE Dice-target patches for the
#                         deployment curve: canonical-align Dataset835's
#                         per-source dense prediction
#                         (${work_dir}/prediction/native/<sid>.nii.gz)
#                         and write the per-eye patches + sidecar metadata to
#                         ${aligned_dir}/labels_dataset835/ and
#                         ${aligned_dir}/metadata_dataset835/. Atlas sources
#                         are aligned too (no-op for Dice but used as the
#                         step_01 latent-opt input in deployment mode).
#                         Patch size is auto-pinned to the training-time
#                         ``patch_size_mm`` recorded in
#                         ${aligned_dir}/metadata/*.json so it can never
#                         drift away from what the model was trained on.
#                         Requires: nnunet-predict + cnisp-align (so the
#                         training metadata exists).
#                         (nnunet/engine/build_dataset835_canonical_patches.py)
#
#   cnisp-prep-dataset835-sparse
#                         Build per-step canonical-aligned Dataset835
#                         SPARSE-CT predictions:
#                         ${aligned_dir}/labels_dataset835_step_{XX}/.
#                         These are CNISP's latent-opt INPUT in deployment
#                         mode (one .nii.gz per (case, step)). Skipped
#                         rows -- e.g. where nnUNet dropped a globe at
#                         high sparsity -- are silently absent on disk;
#                         engine/infer.py logs and skips them at run-time.
#                         Patch size is auto-pinned to the same value as
#                         cnisp-prep-dataset835-gt (training-time
#                         ``patch_size_mm``).
#                         Requires: nnunet-predict + nnunet-predict-sweep
#                         + cnisp-align.
#                         (nnunet/engine/build_dataset835_sparse_patches.py)
#
#   cnisp-infer-nnunet-pred
#                         CNISP test-time latent optimization for the
#                         DEPLOYMENT curve (run_tag=nnunet_pred). Same
#                         model weights, same test set, but the latent-opt
#                         input is Dataset835's sparse-CT pred at each
#                         step. Writes output_basedir/<model>/runs/
#                         nnunet_pred/.
#                         Requires: cnisp-prep-dataset835-gt + sparse.
#                         (orbital_shape_prior_st1/scripts/run_03_test.sh
#                          with TEST_LABEL_SOURCE=nnunet_pred, RUN_TAG=nnunet_pred)
#
#   cnisp-native-remap    Per CNISP run, re-apply the canonical -> native CT
#                         frame mapping to every (case, step) row in
#                         ``sweep_results.pkl``, writing
#                           runs/<run_tag>/native_space_step_XX/
#                             <source>_cnisp_stepNN.nii.gz    # OD+OS merged
#                             manifest.json                   # source_id -> nifti
#                           runs/<run_tag>/native_sweep_manifest.json
#                         The script reads the cached ``pred_class_map`` straight
#                         out of ``sweep_results.pkl`` and calls the current
#                         ``orbital_shape_prior_st1/engine/native_mapping.py``,
#                         so no GPU / latent optimisation is involved.
#                         Idempotent: per-step ``manifest.json`` acts as the
#                         skip marker; pass --force (or ``rm -rf
#                         native_space_step_*/``) to overwrite existing masks
#                         after patching ``native_mapping.py``.
#                         This phase is the explicit re-render entry point
#                         shared by both ``cnisp-viz`` (which audits the
#                         outputs) and ``compare`` (which Dice's against them).
#                         (nnunet/engine/build_cnisp_native_sweep.py)
#
#   cnisp-viz             CNISP-only artifacts (the bits no method-agnostic
#                         viewer can reproduce):
#                         recon_layout.txt (file-tree dump),
#                         cross_resolution_analysis/ (iso-space prior
#                         self-consistency heatmaps), and
#                         native_sweep_summary.json (file audit of the
#                         native_space_step_XX/ tree produced by
#                         cnisp-native-remap or cnisp-infer). One
#                         viz/ tree per run_tag, written under
#                         output_basedir/<model>/runs/<run_tag>/.
#                         Per-step Dice trend / per-class / per-case
#                         figures land under viz/<run_tag>/ during the
#                         `compare` phase.
#                         (orbital_shape_prior_st1/scripts/run_04_visualization.sh)
#
#   compare               Per CNISP run declared in
#                         configs.yaml::cnisp_runs_to_compare:
#                          (a) paired Dice tables under ${work_dir}/comparison/:
#                                paired_per_source__<run_tag>.csv
#                                paired_summary__<run_tag>.csv
#                                paired_summary__<run_tag>.txt
#                          (b) per-method by-eff_res viz bundle (single-
#                              method curves). Each bundle =
#                              {method}_per_source.csv +
#                              {method}_summary_by_eff_res.csv +
#                              {method}_summary_by_eff_res.txt +
#                              {method}_recon_summary.png +
#                              {method}_overall_dice_vs_eff_res.png +
#                              {method}_per_class_dice_vs_eff_res.png +
#                              {method}_per_case_dice_distribution.png.
#                              Output dirs:
#                                CNISP        -> ${cnisp_output_basedir}/<model>/viz/<run_tag>/
#                                              (one bundle per run_tag because
#                                               the CNISP curve depends on which
#                                               latent-opt input the run used)
#                                nnUNet-sparse -> ${work_dir}/comparison/viz/nnUNet-sparse/
#                                              (rendered ONCE outside the
#                                               per-run-tag loop because nnUNet's
#                                               sparse predictions are independent
#                                               of which CNISP run is in flight;
#                                               canonical CSV = nnunet_pred, which
#                                               is a strict superset of atlas_gt's
#                                               nnUNet-sparse rows)
#                          (c) head-to-head paired plots that overlay
#                              both methods on shared axes (this is the
#                              dir to look at to actually SEE the
#                              comparison):
#                                paired_overall_dice_vs_eff_res.png
#                                paired_per_class_dice_vs_eff_res.png
#                                paired_delta_dice_vs_eff_res.png
#                                paired_dice_vs_eff_res.png  (combined)
#                                paired_summary_by_eff_res.csv
#                              Output dir:
#                                ${work_dir}/comparison/viz/paired__<run_tag>/
#                         Prerequisite: ``runs/<run_tag>/native_space_step_XX/``
#                         must already exist. Produced either by
#                         ``cnisp-native-remap`` (explicit re-render entry
#                         point) or ``cnisp-infer`` (as a side effect of
#                         fresh inference). ``compare`` pre-flights this
#                         and bails out with an instructional error if
#                         masks are missing, rather than silently emitting
#                         a half-populated paired CSV.
#                         (nnunet/compare_native.py
#                          + nnunet/engine/build_method_summary.py
#                          + nnunet/engine/build_paired_summary.py)
#
# Dependency order (the order phases run when none are specified):
#   cnisp-train
#     -> nnunet-predict
#     -> cnisp-infer                                  (atlas_gt run)
#     -> nnunet-predict-sweep
#     -> nnunet-predict-smore
#     -> cnisp-prep-dataset835-gt
#     -> cnisp-prep-dataset835-sparse
#     -> cnisp-infer-nnunet-pred                      (nnunet_pred run)
#     -> cnisp-native-remap                           (canonical->native masks)
#     -> cnisp-viz
#     -> compare
#
# Idempotency / skip-if-done:
#   Each expensive phase auto-detects when its outputs are already complete
#   and short-circuits with a "[skip]" line. The checks are pure file-
#   existence tests so they are essentially free (~ms total) compared
#   to the GPU work they gate. Markers used:
#     cnisp-train                       best_checkpoint.pth
#     nnunet-predict                    prediction/native/ has 1 file per source
#     cnisp-infer (atlas_gt)            runs/atlas_gt/sweep_results.pkl
#                                       + runs/atlas_gt/native_sweep_manifest.json
#     nnunet-predict-sweep              prediction/sweep_manifest.json
#     nnunet-predict-smore              prediction/smore/ has 1 file per source
#     cnisp-prep-dataset835-gt          labels_dataset835/ + metadata_dataset835/
#                                       cover every source (OD + OS)
#     cnisp-prep-dataset835-sparse      labels_dataset835_step_01/ covers every source
#                                       (per-step files for higher steps are
#                                       allowed to be partial -- see phase doc)
#     cnisp-infer-nnunet-pred           runs/nnunet_pred/sweep_results.pkl
#                                       + runs/nnunet_pred/native_sweep_manifest.json
#     cnisp-native-remap                per-step manifest checked inside
#                                       build_cnisp_native_sweep.py; skips
#                                       any step whose
#                                       native_space_step_XX/manifest.json
#                                       already exists. With --force every
#                                       step is re-rendered.
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
PHASES_DEFAULT=(
    cnisp-train
    nnunet-predict
    cnisp-infer
    nnunet-predict-sweep
    nnunet-predict-smore
    cnisp-prep-dataset835-gt
    cnisp-prep-dataset835-sparse
    cnisp-infer-nnunet-pred
    cnisp-native-remap
    cnisp-viz
    compare
)
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
VALID_PHASES=(
    cnisp-train
    nnunet-predict
    cnisp-infer
    nnunet-predict-sweep
    nnunet-predict-smore
    cnisp-prep-dataset835-gt
    cnisp-prep-dataset835-sparse
    cnisp-infer-nnunet-pred
    cnisp-viz
    compare
)
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

read_cnisp_runs_to_compare() {
    # Echo one "<run_tag>\t<method_label>" line per entry in
    # configs.yaml::cnisp_runs_to_compare. Falls back to the legacy
    # (atlas_gt, CNISP-atlasGT) pair if the section is absent so older
    # configs don't break.
    python3 - "$1" <<'PY'
import sys, yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f) or {}
runs = cfg.get("cnisp_runs_to_compare") or [
    {"run_tag": "atlas_gt", "method_label": "CNISP-atlasGT"},
]
for e in runs:
    rt = str(e.get("run_tag", ""))
    ml = str(e.get("method_label", f"CNISP-{rt}"))
    if rt:
        print(f"{rt}\t{ml}")
PY
}

CNISP_PATHS_YAML_REL="$(read_yaml_field "$CONFIG" "cnisp_paths_yaml")"
if [[ -z "$CNISP_PATHS_YAML_REL" ]]; then
    echo "[run_pipeline] $CONFIG: missing 'cnisp_paths_yaml'" >&2
    exit 2
fi
if [[ "$CNISP_PATHS_YAML_REL" = /* ]]; then
    CNISP_PATHS_YAML="$CNISP_PATHS_YAML_REL"
else
    CNISP_PATHS_YAML="$REPO_ROOT/$CNISP_PATHS_YAML_REL"
fi

CNISP_MODEL_NAME="$(read_yaml_field "$CONFIG" "cnisp_model_name")"
CNISP_MODEL_BASEDIR="$(read_yaml_field "$CNISP_PATHS_YAML" "model_basedir")"
CNISP_OUTPUT_BASEDIR="$(read_yaml_field "$CNISP_PATHS_YAML" "output_basedir")"
CNISP_ALIGNED_DIR="$(read_yaml_field "$CNISP_PATHS_YAML" "aligned_dir")"
WORK_DIR="$(read_yaml_field "$CONFIG" "work_dir")"
LABELS835_DIRNAME="$(read_yaml_field "$CNISP_PATHS_YAML" "labels_dataset835_dirname")"
LABELS835_DIRNAME="${LABELS835_DIRNAME:-labels_dataset835}"
META835_DIRNAME="$(read_yaml_field "$CNISP_PATHS_YAML" "metadata_dataset835_dirname")"
META835_DIRNAME="${META835_DIRNAME:-metadata_dataset835}"
SPARSE835_PREFIX="$(read_yaml_field "$CNISP_PATHS_YAML" "labels_dataset835_step_prefix")"
SPARSE835_PREFIX="${SPARSE835_PREFIX:-labels_dataset835_step_}"

# Resolve the patch_size_mm that the model was trained on so we can
# echo it in the run banner and so the two `cnisp-prep-dataset835-*`
# phases inherit the *same* physical extent as the original CNISP
# training crops. This closes the silent-drift hole that previously
# let build_dataset835_*_patches.py default to 64 mm even when
# run_01_prepare.sh used 80 mm -- a mismatch that translated the
# nnunet_pred predictions by (80-64)/2 = 8 mm per axis. We treat
# absence of training metadata as an early-fail (`unset`) for the
# dataset835 phases; other phases don't need it.
TRAINING_META_DIR="$CNISP_ALIGNED_DIR/metadata"
CNISP_PATCH_SIZE_MM="$(
    python3 - "$TRAINING_META_DIR" <<'PY' 2>/dev/null || true
import json, sys
from pathlib import Path
meta_dir = Path(sys.argv[1])
if not meta_dir.is_dir():
    sys.exit(0)
sizes = set()
for p in sorted(meta_dir.glob("*.json")):
    try:
        v = float(json.load(open(p)).get("patch_size_mm"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        continue
    sizes.add(round(v, 3))
if len(sizes) == 1:
    print(f"{next(iter(sizes)):.3f}")
PY
)"

# Snapshot the (run_tag, method_label) list once so every phase uses
# the same order (cnisp-viz, compare, and run-summary at the bottom).
CNISP_RUNS_RAW="$(read_cnisp_runs_to_compare "$CONFIG")"
declare -a CNISP_RUN_TAGS=()
declare -a CNISP_METHOD_LABELS=()
while IFS=$'\t' read -r tag label; do
    [[ -z "$tag" ]] && continue
    CNISP_RUN_TAGS+=("$tag")
    CNISP_METHOD_LABELS+=("$label")
done <<<"$CNISP_RUNS_RAW"
if [[ ${#CNISP_RUN_TAGS[@]} -eq 0 ]]; then
    echo "[run_pipeline] $CONFIG produced 0 (run_tag, method_label) " \
         "pairs. Add at least one entry under cnisp_runs_to_compare." >&2
    exit 2
fi

echo "============================================================"
echo "CNISP <-> nnUNet pipeline"
echo "  repo_root:           $REPO_ROOT"
echo "  config:              $CONFIG"
echo "  cnisp_paths_yaml:    $CNISP_PATHS_YAML"
echo "  cnisp_model_name:    $CNISP_MODEL_NAME"
echo "  cnisp_model_dir:     $CNISP_MODEL_BASEDIR/$CNISP_MODEL_NAME"
echo "  cnisp_output_dir:    $CNISP_OUTPUT_BASEDIR/$CNISP_MODEL_NAME"
echo "  cnisp_aligned_dir:   $CNISP_ALIGNED_DIR"
if [[ -n "$CNISP_PATCH_SIZE_MM" ]]; then
    echo "  cnisp_patch_size_mm: $CNISP_PATCH_SIZE_MM (from $TRAINING_META_DIR)"
else
    echo "  cnisp_patch_size_mm: <unresolved> (no aligned/metadata/ yet)"
fi
echo "  nnunet work_dir:     $WORK_DIR"
echo "  cnisp runs:"
for i in "${!CNISP_RUN_TAGS[@]}"; do
    echo "    - run_tag=${CNISP_RUN_TAGS[$i]}  method_label=${CNISP_METHOD_LABELS[$i]}"
done
[[ -n "$TEST_CONFIG"   ]] && echo "  cnisp test yaml:     $TEST_CONFIG"
[[ -n "$GPU_OVERRIDE"  ]] && echo "  CUDA_VISIBLE_DEVICES=$GPU_OVERRIDE"
echo "  phases:              ${PHASES[*]}"
echo "============================================================"

# ── Skip-if-done helpers ─────────────────────────────────────

_count_sources_json() {
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

_eye_dir_complete() {
    # Returns 0 (done) when $1 contains both OD and OS for every source.
    # The dense canonical-align phase writes 2 files per source (or 1
    # when nnUNet dropped one globe). We require >= 2 * n_src - small
    # slack to allow occasional dropped eyes without retriggering.
    local d="$1"
    [[ -d "$d" ]] || return 1
    local n_src; n_src="$(_count_sources_json)"
    [[ -n "$n_src" && "$n_src" -gt 0 ]] || return 1
    local n_files
    n_files=$(find "$d" -maxdepth 1 -name '*.nii.gz' 2>/dev/null | wc -l | tr -d ' ')
    # Be tolerant: at least n_src files = one eye per source minimum.
    [[ "$n_files" -ge "$n_src" ]]
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
    if [[ $FORCE -eq 0 ]] && _predict_dir_complete "${WORK_DIR}/prediction/native"; then
        echo "  ${WORK_DIR}/prediction/native/ already covers every source"
        echo "  -> skipping (pass --force to re-predict)."
        return 0
    fi
    CONFIG="$CONFIG" bash "$REPO_ROOT/nnunet/run_predict_native.sh"
}

phase_nnunet_predict_sweep() {
    echo ""
    echo "[phase] nnunet-predict-sweep --------------------------"
    local marker="${WORK_DIR}/prediction/sweep_manifest.json"
    if [[ $FORCE -eq 0 && -f "$marker" ]]; then
        echo "  sweep manifest already present:"
        echo "    $marker"
        echo "  -> skipping (pass --force or delete the manifest to rebuild)."
        return 0
    fi
    python3 "$REPO_ROOT/nnunet/data_prep/sparsify_inputs.py"   --config "$CONFIG"
    CONFIG="$CONFIG" bash "$REPO_ROOT/nnunet/run_predict_sparse_sweep.sh"
    python3 "$REPO_ROOT/nnunet/engine/upsample_sparse_preds.py" --config "$CONFIG"
}

phase_nnunet_predict_smore() {
    echo ""
    echo "[phase] nnunet-predict-smore --------------------------"
    if [[ $FORCE -eq 0 ]] && _predict_dir_complete "${WORK_DIR}/prediction/smore"; then
        echo "  ${WORK_DIR}/prediction/smore/ already covers every source"
        echo "  -> skipping (pass --force to re-predict)."
        return 0
    fi
    python3 "$REPO_ROOT/nnunet/data_prep/prepare_smore_inputs.py" --config "$CONFIG"
    CONFIG="$CONFIG" bash "$REPO_ROOT/nnunet/run_predict_smore.sh"
}

_run_cnisp_infer_for() {
    # $1 = test_label_source, $2 = run_tag
    local label_src="$1" run_tag="$2"
    if [[ -n "$TEST_CONFIG" ]]; then
        bash "$REPO_ROOT/orbital_shape_prior_st1/scripts/run_03_test.sh" \
             "$TEST_CONFIG" "$label_src" "$run_tag"
    else
        bash "$REPO_ROOT/orbital_shape_prior_st1/scripts/run_03_test.sh" \
             "" "$label_src" "$run_tag"
    fi
}

_skip_cnisp_infer_if_done() {
    # Returns 0 (caller should skip) iff the per-run sweep + manifest exist.
    local run_tag="$1"
    local run_dir="$CNISP_OUTPUT_BASEDIR/$CNISP_MODEL_NAME/runs/$run_tag"
    local sweep_pkl="$run_dir/sweep_results.pkl"
    local native_mf="$run_dir/native_sweep_manifest.json"
    if [[ $FORCE -eq 0 && -f "$sweep_pkl" && -f "$native_mf" ]]; then
        echo "  $run_dir already has sweep_results.pkl + native_sweep_manifest.json"
        echo "  -> skipping (pass --force or delete a marker to rerun)."
        return 0
    fi
    return 1
}

phase_cnisp_infer() {
    echo ""
    echo "[phase] cnisp-infer (run_tag=atlas_gt) ----------------"
    _skip_cnisp_infer_if_done "atlas_gt" && return 0
    _run_cnisp_infer_for "atlas_gt" "atlas_gt"
}

_require_training_patch_size() {
    # Both dataset835 build scripts inherit patch_size_mm from
    # $TRAINING_META_DIR. If that directory is empty / missing we
    # bail out here with a precise hint -- silently letting the
    # python scripts default to some other value is exactly the bug
    # the auto-detect path was added to prevent.
    if [[ -z "$CNISP_PATCH_SIZE_MM" ]]; then
        echo "[run_pipeline] $1: cannot resolve training-time patch_size_mm" >&2
        echo "  searched: $TRAINING_META_DIR" >&2
        echo "  Run 'bash run_preprocessing.sh cnisp-align' first so the" >&2
        echo "  CNISP training metadata is on disk, then re-run this phase." >&2
        exit 2
    fi
    echo "  using training patch_size_mm=$CNISP_PATCH_SIZE_MM "\
         "(auto-detected from $TRAINING_META_DIR)"
}

phase_cnisp_prep_dataset835_gt() {
    echo ""
    echo "[phase] cnisp-prep-dataset835-gt ----------------------"
    local labels_dir="$CNISP_ALIGNED_DIR/$LABELS835_DIRNAME"
    local meta_dir="$CNISP_ALIGNED_DIR/$META835_DIRNAME"
    if [[ $FORCE -eq 0 ]] \
        && _eye_dir_complete "$labels_dir" \
        && _eye_dir_complete "$meta_dir"; then
        echo "  $labels_dir + $meta_dir already cover every source"
        echo "  -> skipping (pass --force to rebuild)."
        return 0
    fi
    _require_training_patch_size "cnisp-prep-dataset835-gt"
    # The python script reads the same $TRAINING_META_DIR so passing
    # --patch-size explicitly is redundant; we still forward it to
    # surface a single value in the log and so a future user can
    # override it from the shell without editing python.
    python3 "$REPO_ROOT/nnunet/engine/build_dataset835_canonical_patches.py" \
            --config "$CONFIG" \
            --patch-size "$CNISP_PATCH_SIZE_MM"
}

phase_cnisp_prep_dataset835_sparse() {
    echo ""
    echo "[phase] cnisp-prep-dataset835-sparse ------------------"
    # Use step_01 as the "complete" marker. Higher steps are
    # allowed to be partial (nnUNet may have dropped globes at high
    # sparsity); the inference loader handles missing rows.
    local step01_dir="$CNISP_ALIGNED_DIR/${SPARSE835_PREFIX}01"
    if [[ $FORCE -eq 0 ]] && _eye_dir_complete "$step01_dir"; then
        echo "  $step01_dir already covers every source"
        echo "  -> skipping (pass --force to rebuild)."
        return 0
    fi
    _require_training_patch_size "cnisp-prep-dataset835-sparse"
    python3 "$REPO_ROOT/nnunet/engine/build_dataset835_sparse_patches.py" \
            --config "$CONFIG" \
            --patch-size "$CNISP_PATCH_SIZE_MM"
}

phase_cnisp_infer_nnunet_pred() {
    echo ""
    echo "[phase] cnisp-infer-nnunet-pred (run_tag=nnunet_pred) -"
    _skip_cnisp_infer_if_done "nnunet_pred" && return 0
    _run_cnisp_infer_for "nnunet_pred" "nnunet_pred"
}

phase_cnisp_native_remap() {
    echo ""
    echo "[phase] cnisp-native-remap ----------------------------"
    # Rebuilds runs/<run_tag>/native_space_step_XX/<source>_cnisp_stepNN.nii.gz
    # from the cached pred_class_map fields in sweep_results.pkl, using the
    # current canonical->native mapping in
    # orbital_shape_prior_st1/engine/native_mapping.py. No GPU, no latent
    # optimisation -- the dense pred is already cached inside the pickle.
    #
    # Idempotency: per-step native_space_step_XX/manifest.json is the skip
    # marker. ``--force`` (global flag, also exported via $FORCE) overrides
    # the skip so every step is re-rendered. To selectively re-render some
    # steps, manually ``rm -rf native_space_step_XX/`` for those step ids
    # and run this phase without --force.
    local force_flag=""
    if [[ $FORCE -eq 1 ]]; then
        force_flag="--force"
        echo "  (--force: existing native_space_step_XX/manifest.json files will be ignored)"
    fi
    for i in "${!CNISP_RUN_TAGS[@]}"; do
        local run_tag="${CNISP_RUN_TAGS[$i]}"
        echo "  ── native remap for run_tag=$run_tag ──"
        # shellcheck disable=SC2086  # force_flag is intentionally word-split
        python3 "$REPO_ROOT/nnunet/engine/build_cnisp_native_sweep.py" \
                --config "$CONFIG" --run-tag "$run_tag" $force_flag
    done
}

phase_cnisp_viz() {
    echo ""
    echo "[phase] cnisp-viz -------------------------------------"
    for i in "${!CNISP_RUN_TAGS[@]}"; do
        local run_tag="${CNISP_RUN_TAGS[$i]}"
        echo "  viz for run_tag=$run_tag"
        if [[ -n "$TEST_CONFIG" ]]; then
            bash "$REPO_ROOT/orbital_shape_prior_st1/scripts/run_04_visualization.sh" \
                 "$TEST_CONFIG" "$run_tag"
        else
            bash "$REPO_ROOT/orbital_shape_prior_st1/scripts/run_04_visualization.sh" \
                 "" "$run_tag"
        fi
    done
}

_require_native_masks() {
    # Fail fast if compare_native.py would have nothing to consume.
    # ``compare`` is strictly a downstream-of-mask phase now; the
    # canonical mask producer is ``cnisp-native-remap`` (or, as a side
    # effect, a fresh ``cnisp-infer`` run).
    local run_tag="$1"
    local run_dir="$CNISP_OUTPUT_BASEDIR/$CNISP_MODEL_NAME/runs/$run_tag"
    local found=0
    # shellcheck disable=SC2231  # we want word-splitting + glob expansion
    for d in "$run_dir"/native_space_step_*; do
        if [[ -d "$d" && -f "$d/manifest.json" ]]; then
            found=1
            break
        fi
    done
    if [[ $found -eq 0 ]]; then
        echo "[run_pipeline] compare: no native_space_step_XX/manifest.json"  >&2
        echo "  under $run_dir/."                                             >&2
        echo "  Native masks must be produced before compare can run."        >&2
        echo "  Fix:"                                                         >&2
        echo "    bash run_pipeline.sh cnisp-native-remap compare"            >&2
        echo "  (or rerun cnisp-infer to produce them as an inference side"   >&2
        echo "   effect; --force forces a rebuild past the per-step skip"     >&2
        echo "   marker.)"                                                    >&2
        exit 2
    fi
}

phase_compare() {
    echo ""
    echo "[phase] compare ---------------------------------------"

    # ── Per-run_tag stages: compare_native, CNISP viz, paired viz ──────────
    # The CNISP method label and the head-to-head plots are intrinsically
    # per-run-tag (different latent-opt inputs → different CNISP curves).
    # The nnUNet-sparse panel is NOT per-run-tag and is rendered separately
    # below, see the rationale block before the post-loop render.
    #
    # Native masks are the responsibility of ``cnisp-native-remap``
    # (or ``cnisp-infer``, as a side effect of fresh inference); this
    # phase only consumes them. We pre-flight every run_tag here so a
    # missing-mask state surfaces with an explicit instruction instead
    # of leaking through compare_native.py as a stream of per-source
    # warnings and a half-populated paired_per_source.csv.
    for i in "${!CNISP_RUN_TAGS[@]}"; do
        local run_tag="${CNISP_RUN_TAGS[$i]}"
        _require_native_masks "$run_tag"
    done

    for i in "${!CNISP_RUN_TAGS[@]}"; do
        local run_tag="${CNISP_RUN_TAGS[$i]}"
        local method="${CNISP_METHOD_LABELS[$i]}"
        echo "  ─── compare for run_tag=$run_tag (method=$method) ───"

        # 1) Per-source paired Dice CSV/TXT for THIS run.
        python3 "$REPO_ROOT/nnunet/compare_native.py" \
                --config "$CONFIG" --cnisp-run-tag "$run_tag"

        local paired_csv="$WORK_DIR/comparison/paired_per_source__${run_tag}.csv"
        local cnisp_viz_dir="$CNISP_OUTPUT_BASEDIR/$CNISP_MODEL_NAME/viz/$run_tag"
        local paired_viz_dir="$WORK_DIR/comparison/viz/paired__${run_tag}"

        # 2) CNISP-only by-eff_res bundle for THIS run.
        python3 "$REPO_ROOT/nnunet/engine/build_method_summary.py" \
                --config "$CONFIG" \
                --method "$method" \
                --paired-csv "$paired_csv" \
                --out-dir "$cnisp_viz_dir"

        # 3) Head-to-head paired plots (both methods overlaid). This is
        #    the dir a reviewer should open to actually SEE the
        #    comparison; the per-method bundles are the raw single-method
        #    view.
        python3 "$REPO_ROOT/nnunet/engine/build_paired_summary.py" \
                --config "$CONFIG" \
                --cnisp-method "$method" \
                --paired-csv "$paired_csv" \
                --out-dir "$paired_viz_dir"
    done

    # ── nnUNet-sparse standalone bundle (rendered ONCE) ──────────────────
    # Rationale for not putting this inside the loop above:
    #   The nnUNet-sparse predictions are produced by the dense Dataset835
    #   sweep, which is *independent of which CNISP run is happening*.
    #   For atlas_* sources the Dice GT is the atlas manual mask under
    #   both run_tags, so the nnUNet-sparse Dice in atlas_gt vs nnunet_pred
    #   paired CSVs is bit-identical for every atlas_* row. Under the
    #   default ``viz_exclude_source_prefixes: ["chk_"]`` filter, chk_
    #   sources are also dropped, so the two CSVs render to identical
    #   plots -- which used to create two run-tag-suffixed directories
    #   (``viz/nnUNet-sparse__atlas_gt/`` and ``viz/nnUNet-sparse__nnunet_pred/``)
    #   with byte-equal contents.
    #
    #   We now render a single ``viz/nnUNet-sparse/`` bundle from the
    #   nnunet_pred CSV (a strict superset: same atlas_ rows + chk_ rows
    #   that don't exist in atlas_gt mode), so the chk_-inclusive case
    #   stays informative without paying for the duplicate render in the
    #   atlas-only default case.
    if [[ ${#CNISP_RUN_TAGS[@]} -gt 0 ]]; then
        local canonical_tag="${CNISP_RUN_TAGS[-1]}"
        # If "nnunet_pred" is present, prefer it (strict superset of
        # atlas_gt for nnUNet-sparse rows). Otherwise fall back to
        # whatever the last run_tag is.
        for t in "${CNISP_RUN_TAGS[@]}"; do
            if [[ "$t" == "nnunet_pred" ]]; then
                canonical_tag="$t"
                break
            fi
        done
        local canonical_csv="$WORK_DIR/comparison/paired_per_source__${canonical_tag}.csv"
        local nnunet_viz_dir="$WORK_DIR/comparison/viz/nnUNet-sparse"

        echo "  ─── nnUNet-sparse standalone (canonical CSV = ${canonical_tag}) ───"
        python3 "$REPO_ROOT/nnunet/engine/build_method_summary.py" \
                --config "$CONFIG" \
                --method nnUNet-sparse \
                --paired-csv "$canonical_csv" \
                --out-dir "$nnunet_viz_dir"
    fi
}

# ── Dispatch ─────────────────────────────────────────────────
START_TS="$(date +%s)"
for phase in "${PHASES[@]}"; do
    case "$phase" in
        cnisp-train)                   phase_cnisp_train ;;
        nnunet-predict)                phase_nnunet_predict ;;
        cnisp-infer)                   phase_cnisp_infer ;;
        nnunet-predict-sweep)          phase_nnunet_predict_sweep ;;
        nnunet-predict-smore)          phase_nnunet_predict_smore ;;
        cnisp-prep-dataset835-gt)      phase_cnisp_prep_dataset835_gt ;;
        cnisp-prep-dataset835-sparse)  phase_cnisp_prep_dataset835_sparse ;;
        cnisp-infer-nnunet-pred)       phase_cnisp_infer_nnunet_pred ;;
        cnisp-native-remap)            phase_cnisp_native_remap ;;
        cnisp-viz)                     phase_cnisp_viz ;;
        compare)                       phase_compare ;;
    esac
done
END_TS="$(date +%s)"

echo ""
echo "============================================================"
printf "Pipeline complete in %ds. Phases run: %s\n" \
    "$((END_TS - START_TS))" "${PHASES[*]}"
echo ""
echo "Where to look for results:"
echo "  CNISP artifacts (per run_tag):"
for i in "${!CNISP_RUN_TAGS[@]}"; do
    rt="${CNISP_RUN_TAGS[$i]}"
    ml="${CNISP_METHOD_LABELS[$i]}"
    base="$CNISP_OUTPUT_BASEDIR/$CNISP_MODEL_NAME/runs/$rt"
    echo "    ── $rt ($ml) ──"
    echo "    $base/recon_layout.txt"
    echo "    $base/cross_resolution_analysis/"
    echo "    $base/native_sweep_summary.json"
    echo "    $base/sweep_results.pkl"
    echo "    $CNISP_OUTPUT_BASEDIR/$CNISP_MODEL_NAME/viz/$rt/${ml}_recon_summary.png"
done
echo "  nnUNet sparse-CT sweep (per-step preds on native CT grid):"
echo "    $WORK_DIR/prediction/sparse_step_XX_upsampled/"
echo "    $WORK_DIR/prediction/sweep_manifest.json"
echo "  nnUNet on SMORE'd CTs (mask only):"
echo "    $WORK_DIR/prediction/smore/"
echo "  Paired comparison tables (one set per CNISP run):"
for i in "${!CNISP_RUN_TAGS[@]}"; do
    rt="${CNISP_RUN_TAGS[$i]}"
    echo "    $WORK_DIR/comparison/paired_per_source__${rt}.csv"
    echo "    $WORK_DIR/comparison/paired_summary__${rt}.csv"
    echo "    $WORK_DIR/comparison/paired_summary__${rt}.txt"
    echo "    $WORK_DIR/comparison/viz/paired__${rt}/paired_dice_vs_eff_res.png"
    echo "      (+ paired_{overall,per_class,delta}_dice_vs_eff_res.png "
    echo "         + paired_summary_by_eff_res.csv -- the head-to-head view)"
done
echo "  nnUNet-sparse standalone bundle (run-tag-agnostic; rendered once):"
echo "    $WORK_DIR/comparison/viz/nnUNet-sparse/nnUNet-sparse_recon_summary.png"
echo "============================================================"
