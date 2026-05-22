#!/usr/bin/env bash
# ============================================================
# Pre-processing orchestrator (runs on a host that can read the
# *source-data* paths -- the atlas image/label dirs, the PHOTON pivot
# table, the QA checklist CSV, the raw CT files referenced therein).
#
# These paths are typically NOT mounted on the GPU server. Run this
# script on the data-side machine first, then move the produced
# directories (aligned_dir, work_dir/nnunet_input/, SMORE outputs) to
# the GPU host -- or mount them via a shared filesystem -- before
# running ./run_pipeline.sh there.
#
# Source-data paths touched (per nnunet/configs.yaml + paths.yaml):
#   cnisp-align    paths.yaml::checklist_csv, paths.yaml::atlas_label_dir
#                  -> writes paths.yaml::aligned_dir
#   nnunet-stage   configs.yaml::atlas_image_dir, configs.yaml::pivot_csv
#                  -> writes configs.yaml::work_dir/nnunet_input/
#                  NOTE: prepare_inputs.py *symlinks* the original CTs
#                  into nnunet_input/. The GPU host must still resolve
#                  those symlinks at predict time. Either keep the
#                  source CTs on a shared filesystem, copy them onto
#                  the GPU host first, or edit prepare_inputs.py to
#                  copy instead of symlink.
#   smore          same source CTs as nnunet-stage; writes to the
#                  shared filesystem configs.yaml::smore_out_root
#                  (default /fs5/p_masi/linz18/data/smore_resolved_images).
#                  Phase 1.5 prep -- expensive (hours per case), opt-in.
#
# Phases (run in this order if any are omitted):
#   cnisp-align    Canonical alignment + train/val/test caselists
#                  (orbital_shape_prior_st1/scripts/run_01_prepare.sh)
#                  Auto-skipped if aligned_dir/labels/ already has files;
#                  pass --force-align to override.
#   nnunet-stage   Resolve 31 source CTs and symlink them as nnUNet
#                  channel-0 inputs + source_to_path.json manifest.
#                  (nnunet/prepare_inputs.py)
#   smore          (opt-in, NOT in default) Super-resolve the 31 CTs
#                  with SMORE for the deferred iso-grid comparison.
#                  (nnunet/build_smore_test_images.py)
#
# Default (no phases given): cnisp-align nnunet-stage
#   -- smore is opt-in because it can take many GPU-hours.
#
# Usage:
#   bash run_preprocessing.sh                                 # cnisp-align + nnunet-stage
#   bash run_preprocessing.sh cnisp-align                     # one phase
#   bash run_preprocessing.sh nnunet-stage smore              # stage + run SMORE
#   bash run_preprocessing.sh --force-align                   # re-do canonical alignment
#   bash run_preprocessing.sh smore --smore-gpu-ids 0,1       # SMORE on two GPUs
#   bash run_preprocessing.sh --config <path>                 # override nnunet/configs.yaml
#   bash run_preprocessing.sh --paths <path>                  # override CNISP paths.yaml
#   bash run_preprocessing.sh -h
# ============================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ── Defaults ─────────────────────────────────────────────────
CONFIG="$REPO_ROOT/nnunet/configs.yaml"
PATHS_YAML=""                        # override CNISP paths.yaml
TRAIN_YAML=""                        # for cnisp-align (test/val fractions)
PATCH_SIZE_MM=""                     # cnisp-align patch size override
FORCE_ALIGN=0
SMORE_EXTRA_ARGS=()                  # collected --smore-* flags
PHASES_DEFAULT=(cnisp-align nnunet-stage)
PHASES=()

usage() {
    sed -n '2,/^# ====/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
}

# ── Arg parse ────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)              usage ;;
        --config)               CONFIG="$2"; shift 2 ;;
        --config=*)             CONFIG="${1#*=}"; shift ;;
        --paths)                PATHS_YAML="$2"; shift 2 ;;
        --paths=*)              PATHS_YAML="${1#*=}"; shift ;;
        --train-config)         TRAIN_YAML="$2"; shift 2 ;;
        --train-config=*)       TRAIN_YAML="${1#*=}"; shift ;;
        --patch-size)           PATCH_SIZE_MM="$2"; shift 2 ;;
        --patch-size=*)         PATCH_SIZE_MM="${1#*=}"; shift ;;
        --force-align)          FORCE_ALIGN=1; shift ;;
        # SMORE pass-through: collect every --smore-* flag for phase_smore
        --smore-*)
            if [[ "$1" == *=* ]]; then
                SMORE_EXTRA_ARGS+=("$1")
                shift
            else
                SMORE_EXTRA_ARGS+=("$1" "$2")
                shift 2
            fi
            ;;
        --) shift; while [[ $# -gt 0 ]]; do PHASES+=("$1"); shift; done ;;
        -*)
            echo "[run_preprocessing] unknown option: $1" >&2
            usage
            ;;
        *) PHASES+=("$1"); shift ;;
    esac
done

if [[ ${#PHASES[@]} -eq 0 ]]; then
    PHASES=("${PHASES_DEFAULT[@]}")
fi

if [[ ! -f "$CONFIG" ]]; then
    echo "[run_preprocessing] config not found: $CONFIG" >&2
    exit 2
fi

# ── Validate phase names (no PyYAML needed) ──────────────────
VALID_PHASES=(cnisp-align nnunet-stage smore)
for phase in "${PHASES[@]}"; do
    found=0
    for v in "${VALID_PHASES[@]}"; do [[ "$phase" == "$v" ]] && found=1; done
    if [[ $found -eq 0 ]]; then
        echo "[run_preprocessing] unknown phase: '$phase'" >&2
        echo "  valid phases: ${VALID_PHASES[*]}" >&2
        exit 2
    fi
done

# ── YAML field reader ────────────────────────────────────────
read_yaml_field() {
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

# Resolve CNISP paths.yaml (may be overridden, default = configs.yaml::cnisp_paths_yaml)
if [[ -z "$PATHS_YAML" ]]; then
    REL_PATHS="$(read_yaml_field "$CONFIG" "cnisp_paths_yaml")"
    if [[ -z "$REL_PATHS" ]]; then
        echo "[run_preprocessing] $CONFIG: missing 'cnisp_paths_yaml'" >&2
        exit 2
    fi
    if [[ "$REL_PATHS" = /* ]]; then PATHS_YAML="$REL_PATHS"
    else                              PATHS_YAML="$REPO_ROOT/$REL_PATHS"
    fi
fi

ALIGNED_DIR="$(read_yaml_field "$PATHS_YAML" "aligned_dir")"
CASEFILES_DIR="$(read_yaml_field "$PATHS_YAML" "casefiles_dir")"
ATLAS_IMG_DIR="$(read_yaml_field "$CONFIG" "atlas_image_dir")"
PIVOT_CSV="$(read_yaml_field "$CONFIG" "pivot_csv")"
WORK_DIR="$(read_yaml_field "$CONFIG" "work_dir")"
SMORE_OUT_ROOT="$(read_yaml_field "$CONFIG" "smore_out_root")"

echo "============================================================"
echo "CNISP <-> nnUNet preprocessing"
echo "  repo_root:           $REPO_ROOT"
echo "  config:              $CONFIG"
echo "  cnisp paths.yaml:    $PATHS_YAML"
echo ""
echo "  source-data paths these phases will READ:"
echo "    checklist_csv      $(read_yaml_field "$PATHS_YAML" "checklist_csv")"
echo "    atlas_label_dir    $(read_yaml_field "$PATHS_YAML" "atlas_label_dir")"
echo "    atlas_image_dir    $ATLAS_IMG_DIR"
echo "    pivot_csv          $PIVOT_CSV"
echo ""
echo "  destinations these phases WRITE:"
echo "    aligned_dir        $ALIGNED_DIR"
echo "    casefiles_dir      $CASEFILES_DIR"
echo "    nnunet_input/      ${WORK_DIR%/}/nnunet_input"
echo "    smore_out_root     $SMORE_OUT_ROOT"
echo ""
echo "  phases:              ${PHASES[*]}"
echo "============================================================"

# ── Phase implementations ────────────────────────────────────

phase_cnisp_align() {
    echo ""
    echo "[phase] cnisp-align -----------------------------------"
    local labels_dir="$ALIGNED_DIR/labels"
    if [[ -d "$labels_dir" && -n "$(ls -A "$labels_dir" 2>/dev/null || true)" \
          && $FORCE_ALIGN -eq 0 ]]; then
        echo "  aligned_dir/labels already populated:"
        echo "    $labels_dir"
        echo "  -> skipping alignment (pass --force-align to override)."
        return 0
    fi
    if [[ -n "$TRAIN_YAML$PATCH_SIZE_MM" ]]; then
        # Bypass run_01_prepare.sh defaults and call 01_prepare_data.py directly
        export PYTHONPATH="$REPO_ROOT/orbital_shape_prior_st1:${PYTHONPATH:-}"
        local args=(-p "$PATHS_YAML")
        [[ -n "$TRAIN_YAML"     ]] && args+=(-c "$TRAIN_YAML")
        [[ -n "$PATCH_SIZE_MM"  ]] && args+=(--patch_size "$PATCH_SIZE_MM")
        python3 "$REPO_ROOT/orbital_shape_prior_st1/scripts/01_prepare_data.py" \
            "${args[@]}"
    else
        bash "$REPO_ROOT/orbital_shape_prior_st1/scripts/run_01_prepare.sh"
    fi
}

phase_nnunet_stage() {
    echo ""
    echo "[phase] nnunet-stage ----------------------------------"
    python3 "$REPO_ROOT/nnunet/prepare_inputs.py" --config "$CONFIG"
    echo ""
    echo "  Staged inputs (symlinks pointing at the source CTs):"
    echo "    ${WORK_DIR%/}/nnunet_input/"
    echo "    ${WORK_DIR%/}/source_to_path.json"
    echo ""
    echo "  Reminder: the GPU host must be able to follow these symlinks."
    echo "  Either keep source CTs on a shared filesystem, copy them"
    echo "  onto the GPU host, or replace symlinks with copies."
}

phase_smore() {
    echo ""
    echo "[phase] smore -----------------------------------------"
    if [[ ${#SMORE_EXTRA_ARGS[@]} -gt 0 ]]; then
        echo "  smore extra args: ${SMORE_EXTRA_ARGS[*]}"
    fi
    python3 "$REPO_ROOT/nnunet/build_smore_test_images.py" \
        --config "$CONFIG" "${SMORE_EXTRA_ARGS[@]}"
}

# ── Dispatch ─────────────────────────────────────────────────
START_TS="$(date +%s)"
for phase in "${PHASES[@]}"; do
    case "$phase" in
        cnisp-align)  phase_cnisp_align ;;
        nnunet-stage) phase_nnunet_stage ;;
        smore)        phase_smore ;;
    esac
done
END_TS="$(date +%s)"

echo ""
echo "============================================================"
printf "Preprocessing complete in %ds. Phases run: %s\n" \
    "$((END_TS - START_TS))" "${PHASES[*]}"
echo ""
echo "What to move to the GPU host before running ./run_pipeline.sh:"
echo "  CNISP aligned patches:    $ALIGNED_DIR"
echo "  CNISP caselists:          $CASEFILES_DIR"
echo "  nnUNet staged inputs:     ${WORK_DIR%/}/nnunet_input/"
echo "                            ${WORK_DIR%/}/source_to_path.json"
echo "  (if smore was run)        $SMORE_OUT_ROOT   # already on shared FS"
echo "============================================================"
