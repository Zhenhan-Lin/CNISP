#!/usr/bin/env bash
# ============================================================
# Predict PHOTON-CT scans with the Dataset835-trained nnUNet.
#
# Selects EVERY scan in an image_info CSV that passes the known
# fail-safe filters (a real 3D volume and an existing image_path),
# symlinks each source CT into the images/
# dir using nnUNet's `<case>_0000.nii.gz` convention (no copy), runs
# nnUNetv2_predict into PRED_DIR, then writes a new CSV that keeps only
# the rows whose prediction actually landed and adds a `seg_path`
# column pointing at each prediction mask.
#
# Set N_SCANS > 0 to cap the number of scans (handy for smoke tests);
# the default of 0 means "process all matching scans".
#
# All knobs are env-overridable. Defaults match nnunet/configs.yaml.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIN_VOX="${MIN_VOX:-4}"
TARGET_SPACING_MM="${TARGET_SPACING_MM:-1.0}"
MIN_EXTENT_MM="${MIN_EXTENT_MM:-3.0}"

# ── Selection inputs ──────────────────────────────────────────────
INFO_CSV="${INFO_CSV:-/fs5/p_masi/linz18/QA_record/table_collection/PHOTON_CP_image_only_image_info.csv}"
# 0 (default) = no cap, process every matching scan. >0 caps the count.
N_SCANS="${N_SCANS:-0}"

# ── Output layout ─────────────────────────────────────────────────
BASE_DIR="${BASE_DIR:-/fs5/p_masi/linz18/data/nnUNet_prediction/PHOTON_CT_part2}"
IMG_DIR="${IMG_DIR:-${BASE_DIR}/images}"
PRED_DIR="${PRED_DIR:-/fs5/p_masi/linz18/EyeSegmentation/nnUNet_results/predictions/PHOTON_CT_part2}"
# New CSV (kept rows + seg_path column), written next to INFO_CSV.
OUT_CSV="${OUT_CSV:-$(dirname "$INFO_CSV")/PHOTON_CP_QCpart2_image_info.csv}"

# ── Visualization (Step 5) ────────────────────────────────────────
# Path to MRIcroGL_screenshots.py on the cluster.
VIS_SCRIPT="${VIS_SCRIPT:-/home-local/linz18/MRIcroGL_screenshots.py}"
VIS_COMBINED_DIR="${VIS_COMBINED_DIR:-/fs5/p_masi/linz18/QA_record/QA/PHOTON_CP/image-seg2}"
VIS_OPACITY="${VIS_OPACITY:-20}"
VIS_MRICROGL="${VIS_MRICROGL:-/home-local/linz18/MRIcroGL/MRIcroGL}"
# Screenshot resume: MRIcroGL_screenshots.py skips rows whose combined PNG already
# exists under VIS_COMBINED_DIR (default). Set VIS_OVERWRITE=1 to regenerate all.
VIS_OVERWRITE="${VIS_OVERWRITE:-0}"
# Set RUN_VIS=0 to skip the screenshot step entirely.
RUN_VIS="${RUN_VIS:-1}"

# ── nnUNet inference identity (Dataset835 training) ───────────────
DATASET_ID="${DATASET_ID:-835}"
DATASET_NAME="${DATASET_NAME:-PHOTON_CT_QAfiltered}"
CFG="${CFG:-3d_fullres}"
PLAN="${PLAN:-nnUNetPlans}"
TRAINER="${TRAINER:-nnUNetTrainer}"
# Which checkpoint to load. Default = final EMA weights (nnUNet default,
# more stable). Set CHECKPOINT=checkpoint_best.pth to use the best
# online-pseudo-dice epoch instead.
CHECKPOINT="${CHECKPOINT:-checkpoint_final.pth}"
# FOLDS: leave empty to auto-pick the single best-Dice fold from each
# fold's validation/summary.json. Set explicitly (e.g. "0 1 2 3 4") to
# override and ensemble.
FOLDS="${FOLDS:-}"
GPU_ID="${GPU_ID:-0}"
# nnUNet RAM knobs: each preprocessing worker holds a full preprocessed volume in RAM.
# Lower -npp/-nps if you see "Background workers died" (often OOM). Defaults: 1/1.
NPP="${NPP:-1}"
NPS="${NPS:-1}"
# Set NNUNET_VERBOSE=1 to pass --verbose (full stack trace on failure).
NNUNET_VERBOSE="${NNUNET_VERBOSE:-0}"
# Set PREDICT_SEQUENTIAL=1 to predict one case at a time (slower; logs case name on failure).
PREDICT_SEQUENTIAL="${PREDICT_SEQUENTIAL:-0}"
FAILED_CASES_LOG="${FAILED_CASES_LOG:-${PRED_DIR}/failed_predict_cases.log}"
DS_DIR_NAME="$(printf "Dataset%03d_%s" "${DATASET_ID}" "${DATASET_NAME}")"

mkdir -p "$IMG_DIR" "$PRED_DIR"

if [[ ! -f "$INFO_CSV" ]]; then
    echo "[ERROR] image_info CSV not found: $INFO_CSV" >&2
    exit 2
fi

if [[ "${N_SCANS}" -gt 0 ]]; then
    SEL_DESC="first ${N_SCANS} matching scans"
else
    SEL_DESC="all matching scans"
fi
echo "[predict] info_csv : ${INFO_CSV}"
echo "[predict] filter   : 3D volume (size_z>1), valid image_path; ${SEL_DESC}"
echo "[predict] images -> ${IMG_DIR}"
echo "[predict] preds  -> ${PRED_DIR}"
echo "[predict] out_csv-> ${OUT_CSV}"

# ── Step 1: select scans + symlink CTs as <case>_0000.nii.gz ──────
# A case name is session_label + '_' + image_label so multiple images
# per session stay distinct (matches the NIfTI naming in the CSV).
echo -e "\n--- Step 1: select + symlink images ---"
python3 - "$INFO_CSV" "$N_SCANS" "$IMG_DIR" "$MIN_VOX" "$TARGET_SPACING_MM" "$MIN_EXTENT_MM" <<'PY'
import csv, os, sys
from pathlib import Path

info_csv, n_scans, img_dir, min_vox, target_sp, min_ext = sys.argv[1:7]
n_scans = int(n_scans)
img_dir = Path(img_dir)
min_vox, target_sp, min_ext = int(min_vox), float(target_sp), float(min_ext)

try:
    import nibabel as nib
except ImportError:
    nib = None


def volume_ok(path: str):
    """Header-only check: reject volumes that would crash nnUNet resampling."""
    if nib is None:
        return True, ""
    try:
        img = nib.load(path)
        sh = tuple(int(x) for x in img.shape[:3])
        if len(sh) < 3 or any(d <= 0 for d in sh):
            return False, f"degenerate shape {sh}"
        zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
        if len(zooms) < 3 or any(z <= 0 for z in zooms):
            return False, f"invalid voxel spacing {zooms}"
        if min(sh) < min_vox:
            return False, f"axis too short shape={sh} (min_vox={min_vox})"
        for i, (n, sp) in enumerate(zip(sh, zooms)):
            if n * sp < min_ext:
                return False, f"axis {i} extent {n*sp:.2f}mm < {min_ext}mm (shape={sh}, spacing={zooms})"
            if int(round(n * sp / target_sp)) <= 0:
                return False, f"resample would zero axis {i}: shape={sh} spacing={zooms} -> target={target_sp}mm"
        return True, ""
    except Exception as e:
        return False, str(e)


picked = 0
with open(info_csv, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if n_scans > 0 and picked >= n_scans:
            break
        try:
            nz = int(float(row.get("size_z_vox") or 0))
        except ValueError:
            nz = 0
        if nz <= 1:
            continue
        src = (row.get("image_path") or "").strip()
        if not src or not Path(src).is_file():
            continue
        case = f"{row['session_label']}_{row['image_label']}"
        dst = img_dir / f"{case}_0000.nii.gz"
        ok, reason = volume_ok(src)
        if not ok:
            print(f"[skip] unusable volume ({reason}): {case}  <- {src}", file=sys.stderr)
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            continue
        src_abs = os.path.abspath(src)
        if dst.is_symlink() and os.path.realpath(dst) == os.path.realpath(src_abs):
            picked += 1
            continue
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        os.symlink(src_abs, dst)
        picked += 1
        print(f"[link] {case}  <- {src}")

if picked == 0:
    print("[ERROR] no scans matched the filter", file=sys.stderr)
    sys.exit(3)
print(f"[done] staged {picked} scan(s) into {img_dir}")
PY

# ── Step 2: pick the best-Dice fold (unless FOLDS is given) ───────
if [[ -z "${FOLDS// }" ]]; then
    if [[ -z "${nnUNet_results:-}" ]]; then
        echo "[ERROR] nnUNet_results not set; cannot locate fold summaries" >&2
        exit 2
    fi
    MODEL_DIR="${nnUNet_results}/${DS_DIR_NAME}/${TRAINER}__${PLAN}__${CFG}"
    echo -e "\n--- Step 2: select best-Dice fold ---"
    echo "[predict] model_dir: ${MODEL_DIR}"
    _FOLD_PY=$(mktemp /tmp/_pick_fold_XXXXXX.py)
    cat > "$_FOLD_PY" <<'PY'
import json, sys
from pathlib import Path

model_dir = Path(sys.argv[1])
best_fold, best_dice = None, float("-inf")
for summary in sorted(model_dir.glob("fold_*/validation/summary.json")):
    fold = summary.parent.parent.name.split("_", 1)[1]
    try:
        data = json.loads(summary.read_text())
        dice = float(data["foreground_mean"]["Dice"])
    except Exception as e:
        print(f"[skip] {summary}: {e}", file=sys.stderr)
        continue
    print(f"[fold {fold}] mean Dice = {dice:.4f}", file=sys.stderr)
    if dice > best_dice:
        best_fold, best_dice = fold, dice

if best_fold is None:
    print("[ERROR] no fold validation/summary.json found under "
          f"{model_dir}", file=sys.stderr)
    sys.exit(4)
print(f"[best] fold {best_fold} (Dice={best_dice:.4f})", file=sys.stderr)
print(best_fold)
PY
    FOLDS="$(python3 "$_FOLD_PY" "$MODEL_DIR")"
    rm -f "$_FOLD_PY"
    echo "[predict] selected fold: ${FOLDS}"
fi

# ── Step 3: nnUNetv2_predict ──────────────────────────────────────
echo -e "\n--- Step 3: nnUNetv2_predict (Dataset${DATASET_ID}, fold ${FOLDS}) ---"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
echo "[predict] nnUNet workers: -npp ${NPP} -nps ${NPS} (raise if RAM allows, lower on OOM)"
NNUNET_VERBOSE_ARG=()
if [[ "${NNUNET_VERBOSE}" == "1" ]]; then
    NNUNET_VERBOSE_ARG=(--verbose)
fi

# Resume support: by default skip cases whose mask already exists in
# PRED_DIR (nnUNet's --continue_prediction). So if the run is interrupted,
# just re-run the script and it picks up where it left off.
# Set CONTINUE=0 to force a clean re-prediction (overwrite everything).
CONTINUE="${CONTINUE:-1}"
CONTINUE_ARG=()
if [[ "${CONTINUE}" != "0" ]]; then
    CONTINUE_ARG=(--continue_prediction)
    echo "[predict] resume mode: skipping cases already in ${PRED_DIR}"
else
    echo "[predict] CONTINUE=0: re-predicting all cases (overwrite)"
fi

_predict_batch() {
    # shellcheck disable=SC2086
    nnUNetv2_predict \
        -d "${DATASET_ID}" \
        -c "${CFG}" \
        -p "${PLAN}" \
        -tr "${TRAINER}" \
        -f ${FOLDS} \
        -chk "${CHECKPOINT}" \
        -i "${IMG_DIR}" \
        -o "${PRED_DIR}" \
        -npp "${NPP}" \
        -nps "${NPS}" \
        "${NNUNET_VERBOSE_ARG[@]}" \
        "${CONTINUE_ARG[@]}" \
        || echo "[warn] nnUNetv2_predict returned non-zero; continuing with whatever masks were produced" >&2
}

_predict_sequential() {
    local n=0 failed=0 total=0
    for _c in "${IMG_DIR}"/*_0000.nii.gz; do
        [[ -e "${_c}" ]] && total=$((total + 1))
    done
    : > "${FAILED_CASES_LOG}"
    echo "[predict] sequential mode: ${total} case(s); failures -> ${FAILED_CASES_LOG}"

    for src_link in "${IMG_DIR}"/*_0000.nii.gz; do
        [[ -e "${src_link}" ]] || continue
        local case
        case="$(basename "${src_link}" _0000.nii.gz)"
        if [[ "${CONTINUE}" != "0" && -f "${PRED_DIR}/${case}.nii.gz" ]]; then
            echo "[predict] skip existing (${n}/${total}): ${case}"
            continue
        fi
        n=$((n + 1))
        echo -e "\n[predict] ===== case ${n}/${total}: ${case} ====="
        local tmp_in
        tmp_in="$(mktemp -d "/tmp/nnunet_in_${case}_XXXXXX")"
        ln -s "$(readlink -f "${src_link}")" "${tmp_in}/${case}_0000.nii.gz"
        set +e
        # shellcheck disable=SC2086
        nnUNetv2_predict \
            -d "${DATASET_ID}" \
            -c "${CFG}" \
            -p "${PLAN}" \
            -tr "${TRAINER}" \
            -f ${FOLDS} \
            -chk "${CHECKPOINT}" \
            -i "${tmp_in}" \
            -o "${PRED_DIR}" \
            -npp "${NPP}" \
            -nps "${NPS}" \
            "${NNUNET_VERBOSE_ARG[@]}" \
            "${CONTINUE_ARG[@]}"
        local rc=$?
        set -e
        rm -rf "${tmp_in}"
        if [[ "${rc}" -ne 0 ]]; then
            failed=$((failed + 1))
            echo "${case}" >> "${FAILED_CASES_LOG}"
            echo "[FAIL] case=${case} (rc=${rc})" >&2
        fi
    done
    echo "[predict] sequential done: failed ${failed} case(s); see ${FAILED_CASES_LOG}"
}

if [[ "${PREDICT_SEQUENTIAL}" == "1" ]]; then
    _predict_sequential
else
    echo "[predict] batch mode (set PREDICT_SEQUENTIAL=1 to log each case name on failure)"
    _predict_batch
fi

echo -e "\n=== predictions written: ${PRED_DIR} ==="

# ── Step 4: build output CSV (kept rows + seg_path column) ────────
echo -e "\n--- Step 4: write ${OUT_CSV} (rows with a landed prediction) ---"
python3 - "$INFO_CSV" "$N_SCANS" "$PRED_DIR" "$OUT_CSV" <<'PY'
import csv, sys
from pathlib import Path

info_csv, n_scans, pred_dir, out_csv = sys.argv[1:5]
n_scans, pred_dir = int(n_scans), Path(pred_dir)

kept_rows = []
selected = 0
missing_pred = []
with open(info_csv, newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = list(reader.fieldnames or [])
    for row in reader:
        if n_scans > 0 and selected >= n_scans:
            break
        # Mirror Step 1's fail-safe filters exactly so case names line up.
        try:
            if int(float(row.get("size_z_vox") or 0)) <= 1:
                continue
        except ValueError:
            continue
        src = (row.get("image_path") or "").strip()
        if not src or not Path(src).is_file():
            continue
        selected += 1
        case = f"{row['session_label']}_{row['image_label']}"
        pred_p = pred_dir / f"{case}.nii.gz"
        if not pred_p.is_file():
            missing_pred.append(case)
            continue
        row["seg_path"] = str(pred_p)
        kept_rows.append(row)

if "seg_path" not in fieldnames:
    fieldnames.append("seg_path")

Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, restval="", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(kept_rows)

print(f"[done] selected {selected} scan(s); "
      f"kept {len(kept_rows)} with a prediction -> {out_csv}")
if missing_pred:
    print(f"[INCOMPLETE] {len(missing_pred)} of {selected} selected scan(s) "
          f"still have NO prediction (dropped from CSV). Re-run the script to "
          f"resume those: {', '.join(missing_pred)}", file=sys.stderr)
else:
    print(f"[COMPLETE] all {selected} selected scan(s) were predicted.")
PY

echo -e "\n=== Done. predictions: ${PRED_DIR} ==="
echo "=== CSV: ${OUT_CSV} ==="

# ── Step 5 (optional): MRIcroGL QA screenshots ───────────────────
if [[ "${RUN_VIS}" != "1" ]]; then
    echo -e "\n[vis] RUN_VIS!=1, skipping screenshot step."
    exit 0
fi

echo -e "\n--- Step 5: MRIcroGL QA screenshots ---"
echo "[vis] combined : ${VIS_COMBINED_DIR}"
echo "[vis] opacity  : ${VIS_OPACITY}%"

VIS_CSV="${VIS_CSV:-$(dirname "$OUT_CSV")/PHOTON_CP_QCpart2_vis.csv}"

# Step 5a: Always generate VIS_CSV (column mapping for MRIcroGL_screenshots.py)

python3 - "$OUT_CSV" "$VIS_CSV" <<'VIS_PY'
import csv, sys
from pathlib import Path

out_csv, vis_csv = sys.argv[1:3]

REQUIRED = [
    "subject_label", "session_label", "image_label",
    "image_type", "segmentation_type", "image_path", "seg_file_path",
]

rows = []
with open(out_csv, newline="") as f:
    reader = csv.DictReader(f)
    src_fields = list(reader.fieldnames or [])
    for row in reader:
        seg = (row.get("seg_file_path") or row.get("seg_path") or "").strip()
        img = (row.get("image_path") or "").strip()
        if not img or not seg:
            continue
        vis_row = {}
        vis_row["subject_label"] = row.get("subject_label",
                                    row.get("subject",
                                    row.get("session_label", "unknown")))
        vis_row["session_label"] = row.get("session_label",
                                    row.get("session", "unknown"))
        vis_row["image_label"] = row.get("image_label", "1")
        vis_row["image_type"] = row.get("image_type", "CT")
        vis_row["segmentation_type"] = "nnUNet835"
        vis_row["image_path"] = img
        vis_row["seg_file_path"] = seg
        rows.append(vis_row)

Path(vis_csv).parent.mkdir(parents=True, exist_ok=True)
with open(vis_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=REQUIRED)
    writer.writeheader()
    writer.writerows(rows)

print(f"[vis] wrote {vis_csv} ({len(rows)} rows)")
VIS_PY

# Step 5b: Run MRIcroGL only if we have the script and a display
if [[ -z "${VIS_SCRIPT}" || ! -f "${VIS_SCRIPT}" ]]; then
    echo "[vis] VIS_SCRIPT not set or not found (${VIS_SCRIPT:-<empty>})." >&2
    echo "[vis] CSV generated at ${VIS_CSV}" >&2
    echo "[vis] Run MRIcroGL_screenshots.py manually with: --pivot-csv ${VIS_CSV}" >&2
    exit 0
fi
if [[ -z "${DISPLAY:-}" ]] && ! command -v xvfb-run >/dev/null 2>&1; then
    echo "[vis] No DISPLAY and no xvfb-run in PATH; cannot run MRIcroGL here." >&2
    echo "[vis] CSV generated at ${VIS_CSV}" >&2
    echo "[vis] Options: install xvfb, ssh -X, or run on a GUI node:" >&2
    echo "[vis]   python3 ${VIS_SCRIPT} \\" >&2
    echo "[vis]     --pivot-csv ${VIS_CSV} \\" >&2
    echo "[vis]     --combined-dir ${VIS_COMBINED_DIR} \\" >&2
    echo "[vis]     --combined-flat \\" >&2
    echo "[vis]     --overlay-opacity ${VIS_OPACITY}" >&2
    exit 0
fi

echo "[vis] script   : ${VIS_SCRIPT}"
if [[ "${VIS_OVERWRITE}" == "1" ]]; then
    echo "[vis] resume   : off (VIS_OVERWRITE=1, regenerate all combined PNGs)"
    VIS_OVERWRITE_ARG=(--overwrite)
else
    echo "[vis] resume   : on (skip cases whose combined PNG already exists in ${VIS_COMBINED_DIR})"
    VIS_OVERWRITE_ARG=()
fi
echo "[vis] running MRIcroGL_screenshots.py ..."
python3 "${VIS_SCRIPT}" \
    --pivot-csv "${VIS_CSV}" \
    --combined-dir "${VIS_COMBINED_DIR}" \
    --combined-flat \
    --out-dir "${VIS_COMBINED_DIR}/_raw" \
    --script-dir "${VIS_COMBINED_DIR}/_raw/_scripts" \
    --log-csv "${VIS_COMBINED_DIR}/screenshot_log.csv" \
    --skip-log "${VIS_COMBINED_DIR}/skip_screenshot.log" \
    --mricrogl-cmd "${VIS_MRICROGL}" \
    --overlay-opacity "${VIS_OPACITY}" \
    "${VIS_OVERWRITE_ARG[@]}" \
    || echo "[warn] MRIcroGL_screenshots.py returned non-zero" >&2

echo -e "\n=== Step 5 done. Screenshots: ${VIS_COMBINED_DIR} ==="
