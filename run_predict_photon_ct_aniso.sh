#!/usr/bin/env bash
# ============================================================
# Predict PHOTON-CT scans with the Dataset835-trained nnUNet.
#
# Selects the first N scans (default 8) from an image_info CSV whose
# `anisotropy_ratio` < threshold (default 1.3), copies each source
# CT into the images/ dir using nnUNet's `<case>_0000.nii.gz`
# convention, then runs nnUNetv2_predict into the preds/ dir.
#
# All knobs are env-overridable. Defaults match nnunet/configs.yaml.
# ============================================================
set -euo pipefail

# ── Selection inputs ──────────────────────────────────────────────
INFO_CSV="${INFO_CSV:-/fs5/p_masi/linz18/QA_record/table_collection/PHOTON_CP_image_only_image_info.csv}"
ANISO_MAX="${ANISO_MAX:-1.3}"
N_SCANS="${N_SCANS:-8}"

# ── Output layout ─────────────────────────────────────────────────
BASE_DIR="${BASE_DIR:-/fs5/p_masi/linz18/data/nnUNet_prediction/PHOTON_CT}"
IMG_DIR="${IMG_DIR:-${BASE_DIR}/images}"
PRED_DIR="${PRED_DIR:-${BASE_DIR}/preds}"

# ── nnUNet inference identity (Dataset835 training) ───────────────
DATASET_ID="${DATASET_ID:-835}"
DATASET_NAME="${DATASET_NAME:-PHOTON_CT_QAfiltered}"
CFG="${CFG:-3d_fullres}"
PLAN="${PLAN:-nnUNetPlans}"
TRAINER="${TRAINER:-nnUNetTrainer}"
# FOLDS: leave empty to auto-pick the single best-Dice fold from each
# fold's validation/summary.json. Set explicitly (e.g. "0 1 2 3 4") to
# override and ensemble.
FOLDS="${FOLDS:-}"
GPU_ID="${GPU_ID:-0}"
DS_DIR_NAME="$(printf "Dataset%03d_%s" "${DATASET_ID}" "${DATASET_NAME}")"

mkdir -p "$IMG_DIR" "$PRED_DIR"

if [[ ! -f "$INFO_CSV" ]]; then
    echo "[ERROR] image_info CSV not found: $INFO_CSV" >&2
    exit 2
fi

echo "[predict] info_csv : ${INFO_CSV}"
echo "[predict] filter   : anisotropy_ratio < ${ANISO_MAX}, first ${N_SCANS} scans"
echo "[predict] images -> ${IMG_DIR}"
echo "[predict] preds  -> ${PRED_DIR}"

# ── Step 1: select scans + stage CTs as <case>_0000.nii.gz ────────
# A case name is session_label + '_' + image_label so multiple images
# per session stay distinct (matches the NIfTI naming in the CSV).
echo -e "\n--- Step 1: select + copy images ---"
python3 - "$INFO_CSV" "$ANISO_MAX" "$N_SCANS" "$IMG_DIR" <<'PY'
import csv, shutil, sys
from pathlib import Path

info_csv, aniso_max, n_scans, img_dir = sys.argv[1:5]
aniso_max = float(aniso_max)
n_scans = int(n_scans)
img_dir = Path(img_dir)

picked = 0
with open(info_csv, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if picked >= n_scans:
            break
        raw = (row.get("anisotropy_ratio") or "").strip()
        try:
            aniso = float(raw)
        except ValueError:
            continue
        if aniso >= aniso_max:
            continue
        # Skip single-slice/2D images (scout, MEDRAD injection localizers).
        # They are isotropic 1x1x1 so they pass the aniso filter, but a
        # 1-slice volume breaks nnUNet 3d_fullres preprocessing.
        try:
            nz = int(float(row.get("size_z_vox") or 0))
        except ValueError:
            nz = 0
        if nz <= 1:
            print(f"[skip] not a 3D volume (size_z={nz}): "
                  f"{row['session_label']}_{row['image_label']}", file=sys.stderr)
            continue
        src = (row.get("image_path") or "").strip()
        if not src or not Path(src).is_file():
            print(f"[skip] missing image_path: {src!r}", file=sys.stderr)
            continue
        case = f"{row['session_label']}_{row['image_label']}"
        dst = img_dir / f"{case}_0000.nii.gz"
        shutil.copyfile(src, dst)
        picked += 1
        print(f"[copy] {case}  (aniso={aniso:.3f})  <- {src}")

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
    FOLDS="$(python3 - "$MODEL_DIR" <<'PY'
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
)"
    echo "[predict] selected fold: ${FOLDS}"
fi

# ── Step 3: nnUNetv2_predict ──────────────────────────────────────
echo -e "\n--- Step 3: nnUNetv2_predict (Dataset${DATASET_ID}, fold ${FOLDS}) ---"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# shellcheck disable=SC2086
nnUNetv2_predict \
    -d "${DATASET_ID}" \
    -c "${CFG}" \
    -p "${PLAN}" \
    -tr "${TRAINER}" \
    -f ${FOLDS} \
    -i "${IMG_DIR}" \
    -o "${PRED_DIR}"

echo -e "\n=== Done. predictions: ${PRED_DIR} ==="
