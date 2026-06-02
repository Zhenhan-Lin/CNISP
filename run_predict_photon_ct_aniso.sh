#!/usr/bin/env bash
# ============================================================
# Predict PHOTON-CT scans with the Dataset835-trained nnUNet.
#
# Selects EVERY scan in an image_info CSV that passes the known
# fail-safe filters (anisotropy_ratio < threshold, a real 3D volume,
# and an existing image_path), symlinks each source CT into the images/
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

# ── Selection inputs ──────────────────────────────────────────────
INFO_CSV="${INFO_CSV:-/fs5/p_masi/linz18/QA_record/table_collection/PHOTON_CP_image_only_image_info.csv}"
ANISO_MAX="${ANISO_MAX:-1.3}"
# 0 (default) = no cap, process every matching scan. >0 caps the count.
N_SCANS="${N_SCANS:-0}"

# ── Output layout ─────────────────────────────────────────────────
BASE_DIR="${BASE_DIR:-/fs5/p_masi/linz18/data/nnUNet_prediction/PHOTON_CT_part2}"
IMG_DIR="${IMG_DIR:-${BASE_DIR}/images}"
PRED_DIR="${PRED_DIR:-/fs5/p_masi/linz18/EyeSegmentation/nnUNet_results/predictions/PHOTON_CT_part2}"
# New CSV (kept rows + seg_path column), written next to INFO_CSV.
OUT_CSV="${OUT_CSV:-$(dirname "$INFO_CSV")/PHOTON_CP_QCpart2_image_info.csv}"

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
echo "[predict] filter   : anisotropy_ratio < ${ANISO_MAX}, 3D volume, valid path; ${SEL_DESC}"
echo "[predict] images -> ${IMG_DIR}"
echo "[predict] preds  -> ${PRED_DIR}"
echo "[predict] out_csv-> ${OUT_CSV}"

# ── Step 1: select scans + symlink CTs as <case>_0000.nii.gz ──────
# A case name is session_label + '_' + image_label so multiple images
# per session stay distinct (matches the NIfTI naming in the CSV).
echo -e "\n--- Step 1: select + symlink images ---"
python3 - "$INFO_CSV" "$ANISO_MAX" "$N_SCANS" "$IMG_DIR" <<'PY'
import csv, os, sys
from pathlib import Path

info_csv, aniso_max, n_scans, img_dir = sys.argv[1:5]
aniso_max = float(aniso_max)
n_scans = int(n_scans)
img_dir = Path(img_dir)

picked = 0
with open(info_csv, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if n_scans > 0 and picked >= n_scans:
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
        # Symlink instead of copy: nnUNet just needs the <case>_0000.nii.gz
        # naming convention in the input dir, no need to duplicate the CT.
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        os.symlink(os.path.abspath(src), dst)
        picked += 1
        print(f"[link] {case}  (aniso={aniso:.3f})  <- {src}")

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

# Don't let a single bad case abort the run: keep whatever masks landed
# so Step 4 can still build the CSV from the successful predictions.
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
    || echo "[warn] nnUNetv2_predict returned non-zero; continuing with whatever masks were produced" >&2

echo -e "\n=== predictions written: ${PRED_DIR} ==="

# ── Step 4: build output CSV (kept rows + seg_path column) ────────
echo -e "\n--- Step 4: write ${OUT_CSV} (rows with a landed prediction) ---"
python3 - "$INFO_CSV" "$ANISO_MAX" "$N_SCANS" "$PRED_DIR" "$OUT_CSV" <<'PY'
import csv, sys
from pathlib import Path

info_csv, aniso_max, n_scans, pred_dir, out_csv = sys.argv[1:6]
aniso_max, n_scans, pred_dir = float(aniso_max), int(n_scans), Path(pred_dir)

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
            if float(row.get("anisotropy_ratio") or "nan") >= aniso_max:
                continue
        except ValueError:
            continue
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
    print(f"[note] {len(missing_pred)} selected scan(s) had no prediction "
          f"(dropped): {', '.join(missing_pred)}", file=sys.stderr)
PY

echo -e "\n=== Done. predictions: ${PRED_DIR} ==="
echo "=== CSV: ${OUT_CSV} ==="
