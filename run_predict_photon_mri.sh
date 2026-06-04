#!/usr/bin/env bash
# ============================================================
# Predict PHOTON-MRI scans with per-modality nnUNet weights.
#
# Reads PHOTON_MRI_image_only_image_info.csv, classifies each row's
# image_type into T1 / T2 / FLAIR / T1CE / OTHER (OTHER -> T1 weights),
# symlinks into family-specific image dirs, runs nnUNetv2_predict per
# dataset (801-804), writes OUT_CSV with seg_path, then optional
# MRIcroGL QA screenshots.
#
# Modality weights (dataset_id, nnUNet_results root):
#   T1/T1CE fallback-OTHER -> 801 MR-T1w
#   T2                     -> 802 MR-T2w
#   FLAIR                  -> 803 MR-FLAIR
#   T1CE                   -> 804 MR-T1CE
#
# Set N_SCANS > 0 to cap matching scans (smoke tests); 0 = all.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIN_VOX="${MIN_VOX:-4}"
TARGET_SPACING_MM="${TARGET_SPACING_MM:-1.0}"
MIN_EXTENT_MM="${MIN_EXTENT_MM:-3.0}"

# ── Selection inputs ──────────────────────────────────────────────
INFO_CSV="${INFO_CSV:-/fs5/p_masi/linz18/local_projects/eye_segmentation/table_collection/PHOTON_MRI_QC/PHOTON_MRI_image_only_image_info.csv}"
N_SCANS="${N_SCANS:-0}"

# ── Output layout ─────────────────────────────────────────────────
BASE_DIR="${BASE_DIR:-/fs5/p_masi/linz18/data/nnUNet_prediction/PHOTON_MRI_QC}"
IMG_DIR="${IMG_DIR:-${BASE_DIR}/images}"
PRED_DIR="${PRED_DIR:-/fs5/p_masi/linz18/EyeSegmentation/nnUNet_results/predictions/PHOTON_MRI_QC}"
OUT_CSV="${OUT_CSV:-$(dirname "$INFO_CSV")/PHOTON_MRI_QCpart2_image_info.csv}"
MANIFEST_CSV="${MANIFEST_CSV:-${BASE_DIR}/staging_manifest.csv}"

# ── Per-modality nnUNet (801-804 on liux64 pipeline) ──────────────
NNUNET_RESULTS_ROOT="${NNUNET_RESULTS_ROOT:-/fs5/p_masi/liux64/eye_nnunet_pipeline/nnUNet_results}"
# Dataset folder names under nnUNet_results (Dataset###_<name>)
DS801_NAME="${DS801_NAME:-MR-T1w}"
DS802_NAME="${DS802_NAME:-MR-T2w}"
DS803_NAME="${DS803_NAME:-MR-FLAIR}"
DS804_NAME="${DS804_NAME:-MR-T1CE}"

CFG="${CFG:-3d_fullres}"
PLAN="${PLAN:-nnUNetPlans}"
TRAINER="${TRAINER:-nnUNetTrainer}"
CHECKPOINT="${CHECKPOINT:-checkpoint_final.pth}"
# Per-family folds: FOLDS_T1, FOLDS_T2, ... or empty -> auto best-Dice
FOLDS_T1="${FOLDS_T1:-}"
FOLDS_T2="${FOLDS_T2:-}"
FOLDS_FLAIR="${FOLDS_FLAIR:-}"
FOLDS_T1CE="${FOLDS_T1CE:-}"
GPU_ID="${GPU_ID:-1}"
CONTINUE="${CONTINUE:-1}"
# nnUNet RAM: -npp/-nps = preprocessing / export worker count (default 1/1).
NPP="${NPP:-1}"
NPS="${NPS:-1}"
NNUNET_VERBOSE="${NNUNET_VERBOSE:-0}"

# ── Visualization (Step 5) ────────────────────────────────────────
VIS_SCRIPT="${VIS_SCRIPT:-/home-local/linz18/MRIcroGL_screenshots.py}"
VIS_COMBINED_DIR="${VIS_COMBINED_DIR:-/fs5/p_masi/linz18/QA_record/QA/PHOTON_MRI/image-seg2}"
VIS_OPACITY="${VIS_OPACITY:-20}"
VIS_MRICROGL="${VIS_MRICROGL:-/home-local/linz18/MRIcroGL/MRIcroGL}"
VIS_OVERWRITE="${VIS_OVERWRITE:-0}"
RUN_VIS="${RUN_VIS:-1}"

mkdir -p "$IMG_DIR" "$PRED_DIR" "${BASE_DIR}"

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
echo "[predict] images -> ${IMG_DIR}/{T1,T2,FLAIR,T1CE}"
echo "[predict] preds  -> ${PRED_DIR}"
echo "[predict] out_csv-> ${OUT_CSV}"
echo "[predict] nnUNet_results root: ${NNUNET_RESULTS_ROOT}"

# ── Step 1: classify modality + symlink per family ───────────────
echo -e "\n--- Step 1: classify + symlink images ---"
python3 - "$INFO_CSV" "$N_SCANS" "$IMG_DIR" "$MANIFEST_CSV" "$MIN_VOX" "$TARGET_SPACING_MM" "$MIN_EXTENT_MM" <<'PY'
import csv, os, re, sys
from pathlib import Path

info_csv, n_scans, img_dir, manifest_csv, min_vox, target_sp, min_ext = sys.argv[1:8]
n_scans = int(n_scans)
img_dir = Path(img_dir)
manifest_csv = Path(manifest_csv)
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


_re_t1like = re.compile(r"(t1|t1w|mpr|fspgr|spgr)")
_re_t2like = re.compile(r"(t2|t2w)")
_re_contrast = re.compile(r"(t1gd|t1ce|post.?g(ad|d)|\bgd\b|contrast|gad|gadol)")
_re_po = re.compile(r"(po$|\bpo\b)")


def classify_family(image_type: str) -> str:
    s = str(image_type).strip().lower()
    if "flair" in s:
        return "FLAIR"
    if _re_contrast.search(s):
        return "T1CE"
    if _re_t1like.search(s) and ("post" in s or _re_po.search(s)):
        return "T1CE"
    if _re_t2like.search(s):
        return "T2"
    if _re_t1like.search(s):
        return "T1"
    return "OTHER"


FAMILY_ROUTE = {"T1": "T1", "T2": "T2", "FLAIR": "FLAIR", "T1CE": "T1CE", "OTHER": "T1"}
FAMILY_TO_DATASET = {"T1": 801, "T2": 802, "FLAIR": 803, "T1CE": 804, "OTHER": 801}

picked = 0
manifest_rows = []
counts = {"T1": 0, "T2": 0, "FLAIR": 0, "T1CE": 0, "OTHER": 0}

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

        image_type = row.get("image_type") or ""
        family = classify_family(image_type)
        route = FAMILY_ROUTE[family]
        case = f"{row['session_label']}_{row['image_label']}"
        subdir = img_dir / route
        subdir.mkdir(parents=True, exist_ok=True)
        dst = subdir / f"{case}_0000.nii.gz"

        ok, reason = volume_ok(src)
        if not ok:
            print(f"[skip] unusable volume ({reason}): {case}  <- {src}", file=sys.stderr)
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            continue

        counts[family] += 1
        src_abs = os.path.abspath(src)
        if dst.is_symlink() and os.path.realpath(dst) == os.path.realpath(src_abs):
            picked += 1
        else:
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            os.symlink(src_abs, dst)
            picked += 1
            print(f"[link] {case}  family={family} -> route={route} "
                  f"(ds={FAMILY_TO_DATASET[family]})  <- {src}")

        manifest_rows.append({
            "case": case,
            "session_label": row.get("session_label", ""),
            "image_label": row.get("image_label", ""),
            "image_type": image_type,
            "modality_family": family,
            "predict_route": route,
            "dataset_id": str(FAMILY_TO_DATASET[family]),
        })

if picked == 0:
    print("[ERROR] no scans matched the filter", file=sys.stderr)
    sys.exit(3)

manifest_csv.parent.mkdir(parents=True, exist_ok=True)
with open(manifest_csv, "w", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=[
            "case", "session_label", "image_label", "image_type",
            "modality_family", "predict_route", "dataset_id",
        ],
    )
    w.writeheader()
    w.writerows(manifest_rows)

print(f"[done] staged {picked} scan(s); family counts: {counts}")
print(f"[done] manifest -> {manifest_csv}")
PY

# ── Step 2+3: per-family best fold + nnUNetv2_predict ─────────────
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export nnUNet_results="${NNUNET_RESULTS_ROOT}"
echo "[predict] nnUNet workers: -npp ${NPP} -nps ${NPS} (raise if RAM allows, lower on OOM)"
NNUNET_VERBOSE_ARG=()
if [[ "${NNUNET_VERBOSE}" == "1" ]]; then
    NNUNET_VERBOSE_ARG=(--verbose)
fi

CONTINUE_ARG=()
if [[ "${CONTINUE}" != "0" ]]; then
    CONTINUE_ARG=(--continue_prediction)
    echo "[predict] resume mode: skipping cases already in ${PRED_DIR}"
else
    echo "[predict] CONTINUE=0: re-predicting all cases (overwrite)"
fi

# Resolve model_dir (flexible layout) + best fold. Args:
#   results_root dataset_id dataset_name trainer plan cfg [explicit_model_dir]
# Prints two lines: model_dir<TAB>fold
_resolve_model_and_fold() {
    local _py
    _py=$(mktemp /tmp/_resolve_mri_model_XXXXXX.py)
    cat > "$_py" <<'PY'
import json, sys
from pathlib import Path

results_root = Path(sys.argv[1])
ds_id = int(sys.argv[2])
ds_name = sys.argv[3]
trainer = sys.argv[4]
plan = sys.argv[5]
cfg = sys.argv[6]
explicit = sys.argv[7].strip() if len(sys.argv) > 7 else ""

ds_tag = f"Dataset{ds_id:03d}"


def has_fold_summaries(model_dir: Path) -> bool:
    return any(model_dir.glob("fold_*/validation/summary.json"))


def pick_best_fold(model_dir: Path):
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
    return best_fold, best_dice


def candidate_dataset_dirs() -> list:
    out = []
    if explicit:
        p = Path(explicit)
        if p.is_dir():
            out.append(p)
    named = results_root / f"{ds_tag}_{ds_name}"
    if named.is_dir() and named not in out:
        out.append(named)
    for p in sorted(results_root.glob(f"{ds_tag}_*")):
        if p.is_dir() and p not in out:
            out.append(p)
    return out


def candidate_model_dirs(ds_dir: Path) -> list:
    found = []
    preferred = ds_dir / f"{trainer}__{plan}__{cfg}"
    if preferred.is_dir():
        found.append(preferred)
    for p in sorted(ds_dir.iterdir()):
        if not p.is_dir() or p in found:
            continue
        if p.name.startswith(f"{trainer}__") or "Trainer" in p.name:
            found.append(p)
    # Nested e.g. nnUNetTrainer_resampling_results/fold_0/...
    for summary in sorted(ds_dir.rglob("fold_*/validation/summary.json")):
        model = summary.parent.parent.parent
        if model.is_dir() and model not in found:
            found.append(model)
    return found


if not results_root.is_dir():
    print(f"[ERROR] nnUNet_results root not found: {results_root}", file=sys.stderr)
    sys.exit(2)

ds_dirs = candidate_dataset_dirs()
if not ds_dirs:
    print(f"[ERROR] no {ds_tag}_* under {results_root}", file=sys.stderr)
    print(f"[hint] ls {results_root} | grep {ds_tag}", file=sys.stderr)
    sys.exit(3)

best = None  # (dice, model_dir, fold)
for ds_dir in ds_dirs:
    print(f"[scan] dataset dir: {ds_dir}", file=sys.stderr)
    for model_dir in candidate_model_dirs(ds_dir):
        if not has_fold_summaries(model_dir):
            print(f"[skip] no fold summaries: {model_dir}", file=sys.stderr)
            continue
        fold, dice = pick_best_fold(model_dir)
        if fold is None:
            continue
        print(f"[candidate] {model_dir} fold={fold} Dice={dice:.4f}", file=sys.stderr)
        if best is None or dice > best[0]:
            best = (dice, model_dir, fold)

if best is None:
    print("[ERROR] no fold_*/validation/summary.json under any candidate model dir.", file=sys.stderr)
    print("[hint] Check NNUNET_RESULTS_ROOT, DS*_NAME, TRAINER/PLAN/CFG, or set MODEL_DIR_T1=...", file=sys.stderr)
    for ds_dir in ds_dirs:
        print(f"  contents of {ds_dir}:", file=sys.stderr)
        for child in sorted(ds_dir.iterdir())[:20]:
            print(f"    {child.name}", file=sys.stderr)
    sys.exit(4)

def infer_trainer_plan_cfg(model_dir: Path):
    name = model_dir.name
    if "__" in name:
        parts = name.split("__")
        if len(parts) >= 3:
            return parts[0], parts[1], parts[2]
    if name.startswith("nnUNetTrainer"):
        return name, plan, cfg
    return trainer, plan, cfg


_, model_dir, fold = best
tr_run, plan_run, cfg_run = infer_trainer_plan_cfg(model_dir)
print(f"[best] {model_dir} fold {fold} (Dice={best[0]:.4f})", file=sys.stderr)
print(f"[best] nnUNet -tr {tr_run} -p {plan_run} -c {cfg_run}", file=sys.stderr)
print(f"{model_dir}\t{fold}\t{tr_run}\t{plan_run}\t{cfg_run}")
PY
    python3 "$_py" "$@"
    rm -f "$_py"
}

_run_family_predict() {
    local route="$1"       # T1 | T2 | FLAIR | T1CE
    local ds_id="$2"
    local ds_name="$3"
    local folds_var="FOLDS_${route}"
    local modeldir_var="MODEL_DIR_${route}"
    local folds="${!folds_var:-}"
    local explicit_model_dir="${!modeldir_var:-}"
    local img_sub="${IMG_DIR}/${route}"

    if [[ ! -d "$img_sub" ]] || [[ -z "$(find "$img_sub" -maxdepth 1 -name '*_0000.nii.gz' -print -quit 2>/dev/null)" ]]; then
        echo "[predict] ${route}: no staged images, skipping nnUNet"
        return 0
    fi

    echo -e "\n--- Step 3: nnUNetv2_predict ${route} (Dataset${ds_id}, ${ds_name}) ---"
    echo "[predict] images: ${img_sub}"

    local resolved model_dir tr_run plan_run cfg_run
    tr_run="${TRAINER}"
    plan_run="${PLAN}"
    cfg_run="${CFG}"
    if [[ -z "${folds// }" ]]; then
        echo "--- Step 2 (${route}): resolve model dir + best-Dice fold ---"
        resolved="$(_resolve_model_and_fold \
            "${NNUNET_RESULTS_ROOT}" "${ds_id}" "${ds_name}" \
            "${TRAINER}" "${PLAN}" "${CFG}" "${explicit_model_dir}")"
        IFS=$'\t' read -r model_dir folds tr_run plan_run cfg_run <<< "${resolved}"
    else
        if [[ -n "${explicit_model_dir}" ]]; then
            model_dir="${explicit_model_dir}"
        else
            model_dir="${NNUNET_RESULTS_ROOT}/Dataset$(printf '%03d' "${ds_id}")_${ds_name}/${TRAINER}__${PLAN}__${CFG}"
        fi
        echo "[predict] ${route} using FOLDS_${route}=${folds}"
    fi

    if [[ -z "${folds// }" ]]; then
        echo "[ERROR] ${route}: empty fold; cannot run nnUNetv2_predict" >&2
        return 1
    fi
    if [[ ! -d "${model_dir}" ]]; then
        echo "[ERROR] ${route}: model_dir not found: ${model_dir}" >&2
        return 1
    fi

    echo "[predict] model_dir: ${model_dir}"
    echo "[predict] ${route} selected fold: ${folds}"
    echo "[predict] ${route} trainer/plan/cfg: ${tr_run} / ${plan_run} / ${cfg_run}"

    # shellcheck disable=SC2086
    nnUNetv2_predict \
        -d "${ds_id}" \
        -c "${cfg_run}" \
        -p "${plan_run}" \
        -tr "${tr_run}" \
        -f ${folds} \
        -chk "${CHECKPOINT}" \
        -i "${img_sub}" \
        -o "${PRED_DIR}" \
        -npp "${NPP}" \
        -nps "${NPS}" \
        "${NNUNET_VERBOSE_ARG[@]}" \
        "${CONTINUE_ARG[@]}" \
        || echo "[warn] nnUNetv2_predict ${route} (Dataset${ds_id}) returned non-zero" >&2
}

_run_family_predict "T1" 801 "${DS801_NAME}"
_run_family_predict "T2" 802 "${DS802_NAME}"
_run_family_predict "FLAIR" 803 "${DS803_NAME}"
_run_family_predict "T1CE" 804 "${DS804_NAME}"

echo -e "\n=== predictions written: ${PRED_DIR} ==="

# ── Step 4: build output CSV (kept rows + seg_path + modality) ────
echo -e "\n--- Step 4: write ${OUT_CSV} (rows with a landed prediction) ---"
python3 - "$INFO_CSV" "$N_SCANS" "$PRED_DIR" "$OUT_CSV" "$MANIFEST_CSV" <<'PY'
import csv, re, sys
from pathlib import Path

info_csv, n_scans, pred_dir, out_csv, manifest_csv = sys.argv[1:6]
n_scans = int(n_scans)
pred_dir = Path(pred_dir)

_re_t1like = re.compile(r"(t1|t1w|mpr|fspgr|spgr)")
_re_t2like = re.compile(r"(t2|t2w)")
_re_contrast = re.compile(r"(t1gd|t1ce|post.?g(ad|d)|\bgd\b|contrast|gad|gadol)")
_re_po = re.compile(r"(po$|\bpo\b)")


def classify_family(image_type: str) -> str:
    s = str(image_type).strip().lower()
    if "flair" in s:
        return "FLAIR"
    if _re_contrast.search(s):
        return "T1CE"
    if _re_t1like.search(s) and ("post" in s or _re_po.search(s)):
        return "T1CE"
    if _re_t2like.search(s):
        return "T2"
    if _re_t1like.search(s):
        return "T1"
    return "OTHER"


FAMILY_TO_DATASET = {"T1": 801, "T2": 802, "FLAIR": 803, "T1CE": 804, "OTHER": 801}

# Optional manifest for consistency with Step 1
manifest_by_case = {}
if Path(manifest_csv).is_file():
    with open(manifest_csv, newline="") as mf:
        for mrow in csv.DictReader(mf):
            manifest_by_case[mrow["case"]] = mrow

kept_rows = []
selected = 0
missing_pred = []
with open(info_csv, newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = list(reader.fieldnames or [])
    for row in reader:
        if n_scans > 0 and selected >= n_scans:
            break
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

        m = manifest_by_case.get(case, {})
        family = m.get("modality_family") or classify_family(row.get("image_type") or "")
        ds_id = m.get("dataset_id") or str(FAMILY_TO_DATASET.get(family, 801))

        row["modality_family"] = family
        row["nnunet_dataset_id"] = ds_id
        row["seg_path"] = str(pred_p)
        kept_rows.append(row)

for col in ("modality_family", "nnunet_dataset_id", "seg_path"):
    if col not in fieldnames:
        fieldnames.append(col)

Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, restval="", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(kept_rows)

print(f"[done] selected {selected} scan(s); "
      f"kept {len(kept_rows)} with a prediction -> {out_csv}")
if missing_pred:
    print(
        f"[INCOMPLETE] {len(missing_pred)} of {selected} selected scan(s) "
        f"still have NO prediction (dropped from CSV). Re-run to resume: "
        f"{', '.join(missing_pred[:20])}"
        + (" ..." if len(missing_pred) > 20 else ""),
        file=sys.stderr,
    )
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

VIS_CSV="${VIS_CSV:-$(dirname "$OUT_CSV")/PHOTON_MRI_QCpart2_vis.csv}"

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
    for row in csv.DictReader(f):
        seg = (row.get("seg_file_path") or row.get("seg_path") or "").strip()
        img = (row.get("image_path") or "").strip()
        if not img or not seg:
            continue
        ds = (row.get("nnunet_dataset_id") or "").strip()
        seg_type = f"nnUNet{ds}" if ds else "nnUNetMRI"
        rows.append({
            "subject_label": row.get("subject_label", row.get("subject", row.get("session_label", "unknown"))),
            "session_label": row.get("session_label", row.get("session", "unknown")),
            "image_label": row.get("image_label", "1"),
            "image_type": row.get("image_type", "MRI"),
            "segmentation_type": seg_type,
            "image_path": img,
            "seg_file_path": seg,
        })

Path(vis_csv).parent.mkdir(parents=True, exist_ok=True)
with open(vis_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=REQUIRED)
    w.writeheader()
    w.writerows(rows)

print(f"[vis] wrote {vis_csv} ({len(rows)} rows)")
VIS_PY

if [[ -z "${VIS_SCRIPT}" || ! -f "${VIS_SCRIPT}" ]]; then
    echo "[vis] VIS_SCRIPT not set or not found (${VIS_SCRIPT:-<empty>})." >&2
    echo "[vis] CSV generated at ${VIS_CSV}" >&2
    exit 0
fi
if [[ -z "${DISPLAY:-}" ]] && ! command -v xvfb-run >/dev/null 2>&1; then
    echo "[vis] No DISPLAY and no xvfb-run; CSV at ${VIS_CSV}" >&2
    exit 0
fi

echo "[vis] script   : ${VIS_SCRIPT}"
if [[ "${VIS_OVERWRITE}" == "1" ]]; then
    echo "[vis] resume   : off (VIS_OVERWRITE=1)"
    VIS_OVERWRITE_ARG=(--overwrite)
else
    echo "[vis] resume   : on (skip existing combined PNGs)"
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
