#!/usr/bin/env bash
# One-shot diagnostic for the cascade-predict IndexError(normalization_schemes[c]).
# Prints: (1) actual channel files per case in the predict input folder,
#         (2) the plan the PREDICTOR loads (results-folder plans.json) -> normalization_schemes,
#         (3) the dataset.json the predictor loads -> channel_names count.
set -euo pipefail
IMG="${1:?usage: _diag_predict_channels.sh <imagesTs_dir> [results_model_dir]}"
MODEL="${2:-}"

echo "== (1) channel files per case in: $IMG =="
if [[ -d "$IMG" ]]; then
  # strip the trailing _NNNN.nii.gz and count how many channel files each case has
  ls -1 "$IMG" | sed -E 's/_[0-9]{4}\.nii\.gz$//' | sort | uniq -c
  echo "-- raw listing (first 12) --"
  ls -1 "$IMG" | head -12
else
  echo "  (dir not found)"
fi

echo
echo "== (2)+(3) plan/dataset the PREDICTOR actually loads =="
python3 - "$MODEL" <<'PY'
import json, os, sys, glob
model = sys.argv[1] if len(sys.argv) > 1 else ""
res = os.environ.get("nnUNet_results", "")
if not model:
    # best-effort: find the OrbitalCascade fold_0 model dir for 855
    cands = glob.glob(os.path.join(res, "Dataset855_*",
                     "nnUNetTrainer_OrbitalCascade__nnUNetPlansFinetune__*"))
    model = cands[0] if cands else ""
print("model dir:", model or "(not found; pass it as arg 2)")
if model:
    pj = os.path.join(model, "plans.json")
    dj = os.path.join(model, "dataset.json")
    if os.path.isfile(pj):
        p = json.load(open(pj))
        for cfgname, c in p.get("configurations", {}).items():
            ns = c.get("normalization_schemes")
            if ns is not None or c.get("previous_stage"):
                print(f"  cfg {cfgname}: normalization_schemes={ns} "
                      f"(len={len(ns) if isinstance(ns,list) else 'n/a'})  "
                      f"previous_stage={c.get('previous_stage')}")
    else:
        print("  no plans.json in model dir")
    if os.path.isfile(dj):
        d = json.load(open(dj))
        cn = d.get("channel_names") or d.get("modality")
        print(f"  dataset.json channel_names={cn} (n={len(cn) if cn else 'n/a'})")
    else:
        print("  no dataset.json in model dir")
PY
