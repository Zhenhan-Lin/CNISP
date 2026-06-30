#!/usr/bin/env python3
"""Select QA-checklist images, thick-degrade them, and stage under data/.

Selection (corrector.yaml::corrector_data.select):
    keep == False AND qa_status == "yes", scanning the checklist in order.

How many to take:
    * target_samples (preferred): stop once this many kept (case, step) SAMPLES
      have been produced (image + prelabel units). Use this to hit an exact
      training-set size from a larger pool. Folder-aware (existing degraded
      images are reused and counted).
    * n (legacy): take the first `n` SOURCE IMAGES (cases); each fans out into
      several (case, step) samples, so this can't target an exact sample count.

For each selected image, apply THICK degradation (reusing the existing pipeline's
nnunet.sparsify_inputs._sparsify_one_ct) along the through-plane axis at each
configured step size. For steps in `thick_threshold_steps`, the (case, step)
variant is dropped if slice thickness (through-plane spacing * step) exceeds
`thick_threshold_mm`.

Outputs (under data_root):
    images/{case_id}_step{XX}_0000.nii.gz     degraded CT (nnUNet channel-0 name)
    corrector_data_manifest.json              per-case provenance + per-step status
    corrector_cases.txt                       one "{case_id}_step{XX}" per line
(nnunet_pred/ and cnisp_pred/ are created empty for the downstream predictions.)

Usage:
    python nnunet-c/scripts/build_corrector_data.py
    python nnunet-c/scripts/build_corrector_data.py --n 200 --steps 3,6,9
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.config import add_repo_to_syspath, load_corrector_config  # noqa: E402

_REPO = add_repo_to_syspath(__file__)

import numpy as np  # noqa: E402
import nibabel as nib  # noqa: E402

from nnunet.sparsify_inputs import _sparsify_one_ct  # noqa: E402

_DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "corrector.yaml"


def _match(row_val: str, want) -> bool:
    """Match a CSV cell against a YAML selector (bool or string)."""
    rv = str(row_val).strip().lower()
    if isinstance(want, bool):
        return rv in ({"true", "yes", "1"} if want else {"false", "no", "0"})
    return rv == str(want).strip().lower()


def _resolve(path_str: str, root: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (root / p)


def iter_candidate_rows(csv_path: Path, sel: dict):
    """Yield checklist rows matching the selector whose source image exists.

    Lazy on purpose: the caller stops pulling once it has enough (either `n`
    source images in legacy mode, or `target_samples` (case, step) samples in
    target mode), so we never scan the whole (large) checklist when only a few
    hundred samples are needed.
    """
    keep_want = sel.get("keep", False)
    qa_want = sel.get("qa_status", True)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if not _match(row.get("keep", ""), keep_want):
                continue
            if not _match(row.get("qa_status", ""), qa_want):
                continue
            img = (row.get("image_path") or "").strip()
            if not img or not Path(img).exists():
                continue
            yield row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(_DEFAULT_CONFIG))
    ap.add_argument("--n", type=int, default=None,
                    help="legacy: number of SOURCE IMAGES (cases) to take. Used "
                         "only when --target-samples / corrector_data.target_samples "
                         "is unset.")
    ap.add_argument("--target-samples", type=int, default=None,
                    help="stop once this many kept (case, step) SAMPLES "
                         "(image+prelabel units) have been selected, scanning the "
                         "pool in checklist order. Overrides --n. Folder-aware: "
                         "existing degraded images are reused (not regenerated) "
                         "and still count toward the target, so re-running tops up "
                         "to the target and stops.")
    ap.add_argument("--steps", default=None, help="override steps, e.g. 3,6,9")
    ap.add_argument("--force", action="store_true",
                    help="re-degrade even if the output image exists")
    args = ap.parse_args()

    cfg = load_corrector_config(args.config, caller_file=__file__)
    cd = cfg["corrector_data"]
    root = cfg["_resolved"]["repo_root"]

    csv_path = _resolve(cd["checklist_csv"], root)
    n = int(args.n if args.n is not None else cd["n"])
    steps = ([int(s) for s in args.steps.split(",")] if args.steps
             else [int(s) for s in cd["steps"]])
    thresh_mm = float(cd.get("thick_threshold_mm", 10.0))
    thresh_steps = {int(s) for s in cd.get("thick_threshold_steps", [])}
    modality = cd.get("modality", "ct")

    data_root = _resolve(cd["data_root"], root)
    images_dir = data_root / cd.get("images_dirname", "images")
    images_dir.mkdir(parents=True, exist_ok=True)
    (data_root / cd.get("nnunet_pred_dirname", "nnunet_pred")).mkdir(parents=True, exist_ok=True)
    (data_root / cd.get("cnisp_pred_dirname", "cnisp_pred")).mkdir(parents=True, exist_ok=True)

    print(f"[build_corrector_data] csv={csv_path}")
    print(f"[build_corrector_data] select keep={cd['select'].get('keep')} "
          f"qa_status={cd['select'].get('qa_status')} n={n} steps={steps} "
          f"thresh={thresh_mm}mm on steps {sorted(thresh_steps)}")

    # target_samples (CLI > config) takes precedence over the legacy case-count n.
    target_samples = (args.target_samples if args.target_samples is not None
                      else cd.get("target_samples"))
    target_samples = int(target_samples) if target_samples else None

    # Folder check: how many (case, step) image samples already exist on disk.
    n_on_disk = len(list(images_dir.glob("*_step*_0000.nii.gz")))
    if target_samples:
        print(f"[build_corrector_data] target_samples={target_samples} "
              f"(stop after this many kept (case,step) samples); "
              f"{n_on_disk} image sample(s) already on disk")
        if n_on_disk > target_samples:
            print(f"[build_corrector_data] WARN: {n_on_disk} image samples on "
                  f"disk EXCEEDS target_samples={target_samples}. Selection is "
                  f"manifest-driven (downstream uses only the selected "
                  f"{target_samples}), but stale files linger in {images_dir}; "
                  f"delete extras (+ matching prelabels) to keep the folder "
                  f"exactly at {target_samples}.", file=sys.stderr)
    else:
        print(f"[build_corrector_data] legacy case cap n={n} "
              f"({n_on_disk} image sample(s) already on disk)")

    manifest = {"csv": str(csv_path), "n_requested": n,
                "target_samples": target_samples, "steps": steps,
                "thick_threshold_mm": thresh_mm,
                "thick_threshold_steps": sorted(thresh_steps),
                "cases": {}}
    case_lines = []
    n_written = n_dropped = 0
    n_samples = 0          # kept (case, step) samples selected so far
    n_cases = 0            # source images consumed
    stop = False

    for row in iter_candidate_rows(csv_path, cd["select"]):
        if stop:
            break
        # Legacy mode: cap on the number of source images.
        if target_samples is None and n_cases >= n:
            break
        n_cases += 1
        case_id = row["case_id"].strip()
        src = Path(row["image_path"].strip())
        gt_candidate = (row.get("pred_path") or "").strip()  # full-res 835 pred

        img = nib.load(str(src))
        zooms = np.asarray(img.header.get_zooms()[:3], dtype=float)
        axis = int(np.argmax(zooms))
        thru_sp = float(zooms[axis])

        entry = {"source_image": str(src), "gt_candidate_pred": gt_candidate,
                 "csv_z_spacing": row.get("z_spacing", ""),
                 "through_plane_spacing": thru_sp, "step_axis": axis, "steps": {}}

        for step in steps:
            thickness = thru_sp * step
            if step in thresh_steps and thickness > thresh_mm:
                entry["steps"][str(step)] = {
                    "kept": False, "thickness_mm": round(thickness, 4),
                    "reason": f"thickness {thickness:.2f} > {thresh_mm}mm",
                }
                n_dropped += 1
                continue
            out = images_dir / f"{case_id}_step{step:02d}_0000.nii.gz"
            if out.exists() and not args.force:
                pass
            else:
                arr, affine = _sparsify_one_ct(
                    src, step_axis=axis, step_size=step,
                    mode="thick", modality=modality, start=0,
                )
                nib.save(nib.Nifti1Image(arr.astype(np.float32), affine), str(out))
            entry["steps"][str(step)] = {
                "kept": True, "thickness_mm": round(thickness, 4),
                "image": str(out),
            }
            case_lines.append(f"{case_id}_step{step:02d}")
            n_written += 1
            n_samples += 1
            # Target reached -> stop mid-case so we land exactly on the target.
            if target_samples is not None and n_samples >= target_samples:
                stop = True
                break

        manifest["cases"][case_id] = entry
        print(f"  {case_id}: axis={axis} thru_sp={thru_sp:.3f} "
              f"steps={[s for s in steps if entry['steps'][str(s)].get('kept')]}"
              f"  (samples so far: {n_samples})")

    if target_samples is not None and n_samples < target_samples:
        print(f"[build_corrector_data] WARN: pool exhausted at {n_samples} "
              f"sample(s) (< target_samples={target_samples}); widen the "
              f"selection or add steps.", file=sys.stderr)

    with open(data_root / "corrector_data_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    with open(data_root / "corrector_cases.txt", "w") as f:
        f.write("\n".join(case_lines) + ("\n" if case_lines else ""))

    print(f"[build_corrector_data] selected {n_samples} sample(s) from "
          f"{n_cases} source image(s); wrote/kept {n_written} degraded image(s), "
          f"dropped {n_dropped} (threshold) -> {images_dir}")
    print(f"[build_corrector_data] manifest -> {data_root}/corrector_data_manifest.json")
    print(f"[build_corrector_data] cases    -> {data_root}/corrector_cases.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
