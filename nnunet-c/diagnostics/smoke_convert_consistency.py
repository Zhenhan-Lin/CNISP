#!/usr/bin/env python3
"""Audit-7 smoke check: prove train and test go through the SAME conversion.

Takes ONE (degraded CT, prelabel mask, GT) triplet and runs it through
``engine/convert.py::convert_case`` TWICE:

  * TRAIN mode  -> gt_path + labels_dir set  (writes 5 channels + a GT label)
  * TEST  mode  -> no gt_path                (writes 5 channels only)

with otherwise IDENTICAL arguments, then byte-compares the five channel files
(_0000.._0004). If they are identical, ``convert_case`` provably produces the
same network input regardless of train/test mode -- i.e. no train/test drift.

Train and test datasets are patient-disjoint by design, so we can't grab "the
same case from both built datasets"; instead we feed one triplet through both
code paths, which is the exact thing under test (the conversion function).

Run on the data box (needs numpy/nibabel + a real triplet); cannot run where the
data / deps are absent.

Usage:
    python nnunet-c/diagnostics/smoke_convert_consistency.py \
        --ct  <degraded CT _0000.nii.gz> \
        --prelabel <CNISP/nnUNet mask {1,2,3,4}.nii.gz> \
        --gt  <GT label .nii.gz> \
        [--degraded-marker /images/]      # marker that matches --ct's path
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.config import add_repo_to_syspath  # noqa: E402

add_repo_to_syspath(__file__)

import numpy as np  # noqa: E402
import nibabel as nib  # noqa: E402

from engine.convert import convert_case, STRUCTS, N_CHANNELS  # noqa: E402


def _load(p: Path):
    img = nib.load(str(p))
    return np.asanyarray(img.dataobj), np.asarray(img.affine)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ct", required=True, help="degraded CT (ch0 source)")
    ap.add_argument("--prelabel", required=True, help="prelabel mask {1,2,3,4}")
    ap.add_argument("--gt", required=True, help="GT label (defines ref grid)")
    ap.add_argument("--experiment", default="thick")
    ap.add_argument("--degraded-marker", default=None,
                    help="ch0-source marker substring that matches --ct's path; "
                         "passed IDENTICALLY to both modes so only gt_path differs")
    args = ap.parse_args()

    ct, prelabel, gt = Path(args.ct), Path(args.prelabel), Path(args.gt)
    for p in (ct, prelabel, gt):
        if not p.exists():
            print(f"[smoke] missing: {p}", file=sys.stderr)
            return 2

    gt_img = nib.load(str(gt))
    ref_grid = (gt_img.shape[:3], np.asarray(gt_img.affine))
    cid = "smoke_case"

    with tempfile.TemporaryDirectory() as td:
        tr_imgs = Path(td) / "train" / "imagesTr"
        tr_lbls = Path(td) / "train" / "labelsTr"
        te_imgs = Path(td) / "test" / "imagesTs"
        for d in (tr_imgs, tr_lbls, te_imgs):
            d.mkdir(parents=True, exist_ok=True)

        # TRAIN mode: gt_path + labels_dir set.
        convert_case(
            case_id=cid, ct_path=ct, prelabel_path=prelabel, ref_grid=ref_grid,
            experiment=args.experiment, images_dir=tr_imgs,
            gt_path=gt, labels_dir=tr_lbls,
            degraded_marker=args.degraded_marker,
        )
        # TEST mode: no gt_path.
        convert_case(
            case_id=cid, ct_path=ct, prelabel_path=prelabel, ref_grid=ref_grid,
            experiment=args.experiment, images_dir=te_imgs,
            degraded_marker=args.degraded_marker,
        )

        print("=" * 64)
        print(f"[smoke] byte-compare {N_CHANNELS} channels (train vs test) for {cid}")
        print(f"[smoke] channel order: {STRUCTS} -> ch1..ch{N_CHANNELS - 1}")
        print("=" * 64)
        all_ok = True
        for i in range(N_CHANNELS):
            name = f"{cid}_{i:04d}.nii.gz"
            a_arr, a_aff = _load(tr_imgs / name)
            b_arr, b_aff = _load(te_imgs / name)
            same_vox = np.array_equal(a_arr, b_arr)
            same_aff = np.allclose(a_aff, b_aff, atol=1e-6)
            ok = same_vox and same_aff
            all_ok &= ok
            tag = "ch0(CT)" if i == 0 else f"ch{i}({STRUCTS[i-1]})"
            print(f"  {tag:14s} voxels_equal={same_vox}  affine_equal={same_aff}  "
                  f"-> {'IDENTICAL' if ok else 'DIFFERENT'}")

        # Sanity: train wrote a label, test did not.
        has_label = (tr_lbls / f"{cid}.nii.gz").exists()
        no_test_label = not (Path(td) / "test" / "labelsTs" / f"{cid}.nii.gz").exists()
        print("-" * 64)
        print(f"[smoke] train wrote GT label: {has_label}  (test wrote none: {no_test_label})")
        print(f"[smoke] RESULT: {'PASS - train/test channels byte-identical' if all_ok else 'FAIL - DRIFT DETECTED'}")
        return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
