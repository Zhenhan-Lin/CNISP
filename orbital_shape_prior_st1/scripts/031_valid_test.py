#!/usr/bin/env python3
"""
031: CNISP model valid_test.

A quick, self-contained sanity check that answers two questions for a trained
CNISP model, on the FIRST image of the test set only (its OD+OS, merged):

  1. Is the reconstructed mask reasonable?
       - canonical dense prediction has labels in {0..4}, non-empty per eye;
       - native merged mask is non-empty, no NaN/inf, has a plausible number of
         foreground structures (1..4), and reports per-class voxel counts.
  2. Can it be mapped back to original space correctly?
       - native merged mask shape == metadata original_shape;
       - native merged mask affine == metadata original_affine;
       - (these use the EXACT production native-mapping path, so a pass means
         the real inference -> native pipeline round-trips geometry correctly.)

It runs the normal inference (engine.infer.infer_test_set) restricted to the
first source via a temporary 2-case casefile, at step_size=1 only (the dense
baseline; no degradation), with output redirected under ``reconstructions/tmp``
so it never touches the real run trees. The chosen native mask is also copied to
``reconstructions/tmp/<stem>_valid_stepXX.nii.gz`` and a ``valid_report.json``
is written there.

Usage:
    python orbital_shape_prior_st1/scripts/031_valid_test.py \
        -m orbital_ad_v6_5_gt \
        -t configs/train_v6_5_gt.yaml \
        -c configs/test_default.yaml \
        --checkpoint best --test-label-source atlas_gt --experiment thick

Exit code 0 = PASS, 1 = FAIL (hard check failed).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]   # orbital_shape_prior_st1/
REPO_ROOT = PROJECT_ROOT.parent                       # repo root
for _p in (str(PROJECT_ROOT), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from engine.dataset import load_casenames           # noqa: E402
from engine.infer import infer_test_set             # noqa: E402
from engine.test_label_sources import build_run_layout  # noqa: E402


# ── helpers ──────────────────────────────────────────────────────────

def _source_of(casename: str) -> str:
    return casename[:-3] if casename.endswith(("_OD", "_OS")) else casename


def _resolve_cfg(path_arg: str) -> Path:
    """Resolve a config path: absolute, repo-relative, or under configs/."""
    p = Path(path_arg)
    for cand in (p, REPO_ROOT / p, PROJECT_ROOT / p, PROJECT_ROOT / "configs" / p.name):
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"config not found: {path_arg}")


def _load_params(paths_yaml: Path, train_yaml: Path, test_yaml: Path) -> dict:
    params: dict = {}
    for y in (paths_yaml, train_yaml, test_yaml):
        with open(y) as f:
            params.update(yaml.safe_load(f) or {})
    return params


def _find_native_mask(out_dir: Path, source_id: str):
    """Return (step, mask_path) for the source's native mask (smallest step)."""
    step_dirs = sorted(out_dir.glob("native_space_step_*"))
    for sd in step_dirs:
        if not sd.name[len("native_space_step_"):].isdigit():
            continue  # skip start-offset dirs like _o1
        manifest = sd / "manifest.json"
        if not manifest.is_file():
            continue
        mf = json.load(open(manifest))
        by_sid = mf.get("by_source_id", mf)
        if source_id in by_sid:
            step = int(sd.name[len("native_space_step_"):])
            return step, sd / by_sid[source_id]
    return None, None


# ── validation ───────────────────────────────────────────────────────

def validate(out_dir: Path, source_id: str, casenames, metadata_dir: Path,
             tmp_base: Path) -> dict:
    report: dict = {"source_id": source_id, "casenames": list(casenames),
                    "checks": [], "warnings": [], "passed": True}

    def hard(name, ok, detail=""):
        report["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
        if not ok:
            report["passed"] = False
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))

    def warn(msg):
        report["warnings"].append(msg)
        print(f"  [WARN] {msg}")

    # ── (A) canonical dense prediction sanity (per eye) ──────────────
    for cn in casenames:
        hits = list(out_dir.glob(f"step_*/pred/{cn}_pred.nii.gz"))
        if not hits:
            warn(f"canonical pred not found for {cn} (step_*/pred/{cn}_pred.nii.gz)")
            continue
        arr = np.asanyarray(nib.load(str(hits[0])).dataobj)
        uniq = sorted(int(v) for v in np.unique(arr))
        hard(f"canonical[{cn}] labels subset of 0..4", set(uniq) <= {0, 1, 2, 3, 4},
             f"unique={uniq}")
        hard(f"canonical[{cn}] non-empty foreground", arr.any(), f"fg_vox={int((arr>0).sum())}")

    # ── (B) native mask reasonableness + geometry round-trip ─────────
    step, mask_path = _find_native_mask(out_dir, source_id)
    if mask_path is None or not Path(mask_path).exists():
        hard("native mask produced", False,
             f"no native_space_step_*/manifest entry for {source_id} in {out_dir}")
        return report
    report["native_mask"] = str(mask_path)
    report["native_step"] = step
    print(f"  native mask (step {step:02d}): {mask_path}")

    meta_path = metadata_dir / f"{casenames[0]}.json"
    hard("metadata exists", meta_path.is_file(), str(meta_path))
    if not meta_path.is_file():
        return report
    meta = json.load(open(meta_path))
    orig_shape = tuple(int(v) for v in meta["original_shape"])
    orig_aff = np.array(meta["original_affine"], dtype=float)

    img = nib.load(str(mask_path))
    arr = np.asanyarray(img.dataobj)

    hard("native shape == original_shape", tuple(img.shape[:3]) == orig_shape,
         f"native={tuple(img.shape[:3])} original={orig_shape}")
    hard("native affine == original_affine", np.allclose(img.affine, orig_aff, atol=1e-3),
         f"max|diff|={float(np.max(np.abs(img.affine - orig_aff))):.4g}")
    hard("native finite (no NaN/inf)", np.isfinite(arr).all())
    hard("native non-empty foreground", (arr != 0).any())

    # background = most frequent value; foreground = the rest.
    vals, counts = np.unique(arr, return_counts=True)
    bg = int(vals[int(np.argmax(counts))])
    fg_vals = [int(v) for v in vals if int(v) != bg]
    fg_counts = {int(v): int(c) for v, c in zip(vals, counts) if int(v) != bg}
    report["background_value"] = bg
    report["foreground_value_counts"] = fg_counts
    print(f"  native labels: bg={bg} fg_counts={fg_counts}")
    hard("plausible #foreground structures (1..4)", 1 <= len(fg_vals) <= 4,
         f"n_fg_distinct={len(fg_vals)} ({fg_vals})")
    if len(fg_vals) < 4:
        warn(f"only {len(fg_vals)} foreground structure(s) present in the merged "
             f"native mask (expected up to 4: ON/Recti/Globe/Fat)")

    # ── copy the validated mask to reconstructions/tmp for easy access ─
    tmp_base.mkdir(parents=True, exist_ok=True)
    base_stem = Path(mask_path).stem.replace(".nii", "")
    dst = tmp_base / f"{base_stem}_valid_step{step:02d}.nii.gz"
    shutil.copyfile(mask_path, dst)
    report["copied_to"] = str(dst)
    print(f"  copied native mask -> {dst}")

    # ── ALSO write a clean nnUNet-scheme copy {0,1,2,3,4} ────────────
    # The production native mask stays in the original scheme (labelfusion
    # ±offset / nnunet) so compare_native native-Dice still matches the GT.
    # This extra copy remaps by STRUCTURE NAME -> {ON:1,Recti:2,Globe:3,Fat:4},
    # which is what nnUNet-C consumes as a prelabel channel.
    try:
        from nnunet.data_prep.resolve_gt import build_struct_to_value, NNUNET_LABELS
        scheme = meta.get("input_label_scheme", "nnunet")
        offset = int(arr.min()) if int(arr.min()) < 0 else 0
        stv = build_struct_to_value(scheme, offset)  # name -> original value
        remapped = np.zeros(arr.shape, dtype=np.uint8)
        mapping = {}
        for name, val in stv.items():
            remapped[arr == val] = NNUNET_LABELS[name]
            mapping[name] = {"from": int(val), "to": int(NNUNET_LABELS[name])}
        nnu_path = tmp_base / f"{base_stem}_valid_step{step:02d}_nnunet.nii.gz"
        nib.save(nib.Nifti1Image(remapped, img.affine), str(nnu_path))
        ru = sorted(int(v) for v in np.unique(remapped))
        report["nnunet_scheme_mask"] = str(nnu_path)
        report["nnunet_scheme_mapping"] = mapping
        report["nnunet_scheme_values"] = ru
        print(f"  nnUNet-scheme copy ({scheme}, offset={offset}) -> {nnu_path}")
        print(f"    label remap: " +
              ", ".join(f"{n}:{m['from']}->{m['to']}" for n, m in mapping.items()))
        hard("nnUNet-scheme mask values subset of 0..4", set(ru) <= {0, 1, 2, 3, 4},
             f"values={ru}")
    except Exception as e:  # noqa: BLE001
        warn(f"could not write nnUNet-scheme copy: {e}")

    return report


# ── main ─────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-p", "--paths", default="configs/paths.yaml")
    ap.add_argument("-t", "--train_config", default="configs/train_v6_5_gt.yaml")
    ap.add_argument("-c", "--config", default="configs/test_default.yaml")
    ap.add_argument("-m", "--model_name", default="orbital_ad_v6_5_gt")
    ap.add_argument("--checkpoint", default="best", choices=["best", "latest"])
    ap.add_argument("--test-label-source", default="atlas_gt",
                    choices=["atlas_gt", "nnunet_pred", "real_pair"])
    ap.add_argument("--run-tag", default="valid_tmp")
    ap.add_argument("--experiment", default="thick", choices=["thin", "thick", "real", "fov"])
    args = ap.parse_args()

    paths_yaml = _resolve_cfg(args.paths)
    train_yaml = _resolve_cfg(args.train_config)
    test_yaml = _resolve_cfg(args.config)
    params = _load_params(paths_yaml, train_yaml, test_yaml)

    params["model_name"] = args.model_name
    params["checkpoint"] = args.checkpoint
    params["test_label_source"] = args.test_label_source
    params["run_tag"] = args.run_tag
    params["experiment"] = args.experiment

    # ── restrict to the FIRST test source (its OD+OS) ────────────────
    casefiles_dir = Path(params["casefiles_dir"])
    all_cn = load_casenames(casefiles_dir / params["test_casefile"])
    if not all_cn:
        print(f"[031] empty test casefile: {casefiles_dir/params['test_casefile']}",
              file=sys.stderr)
        return 1
    first_sid = _source_of(all_cn[0])
    sel = [c for c in all_cn if _source_of(c) == first_sid]
    tmp_casefile = casefiles_dir / "_valid_first_source.txt"
    tmp_casefile.write_text("\n".join(sel) + "\n")
    params["test_casefile"] = tmp_casefile.name

    # ── redirect ALL output under reconstructions/tmp ────────────────
    tmp_base = Path(params["output_basedir"]) / "tmp"
    params["output_basedir"] = str(tmp_base)

    # Save every mask (override the 12-source whitelist).
    params["save_mask_source_ids"] = None
    # Only step_size=1 (dense baseline, no degradation) per case. A huge
    # target_eff_res_increment_mm makes delta_step huge, so the 2nd candidate
    # step's eff_res exceeds max_eff_resolution_mm and adaptive_steps_for_case
    # returns [1] (the dense baseline is always kept).
    ass = dict(params.get("adaptive_step_sweep", {}) or {})
    ass["max_num_steps_per_case"] = 1
    ass["target_eff_res_increment_mm"] = 1.0e6
    ass["start_offsets"] = [0]
    params["adaptive_step_sweep"] = ass

    print("=" * 64)
    print("031 CNISP valid_test")
    print(f"  model          : {args.model_name} (checkpoint={args.checkpoint})")
    print(f"  label source   : {args.test_label_source}  experiment={args.experiment}")
    print(f"  first source   : {first_sid}  cases={sel}")
    print(f"  sweep          : step_size=1 only (dense baseline)")
    print(f"  output (tmp)   : {tmp_base}")
    print("=" * 64)

    infer_test_set(params)

    layout = build_run_layout(params)
    print("\n[031] validating reconstruction + native mapping ...")
    report = validate(layout.output_dir, first_sid, sel, layout.metadata_dir, tmp_base)

    report_path = tmp_base / "valid_report.json"
    tmp_base.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[031] report -> {report_path}")
    print(f"[031] RESULT: {'PASS' if report['passed'] else 'FAIL'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
