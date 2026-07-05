#!/usr/bin/env python3
"""Assemble the 5-arm (A-E) MASK_INDEX json consumed by ``build_metrics.py``.

The evaluation subsystem (:mod:`simulation.evaluation`) renders its per-structure
figures from a ``metrics_long.csv`` built out of a MASK_INDEX -- a flat list of
per-(case, arm, step) mask entries. This driver builds that MASK_INDEX for the
five pipelines, reusing the SAME on-disk artefacts every other phase already
produced (no re-inference):

    A. nnUNet        image-conditioned nnUNet on the sparse CT (baseline)
                     -> ${work_dir}/prediction/native/<sid>.nii.gz        (step 1)
                     -> ${work_dir}/prediction/<exp>/sparse_step_XX_native/<sid>.nii.gz
    B. Cascade UNet  nnU->nnU self-correction (control B corrector)
                     -> <cascade-pred-dir>/<pred_file from the arm-B map>
    C. CNISP         CNISP shape prior with the nnUNet sparse pred as input
                     -> <cnisp-run-dir>/native_space_step_XX/<by_source_id[sid]>
    D. Proposed      nnU->CNISP->nnU corrector (control C corrector)
                     -> <proposed-pred-dir>/<pred_file from the arm-C map>
    E. Oracle        CNISP shape prior with the GT as input (cnisp-gt ceiling)
                     -> <oracle-run-dir>/native_space_step_XX/<by_source_id[sid]>
       GT            the true GT itself, a DISTINCT reference arm (default on;
                     --no-gt-arm to omit) -> the case's gt_label_path
                     (GT-vs-GT => Dice 1 / ASSD 0, a reference line not a method)

The (case, step, GT path, GT label scheme) universe comes from the corrector
``test_cases_map.json`` sidecar(s) (written by build_corrector_testset.py) -- the
authoritative record of which (source, step) each corrector actually produced and
of the native GT + its struct->value scheme. Effective resolution per (source,
step) is joined from the CNISP ``sweep_results.pkl`` so every arm lands on the
same eff_res the comparison figures use.

Every arm mask is Diced against the SAME native GT downstream
(``metrics.compute_case_metrics``), which resamples the prediction onto the GT
voxel grid by world coordinates (order 0) when the grids differ -- so masks
exported on the iso-0.5 head grid (the corrector output) or any other grid are
handled there; this builder does NOT gate on geometry. nnUNet-scheme arms (A/B/D)
are recorded as ``pred_scheme=nnunet, offset_pred=0``; the CNISP native masks
(C/E) are written in each source's ORIGINAL scheme (nnUNet for chk_/deployment,
labelfusion for atlas, possibly with a negative offset), so their scheme+offset
is auto-detected per mask.

Usage
-----
    python simulation/evaluation/build_mask_index.py \
        --config nnunet/configs_v7.yaml --experiment thick \
        --cascade-map  nnunet-c/test_input/PHOTON_CT_CORR_B_stacked/test_cases_map.json \
        --cascade-pred-dir /fs5/.../predictions/PHOTON_CT_CORR_B_stacked/fold_0 \
        --proposed-map nnunet-c/test_input/PHOTON_CT_CORR_C_cnisp/test_cases_map.json \
        --proposed-pred-dir /fs5/.../predictions/backup/PHOTON_CT_CORR_C_cnisp/fold_0 \
        --cnisp-run-tag  nnunet_pred_nodelta \
        --oracle-run-tag atlas_gt \
        --out comparison/viz/evaluation__thick/mask_index.json

Any auto-derived path (``--nnunet-pred-root`` / ``--cnisp-run-dir`` /
``--oracle-run-dir`` / ``--sweep-pkl``) can be overridden explicitly; that is the
intended way to point arm C at a ``predictions/backup/...`` directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import nibabel as nib

# Make ``nnunet.*`` importable when run as a script (repo root = parents[2]).
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from nnunet.helpers.config import load_yaml  # noqa: E402
from nnunet.lib.metrics import build_eff_res_index  # noqa: E402

# A-E arm display labels; A-E MUST match simulation.evaluation.metrics.METHODS.
# ARM_GT is an optional, non-plotted reference arm (see --gt-arm), so it is
# intentionally present here but absent from METHODS.
ARM_NNUNET = "nnUNet"
ARM_CASCADE = "Cascade UNet"
ARM_CNISP = "CNISP"
ARM_PROPOSED = "Proposed"
ARM_ORACLE = "Oracle"
ARM_GT = "GT"
ALL_ARMS = (ARM_NNUNET, ARM_CASCADE, ARM_CNISP, ARM_PROPOSED, ARM_ORACLE, ARM_GT)

# struct order used to read the corrector map's gt_struct_to_value.
_ON, _RECTI, _GLOBE, _FAT = "ON", "Recti", "Globe", "Fat"


def _detect_scheme_offset(arr: np.ndarray) -> Tuple[str, int]:
    """Infer ``(scheme, background)`` from a native label array's value set.

    Mirrors ``nnunet-c/lib/labels.detect_scheme_and_offset`` (not importable
    here -- the package dir is hyphenated). labelfusion foreground bases are a
    subset of {1,3,5,7} and contain 5 or 7; nnUNet bases are a subset of
    {1,2,3,4}. ``background`` is the most common value (e.g. -1000 for
    atlas-offset volumes, else 0). Canonical {1,2,3,4} never appears in a
    native_space mask (those are remapped back to the source's original
    scheme), so the {1,2,3,4}->nnUNet branch is unambiguous here.
    """
    vals, counts = np.unique(arr, return_counts=True)
    bg = int(vals[int(np.argmax(counts))])
    bases = {int(v) - bg for v in vals if int(v) != bg}
    if bases <= {1, 3, 5, 7} and (bases & {5, 7}):
        return "labelfusion", bg
    if bases <= {1, 2, 3, 4}:
        return "nnunet", bg
    raise ValueError(f"cannot infer scheme from foreground bases "
                     f"{sorted(bases)} (bg={bg})")


def _gt_scheme_offset(stv: Dict[str, int]) -> Tuple[str, int]:
    """Corrector-map ``gt_struct_to_value`` -> ``(scheme, offset_arg)``.

    ``offset_arg`` is what ``metrics.load_labelmap`` must ADD to bring the GT
    onto its bare scheme (ON base == 1 in both schemes -> offset = 1 - ON).
    The Globe base then disambiguates labelfusion (5) vs nnUNet (3).
    """
    on = int(stv[_ON])
    globe = int(stv[_GLOBE])
    offset = 1 - on
    base_globe = globe + offset
    if base_globe == 5:
        return "labelfusion", offset
    if base_globe == 3:
        return "nnunet", offset
    raise ValueError(f"cannot classify GT scheme from struct_to_value={stv}")


def _load_map(map_path: Optional[str], pred_dir: Optional[str]):
    """Read a corrector test_cases_map.json -> ``{(sid, step): {pred, gt, stv}}``.

    ``pred`` is the resolved prediction path (absolute, or relative to
    ``pred_dir``). Returns an empty dict when ``map_path`` is None/missing.
    """
    out: Dict[Tuple[str, int], Dict] = {}
    if not map_path:
        return out
    p = Path(map_path)
    if not p.is_file():
        print(f"[mask_index] map not found: {p}; skipping this arm.",
              file=sys.stderr)
        return out
    base = Path(pred_dir) if pred_dir else None
    mp = json.load(open(p))
    for c in mp.get("cases", {}).values():
        sid = c.get("source_id")
        step = c.get("step")
        if sid is None or step is None:
            continue
        pf = c["pred_file"]
        pred_path = Path(pf) if Path(pf).is_absolute() else (
            (base / pf) if base else Path(pf))
        out[(sid, int(step))] = {
            "pred": pred_path,
            "gt": Path(c["gt_label_path"]),
            "stv": {k: int(v) for k, v in c["gt_struct_to_value"].items()},
        }
    return out


def _load_step_manifest(run_dir: Path, step: int) -> Dict[str, str]:
    """``native_space_step_XX/manifest.json`` -> ``{source_id: filename}``."""
    step_dir = run_dir / f"native_space_step_{step:02d}"
    mf = step_dir / "manifest.json"
    if not mf.is_file():
        return {}
    try:
        data = json.load(open(mf))
    except (OSError, json.JSONDecodeError):
        return {}
    return data.get("by_source_id", data if isinstance(data, dict) else {})


def _nnunet_sparse_path(root: Path, exp: str, sid: str, step: int) -> Path:
    """nnUNet sparse native pred path for (sid, step). step 1 = dense baseline."""
    if step == 1:
        return root / "native" / f"{sid}.nii.gz"
    return root / exp / f"sparse_step_{step:02d}_native" / f"{sid}.nii.gz"


def run(args) -> int:
    exp = str(args.experiment)

    # ── Resolve auto-derivable paths from the config (overridable) ──
    nnunet_root = Path(args.nnunet_pred_root) if args.nnunet_pred_root else None
    cnisp_run = Path(args.cnisp_run_dir) if args.cnisp_run_dir else None
    oracle_run = Path(args.oracle_run_dir) if args.oracle_run_dir else None
    sweep_pkl = Path(args.sweep_pkl) if args.sweep_pkl else None
    if args.config:
        cfg = load_yaml(Path(args.config))
        cnisp_paths = load_yaml(Path(cfg["cnisp_paths_yaml"]))
        work_dir = Path(cfg["work_dir"])
        runs_base = (Path(cnisp_paths["output_basedir"]) / cfg["cnisp_model_name"]
                     / "runs" / exp)
        if nnunet_root is None:
            nnunet_root = work_dir / "prediction"
        if cnisp_run is None:
            cnisp_run = runs_base / args.cnisp_run_tag
        if oracle_run is None:
            oracle_run = runs_base / args.oracle_run_tag
        if sweep_pkl is None:
            sweep_pkl = cnisp_run / "sweep_results.pkl"

    # ── Case universe + GT come from the corrector maps ──
    cascade = _load_map(args.cascade_map, args.cascade_pred_dir)
    proposed = _load_map(args.proposed_map, args.proposed_pred_dir)
    case_gt: Dict[Tuple[str, int], Dict] = {}
    for m in (cascade, proposed):            # proposed wins ties (same GT anyway)
        for k, v in m.items():
            case_gt.setdefault(k, {"gt": v["gt"], "stv": v["stv"]})
            case_gt[k] = {"gt": v["gt"], "stv": v["stv"]}
    if not case_gt:
        print("[mask_index] no cases -- pass at least one of --cascade-map / "
              "--proposed-map.", file=sys.stderr)
        return 2

    exclude = tuple(p for p in (args.exclude_source_prefix or "").split(",") if p)
    eff_idx = build_eff_res_index(sweep_pkl) if sweep_pkl else {}

    index: List[Dict] = []
    stats = {a: 0 for a in ALL_ARMS}
    n_missing = 0

    for (sid, step) in sorted(case_gt):
        if any(sid.startswith(pref) for pref in exclude):
            continue
        gt_path = case_gt[(sid, step)]["gt"]
        stv = case_gt[(sid, step)]["stv"]
        if not gt_path.exists():
            print(f"  [skip] {sid} step{step:02d}: GT missing ({gt_path})",
                  file=sys.stderr)
            continue
        try:
            gt_scheme, gt_off = _gt_scheme_offset(stv)
        except ValueError as e:
            print(f"  [skip] {sid} step{step:02d}: {e}", file=sys.stderr)
            continue
        eff = eff_idx.get((sid, step))

        def _emit(arm: str, pred_path: Optional[Path], pred_scheme: str,
                  off_pred: int) -> None:
            # metrics.compute_case_metrics resamples pred onto the GT grid
            # (world-aware, order 0), so a differing pred grid is expected and
            # handled there -- the builder only records the mask, it does not
            # gate on geometry.
            nonlocal n_missing
            if pred_path is None or not pred_path.exists():
                n_missing += 1
                return
            index.append({
                "case": sid, "arm": arm, "step": int(step), "mode": exp,
                "eff_res": (float(eff) if eff is not None else None),
                "pred_path": str(pred_path), "gt_path": str(gt_path),
                "pred_scheme": pred_scheme, "gt_scheme": gt_scheme,
                "offset_pred": int(off_pred), "offset_gt": int(gt_off),
            })
            stats[arm] += 1

        # A. nnUNet -- nnunet scheme {1,2,3,4}, no offset.
        if nnunet_root is not None:
            _emit(ARM_NNUNET, _nnunet_sparse_path(nnunet_root, exp, sid, step),
                  "nnunet", 0)
        # B. Cascade UNet -- corrector output, nnunet scheme.
        cb = cascade.get((sid, step))
        if cb is not None:
            _emit(ARM_CASCADE, cb["pred"], "nnunet", 0)
        # D. Proposed -- corrector output, nnunet scheme.
        cp = proposed.get((sid, step))
        if cp is not None:
            _emit(ARM_PROPOSED, cp["pred"], "nnunet", 0)
        # GT reference arm: the true GT itself, DISTINCT from Oracle (=cnisp-gt).
        # GT-vs-GT is perfect by construction (Dice 1 / ASSD 0 / CoV 0) -- an
        # explicit reference line, not a competing method.
        if args.gt_arm:
            _emit(ARM_GT, gt_path, gt_scheme, gt_off)

        # C. CNISP + E. Oracle (cnisp-gt) -- native_space masks, scheme auto-detected.
        for arm, run_dir in ((ARM_CNISP, cnisp_run), (ARM_ORACLE, oracle_run)):
            if run_dir is None:
                continue
            fname = _load_step_manifest(run_dir, step).get(sid)
            if not fname:
                n_missing += 1
                continue
            mpath = run_dir / f"native_space_step_{step:02d}" / fname
            if not mpath.exists():
                mpath = Path(fname) if Path(fname).is_absolute() else mpath
            if not mpath.exists():
                n_missing += 1
                continue
            try:
                arr = np.asanyarray(nib.load(str(mpath)).dataobj)
                scheme, bg = _detect_scheme_offset(arr)
            except Exception as e:  # noqa: BLE001
                print(f"    [skip {arm}] {sid} step{step:02d}: scheme "
                      f"detect failed ({e})", file=sys.stderr)
                continue
            _emit(arm, mpath, scheme, -bg)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(index, f, indent=2)

    print(f"[mask_index] {len(index)} entries -> {out}")
    for arm in ALL_ARMS:
        print(f"    {arm:14s}: {stats[arm]} mask(s)")
    if n_missing:
        print(f"    (missing/absent masks skipped: {n_missing})")
    print("    (preds on a different grid than GT are resampled onto the GT "
          "grid at metrics time -- world-aware, order 0)")
    if not index:
        print("[mask_index] EMPTY index -- check the arm paths above.",
              file=sys.stderr)
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="destination MASK_INDEX json.")
    ap.add_argument("--experiment", "--mode", dest="experiment", default="thick",
                    help="sweep mode / experiment layer (thin|thick). Default thick.")
    ap.add_argument("--config", default=None,
                    help="nnunet config (e.g. nnunet/configs_v7.yaml) used to "
                         "auto-derive the nnUNet/CNISP/Oracle/sweep paths. Any of "
                         "those can still be overridden with the flags below.")
    # Corrector arms (authoritative case set + GT).
    ap.add_argument("--cascade-map", default=None,
                    help="arm-B (Cascade UNet) test_cases_map.json.")
    ap.add_argument("--cascade-pred-dir", default=None,
                    help="dir for arm-B relative pred_file entries (e.g. "
                         ".../PHOTON_CT_CORR_B_stacked/fold_0).")
    ap.add_argument("--proposed-map", default=None,
                    help="arm-C (Proposed) test_cases_map.json.")
    ap.add_argument("--proposed-pred-dir", default=None,
                    help="dir for arm-C relative pred_file entries (point this "
                         "at the predictions/backup/... dir when needed).")
    # Auto-derivable arm roots (override the config-derived defaults).
    ap.add_argument("--nnunet-pred-root", default=None,
                    help="nnUNet prediction root (holds native/ + <exp>/"
                         "sparse_step_XX_native/). Default: <work_dir>/prediction.")
    ap.add_argument("--cnisp-run-dir", default=None,
                    help="CNISP (C) run dir holding native_space_step_XX/. "
                         "Default: <output_basedir>/<model>/runs/<exp>/<cnisp-run-tag>.")
    ap.add_argument("--oracle-run-dir", default=None,
                    help="Oracle (E) run dir holding native_space_step_XX/. "
                         "Default: <output_basedir>/<model>/runs/<exp>/<oracle-run-tag>.")
    ap.add_argument("--cnisp-run-tag", default="nnunet_pred",
                    help="run_tag for arm C when derived from --config "
                         "(default nnunet_pred).")
    ap.add_argument("--oracle-run-tag", default="atlas_gt",
                    help="run_tag for arm E (Oracle = cnisp-gt) when derived from "
                         "--config (default atlas_gt).")
    ap.add_argument("--gt-arm", action=argparse.BooleanOptionalAction, default=False,
                    help="Emit a separate 'GT' reference arm from each case's true "
                         "GT (default OFF). This is DISTINCT from Oracle (=cnisp-gt): "
                         "GT-vs-GT is perfect by construction (Dice 1 / ASSD 0 / "
                         "CoV 0). It is NOT plotted (absent from metrics.METHODS), so "
                         "leave it off unless you specifically want a GT-vs-GT column "
                         "in metrics_long.csv. Pass --gt-arm to emit it.")
    ap.add_argument("--sweep-pkl", default=None,
                    help="sweep_results.pkl for the eff_res join. Default: "
                         "<cnisp-run-dir>/sweep_results.pkl.")
    ap.add_argument("--exclude-source-prefix", default="chk_",
                    help="comma-separated source_id prefixes to drop (default "
                         "'chk_', matching the comparison viz filter). Pass '' to "
                         "keep everything.")
    return ap


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
