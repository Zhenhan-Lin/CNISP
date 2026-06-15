#!/usr/bin/env python3
"""Native-space evaluation toolkit: label IO, geometry, Dice, eff_res, sources.

This is the shared *calculation* layer behind the two native-space Dice
drivers -- ``engine/compare_native.py`` (head-to-head nnUNet vs CNISP) and
``build_nnunet_native_summary.py`` (nnUNet-only). Keeping the loaders,
the world-aware resample, the Dice scorer, the eff_res indexers and the
test-source resolver here guarantees the two drivers can never drift apart on
how a mask is read or how Dice is scored (previously the summary driver
imported these as private ``_`` names straight out of ``compare_native``).

Nothing here renders or writes tables -- that lives in :mod:`nnunet.lib.viz`.
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

# Make ``nnunet.*`` importable when this library is imported standalone.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nnunet.helpers.buckets import STRUCT_ORDER  # noqa: E402
from nnunet.data_prep.resolve_gt import (  # noqa: E402
    build_struct_to_value,
    resolve_sources,
)

# Structure columns in display order, including the 4-class mean last.
COLS: List[str] = STRUCT_ORDER + ["mean"]


# ── Label-volume IO ──────────────────────────────────────────────


def load_label_volume_with_affine(p: Path) -> Tuple[np.ndarray, np.ndarray]:
    img = nib.load(str(p))
    arr = np.asarray(img.dataobj)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.rint(arr)
    return arr.astype(np.int32, copy=False), np.asarray(img.affine, dtype=np.float64)


def load_label_volume(p: Path) -> np.ndarray:
    arr, _ = load_label_volume_with_affine(p)
    return arr


# ── Geometry ─────────────────────────────────────────────────────


def affines_consistent(
    pred_aff: np.ndarray, gt_aff: np.ndarray,
    *, rot_atol: float = 1e-3, trans_atol: float = 1e-2,
) -> bool:
    """True iff two affines describe the SAME voxel grid.

    ``compare_native`` Dices element-wise on the raw arrays (affines are
    dropped at load), which is only valid when voxel (i,j,k) maps to the
    same physical point in pred and GT -- i.e. their affines match. We
    check the 3x3 direction/spacing block tightly and the origin within a
    sub-voxel tolerance (nibabel always returns RAS-based affines, so this
    is storage-convention agnostic). A mismatch means a remap-back step
    failed to restore orientation; we refuse to report a Dice computed on
    misaligned voxels.
    """
    pred_aff = np.asarray(pred_aff, dtype=np.float64)
    gt_aff = np.asarray(gt_aff, dtype=np.float64)
    return (
        np.allclose(pred_aff[:3, :3], gt_aff[:3, :3], atol=rot_atol, rtol=0.0)
        and np.allclose(pred_aff[:3, 3], gt_aff[:3, 3], atol=trans_atol, rtol=0.0)
    )


def resample_pred_onto_gt(
    pred: np.ndarray,
    pred_aff: np.ndarray,
    gt_shape: Tuple[int, ...],
    gt_aff: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbour resample a label volume onto the GT voxel grid.

    Used when a prediction legitimately lives on a different grid than its
    GT -- e.g. the nnUNet dense baseline saved on the raw CT grid vs a
    ``chk_*`` pseudo-GT that the previous nnUNet saved on a coarser/resampled
    grid. The mapping is by WORLD coordinates (via the two affines), so the
    anatomy stays aligned; ``order=0`` keeps label values intact. GT is never
    resampled -- only the pred is moved onto the GT grid.
    """
    img = nib.Nifti1Image(np.asarray(pred).astype(np.int16),
                          np.asarray(pred_aff, dtype=np.float64))
    out = resample_from_to(
        img,
        (tuple(int(x) for x in gt_shape), np.asarray(gt_aff, dtype=np.float64)),
        order=0, mode="constant", cval=0,
    )
    return np.asarray(out.dataobj).astype(np.int32)


# ── Dice ─────────────────────────────────────────────────────────


def binary_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)
    inter = int(np.logical_and(pred_bool, gt_bool).sum())
    denom = int(pred_bool.sum()) + int(gt_bool.sum())
    if denom == 0:
        # both empty -> perfect by convention
        return 1.0 if (not pred_bool.any() and not gt_bool.any()) else 0.0
    return 2.0 * inter / denom


def dice_for_source(
    pred: np.ndarray,
    gt: np.ndarray,
    pred_scheme_map: Dict[str, int],
    gt_scheme_map: Dict[str, int],
) -> Dict[str, float]:
    """Compute per-structure Dice, with each side using its own label map."""
    out: Dict[str, float] = {}
    foreground = []
    for name in STRUCT_ORDER:
        pred_mask = pred == pred_scheme_map[name]
        gt_mask = gt == gt_scheme_map[name]
        d = binary_dice(pred_mask, gt_mask)
        out[name] = d
        foreground.append(d)
    out["mean"] = float(np.mean(foreground)) if foreground else float("nan")
    return out


def detect_pred_offset(arr: np.ndarray, scheme: str) -> int:
    """Recover the additive label offset baked into a saved prediction.

    Atlas GT may carry a -1000 (or other negative) offset on every label.
    When CNISP's ``native_mapping.remap_canonical_to_original`` is unable
    to look up ``meta["original_nifti_path"]`` at inference time (e.g.
    after a data move), the saved prediction is written WITHOUT that
    offset even though the GT keeps it. Comparing them with the GT
    scheme map then yields all-zero Dice for atlas sources.

    To make Dice robust to that mismatch we recover the offset directly
    from the prediction array. ``remap_canonical_to_original`` maps the
    canonical BACKGROUND (label 0) to ``offset`` itself (e.g. canonical 0
    -> -1000), so the background is the MOST negative value in the volume
    (ON = offset+1 etc. are all less negative). The offset is therefore
    simply ``min(neg_values)`` -- matching ``resolve_gt._detect_gt_offset``,
    which reads the GT offset as ``arr.min()``. When no negative labels
    are present we assume the prediction uses the bare scheme
    (``offset = 0``).

    NOTE: an earlier version returned ``min(neg_values) - 1`` on the
    assumption that the most-negative label was ON (offset+1). That is
    wrong -- the negative background (offset) is more negative than ON --
    and produced a 1-off offset that shifted every structure by one,
    yielding all-zero CNISP Dice for atlas sources.

    Parameters
    ----------
    arr : ndarray
        Prediction volume as loaded by :func:`load_label_volume`.
    scheme : str
        Source GT scheme name (``"labelfusion"`` or ``"nnunet"``); kept
        for signature symmetry with ``build_struct_to_value`` even though
        the detection itself is scheme-agnostic.
    """
    del scheme  # detection is scheme-agnostic (background maps to offset)
    if arr.size == 0:
        return 0
    if np.any(arr < 0):
        return int(arr[arr < 0].min())
    return 0


# ── Sweep -> (source, step) eff_res lookups ──────────────────────


def build_eff_res_index(sweep_pkl: Path) -> Dict[Tuple[str, int], float]:
    """source_id, step_size -> effective_resolution_mm (averaged over eyes)."""
    if not sweep_pkl.exists():
        print(f"  [warn] sweep_results.pkl not found: {sweep_pkl}; "
              f"eff_res will be inferred from CNISP step manifests only",
              file=sys.stderr)
        return {}
    with open(sweep_pkl, "rb") as f:
        sweep: List[dict] = pickle.load(f)

    accum: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    for r in sweep:
        cn = r.get("casename")
        if cn is None:
            continue
        if not (cn.endswith("_OD") or cn.endswith("_OS")):
            continue
        source_id = cn[:-3]
        accum[(source_id, int(r["step_size"]))].append(
            float(r["effective_resolution_mm"])
        )
    return {k: float(np.mean(v)) for k, v in accum.items()}


def cnisp_canonical_dice_from_pkl(
    sweep_pkl: Path,
) -> Dict[Tuple[str, int, int], Dict]:
    """Per-(source_id, step, start) canonical-space Dice from sweep_results.pkl.

    Reads the CNISP sweep pickle (one row per (case=eye, step, start)) and
    averages the two eyes' per-structure ``dice`` into a per-source value.
    This is what the eff_res aggregate consumes so the cross-method figure
    stays complete even when most native masks are NOT written to disk
    (save_mask_source_ids whitelist) -- the Dice here is canonical-space and
    matches ``test_results.csv``, NOT the native-space merged-mask Dice the
    legacy mask-reading path produced.

    Each row's ``dice['per_class']`` is in canonical class order
    {1:ON, 2:Globe, 3:Fat, 4:Recti}, which is exactly ``STRUCT_ORDER``.

    Returns ``{(source_id, step, start): {"effective_resolution_mm": float,
    "dice": {structure: mean_over_eyes}}}``.
    """
    if not sweep_pkl.exists():
        print(f"  [warn] sweep_results.pkl not found: {sweep_pkl}; "
              f"CNISP rows will be empty", file=sys.stderr)
        return {}
    with open(sweep_pkl, "rb") as f:
        sweep: List[dict] = pickle.load(f)

    structs_all = STRUCT_ORDER + ["mean"]
    accum: Dict[Tuple[str, int, int], Dict[str, List[float]]] = defaultdict(
        lambda: {s: [] for s in structs_all}
    )
    eff: Dict[Tuple[str, int, int], float] = {}
    for r in sweep:
        cn = r.get("casename")
        if cn is None or not (cn.endswith("_OD") or cn.endswith("_OS")):
            continue
        sid = cn[:-3]
        step = int(r["step_size"])
        start = int(r.get("slice_start_id", 0))
        key = (sid, step, start)
        d = r.get("dice") or {}
        per_class = d.get("per_class") or []
        for i, name in enumerate(STRUCT_ORDER):
            if i < len(per_class):
                accum[key][name].append(float(per_class[i]))
        if d.get("mean") is not None:
            accum[key]["mean"].append(float(d["mean"]))
        if r.get("effective_resolution_mm") is not None:
            eff[key] = float(r["effective_resolution_mm"])

    out: Dict[Tuple[str, int, int], Dict] = {}
    for key, structs in accum.items():
        out[key] = {
            "effective_resolution_mm": eff.get(key),
            "dice": {
                name: (float(np.mean(v)) if v else float("nan"))
                for name, v in structs.items()
            },
        }
    return out


def eff_res_from_sparse_manifest(
    work_dir: Path, experiment: str,
) -> Dict[Tuple[str, int], float]:
    """(source_id, step) -> eff_res_mm read from the sparsify manifest.

    ``input/{exp}/sparse_manifest.json`` records ``by_step[XX][sid]`` with
    an ``eff_res_mm`` field (CNISP's row value) and an ``actual_eff_res_mm``
    (= raw base spacing * step).

    step_01 is the un-sparsified dense baseline and is absent from the
    manifest (``sparsify_inputs`` skips it). We still synthesise its eff_res
    here as the source's base through-plane spacing so the dense/raw point
    lands at its true (low) effective resolution -- matching CNISP's step=1
    placement -- instead of falling into the NaN/'unknown' bucket and being
    dropped from the eff_res figure. base = actual_eff_res_mm / step (exact,
    since sparsify writes actual = base*step); falls back to eff_res_mm/step.
    """
    p = work_dir / "input" / experiment / "sparse_manifest.json"
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            m = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    out: Dict[Tuple[str, int], float] = {}
    actual: Dict[Tuple[str, int], float] = {}
    for step_tag, sid_map in m.get("by_step", {}).items():
        try:
            step = int(step_tag)
        except ValueError:
            continue
        for sid, info in sid_map.items():
            eff = info.get("eff_res_mm")
            if eff is not None:
                out[(sid, step)] = float(eff)
            a = info.get("actual_eff_res_mm")
            if a is not None:
                actual[(sid, step)] = float(a)

    # Synthesise step=1 (dense baseline) eff_res = base through-plane spacing.
    per_src_steps: Dict[str, List[int]] = defaultdict(list)
    for (sid, step) in out:
        if step > 1:
            per_src_steps[sid].append(step)
    for sid, steps in per_src_steps.items():
        s = min(steps)
        base = actual.get((sid, s))
        if base is not None:
            base = base / s
        elif (sid, s) in out:
            base = out[(sid, s)] / s
        if base is not None:
            out[(sid, 1)] = float(base)
    return out


# ── Test-source resolution ───────────────────────────────────────


def resolve_test_sources(
    cnisp_paths: Dict,
    *,
    resolve_ct: bool = False,
    detect_atlas_offset: bool = True,
):
    """Resolve the test cohort + native GT from the CURRENT paths.yaml.

    GT roots come from ``cnisp_paths`` (not from whatever absolute path was
    baked into the metadata JSONs at align time), so the Dice drivers stay
    correct after a data move as long as paths.yaml is updated. ``resolve_ct``
    is left False for Dice (which never reads the raw CT). Returns the same
    ``(sources, missing)`` tuple as :func:`resolve_gt.resolve_sources`.
    """
    meta_dir = Path(cnisp_paths["aligned_dir"]) / cnisp_paths.get(
        "metadata_dirname", "metadata"
    )
    test_cases = Path(cnisp_paths["casefiles_dir"]) / "test_cases.txt"
    atlas_label_dir = cnisp_paths.get("atlas_label_dir")
    checklist_csv_str = cnisp_paths.get("checklist_csv")
    chk_pred_dir = (
        Path(checklist_csv_str).parent / "fold_0" / "predictions"
        if checklist_csv_str else None
    )
    return resolve_sources(
        test_cases_path=test_cases,
        meta_dir=meta_dir,
        detect_atlas_offset=detect_atlas_offset,
        resolve_ct=resolve_ct,
        atlas_label_dir=Path(atlas_label_dir) if atlas_label_dir else None,
        chk_pred_dir=chk_pred_dir,
    )


def override_chk_gt_for_deployment(
    sources,
    work_dir: Path,
    deployment_dirname: str,
) -> List:
    """Swap chk_* GT to Dataset835's dense pred for deployment-mode runs.

    Atlas sources are returned unchanged (atlas manual GT remains the
    Dice target there). chk_* sources have their ``gt_label_path`` and
    ``gt_struct_to_value`` rewritten to point at
    ``{work_dir}/{deployment_dirname}/<sid>.nii.gz`` with the
    ``nnunet`` scheme. ``gt_source`` is tagged ``chk_pseudo_dataset835``
    so downstream filters / reports can tell the two GT modes apart.
    """
    out = []
    new_struct_map = build_struct_to_value("nnunet", offset=0)
    for s in sources:
        if s.gt_source != "chk_pseudo":
            out.append(s)
            continue
        new_path = work_dir / deployment_dirname / f"{s.source_id}.nii.gz"
        out.append(dataclass_replace(
            s,
            gt_label_path=new_path,
            gt_scheme="nnunet",
            gt_label_offset=0,
            gt_struct_to_value=new_struct_map,
            gt_source="chk_pseudo_dataset835",
        ))
    return out


def lookup_method_label(cfg: Dict, run_tag: str) -> str:
    """Pick the method_label for a given run_tag from configs.yaml.

    Falls back to ``CNISP-<run_tag>`` if no entry matches; this keeps
    one-off runs (e.g. an ablation with a custom run_tag) working
    without forcing every contributor to update configs.yaml.
    """
    for entry in cfg.get("cnisp_runs_to_compare", []):
        if str(entry.get("run_tag")) == run_tag:
            return str(entry.get("method_label", f"CNISP-{run_tag}"))
    return f"CNISP-{run_tag}"


# ── nnUNet native Dice scorer (compare-independent) ──────────────


def compute_nnunet_native_rows(
    work_dir: Path,
    experiment: str,
    sources: List,
    eff_res_idx: Dict[Tuple[str, int], float],
) -> Tuple[List[Dict], Dict[str, int]]:
    """Dice every nnUNet native-grid pred against its GT, one row per (src, step).

    Returns ``(wide_rows, stats)`` where each wide row has ``source_id``,
    ``gt_source``, ``step_size``, ``eff_res_mm`` and one float per
    :data:`COLS`. ``stats`` carries skip/resample counters for logging.
    """
    sweep_manifest = work_dir / "prediction" / experiment / "sweep_manifest.json"
    if not sweep_manifest.exists():
        raise SystemExit(
            f"nnUNet sweep manifest not found: {sweep_manifest}\n"
            f"  Run the `nnunet-predict-sweep` phase (experiment={experiment}) first."
        )
    with open(sweep_manifest) as f:
        nn_m = json.load(f)

    nn_pred_root = work_dir / "prediction" / experiment
    step_paths: Dict[int, Dict[str, Path]] = {}
    for step_tag, sid_map in nn_m.get("steps", {}).items():
        try:
            step = int(step_tag)
        except ValueError:
            continue
        native_dir = nn_pred_root / f"sparse_step_{step:02d}_native"
        step_paths[step] = {
            sid: native_dir / Path(raw).name for sid, raw in sid_map.items()
        }
    if not step_paths:
        raise SystemExit(
            f"nnUNet sweep manifest has no usable steps: {sweep_manifest}")

    # nnUNet masks are always written in the bare nnunet scheme with no
    # offset; the per-source GT keeps its own scheme/offset.
    nnunet_pred_struct_map = build_struct_to_value("nnunet", 0)

    wide_rows: List[Dict] = []
    stats = {"sources": 0, "skipped_gt": 0, "skipped_pred": 0,
             "skipped_atlas_mismatch": 0, "resampled_chk": 0}

    for src in sources:
        sid = src.source_id
        gt_path = src.gt_label_path
        if not gt_path.exists():
            stats["skipped_gt"] += 1
            print(f"  [skip] {sid}: GT not found at {gt_path}", file=sys.stderr)
            continue
        try:
            gt, gt_affine = load_label_volume_with_affine(gt_path)
        except Exception as e:  # noqa: BLE001
            stats["skipped_gt"] += 1
            print(f"  [skip] {sid}: failed to read GT ({e})", file=sys.stderr)
            continue
        stats["sources"] += 1

        for step in sorted(step_paths):
            path_map = step_paths[step]
            if sid not in path_map:
                continue
            pred_path = path_map[sid]
            if not pred_path.exists():
                stats["skipped_pred"] += 1
                continue
            try:
                nn_pred, nn_aff = load_label_volume_with_affine(pred_path)
            except Exception as e:  # noqa: BLE001
                stats["skipped_pred"] += 1
                print(f"  [skip nnUNet step{step:02d}] {sid}: load failed ({e})",
                      file=sys.stderr)
                continue

            grid_mismatch = (
                nn_pred.shape != gt.shape
                or not affines_consistent(nn_aff, gt_affine)
            )
            if grid_mismatch:
                if not src.gt_source.startswith("chk_"):
                    # atlas pred must already sit on the GT (= raw CT) grid;
                    # a mismatch there is a real remap/orientation bug.
                    stats["skipped_atlas_mismatch"] += 1
                    print(f"  [skip nnUNet step{step:02d}] {sid}: pred grid "
                          f"{nn_pred.shape} != GT grid {gt.shape} (atlas "
                          f"should already be on the GT grid).", file=sys.stderr)
                    continue
                # chk_* pseudo-GT lives on a different grid; resample the PRED
                # onto the GT grid by world coords (GT is never resampled).
                nn_pred = resample_pred_onto_gt(nn_pred, nn_aff, gt.shape, gt_affine)
                stats["resampled_chk"] += 1

            dices = dice_for_source(
                nn_pred, gt,
                pred_scheme_map=nnunet_pred_struct_map,
                gt_scheme_map=src.gt_struct_to_value,
            )
            row: Dict = {
                "source_id": sid,
                "gt_source": src.gt_source,
                "step_size": step,
                "eff_res_mm": eff_res_idx.get((sid, step), float("nan")),
            }
            for c in COLS:
                row[c] = dices[c]
            wide_rows.append(row)

    wide_rows.sort(key=lambda r: (r["source_id"], r["step_size"]))
    return wide_rows, stats
