#!/usr/bin/env python3
"""Per-source paired Dice: nnUNet vs one CNISP run, native head space.

Each invocation handles ONE CNISP run -- it pairs the (always-shared)
nnUNet sparse-CT sweep with the CNISP outputs under
``output_basedir/<model>/runs/<run_tag>/`` and emits the matched
paired CSV/TXT bundle.

To compare ``CNISP-atlasGT`` vs ``CNISP-nnUNetPred`` head-to-head,
the pipeline calls this script once per entry in
``configs.yaml::cnisp_runs_to_compare``.

Inputs
------
* ``{work_dir}/input/native/{source_id}_0000.nii.gz``    - staged input CT
* ``{work_dir}/prediction/sparse_step_{XX}_upsampled/{source_id}.nii.gz``
  - nnUNet per-step prediction NN-upsampled back to the native CT grid
  (step_01 is a symlink to the dense baseline ``prediction/native/``).
  Indexed via ``{work_dir}/prediction/sweep_manifest.json``.
* ``output_basedir/{model}/runs/{run_tag}/native_space_step_{XX}/...``
  -- CNISP per-step predictions for this run, produced by
  ``orbital_shape_prior_st1/engine/infer.py`` (or backfilled by
  ``nnunet/engine/build_cnisp_native_sweep.py``).
* ``output_basedir/{model}/runs/{run_tag}/sweep_results.pkl`` -- eff_res lookup.
* ``output_basedir/{model}/runs/{run_tag}/native_sweep_manifest.json`` --
  records the ``test_label_source`` used for this run, which decides
  whether chk_* sources are Diced against the legacy chk_pseudo GT
  (``atlas_gt`` runs) or against Dataset835's dense pred
  (``nnunet_pred`` runs). Atlas sources always Dice against the
  atlas manual GT.

Comparison
----------
* Both methods contribute one row per (source_id, step_size,
  structure). All Dice computed on the ORIGINAL CT's voxel grid -- GT
  is never resampled. nnUNet's sparse-CT predictions are NN-upsampled
  along the through-plane axis by ``engine/upsample_sparse_preds.py``
  before this step.
* When the CNISP run uses ``test_label_source=nnunet_pred`` we ALSO
  switch nnUNet-sparse's chk_* GT to Dataset835's dense pred so both
  methods Dice against the same target in this bucket. Atlas rows are
  unaffected (they always Dice against atlas manual GT).
* Same effective-resolution bucket edges apply to both methods.

Outputs (under ``{work_dir}/comparison/``)
------------------------------------------
For ``--cnisp-run-tag <T>``:
* ``paired_per_source__<T>.csv`` -- long, one row per
  (source, method, step_size, structure, dice).
* ``paired_summary__<T>.csv``    -- aggregated by
  (method, eff_res_bucket, structure).
* ``paired_summary__<T>.txt``    -- human-readable wide table.

The companion driver ``nnunet/engine/build_method_summary.py`` reads
the per-source CSV and renders the per-method PNG.

Usage
-----
    python nnunet/compare_native.py --config nnunet/configs.yaml \\
        --cnisp-run-tag atlas_gt
    python nnunet/compare_native.py --config nnunet/configs.yaml \\
        --cnisp-run-tag nnunet_pred
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from collections import defaultdict
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import yaml

# Ensure ``nnunet`` is importable when this file is run as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nnunet.resolve_gt import (  # noqa: E402
    build_struct_to_value, resolve_sources,
)


STRUCT_ORDER = ["ON", "Globe", "Fat", "Recti"]
NNUNET_METHOD_LABEL = "nnUNet-sparse"


def _detect_pred_offset(arr: np.ndarray, scheme: str) -> int:
    """Recover the additive label offset baked into a saved prediction.

    Atlas GT may carry a -1000 (or other negative) offset on every label.
    When CNISP's ``native_mapping.remap_canonical_to_original`` is unable
    to look up ``meta["original_nifti_path"]`` at inference time (e.g.
    after a data move), the saved prediction is written WITHOUT that
    offset even though the GT keeps it. Comparing them with the GT
    scheme map then yields all-zero Dice for atlas sources.

    To make Dice robust to that mismatch we recover the offset directly
    from the prediction array. The canonical foreground value for "ON"
    is 1 in both LABELFUSION_LABELS and NNUNET_LABELS, so any negative
    label present means ``offset = min(neg_values) - 1``. When no
    negative labels are present we assume the prediction uses the bare
    scheme (``offset = 0``).

    Parameters
    ----------
    arr : ndarray
        Prediction volume as loaded by ``_load_label_volume``.
    scheme : str
        Source GT scheme name (``"labelfusion"`` or ``"nnunet"``); kept
        for signature symmetry with ``build_struct_to_value`` even though
        the detection itself is scheme-agnostic.
    """
    del scheme  # detection is scheme-agnostic (ON canonical value is 1 in both)
    if arr.size == 0:
        return 0
    if np.any(arr < 0):
        return int(arr[arr < 0].min()) - 1
    return 0


# ── Generic helpers ──────────────────────────────────────────────


def _load_yaml(path: Path) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_label_volume(p: Path) -> np.ndarray:
    img = nib.load(str(p))
    arr = np.asarray(img.dataobj)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.rint(arr)
    return arr.astype(np.int32, copy=False)


def _binary_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)
    inter = int(np.logical_and(pred_bool, gt_bool).sum())
    denom = int(pred_bool.sum()) + int(gt_bool.sum())
    if denom == 0:
        # both empty -> perfect by convention
        return 1.0 if (not pred_bool.any() and not gt_bool.any()) else 0.0
    return 2.0 * inter / denom


def _assign_bucket(eff_res: Optional[float],
                   edges: List[float]) -> Tuple[int, str]:
    """Return (idx, label) for the given eff_res. None -> (-1, 'unknown')."""
    if eff_res is None or (isinstance(eff_res, float) and math.isnan(eff_res)):
        return -1, "unknown"
    for i, ub in enumerate(edges):
        if eff_res <= ub + 1e-6:
            lower = 0.0 if i == 0 else edges[i - 1]
            return i, f"({lower:.1f}, {ub:.1f}]"
    return len(edges), f"({edges[-1]:.1f}, inf]"


# ── Per-source Dice computation ──────────────────────────────────


def _dice_for_source(
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
        d = _binary_dice(pred_mask, gt_mask)
        out[name] = d
        foreground.append(d)
    out["mean"] = float(np.mean(foreground)) if foreground else float("nan")
    return out


# ── Sweep -> (source, step) eff_res lookup ───────────────────────


def _build_eff_res_index(
    sweep_pkl: Path,
    meta_dir: Path,
) -> Dict[Tuple[str, int], float]:
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


# ── Main ─────────────────────────────────────────────────────────


def _lookup_method_label(cfg: Dict, run_tag: str) -> str:
    """Pick the method_label for a given run_tag from configs.yaml.

    Falls back to ``CNISP-<run_tag>`` if no entry matches; this keeps
    one-off runs (e.g. an ablation with a custom run_tag) working
    without forcing every contributor to update configs.yaml.
    """
    for entry in cfg.get("cnisp_runs_to_compare", []):
        if str(entry.get("run_tag")) == run_tag:
            return str(entry.get("method_label", f"CNISP-{run_tag}"))
    return f"CNISP-{run_tag}"


def _override_chk_gt_for_deployment(
    sources,
    work_dir: Path,
    deployment_dirname: str,
) -> List:
    """Swap chk_* GT to Dataset835's dense pred for deployment-mode runs.

    Atlas sources are returned unchanged (atlas manual GT remains the
    Dice target there). chk_* sources have their ``gt_label_path`` and
    ``gt_struct_to_value`` rewritten to point at
    ``{work_dir}/{deployment_dirname}/<sid>.nii.gz`` with the
    ``nnunet`` scheme. ``gt_source`` is tagged
    ``chk_pseudo_dataset835`` so downstream filters / reports can tell
    the two GT modes apart.
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="nnunet/configs.yaml")
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--cnisp-run-tag", default="atlas_gt",
                    help="Which CNISP run to compare against (subdir under "
                         "output_basedir/<model>/runs/). Default atlas_gt "
                         "preserves the ceiling-curve comparison.")
    ap.add_argument("--cnisp-method-label", default=None,
                    help="Override the CNISP method label. If unset, look up "
                         "cnisp_runs_to_compare in the config.")
    ap.add_argument("--out-suffix", default=None,
                    help="Suffix for output filenames. Default is "
                         "'__<cnisp_run_tag>' so multiple runs do not "
                         "collide.")
    ap.add_argument("--strict-shape", action="store_true",
                    help="Fail if a prediction's shape differs from GT "
                         "(default: skip the source with a warning).")
    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config))
    cnisp_paths = _load_yaml(Path(cfg["cnisp_paths_yaml"]))

    model_name = args.model_name or cfg["cnisp_model_name"]
    work_dir = Path(args.work_dir or cfg["work_dir"])
    run_tag = str(args.cnisp_run_tag)
    cnisp_method_label = (
        args.cnisp_method_label or _lookup_method_label(cfg, run_tag)
    )
    out_suffix = (args.out_suffix if args.out_suffix is not None
                  else f"__{run_tag}")

    output_base = (
        Path(cnisp_paths["output_basedir"]) / model_name / "runs" / run_tag
    )
    meta_dir = Path(cnisp_paths["aligned_dir"]) / cnisp_paths.get(
        "metadata_dirname", "metadata"
    )
    casefiles_dir = Path(cnisp_paths["casefiles_dir"])
    test_cases = casefiles_dir / "test_cases.txt"

    nnunet_sweep_manifest = work_dir / "prediction" / "sweep_manifest.json"
    if not nnunet_sweep_manifest.exists():
        print(f"[compare_native] nnUNet sweep manifest not found: "
              f"{nnunet_sweep_manifest}", file=sys.stderr)
        print(f"  Did you run nnunet/engine/upsample_sparse_preds.py? "
              f"(`nnunet-predict-sweep` phase)", file=sys.stderr)
        return 2

    # The CNISP run's manifest tells us whether it was a deployment-mode
    # run (test_label_source=nnunet_pred). When so, chk_* GT switches to
    # Dataset835's dense pred for BOTH methods so the head-to-head Dice
    # stays voxel-for-voxel against the same target.
    cnisp_top_manifest = output_base / "native_sweep_manifest.json"
    cnisp_test_label_source = "atlas_gt"
    if cnisp_top_manifest.exists():
        try:
            with open(cnisp_top_manifest) as f:
                cnisp_test_label_source = json.load(f).get(
                    "test_label_source", "atlas_gt"
                )
        except (OSError, json.JSONDecodeError):
            pass
    deployment_dirname = cfg.get(
        "deployment_gt_dirname_for_chk", "prediction/native"
    )

    bucket_edges = list(cfg.get("summary_bucket_edges_mm",
                                [1.0, 2.0, 3.0, 4.0, 5.0, 6.5, 8.5, 11.0, 13.0]))

    print(f"[compare_native] run_tag                  = {run_tag}")
    print(f"[compare_native] cnisp method label       = {cnisp_method_label}")
    print(f"[compare_native] cnisp test_label_source  = {cnisp_test_label_source}")
    print(f"[compare_native] cnisp run dir            = {output_base}")
    print(f"[compare_native] output suffix            = {out_suffix}")

    # ── Resolve the 31 sources ────────────────────────────────────
    # GT-only: Dice never reads the raw CT, so we explicitly opt out
    # of CT-path resolution. This keeps compare_native independent of
    # the nnUNet-side pivot CSV / atlas_image_dir config (those are
    # only needed by the input-staging phases, not by Dice).
    #
    # GT roots come from the CURRENT paths.yaml -- not from whatever
    # absolute path was baked into the metadata JSONs at align time.
    # Compare therefore stays correct after a data move as long as
    # paths.yaml is updated; the metadata JSONs (and any /home-local
    # softlinks they happen to reference) can stay frozen.
    atlas_label_dir = cnisp_paths.get("atlas_label_dir")
    checklist_csv_str = cnisp_paths.get("checklist_csv")
    chk_pred_dir = (
        Path(checklist_csv_str).parent / "fold_0" / "predictions"
        if checklist_csv_str else None
    )
    print(
        f"[compare_native] atlas GT root            = "
        f"{atlas_label_dir or '<unset; falling back to metadata>'}"
    )
    print(
        f"[compare_native] chk_*  GT root            = "
        f"{chk_pred_dir or '<unset; falling back to metadata>'}"
    )
    sources, missing = resolve_sources(
        test_cases_path=test_cases,
        meta_dir=meta_dir,
        detect_atlas_offset=True,
        resolve_ct=False,
        atlas_label_dir=Path(atlas_label_dir) if atlas_label_dir else None,
        chk_pred_dir=chk_pred_dir,
    )
    if missing:
        # Any entries here come from metadata-scheme problems, not CT
        # lookup (resolve_ct=False suppressed those). Still non-fatal.
        print(f"[compare_native] note: {len(missing)} source(s) had "
              f"metadata problems; comparison itself uses GT only.",
              file=sys.stderr)
        for m in missing[:5]:
            print(f"  - {m}", file=sys.stderr)
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more", file=sys.stderr)

    # Swap chk_* GT to Dataset835's dense pred for deployment-mode runs.
    if cnisp_test_label_source == "nnunet_pred":
        sources = _override_chk_gt_for_deployment(
            sources, work_dir, deployment_dirname,
        )
        n_overridden = sum(1 for s in sources
                           if s.gt_source == "chk_pseudo_dataset835")
        print(f"[compare_native] deployment-mode: chk_* GT swapped to "
              f"{deployment_dirname}/ ({n_overridden} source(s))")

    # ── nnUNet label scheme reminder ──────────────────────────────
    # We trust NNUNET_LABELS from resolve_gt (matches NNUNET_MAP_CT in
    # canonical_align.py). expected_nnunet_labels is surfaced in configs.yaml
    # just for documentation / future runtime checks.
    expected_labels = cfg.get("expected_nnunet_labels", {})
    if expected_labels:
        print(f"[compare_native] documented nnUNet labels (sanity): "
              f"{expected_labels}")

    # ── CNISP per-step manifest loader ────────────────────────────
    # ``by_source_id`` is read as basename + anchored at the manifest's
    # own directory: the manifest lives next to its NIfTI siblings, so
    # whatever subtree it sits in IS the current file tree for that
    # step. ``Path(raw).name`` makes this work for both fresh manifests
    # (basename only) and legacy manifests (full absolute path baked in
    # at write time) without any branching.
    step_dirs = sorted(output_base.glob("native_space_step_*"))
    cnisp_step_paths: Dict[int, Dict[str, Path]] = {}
    for d in step_dirs:
        try:
            step = int(d.name.replace("native_space_step_", ""))
        except ValueError:
            continue
        manifest = d / "manifest.json"
        if not manifest.exists():
            print(f"  [warn] no manifest in {d}; skipping", file=sys.stderr)
            continue
        with open(manifest) as f:
            m = json.load(f)
        cnisp_step_paths[step] = {
            sid: d / Path(raw).name
            for sid, raw in m.get("by_source_id", {}).items()
        }
    if not cnisp_step_paths:
        print(f"[compare_native] no CNISP step manifests under {output_base}.",
              file=sys.stderr)
        print(f"  Did you run nnunet/engine/build_cnisp_native_sweep.py?",
              file=sys.stderr)
        return 2
    print(f"[compare_native] CNISP steps available: {sorted(cnisp_step_paths)}")

    # ── nnUNet per-step manifest loader ───────────────────────────
    # Same idea as the CNISP loader: basenames anchored against the
    # canonical upsampled-output convention written by
    # ``nnunet/engine/upsample_sparse_preds.py``:
    #     ${work_dir}/prediction/sparse_step_{XX}_upsampled/{sid}.nii.gz
    nn_upsampled_root = work_dir / "prediction"
    with open(nnunet_sweep_manifest) as f:
        nn_m = json.load(f)
    nnunet_step_paths: Dict[int, Dict[str, Path]] = {}
    for step_tag, sid_map in nn_m.get("steps", {}).items():
        try:
            step = int(step_tag)
        except ValueError:
            continue
        canonical_dir = nn_upsampled_root / f"sparse_step_{step:02d}_upsampled"
        nnunet_step_paths[step] = {
            sid: canonical_dir / Path(raw).name
            for sid, raw in sid_map.items()
        }
    if not nnunet_step_paths:
        print(f"[compare_native] nnUNet sweep manifest has no usable steps: "
              f"{nnunet_sweep_manifest}", file=sys.stderr)
        return 2
    print(f"[compare_native] nnUNet steps available: {sorted(nnunet_step_paths)}")

    # ── eff_res lookup ────────────────────────────────────────────
    eff_res_idx = _build_eff_res_index(
        output_base / "sweep_results.pkl", meta_dir
    )

    # ── Iterate sources & emit per-source rows ────────────────────
    per_source_rows: List[Dict[str, str]] = []
    n_done = 0
    n_skipped_gt = 0
    n_skipped_nnunet = 0
    n_pred_offset_fixed = 0

    for src in sources:
        sid = src.source_id
        gt_path = src.gt_label_path
        if not gt_path.exists():
            n_skipped_gt += 1
            print(f"  [skip] {sid}: GT not found at {gt_path}", file=sys.stderr)
            continue
        try:
            gt = _load_label_volume(gt_path)
        except Exception as e:  # noqa: BLE001
            n_skipped_gt += 1
            print(f"  [skip] {sid}: failed to read GT ({e})", file=sys.stderr)
            continue

        # ── nnUNet per step ───────────────────────────────────────
        # nnUNet predictions live on the same native CT grid as the GT;
        # for step>1 they've already been NN-upsampled by
        # engine/upsample_sparse_preds.py before reaching this script.
        for step in sorted(nnunet_step_paths):
            path_map = nnunet_step_paths[step]
            if sid not in path_map:
                continue
            nnunet_pred_path = path_map[sid]
            if not nnunet_pred_path.exists():
                n_skipped_nnunet += 1
                print(f"  [skip nnUNet step{step:02d}] {sid}: no pred at "
                      f"{nnunet_pred_path}", file=sys.stderr)
                continue
            try:
                nn_pred = _load_label_volume(nnunet_pred_path)
            except Exception as e:  # noqa: BLE001
                n_skipped_nnunet += 1
                print(f"  [skip nnUNet step{step:02d}] {sid}: load failed "
                      f"({e})", file=sys.stderr)
                continue
            if nn_pred.shape != gt.shape:
                msg = (f"{sid} nnUNet step{step:02d}: pred shape "
                       f"{nn_pred.shape} != GT shape {gt.shape}")
                if args.strict_shape:
                    print(f"  [error] {msg}", file=sys.stderr)
                    return 3
                print(f"  [skip nnUNet step{step:02d}] {msg}", file=sys.stderr)
                continue
            # nnUNet predictions are always written in the bare nnunet
            # scheme {ON:1, Recti:2, Globe:3, Fat:4} with no offset, so
            # the pred scheme map is fixed regardless of source / GT
            # offset. (Dice still works against an offset GT because
            # ``_dice_for_source`` uses each side's own scheme map.)
            nnunet_pred_struct_map = build_struct_to_value("nnunet", 0)
            dices = _dice_for_source(
                nn_pred, gt,
                pred_scheme_map=nnunet_pred_struct_map,
                gt_scheme_map=src.gt_struct_to_value,
            )
            eff_res = eff_res_idx.get((sid, step))
            for name in STRUCT_ORDER + ["mean"]:
                per_source_rows.append({
                    "source_id": sid,
                    "gt_source": src.gt_source,
                    "method": NNUNET_METHOD_LABEL,
                    "step_size": str(step),
                    "eff_res_mm": (f"{eff_res:.4f}" if eff_res is not None else ""),
                    "structure": name,
                    "dice": f"{dices[name]:.6f}",
                })

        # ── CNISP per step ────────────────────────────────────────
        # CNISP's native_mapping.remap_canonical_to_original emits labels
        # in the SAME scheme as the source GT, but the additive offset
        # (e.g. atlas's -1000) is detected at *inference time* by peeking
        # at ``meta["original_nifti_path"]``. If that path was stale on
        # the inference host (data move, missing softlink, ...) the
        # saved pred ends up in the bare scheme (e.g. {1,3,5,7}) even
        # though the GT keeps the offset (e.g. {-999,-997,-995,-993}).
        # We recover the prediction's actual offset per-file below so
        # the Dice match is correct regardless of inference-time path
        # availability. The bare scheme always works when offset == 0,
        # so this also covers chk_* sources and deployment-mode runs.
        for step in sorted(cnisp_step_paths):
            path_map = cnisp_step_paths[step]
            if sid not in path_map:
                continue
            cnisp_path = path_map[sid]
            if not cnisp_path.exists():
                print(f"  [skip CNISP step{step:02d}] {sid}: file missing "
                      f"{cnisp_path}", file=sys.stderr)
                continue
            try:
                cn_pred = _load_label_volume(cnisp_path)
            except Exception as e:  # noqa: BLE001
                print(f"  [skip CNISP step{step:02d}] {sid}: load failed ({e})",
                      file=sys.stderr)
                continue
            if cn_pred.shape != gt.shape:
                msg = (f"{sid} CNISP step{step:02d}: shape "
                       f"{cn_pred.shape} != GT {gt.shape}")
                if args.strict_shape:
                    print(f"  [error] {msg}", file=sys.stderr)
                    return 3
                print(f"  [skip CNISP step{step:02d}] {msg}", file=sys.stderr)
                continue

            pred_offset = _detect_pred_offset(cn_pred, src.gt_scheme)
            if pred_offset != src.gt_label_offset:
                # Inference-time offset detection disagreed with GT-time
                # detection; rebuild the pred scheme map from what the
                # file actually contains so Dice doesn't silently zero
                # out. Logged once per (source, step) so a healthy run
                # stays quiet but a broken pipeline is immediately
                # visible.
                n_pred_offset_fixed += 1
                if n_pred_offset_fixed <= 3:
                    print(
                        f"  [info CNISP step{step:02d}] {sid}: pred offset "
                        f"{pred_offset} != GT offset {src.gt_label_offset}; "
                        f"using pred-detected offset for Dice.",
                        file=sys.stderr,
                    )
            cnisp_pred_struct_map = build_struct_to_value(
                src.gt_scheme, pred_offset,
            )

            dices = _dice_for_source(
                cn_pred, gt,
                pred_scheme_map=cnisp_pred_struct_map,
                gt_scheme_map=src.gt_struct_to_value,
            )
            eff_res = eff_res_idx.get((sid, step))
            for name in STRUCT_ORDER + ["mean"]:
                per_source_rows.append({
                    "source_id": sid,
                    "gt_source": src.gt_source,
                    "method": cnisp_method_label,
                    "step_size": str(step),
                    "eff_res_mm": (f"{eff_res:.4f}" if eff_res is not None else ""),
                    "structure": name,
                    "dice": f"{dices[name]:.6f}",
                })

        n_done += 1

    print(f"\n[compare_native] processed {n_done} source(s); "
          f"skipped: gt={n_skipped_gt} nnUNet={n_skipped_nnunet}")
    if n_pred_offset_fixed:
        print(f"[compare_native] CNISP pred offset auto-corrected for "
              f"{n_pred_offset_fixed} (source, step) row(s) -- pred file's "
              f"label range disagreed with GT's; re-run CNISP infer.py "
              f"after fixing paths.yaml to silence this.")

    # ── Write per-source CSV ──────────────────────────────────────
    out_dir = work_dir / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_source_csv = out_dir / f"paired_per_source{out_suffix}.csv"
    with open(per_source_csv, "w", newline="") as f:
        fieldnames = ["source_id", "gt_source", "method", "step_size",
                      "eff_res_mm", "structure", "dice"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in per_source_rows:
            w.writerow(r)
    print(f"[compare_native] wrote {per_source_csv}")

    # ── Aggregate into summary CSV + TXT ──────────────────────────
    table_by_struct: Dict[str, Dict[str, Tuple[float, float, int]]] = {
        s: {} for s in STRUCT_ORDER + ["mean"]
    }

    # Group: (method, bucket_label) -> {structure: [dice]}
    # Both methods are bucketed by eff_res_mm; rows without an eff_res
    # fall into the "unknown" bucket at the right edge of the table.
    grouped: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in per_source_rows:
        method = r["method"]
        eff = float(r["eff_res_mm"]) if r["eff_res_mm"] else None
        _, label = _assign_bucket(eff, bucket_edges)
        col = f"{method} {label}"
        grouped[(method, col)][r["structure"]].append(float(r["dice"]))

    # Stable column ordering: pair (nnUNet, CNISP) at each eff-res bucket,
    # buckets sorted by lower bound, unknown sinking to the bottom.
    bucket_order: List[str] = []
    seen_buckets = set()
    for r in per_source_rows:
        if r["eff_res_mm"]:
            eff = float(r["eff_res_mm"])
            _, label = _assign_bucket(eff, bucket_edges)
        else:
            label = "unknown"
        if label not in seen_buckets:
            seen_buckets.add(label)
            bucket_order.append(label)

    def _bucket_sort_key(label: str) -> float:
        if label == "unknown":
            return 1e9
        try:
            lo = label.split(",")[0].lstrip("(")
            return float(lo)
        except Exception:  # noqa: BLE001
            return 1e9

    bucket_order.sort(key=_bucket_sort_key)
    methods_in_order = [NNUNET_METHOD_LABEL, cnisp_method_label]
    all_cols: List[str] = []
    for label in bucket_order:
        for m in methods_in_order:
            all_cols.append(f"{m} {label}")

    for col in all_cols:
        # The method label is everything before the bucket; the bucket
        # always starts with "(" or "unknown", so split on the first
        # such delimiter.
        method = next((m for m in methods_in_order
                       if col.startswith(m + " ")), col.split(" ", 1)[0])
        for struct in STRUCT_ORDER + ["mean"]:
            vals = grouped.get((method, col), {}).get(struct, [])
            if not vals:
                table_by_struct[struct][col] = (float("nan"), float("nan"), 0)
                continue
            arr = np.asarray(vals, dtype=np.float64)
            table_by_struct[struct][col] = (float(arr.mean()),
                                            float(arr.std()),
                                            int(len(arr)))

    summary_csv = out_dir / f"paired_summary{out_suffix}.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bucket", "structure", "mean_dice", "std_dice", "n_sources"])
        for col in all_cols:
            for struct in STRUCT_ORDER + ["mean"]:
                mean, std, n = table_by_struct[struct][col]
                w.writerow([
                    col, struct,
                    "" if math.isnan(mean) else f"{mean:.4f}",
                    "" if math.isnan(std) else f"{std:.4f}",
                    n,
                ])
    print(f"[compare_native] wrote {summary_csv}")

    # ── Plaintext table ───────────────────────────────────────────
    txt_path = out_dir / f"paired_summary{out_suffix}.txt"
    if cnisp_test_label_source == "nnunet_pred":
        chk_note = (
            "  - DEPLOYMENT MODE: chk_* sources are Diced against\n"
            "    Dataset835's dense pred (prediction/native/), shared\n"
            "    between both methods; atlas sources Dice against the\n"
            "    atlas manual GT.\n")
    else:
        chk_note = (
            "  - 6 chk_* sources use the legacy chk_pseudo GT (previous\n"
            "    nnUNet's QA-kept predictions). Filter on\n"
            "    gt_source=='atlas' in paired_per_source.csv for the\n"
            "    manual-GT-only view.\n")
    with open(txt_path, "w") as f:
        f.write("=" * 78 + "\n")
        f.write(f"{NNUNET_METHOD_LABEL} vs {cnisp_method_label} -- "
                f"per-source full-head Dice (native space)\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"CNISP run_tag           : {run_tag}\n")
        f.write(f"CNISP test_label_source : {cnisp_test_label_source}\n\n")
        f.write("Caveats\n")
        f.write(f"  - {cnisp_method_label} is GT-conditioned (sparse-slice "
                f"latent optimization).\n")
        f.write(f"  - {NNUNET_METHOD_LABEL} is image-conditioned. Per-step "
                f"rows feed nnUNet a\n")
        f.write("    sparsified CT (drop every Nth axial slice) at the same\n")
        f.write("    eff_res used by CNISP for that (source, step). The nnUNet\n")
        f.write("    plan was trained at iso 0.5 mm, so large z-spacing rows\n")
        f.write("    are intentionally out-of-distribution -- that's the test.\n")
        f.write("  - nnUNet preds are NN-upsampled along the through-plane axis\n")
        f.write("    back to the native CT grid before Dice; GT is never\n")
        f.write("    resampled.\n")
        f.write(chk_note + "\n")

        f.write(f"Sources processed: {n_done}  "
                f"(skipped GT={n_skipped_gt}, skipped nnUNet={n_skipped_nnunet})\n\n")

        f.write("Mean Dice by eff_res bucket (n_sources in parentheses)\n")
        f.write("-" * 78 + "\n")
        # Column width scales with the longest method-label prefix so
        # CNISP-atlasGT / CNISP-nnUNetPred stay legible.
        max_method_w = max(len(m) for m in methods_in_order)
        col_w = max(22, max_method_w + 16)
        header = "structure".ljust(11) + "".join(c.ljust(col_w) for c in all_cols)
        f.write(header + "\n")
        for struct in STRUCT_ORDER + ["mean"]:
            row = struct.ljust(11)
            for col in all_cols:
                mean, std, n = table_by_struct[struct][col]
                cell = "n/a".ljust(col_w) if math.isnan(mean) \
                    else f"{mean:.3f}+/-{std:.3f}(n={n})".ljust(col_w)
                row += cell
            f.write(row + "\n")
        f.write("\n")

        # (CNISP - nnUNet) delta on the mean row, within each shared bucket.
        f.write(f"{cnisp_method_label} mean Dice minus {NNUNET_METHOD_LABEL} "
                f"mean, same eff_res bucket\n")
        f.write(f"(positive => {cnisp_method_label} wins within that bucket)\n")
        f.write("-" * 78 + "\n")
        for label in bucket_order:
            nn_col = f"{NNUNET_METHOD_LABEL} {label}"
            cn_col = f"{cnisp_method_label} {label}"
            nn_mean = table_by_struct["mean"].get(nn_col, (float("nan"),))[0]
            cn_mean = table_by_struct["mean"].get(cn_col, (float("nan"),))[0]
            if math.isnan(nn_mean) or math.isnan(cn_mean):
                continue
            f.write(f"  {label:<25} {cn_mean - nn_mean:+.3f}\n")
        f.write("\n")
    print(f"[compare_native] wrote {txt_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
