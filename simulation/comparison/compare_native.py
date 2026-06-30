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
* ``{work_dir}/prediction/sparse_step_{XX}_native/{source_id}.nii.gz``
  - nnUNet per-step plan-spacing (iso 0.5) prediction resampled onto the
  native CT grid with nnUNet's own segmentation resampler (the iso mask
  itself lives in ``sparse_step_{XX}_upsampled/``). step_01 is a symlink
  to the dense baseline ``prediction/native/``. Indexed via
  ``{work_dir}/prediction/sweep_manifest.json``.
* ``output_basedir/{model}/runs/{run_tag}/sweep_results.pkl`` -- the
  source of BOTH the eff_res lookup AND the CNISP Dice. CNISP Dice is the
  canonical-space per-eye Dice recorded here (averaged over the two eyes
  to a per-source value), NOT the native-space reconstructed mask. This
  keeps every test source in the eff_res aggregate even when its native
  ``.nii.gz`` was skipped by the ``save_mask_source_ids`` whitelist.
  (The native ``native_space_step_{XX}/`` masks are still written for the
  whitelisted sources for inspection, but compare_native no longer reads
  them.)
* ``output_basedir/{model}/runs/{run_tag}/native_sweep_manifest.json`` --
  records the ``test_label_source`` used for this run, which decides
  whether chk_* sources are Diced against the legacy chk_pseudo GT
  (``atlas_gt`` runs) or against Dataset835's dense pred
  (``nnunet_pred`` runs). Atlas sources always Dice against the
  atlas manual GT.

Comparison
----------
* Both methods contribute one row per (source_id, step_size,
  slice_start_id, structure). For coarse eff_res the sweep fans out over
  start offsets {0,1,2}; those rows share an eff_res bucket and just add
  samples to it. nnUNet Dice is computed on the ORIGINAL CT's voxel grid
  (GT never resampled) from the plan (iso 0.5) spacing predictions
  resampled onto the native CT grid by ``nnunet/predict_sparse_iso.py``.
  CNISP Dice is the canonical-space per-eye Dice from sweep_results.pkl
  (averaged over eyes); the two methods therefore live in different
  spaces but share the same eff_res bucketing for the figure.
* When the CNISP run uses ``test_label_source=nnunet_pred`` we ALSO
  switch nnUNet-sparse's chk_* GT to Dataset835's dense pred so both
  methods Dice against the same target in this bucket. Atlas rows are
  unaffected (they always Dice against atlas manual GT).
* Same effective-resolution bucket edges apply to both methods.

Optionally a third method, nnUNet-C (control C corrector), is folded in
when an ``--nnunet-c-eval-csv`` (or the ``nnunet_c_eval_csv`` config key) is
provided: its per-case Dice (from ``nnunet-c/diagnostics/eval_corrector.py``)
is read as-is and joined to the shared eff_res buckets.

Outputs (under the repo-level ``comparison/`` dir by default; override with
``--out-dir`` / the ``comparison_out_dir`` config key)
------------------------------------------------------
For ``--cnisp-run-tag <T>`` / ``--experiment <E>`` (suffix ``__<T>__<E>``):
* ``paired_per_source__<T>__<E>.csv`` -- long, one row per
  (source, method, step_size, structure, dice).
* ``paired_summary__<T>__<E>.csv``    -- aggregated by
  (method, eff_res_bucket, structure).
* ``paired_summary__<T>__<E>.txt``    -- human-readable wide table.

The companion driver ``simulation/comparison/method_summary.py`` reads
the per-source CSV and renders the per-method PNG.

Usage
-----
    python simulation/comparison/compare_native.py \\
        --config nnunet/configs.yaml --cnisp-run-tag atlas_gt
    python simulation/comparison/compare_native.py \\
        --config nnunet/configs_v7.yaml --cnisp-run-tag nnunet_pred \\
        --experiment thick \\
        --nnunet-c-eval-csv nnunet-c/predictions/PHOTON_CT_CORR_C_cnisp/eval_C_fold0.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np

# Make ``nnunet.*`` and ``simulation.*`` importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nnunet.helpers.buckets import (  # noqa: E402
    NNUNET_C_METHOD_LABEL,
    NNUNET_INTERP_METHOD_LABEL,
    NNUNET_METHOD_LABEL,
    resolve_nnunet_c_runs,
    STRUCT_ORDER,
    assign_bucket,
    bucket_sort_key,
)
from nnunet.helpers.config import load_yaml  # noqa: E402
from nnunet.helpers.paired_csv import (  # noqa: E402
    apply_source_filter,
    resolve_source_prefix_filters,
)
from nnunet.data_prep.resolve_gt import build_struct_to_value  # noqa: E402
from nnunet.lib.metrics import (  # noqa: E402
    affines_consistent,
    build_eff_res_index,
    cnisp_canonical_dice_from_pkl,
    dice_for_source,
    load_label_volume_with_affine,
    lookup_method_label,
    override_chk_gt_for_deployment,
    resample_pred_onto_gt,
    resolve_test_sources,
)
try:
    from simulation.comparison.nnunet_c import load_nnunet_c_rows  # noqa: E402
except ModuleNotFoundError:
    # Importing the ``simulation`` package runs simulation/__init__.py, which
    # eagerly pulls in the degradation operators (torch/numpy). The comparison
    # path only needs this one stdlib-only helper, so fall back to a sibling
    # import when those heavy deps are unavailable.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from nnunet_c import load_nnunet_c_rows  # type: ignore  # noqa: E402

# Repo root (parents: [0]=comparison, [1]=simulation, [2]=repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse_step_tag(step_tag: str) -> Tuple[int, int]:
    """Parse a sweep step tag into ``(step, start)``.

    ``"03"`` -> ``(3, 0)``; ``"03_o2"`` -> ``(3, 2)``. Returns
    ``(None, 0)`` for an unparseable tag so callers can skip it.
    """
    base, _, off = step_tag.partition("_o")
    try:
        step = int(base)
    except ValueError:
        return None, 0
    try:
        start = int(off) if off else 0
    except ValueError:
        start = 0
    return step, start


def run(args) -> int:
    cfg = load_yaml(Path(args.config))
    cnisp_paths = load_yaml(Path(cfg["cnisp_paths_yaml"]))

    model_name = args.model_name or cfg["cnisp_model_name"]
    work_dir = Path(args.work_dir or cfg["work_dir"])
    run_tag = str(args.cnisp_run_tag)
    experiment = str(args.experiment)
    cnisp_method_label = (
        args.cnisp_method_label or lookup_method_label(cfg, run_tag)
    )
    out_suffix = (args.out_suffix if args.out_suffix is not None
                  else f"__{run_tag}__{experiment}")

    # nnUNet-C corrector arms (controls C and/or B). Each arm = (method_label,
    # eval_csv) and is folded into the paired CSV as its own method when its
    # eval CSV exists. Multiple arms come from the ``nnunet_c_runs`` config list
    # (preferred) or the legacy single ``nnunet_c_eval_csv`` key; a CLI
    # ``--nnunet-c-eval-csv`` overrides to a single arm. Resolved here so the
    # banner lists which corrector arms will participate.
    if args.nnunet_c_eval_csv:
        _cli_label = (args.nnunet_c_method_label
                      or cfg.get("nnunet_c_method_label", NNUNET_C_METHOD_LABEL))
        nnunet_c_runs = [(_cli_label, args.nnunet_c_eval_csv)]
    else:
        nnunet_c_runs = resolve_nnunet_c_runs(cfg)
    # Config paths are conventionally relative to the repo root.
    nnunet_c_runs = [
        (lbl, csv if Path(csv).is_absolute() else str(REPO_ROOT / csv))
        for lbl, csv in nnunet_c_runs
    ]

    output_base = (
        Path(cnisp_paths["output_basedir"]) / model_name
        / "runs" / experiment / run_tag
    )
    meta_dir = Path(cnisp_paths["aligned_dir"]) / cnisp_paths.get(
        "metadata_dirname", "metadata"
    )
    casefiles_dir = Path(cnisp_paths["casefiles_dir"])
    test_cases = casefiles_dir / "test_cases.txt"

    nnunet_sweep_manifest = (
        work_dir / "prediction" / experiment / "sweep_manifest.json"
    )
    if not nnunet_sweep_manifest.exists():
        print(f"[compare_native] nnUNet sweep manifest not found: "
              f"{nnunet_sweep_manifest}", file=sys.stderr)
        print(f"  Did you run the `nnunet-predict-sweep` phase for "
              f"experiment={experiment}?", file=sys.stderr)
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

    print(f"[compare_native] experiment               = {experiment}")
    print(f"[compare_native] run_tag                  = {run_tag}")
    print(f"[compare_native] cnisp method label       = {cnisp_method_label}")
    print(f"[compare_native] cnisp test_label_source  = {cnisp_test_label_source}")
    print(f"[compare_native] cnisp run dir            = {output_base}")
    print(f"[compare_native] output suffix            = {out_suffix}")
    for _lbl, _csv in nnunet_c_runs:
        print(f"[compare_native] nnUNet-C arm             = {_lbl}  <-  {_csv}")

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
    sources, missing = resolve_test_sources(
        cnisp_paths, resolve_ct=False, detect_atlas_offset=True,
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
        sources = override_chk_gt_for_deployment(
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

    # ── CNISP per-(source, step, start) Dice loader ───────────────
    # CNISP Dice is read from the run's canonical-space sweep_results.pkl
    # (per eye, averaged to per source) -- NOT from the native-space
    # reconstructed masks. This decouples the eff_res aggregate from the
    # save_mask_source_ids whitelist: most sources never write a native
    # .nii.gz, but every source still has its canonical Dice in the pkl,
    # so the cross-method figure keeps all test sources. (Trade-off: the
    # CNISP curve is now canonical-space Dice, matching test_results.csv,
    # rather than native-space merged-mask Dice.)
    cnisp_dice = cnisp_canonical_dice_from_pkl(output_base / "sweep_results.pkl")
    if not cnisp_dice:
        print(f"[compare_native] no CNISP Dice in "
              f"{output_base / 'sweep_results.pkl'}.", file=sys.stderr)
        print(f"  Did the CNISP infer/sweep run for this experiment?",
              file=sys.stderr)
        return 2
    cnisp_keys_by_sid: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for (sid_k, step_k, start_k) in cnisp_dice:
        cnisp_keys_by_sid[sid_k].append((step_k, start_k))
    for sid_k in cnisp_keys_by_sid:
        cnisp_keys_by_sid[sid_k].sort()
    print(f"[compare_native] CNISP (step,start) combos available: "
          f"{sorted({k[1:] for k in cnisp_dice})}")

    # ── nnUNet per-step manifest loader ───────────────────────────
    # Same idea as the CNISP loader: basenames anchored against the
    # canonical native-grid output convention written by
    # ``nnunet/predict_sparse_iso.py``:
    #     ${work_dir}/prediction/sparse_step_{XX}_native/{sid}.nii.gz
    # That mask is the plan-spacing (iso 0.5) network prediction
    # resampled onto the native CT grid with nnUNet's own segmentation
    # resampler (NOT the old NN slice-duplication). The matching
    # iso-spacing prediction lives in ``sparse_step_{XX}_upsampled/``;
    # Dice uses the native one so GT is never resampled.
    nn_pred_root = work_dir / "prediction" / experiment
    with open(nnunet_sweep_manifest) as f:
        nn_m = json.load(f)
    # nnUNet steps are keyed by (step, start). For coarse eff_res the
    # sweep fans out over start offsets {0,1,2}; the manifest key is "03"
    # for start=0 and "03_o1" for start=1, with native masks under
    # sparse_step_03_native/ and sparse_step_03_o1_native/ respectively.
    nnunet_step_paths: Dict[Tuple[int, int], Dict[str, Path]] = {}
    for step_tag, sid_map in nn_m.get("steps", {}).items():
        step, start = _parse_step_tag(str(step_tag))
        if step is None:
            continue
        sd = f"sparse_step_{step:02d}" + (f"_o{start}" if start else "")
        canonical_dir = nn_pred_root / f"{sd}_native"
        nnunet_step_paths[(step, start)] = {
            sid: canonical_dir / Path(raw).name
            for sid, raw in sid_map.items()
        }
    if not nnunet_step_paths:
        print(f"[compare_native] nnUNet sweep manifest has no usable steps: "
              f"{nnunet_sweep_manifest}", file=sys.stderr)
        return 2
    print(f"[compare_native] nnUNet (step,start) combos available: "
          f"{sorted(nnunet_step_paths)}")

    # ── nnUNet Taubin post-processing control (optional) ──────────
    # The `nnunet-interp` phase writes Taubin-smoothed nnUNet preds onto the
    # native grid under prediction/<exp>/interpolation/sparse_step_XX/ and an
    # interp_manifest.json. When present, we add an `nnUNet-interp` column to
    # the paired table; when absent, the comparison runs exactly as before.
    interp_manifest_path = (
        nn_pred_root / "interpolation" / "interp_manifest.json"
    )
    interp_step_paths: Dict[Tuple[int, int], Dict[str, Path]] = {}
    if interp_manifest_path.exists():
        with open(interp_manifest_path) as f:
            interp_m = json.load(f)
        for step_tag, sid_map in interp_m.get("steps", {}).items():
            step, start = _parse_step_tag(str(step_tag))
            if step is None:
                continue
            sd = f"sparse_step_{step:02d}" + (f"_o{start}" if start else "")
            interp_dir = nn_pred_root / "interpolation" / sd
            interp_step_paths[(step, start)] = {
                sid: interp_dir / Path(raw).name
                for sid, raw in sid_map.items()
            }
        print(f"[compare_native] nnUNet-interp (step,start) combos available: "
              f"{sorted(interp_step_paths)}")
    else:
        print(f"[compare_native] no interp manifest at {interp_manifest_path}; "
              f"skipping the {NNUNET_INTERP_METHOD_LABEL} column (run the "
              f"`nnunet-interp` phase to add it).")

    # ── eff_res lookup ────────────────────────────────────────────
    eff_res_idx = build_eff_res_index(output_base / "sweep_results.pkl")

    # ── Iterate sources & emit per-source rows ────────────────────
    per_source_rows: List[Dict[str, str]] = []
    n_done = 0
    n_skipped_gt = 0
    n_skipped_nnunet = 0
    n_nnunet_resampled = 0
    n_skipped_interp = 0
    n_interp_resampled = 0

    for src in sources:
        sid = src.source_id
        gt_path = src.gt_label_path
        if not gt_path.exists():
            n_skipped_gt += 1
            print(f"  [skip] {sid}: GT not found at {gt_path}", file=sys.stderr)
            continue
        try:
            gt, gt_affine = load_label_volume_with_affine(gt_path)
        except Exception as e:  # noqa: BLE001
            n_skipped_gt += 1
            print(f"  [skip] {sid}: failed to read GT ({e})", file=sys.stderr)
            continue

        # ── nnUNet per step ───────────────────────────────────────
        # nnUNet predictions live on the raw CT (native) grid; for step>1
        # they've been resampled from plan (iso 0.5) spacing onto that grid
        # by nnunet/predict_sparse_iso.py (world-coordinate resample) before
        # reaching here. For atlas sources the GT shares that grid, so Dice is
        # voxel-for-voxel. For chk_* sources the pseudo-GT was saved on a
        # different (resampled) grid, so the pred is resampled onto the GT grid
        # below (GT itself is never resampled).
        for (step, start) in sorted(nnunet_step_paths):
            otag = f" o{start}" if start else ""
            path_map = nnunet_step_paths[(step, start)]
            if sid not in path_map:
                continue
            nnunet_pred_path = path_map[sid]
            if not nnunet_pred_path.exists():
                n_skipped_nnunet += 1
                print(f"  [skip nnUNet step{step:02d}{otag}] {sid}: no pred at "
                      f"{nnunet_pred_path}", file=sys.stderr)
                continue
            try:
                nn_pred, nn_aff = load_label_volume_with_affine(nnunet_pred_path)
            except Exception as e:  # noqa: BLE001
                n_skipped_nnunet += 1
                print(f"  [skip nnUNet step{step:02d}{otag}] {sid}: load failed "
                      f"({e})", file=sys.stderr)
                continue
            grid_mismatch = (
                nn_pred.shape != gt.shape
                or not affines_consistent(nn_aff, gt_affine)
            )
            if grid_mismatch:
                is_chk = src.gt_source.startswith("chk_")
                msg = (f"{sid} nnUNet step{step:02d}{otag}: pred grid "
                       f"{nn_pred.shape}/{nib.aff2axcodes(nn_aff)} != GT grid "
                       f"{gt.shape}/{nib.aff2axcodes(gt_affine)}")
                if args.strict_shape:
                    print(f"  [error] {msg}", file=sys.stderr)
                    return 3
                if not is_chk:
                    # atlas sources MUST already sit on the GT (= raw CT) grid;
                    # a mismatch there is a real remap/orientation bug, so keep
                    # the loud skip rather than silently resampling.
                    print(f"  [skip nnUNet step{step:02d}{otag}] {msg} (atlas "
                          f"pred should already be on the GT grid -- NOT "
                          f"comparing).", file=sys.stderr)
                    n_skipped_nnunet += 1
                    continue
                # chk_* pseudo-GT lives on a different (resampled) grid than the
                # raw-CT prediction (e.g. the step01 dense baseline). Resample
                # the PRED onto the GT grid by world coordinates so Dice is
                # voxel-for-voxel; GT is never resampled.
                nn_pred = resample_pred_onto_gt(
                    nn_pred, nn_aff, gt.shape, gt_affine)
                nn_aff = gt_affine
                n_nnunet_resampled += 1
                if n_nnunet_resampled <= 3:
                    print(f"  [info nnUNet step{step:02d}{otag}] {sid}: pred "
                          f"resampled onto chk_* GT grid {gt.shape} "
                          f"(world-aware, order=0).", file=sys.stderr)
            # nnUNet predictions are always written in the bare nnunet
            # scheme {ON:1, Recti:2, Globe:3, Fat:4} with no offset, so
            # the pred scheme map is fixed regardless of source / GT
            # offset. (Dice still works against an offset GT because
            # ``dice_for_source`` uses each side's own scheme map.)
            nnunet_pred_struct_map = build_struct_to_value("nnunet", 0)
            dices = dice_for_source(
                nn_pred, gt,
                pred_scheme_map=nnunet_pred_struct_map,
                gt_scheme_map=src.gt_struct_to_value,
            )
            # eff_res is start-independent (= step * base spacing).
            eff_res = eff_res_idx.get((sid, step))
            for name in STRUCT_ORDER + ["mean"]:
                per_source_rows.append({
                    "source_id": sid,
                    "gt_source": src.gt_source,
                    "method": NNUNET_METHOD_LABEL,
                    "step_size": str(step),
                    "slice_start_id": str(start),
                    "eff_res_mm": (f"{eff_res:.4f}" if eff_res is not None else ""),
                    "structure": name,
                    "dice": f"{dices[name]:.6f}",
                })

        # ── nnUNet-interp (Taubin control) per step ───────────────
        # Same native grid as the nnUNet preds above (the smoothed degraded
        # pred was resampled onto sparse_step_XX_native's grid), so the same
        # atlas/chk_ grid handling applies. nnUNet-only control; absent when
        # the `nnunet-interp` phase has not been run.
        for (step, start) in sorted(interp_step_paths):
            otag = f" o{start}" if start else ""
            path_map = interp_step_paths[(step, start)]
            if sid not in path_map:
                continue
            interp_pred_path = path_map[sid]
            if not interp_pred_path.exists():
                n_skipped_interp += 1
                print(f"  [skip interp step{step:02d}{otag}] {sid}: no pred at "
                      f"{interp_pred_path}", file=sys.stderr)
                continue
            try:
                in_pred, in_aff = load_label_volume_with_affine(interp_pred_path)
            except Exception as e:  # noqa: BLE001
                n_skipped_interp += 1
                print(f"  [skip interp step{step:02d}{otag}] {sid}: load failed "
                      f"({e})", file=sys.stderr)
                continue
            grid_mismatch = (
                in_pred.shape != gt.shape
                or not affines_consistent(in_aff, gt_affine)
            )
            if grid_mismatch:
                is_chk = src.gt_source.startswith("chk_")
                msg = (f"{sid} interp step{step:02d}{otag}: pred grid "
                       f"{in_pred.shape}/{nib.aff2axcodes(in_aff)} != GT grid "
                       f"{gt.shape}/{nib.aff2axcodes(gt_affine)}")
                if args.strict_shape:
                    print(f"  [error] {msg}", file=sys.stderr)
                    return 3
                if not is_chk:
                    print(f"  [skip interp step{step:02d}{otag}] {msg} (atlas "
                          f"pred should already be on the GT grid -- NOT "
                          f"comparing).", file=sys.stderr)
                    n_skipped_interp += 1
                    continue
                in_pred = resample_pred_onto_gt(
                    in_pred, in_aff, gt.shape, gt_affine)
                in_aff = gt_affine
                n_interp_resampled += 1
                if n_interp_resampled <= 3:
                    print(f"  [info interp step{step:02d}{otag}] {sid}: pred "
                          f"resampled onto chk_* GT grid {gt.shape} "
                          f"(world-aware, order=0).", file=sys.stderr)
            # Taubin masks keep the bare nnUNet scheme {ON:1, Recti:2, Globe:3,
            # Fat:4}, same as the nnUNet preds.
            interp_pred_struct_map = build_struct_to_value("nnunet", 0)
            dices = dice_for_source(
                in_pred, gt,
                pred_scheme_map=interp_pred_struct_map,
                gt_scheme_map=src.gt_struct_to_value,
            )
            eff_res = eff_res_idx.get((sid, step))
            for name in STRUCT_ORDER + ["mean"]:
                per_source_rows.append({
                    "source_id": sid,
                    "gt_source": src.gt_source,
                    "method": NNUNET_INTERP_METHOD_LABEL,
                    "step_size": str(step),
                    "slice_start_id": str(start),
                    "eff_res_mm": (f"{eff_res:.4f}" if eff_res is not None else ""),
                    "structure": name,
                    "dice": f"{dices[name]:.6f}",
                })

        # ── CNISP per (step, start) ───────────────────────────────
        # CNISP Dice comes from the canonical-space sweep pkl (read once
        # above into ``cnisp_dice``), averaged over the two eyes. No
        # native mask is read here, so sources outside the
        # save_mask_source_ids whitelist still contribute their Dice to
        # the eff_res aggregate. The eff_res value is taken from the pkl
        # row when present, falling back to the shared eff_res index.
        for (step, start) in cnisp_keys_by_sid.get(sid, []):
            rec = cnisp_dice.get((sid, step, start))
            if rec is None:
                continue
            dices = rec["dice"]
            eff_res = rec.get("effective_resolution_mm")
            if eff_res is None:
                eff_res = eff_res_idx.get((sid, step))
            for name in STRUCT_ORDER + ["mean"]:
                per_source_rows.append({
                    "source_id": sid,
                    "gt_source": src.gt_source,
                    "method": cnisp_method_label,
                    "step_size": str(step),
                    "slice_start_id": str(start),
                    "eff_res_mm": (f"{eff_res:.4f}" if eff_res is not None else ""),
                    "structure": name,
                    "dice": f"{dices.get(name, float('nan')):.6f}",
                })

        n_done += 1

    print(f"\n[compare_native] processed {n_done} source(s); "
          f"skipped: gt={n_skipped_gt} nnUNet={n_skipped_nnunet}"
          + (f" interp={n_skipped_interp}" if interp_step_paths else ""))
    if n_nnunet_resampled:
        print(f"[compare_native] nnUNet pred resampled onto chk_* GT grid for "
              f"{n_nnunet_resampled} (source, step) row(s) (world-aware, "
              f"order=0; GT never resampled).")
    if n_interp_resampled:
        print(f"[compare_native] interp pred resampled onto chk_* GT grid for "
              f"{n_interp_resampled} (source, step) row(s) (world-aware, "
              f"order=0; GT never resampled).")

    # ── nnUNet-C corrector rows (optional third method) ───────────
    # nnUNet-C Dice comes pre-computed from eval_corrector.py's per-case CSV
    # (prediction resampled onto the native GT grid, order 0 -- the SAME
    # convention used above for nnUNet-sparse). We only join the eff_res from
    # the shared per-(source, step) index so the corrector lands on the exact
    # same buckets as the other two methods.
    nnunet_c_methods_added: List[str] = []
    if nnunet_c_runs:
        gt_source_by_sid = {s.source_id: s.gt_source for s in sources}
        for nnc_label, nnc_csv in nnunet_c_runs:
            if not Path(nnc_csv).exists():
                print(f"[compare_native] nnUNet-C arm {nnc_label!r}: eval CSV "
                      f"not found ({nnc_csv}); skipping this arm.")
                continue
            nnunet_c_rows = load_nnunet_c_rows(
                Path(nnc_csv),
                nnc_label,
                eff_res_idx,
                STRUCT_ORDER,
                gt_source_by_sid=gt_source_by_sid,
            )
            if nnunet_c_rows and nnc_label not in nnunet_c_methods_added:
                per_source_rows.extend(nnunet_c_rows)
                nnunet_c_methods_added.append(nnc_label)

    # ── Write per-source CSV ──────────────────────────────────────
    # Outputs land in the repo-level ``comparison/`` dir by default (the
    # shared deliverable location), overridable with --out-dir / the
    # ``comparison_out_dir`` config key.
    out_dir = Path(
        args.out_dir
        or cfg.get("comparison_out_dir")
        or (REPO_ROOT / "comparison")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    per_source_csv = out_dir / f"paired_per_source{out_suffix}.csv"
    with open(per_source_csv, "w", newline="") as f:
        fieldnames = ["source_id", "gt_source", "method", "step_size",
                      "slice_start_id", "eff_res_mm", "structure", "dice"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in per_source_rows:
            w.writerow(r)
    print(f"[compare_native] wrote {per_source_csv}")

    # ── Aggregate into summary CSV + TXT ──────────────────────────
    # The aggregated summary (and the PNGs built from it) is the
    # head-to-head VISUALIZATION; we keep it focused on the human-labelled
    # atlas cohort and drop chk_* rows (the per-source CSV above still holds
    # every source, so chk_* Dice stays on record and correct -- it is just
    # not counted in the comparison summary). Same viz_*_source_prefixes
    # mechanism the plot scripts use (default excludes 'chk_').
    summary_include, summary_exclude = resolve_source_prefix_filters(
        None, None, cfg)
    summary_rows = apply_source_filter(
        per_source_rows, summary_include, summary_exclude)
    n_summary_dropped = len(per_source_rows) - len(summary_rows)
    if n_summary_dropped:
        print(f"[compare_native] summary excludes {n_summary_dropped} row(s) "
              f"by source prefix (include={summary_include or '[]'}, "
              f"exclude={summary_exclude or '[]'}); per-source CSV keeps all.")

    table_by_struct: Dict[str, Dict[str, Tuple[float, float, int]]] = {
        s: {} for s in STRUCT_ORDER + ["mean"]
    }

    # Group: (method, bucket_label) -> {structure: [dice]}
    # Both methods are bucketed by eff_res_mm; rows without an eff_res
    # fall into the "unknown" bucket at the right edge of the table.
    grouped: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in summary_rows:
        method = r["method"]
        eff = float(r["eff_res_mm"]) if r["eff_res_mm"] else None
        _, label = assign_bucket(eff, bucket_edges)
        col = f"{method} {label}"
        grouped[(method, col)][r["structure"]].append(float(r["dice"]))

    # Stable column ordering: pair (nnUNet, CNISP) at each eff-res bucket,
    # buckets sorted by lower bound, unknown sinking to the bottom.
    bucket_order: List[str] = []
    seen_buckets = set()
    for r in summary_rows:
        if r["eff_res_mm"]:
            eff = float(r["eff_res_mm"])
            _, label = assign_bucket(eff, bucket_edges)
        else:
            label = "unknown"
        if label not in seen_buckets:
            seen_buckets.add(label)
            bucket_order.append(label)

    bucket_order.sort(key=bucket_sort_key)
    # Insert the Taubin control between nnUNet and CNISP, only when it was
    # actually scored (interp masks present), so absent runs keep the
    # original two-column layout.
    methods_in_order = [NNUNET_METHOD_LABEL]
    if interp_step_paths:
        methods_in_order.append(NNUNET_INTERP_METHOD_LABEL)
    methods_in_order.append(cnisp_method_label)
    for nnc_label in nnunet_c_methods_added:
        if nnc_label not in methods_in_order:
            methods_in_order.append(nnc_label)
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
    # The summary/visualization below is restricted to the source cohort
    # surviving the viz prefix filter (default: atlas only, chk_* dropped).
    summary_atlas_only = ("chk_" in tuple(summary_exclude)) or bool(summary_include)
    if cnisp_test_label_source == "nnunet_pred":
        chk_note = (
            "  - DEPLOYMENT MODE: chk_* sources Dice against Dataset835's\n"
            "    dense pred (prediction/native/); atlas sources Dice against\n"
            "    the atlas manual GT.\n")
    else:
        chk_note = (
            "  - chk_* sources use the legacy chk_pseudo GT (previous\n"
            "    nnUNet's QA-kept predictions).\n")
    if summary_atlas_only:
        chk_note += (
            "  - This summary table is ATLAS-ONLY: chk_* rows are excluded\n"
            "    from the aggregation/visualization (viz source filter). They\n"
            "    are still computed and kept -- in full -- in\n"
            "    paired_per_source.csv (filter gt_source there to inspect).\n")
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
        f.write("  - nnUNet preds are taken at the plan (iso 0.5) spacing and\n")
        f.write("    resampled onto the native CT grid with nnUNet's own\n")
        f.write("    segmentation resampler before Dice; GT is never resampled.\n")
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


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="nnunet/configs.yaml")
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--out-dir", default=None,
                    help="Where to write the paired CSV/TXT bundle. Default: "
                         "the ``comparison_out_dir`` config key, else the "
                         "repo-level ``comparison/`` directory.")
    ap.add_argument("--cnisp-run-tag", default="atlas_gt",
                    help="Which CNISP run to compare against (subdir under "
                         "output_basedir/<model>/runs/<experiment>/). Default "
                         "atlas_gt preserves the ceiling-curve comparison.")
    ap.add_argument("--experiment", choices=["thin", "thick", "real"],
                    default="thin",
                    help="Experiment directory layer (thin|thick|real). "
                         "Reads CNISP masks from runs/<experiment>/<run-tag>/ "
                         "and nnUNet sparse preds from prediction/<experiment>/"
                         ", and exp-suffixes the output CSVs so thin/thick "
                         "comparisons coexist.")
    ap.add_argument("--cnisp-method-label", default=None,
                    help="Override the CNISP method label. If unset, look up "
                         "cnisp_runs_to_compare in the config.")
    ap.add_argument("--nnunet-c-eval-csv", default=None,
                    help="Path to an nnUNet-C eval CSV (eval_corrector.py "
                         "output). When given (or via the ``nnunet_c_eval_csv``"
                         " config key), nnUNet-C is added as a third method "
                         "in the paired CSV/summary. Its Dice is read as-is "
                         "and joined to the shared eff_res buckets.")
    ap.add_argument("--nnunet-c-method-label", default=None,
                    help="Override the nnUNet-C method label (default: the "
                         "``nnunet_c_method_label`` config key, else "
                         f"{NNUNET_C_METHOD_LABEL!r}).")
    ap.add_argument("--out-suffix", default=None,
                    help="Suffix for output filenames. Default is "
                         "'__<cnisp_run_tag>__<experiment>' so multiple runs "
                         "do not collide.")
    ap.add_argument("--strict-shape", action="store_true",
                    help="Fail if a prediction's shape differs from GT "
                         "(default: skip the source with a warning).")
    return ap


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
