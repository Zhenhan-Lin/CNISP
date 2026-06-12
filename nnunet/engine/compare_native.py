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
* ``output_basedir/{model}/runs/{run_tag}/native_space_step_{XX}/...``
  -- CNISP per-step predictions for this run, produced by
  ``orbital_shape_prior_st1/engine/infer.py`` (or backfilled by
  ``nnunet/build_cnisp_native_sweep.py``).
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
  is never resampled. nnUNet's sparse-CT predictions are produced at the
  plan (iso 0.5) spacing and resampled onto the native CT grid with
  nnUNet's own segmentation resampler by ``nnunet/predict_sparse_iso.py``
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

The companion driver ``nnunet/build_method_summary.py`` reads
the per-source CSV and renders the per-method PNG.

Usage
-----
    python nnunet/compare_native.py --config nnunet/configs.yaml \\
        --cnisp-run-tag atlas_gt
    python nnunet/compare_native.py --config nnunet/configs.yaml \\
        --cnisp-run-tag nnunet_pred
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np

# Make ``nnunet.*`` importable when this library is imported standalone.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nnunet.helpers.buckets import (  # noqa: E402
    NNUNET_METHOD_LABEL,
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
    detect_pred_offset,
    dice_for_source,
    load_label_volume_with_affine,
    lookup_method_label,
    override_chk_gt_for_deployment,
    resolve_test_sources,
)


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
        print(f"  Did you run nnunet/build_cnisp_native_sweep.py?",
              file=sys.stderr)
        return 2
    print(f"[compare_native] CNISP steps available: {sorted(cnisp_step_paths)}")

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
    nnunet_step_paths: Dict[int, Dict[str, Path]] = {}
    for step_tag, sid_map in nn_m.get("steps", {}).items():
        try:
            step = int(step_tag)
        except ValueError:
            continue
        canonical_dir = nn_pred_root / f"sparse_step_{step:02d}_native"
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
    eff_res_idx = build_eff_res_index(output_base / "sweep_results.pkl")

    # ── Iterate sources & emit per-source rows ────────────────────
    per_source_rows: List[Dict[str, str]] = []
    n_done = 0
    n_skipped_gt = 0
    n_skipped_nnunet = 0
    n_nnunet_resampled = 0
    n_pred_offset_fixed = 0

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
                nn_pred, nn_aff = load_label_volume_with_affine(nnunet_pred_path)
            except Exception as e:  # noqa: BLE001
                n_skipped_nnunet += 1
                print(f"  [skip nnUNet step{step:02d}] {sid}: load failed "
                      f"({e})", file=sys.stderr)
                continue
            grid_mismatch = (
                nn_pred.shape != gt.shape
                or not affines_consistent(nn_aff, gt_affine)
            )
            if grid_mismatch:
                is_chk = src.gt_source.startswith("chk_")
                msg = (f"{sid} nnUNet step{step:02d}: pred grid "
                       f"{nn_pred.shape}/{nib.aff2axcodes(nn_aff)} != GT grid "
                       f"{gt.shape}/{nib.aff2axcodes(gt_affine)}")
                if args.strict_shape:
                    print(f"  [error] {msg}", file=sys.stderr)
                    return 3
                if not is_chk:
                    # atlas sources MUST already sit on the GT (= raw CT) grid;
                    # a mismatch there is a real remap/orientation bug, so keep
                    # the loud skip rather than silently resampling.
                    print(f"  [skip nnUNet step{step:02d}] {msg} (atlas pred "
                          f"should already be on the GT grid -- NOT comparing).",
                          file=sys.stderr)
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
                    print(f"  [info nnUNet step{step:02d}] {sid}: pred resampled "
                          f"onto chk_* GT grid {gt.shape} (world-aware, order=0).",
                          file=sys.stderr)
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
                cn_pred, cn_aff = load_label_volume_with_affine(cnisp_path)
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
            if not affines_consistent(cn_aff, gt_affine):
                msg = (f"{sid} CNISP step{step:02d}: pred/GT affine mismatch "
                       f"(orientation not restored on remap). "
                       f"pred axcodes={nib.aff2axcodes(cn_aff)} "
                       f"GT axcodes={nib.aff2axcodes(gt_affine)}. Element-wise "
                       f"Dice would be on misaligned voxels -- NOT comparing.")
                if args.strict_shape:
                    print(f"  [error] {msg}", file=sys.stderr)
                    return 3
                print(f"  [skip CNISP step{step:02d}] {msg}", file=sys.stderr)
                continue

            pred_offset = detect_pred_offset(cn_pred, src.gt_scheme)
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

            dices = dice_for_source(
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
    if n_nnunet_resampled:
        print(f"[compare_native] nnUNet pred resampled onto chk_* GT grid for "
              f"{n_nnunet_resampled} (source, step) row(s) (world-aware, "
              f"order=0; GT never resampled).")
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
