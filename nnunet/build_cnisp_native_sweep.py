#!/usr/bin/env python3
"""Backfill: map every CNISP sweep step back to native head space.

``orbital_shape_prior_st1/engine/infer.py`` now writes
``runs/<run_tag>/native_space_step_XX/`` plus a
``runs/<run_tag>/native_sweep_manifest.json`` itself, so a *fresh* CNISP
inference run does not need this script.

This file is kept as a **backfill helper** for runs that finished before
that change, and as a uniform entry point that ``run_pipeline.sh``'s
``compare`` phase invokes for every CNISP run (it's a no-op when the
per-step manifests already exist). It re-uses the same
``map_results_to_native`` call and produces the same layout, but is
idempotent: any ``native_space_step_XX/manifest.json`` that already
exists is skipped unless ``--force`` is passed.

When the run's top-level ``native_sweep_manifest.json`` records
``test_label_source=nnunet_pred``, the backfill follows the chk_* /
atlas dispatch (``metadata_dataset835/`` vs ``metadata/``) so it never
loses track of which canonical crop a chk_* deployment patch came
from.

Outputs (per step, only when missing or ``--force``):
    output_basedir/{model_name}/runs/{run_tag}/native_space_step_{XX}/
        {original_stem}_cnisp_step{XX}.nii.gz   # OD+OS merged
        manifest.json                            # source_id -> nifti path

Usage
-----
    # Ceiling-curve run (default)
    python nnunet/build_cnisp_native_sweep.py --config nnunet/configs.yaml
    # Deployment-curve run
    python nnunet/build_cnisp_native_sweep.py --config nnunet/configs.yaml \\
        --run-tag nnunet_pred
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

# Make ``nnunet.*`` importable when run as ``python nnunet/...``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nnunet.helpers.config import (  # noqa: E402
    add_cnisp_src_to_syspath,
    load_yaml,
    stem_of,
)

add_cnisp_src_to_syspath(__file__)

from engine.native_mapping import map_results_to_native  # noqa: E402


def _meta_path_for_casename_factory(
    aligned_dir: Path,
    metadata_dirname: str,
    metadata_dataset835_dirname: str,
    test_label_source: str,
):
    """Same dispatch as engine/test_label_sources._meta_path_for_case.

    Duplicated here (rather than imported) because the legacy backfill
    must run without orbital_shape_prior_st1 on the Python path -- in
    practice we still add the path via sys.path manipulation, but the
    duplication avoids importing torch/nibabel transitively from
    engine.test_label_sources just for one dispatch table.
    """
    meta_dir = aligned_dir / metadata_dirname
    meta_dataset835_dir = aligned_dir / metadata_dataset835_dirname

    def _resolve(casename: str) -> Path:
        if (test_label_source == "nnunet_pred"
                and not casename.startswith("atlas_")):
            return meta_dataset835_dir / f"{casename}.json"
        return meta_dir / f"{casename}.json"
    return _resolve


def run(args) -> int:
    cfg = load_yaml(Path(args.config))
    cnisp_paths = load_yaml(Path(cfg["cnisp_paths_yaml"]))

    model_name = args.model_name or cfg["cnisp_model_name"]
    output_base = (
        Path(cnisp_paths["output_basedir"]) / model_name
        / "runs" / args.experiment / args.run_tag
    )
    sweep_pkl = output_base / "sweep_results.pkl"
    aligned_dir = Path(cnisp_paths["aligned_dir"])

    # Read test_label_source from the run's top-level manifest if
    # present (so the backfill picks the right metadata tree for chk_*
    # cases). Falls back to "atlas_gt" for legacy runs that pre-date
    # Option C.
    top_manifest_path = output_base / "native_sweep_manifest.json"
    test_label_source = "atlas_gt"
    save_id_set = None  # None = save all (back-compat / legacy runs)
    if top_manifest_path.exists():
        try:
            with open(top_manifest_path) as f:
                _top = json.load(f)
            test_label_source = _top.get("test_label_source", "atlas_gt")
            _ids = _top.get("save_mask_source_ids")
            if _ids:
                save_id_set = set(_ids)
        except (OSError, json.JSONDecodeError):
            pass

    meta_path_for = _meta_path_for_casename_factory(
        aligned_dir,
        cnisp_paths.get("metadata_dirname", "metadata"),
        cnisp_paths.get("metadata_dataset835_dirname", "metadata_dataset835"),
        test_label_source,
    )

    if not sweep_pkl.exists():
        print(f"[build_cnisp_native_sweep] sweep_results.pkl not found: {sweep_pkl}",
              file=sys.stderr)
        print("  Did infer.py finish? Expected under "
              f"{output_base}", file=sys.stderr)
        return 2

    print(f"[build_cnisp_native_sweep] loading {sweep_pkl}")
    with open(sweep_pkl, "rb") as f:
        all_results: List[dict] = pickle.load(f)
    print(f"[build_cnisp_native_sweep] {len(all_results)} (case, step) rows; "
          f"run_tag={args.run_tag} test_label_source={test_label_source}")

    # ── Group by (step_size, slice_start_id) ──────────────────────
    # The high-eff_res start-offset fan-out adds start>0 rows; start=0 keeps
    # the legacy native_space_step_XX/ name + bare-int manifest key.
    by_step: Dict[tuple, List[dict]] = defaultdict(list)
    for r in all_results:
        if "step_size" not in r or "pred_class_map" not in r:
            continue
        by_step[(int(r["step_size"]),
                 int(r.get("slice_start_id", 0)))].append(r)

    step_whitelist = None
    if args.steps:
        step_whitelist = {int(s) for s in args.steps.split(",") if s.strip()}

    keys_to_map = sorted(by_step)
    if step_whitelist is not None:
        keys_to_map = [(s, o) for (s, o) in keys_to_map if s in step_whitelist]
        missing = sorted(step_whitelist - {s for (s, _o) in by_step})
        if missing:
            print(f"  [warn] requested steps not in sweep: {missing}")
    print(f"[build_cnisp_native_sweep] (step,start) to map: {keys_to_map}")

    overall_manifest: Dict[str, Dict[str, str]] = {}

    for (step, start) in keys_to_map:
        step_results = by_step[(step, start)]
        _sd = f"step_{step:02d}" if start == 0 else f"step_{step:02d}_o{start}"
        _ostr = "" if start == 0 else f"_o{start}"
        step_dir = output_base / f"native_space_{_sd}"
        suffix = f"_cnisp_step{step:02d}{_ostr}"
        manifest_path = step_dir / "manifest.json"

        _mkey = str(step) if start == 0 else f"{step}_o{start}"

        # Skip if infer.py (or a previous backfill run) already produced
        # this (step, start). ``--force`` overrides.
        if manifest_path.exists() and not args.force:
            try:
                with open(manifest_path) as f:
                    existing = json.load(f)
                existing_map = existing.get("by_source_id", {})
            except (OSError, json.JSONDecodeError):
                existing_map = {}
            if existing_map:
                print(f"\n  {_sd}: already mapped "
                      f"({len(existing_map)} sources, manifest={manifest_path})"
                      f" -- skip (--force to override).")
                overall_manifest[_mkey] = existing_map
                continue

        print(f"\n  {_sd}: {len(step_results)} cases -> {step_dir}")

        step_dir.mkdir(parents=True, exist_ok=True)
        native_paths = map_results_to_native(
            step_results, aligned_dir, step_dir, suffix=suffix,
            meta_path_for_casename=meta_path_for,
            save_source_ids=save_id_set,
        )

        # ── source_id <-> output path manifest ────────────────────
        # map_results_to_native names files by the source's
        # ``original_nifti_path`` stem. To make compare_native happy we
        # rebuild the (source_id -> native filename) map here. Only the
        # **basename** is stored: consumers anchor it against the
        # manifest's own directory so the artefact survives any data
        # move (only the location of the manifest matters, not what
        # absolute path was current at write time).
        path_by_stem = {stem_of(p): Path(p).name for p in native_paths}
        step_manifest: Dict[str, str] = {}
        seen_sources = set()
        for r in step_results:
            meta_path = meta_path_for(r["casename"])
            if not meta_path.exists():
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            # ``source_id`` in metadata already includes the source prefix
            # (e.g. "atlas_orbit0001_ubMask_al2_fill", "chk_14455"), so we
            # can use it verbatim.
            source_id = str(meta["source_id"])
            if source_id in seen_sources:
                continue
            # Only list sources whose native mask was actually written.
            if save_id_set is not None and source_id not in save_id_set:
                continue
            seen_sources.add(source_id)
            stem = stem_of(meta["original_nifti_path"])
            # ``map_results_to_native`` writes "{stem}{suffix}.nii.gz"
            output_stem = stem + suffix
            if output_stem in path_by_stem:
                step_manifest[source_id] = path_by_stem[output_stem]
            else:
                # Fall back to constructed name (catches odd suffix cases)
                fname = f"{stem}{suffix}.nii.gz"
                if (step_dir / fname).exists():
                    step_manifest[source_id] = fname
                else:
                    print(f"    [warn] no native file for {source_id} (stem={stem})")

        with open(manifest_path, "w") as f:
            json.dump({
                "step_size": step,
                "slice_start_id": start,
                "model_name": model_name,
                "run_tag": args.run_tag,
                "test_label_source": test_label_source,
                "suffix": suffix,
                "n_sources": len(step_manifest),
                "by_source_id": step_manifest,
            }, f, indent=2)
        print(f"    manifest: {manifest_path} ({len(step_manifest)} sources)")
        overall_manifest[_mkey] = step_manifest

    summary_path = output_base / "native_sweep_manifest.json"
    with open(summary_path, "w") as f:
        json.dump({
            "model_name": model_name,
            "run_tag": args.run_tag,
            "test_label_source": test_label_source,
            "steps": {str(k): v for k, v in overall_manifest.items()},
        }, f, indent=2)
    print(f"\n[build_cnisp_native_sweep] summary: {summary_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="nnunet/configs.yaml")
    ap.add_argument("--model-name", default=None,
                    help="Override cnisp_model_name from config")
    ap.add_argument("--run-tag", default="atlas_gt",
                    help="Which CNISP run to backfill, under "
                         "output_basedir/<model>/runs/<experiment>/<run-tag>/.")
    ap.add_argument("--experiment", choices=["thin", "thick", "real"],
                    default="thin",
                    help="Experiment directory layer (thin|thick|real) under "
                         "runs/. Must match the experiment infer.py wrote.")
    ap.add_argument("--steps", default=None,
                    help="Optional comma-separated step_size whitelist "
                         "(default: every step present in sweep_results.pkl)")
    ap.add_argument("--force", action="store_true",
                    help="Re-map even if native_space_step_XX/manifest.json "
                         "already exists (default: skip steps already done).")
    return ap


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
