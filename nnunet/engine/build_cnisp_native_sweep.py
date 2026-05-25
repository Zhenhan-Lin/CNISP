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
    python nnunet/engine/build_cnisp_native_sweep.py --config nnunet/configs.yaml
    # Deployment-curve run
    python nnunet/engine/build_cnisp_native_sweep.py --config nnunet/configs.yaml \\
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

import yaml


# ── Make orbital_shape_prior_st1 importable ───────────────────────
# This file lives at nnunet/engine/build_cnisp_native_sweep.py;
# repo root is two directories up.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CNISP_SRC = _REPO_ROOT / "orbital_shape_prior_st1"
if str(_CNISP_SRC) not in sys.path:
    sys.path.insert(0, str(_CNISP_SRC))


from engine.native_mapping import map_results_to_native  # noqa: E402


def _load_yaml(path: Path) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _stem_of(p: Path | str) -> str:
    name = Path(p).name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    if name.endswith(".nii"):
        return name[: -len(".nii")]
    return Path(name).stem


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="nnunet/configs.yaml")
    ap.add_argument("--model-name", default=None,
                    help="Override cnisp_model_name from config")
    ap.add_argument("--run-tag", default="atlas_gt",
                    help="Which CNISP run to backfill, under "
                         "output_basedir/<model>/runs/<run-tag>/.")
    ap.add_argument("--steps", default=None,
                    help="Optional comma-separated step_size whitelist "
                         "(default: every step present in sweep_results.pkl)")
    ap.add_argument("--force", action="store_true",
                    help="Re-map even if native_space_step_XX/manifest.json "
                         "already exists (default: skip steps already done).")
    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config))
    cnisp_paths = _load_yaml(Path(cfg["cnisp_paths_yaml"]))

    model_name = args.model_name or cfg["cnisp_model_name"]
    output_base = (
        Path(cnisp_paths["output_basedir"]) / model_name / "runs" / args.run_tag
    )
    sweep_pkl = output_base / "sweep_results.pkl"
    aligned_dir = Path(cnisp_paths["aligned_dir"])

    # Read test_label_source from the run's top-level manifest if
    # present (so the backfill picks the right metadata tree for chk_*
    # cases). Falls back to "atlas_gt" for legacy runs that pre-date
    # Option C.
    top_manifest_path = output_base / "native_sweep_manifest.json"
    test_label_source = "atlas_gt"
    if top_manifest_path.exists():
        try:
            with open(top_manifest_path) as f:
                test_label_source = json.load(f).get(
                    "test_label_source", "atlas_gt"
                )
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

    # ── Group by step_size ────────────────────────────────────────
    by_step: Dict[int, List[dict]] = defaultdict(list)
    for r in all_results:
        if "step_size" not in r or "pred_class_map" not in r:
            continue
        by_step[int(r["step_size"])].append(r)

    step_whitelist = None
    if args.steps:
        step_whitelist = {int(s) for s in args.steps.split(",") if s.strip()}

    steps_to_map = sorted(by_step)
    if step_whitelist is not None:
        steps_to_map = [s for s in steps_to_map if s in step_whitelist]
        missing = sorted(step_whitelist - set(by_step))
        if missing:
            print(f"  [warn] requested steps not in sweep: {missing}")
    print(f"[build_cnisp_native_sweep] steps to map: {steps_to_map}")

    overall_manifest: Dict[int, Dict[str, str]] = {}

    for step in steps_to_map:
        step_results = by_step[step]
        step_dir = output_base / f"native_space_step_{step:02d}"
        suffix = f"_cnisp_step{step:02d}"
        manifest_path = step_dir / "manifest.json"

        # Skip if infer.py (or a previous backfill run) already produced
        # this step. ``--force`` overrides.
        if manifest_path.exists() and not args.force:
            try:
                with open(manifest_path) as f:
                    existing = json.load(f)
                existing_map = existing.get("by_source_id", {})
            except (OSError, json.JSONDecodeError):
                existing_map = {}
            if existing_map:
                print(f"\n  step {step:02d}: already mapped "
                      f"({len(existing_map)} sources, manifest={manifest_path})"
                      f" -- skip (--force to override).")
                overall_manifest[step] = existing_map
                continue

        print(f"\n  step {step:02d}: {len(step_results)} cases -> {step_dir}")

        step_dir.mkdir(parents=True, exist_ok=True)
        native_paths = map_results_to_native(
            step_results, aligned_dir, step_dir, suffix=suffix,
            meta_path_for_casename=meta_path_for,
        )

        # ── source_id <-> output path manifest ────────────────────
        # map_results_to_native names files by the source's
        # ``original_nifti_path`` stem. To make compare_native happy we
        # rebuild the (source_id -> native filename) map here. Only the
        # **basename** is stored: consumers anchor it against the
        # manifest's own directory so the artefact survives any data
        # move (only the location of the manifest matters, not what
        # absolute path was current at write time).
        path_by_stem = {_stem_of(p): Path(p).name for p in native_paths}
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
            seen_sources.add(source_id)
            stem = _stem_of(meta["original_nifti_path"])
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
                "model_name": model_name,
                "run_tag": args.run_tag,
                "test_label_source": test_label_source,
                "suffix": suffix,
                "n_sources": len(step_manifest),
                "by_source_id": step_manifest,
            }, f, indent=2)
        print(f"    manifest: {manifest_path} ({len(step_manifest)} sources)")
        overall_manifest[step] = step_manifest

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


if __name__ == "__main__":
    sys.exit(main())
