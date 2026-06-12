#!/usr/bin/env python3
"""Visualise nnUNet's own Dice on the degraded (sparsified) images, by step.

This answers a single question -- "how good is nnUNet's segmentation of the
sparse-CT scan, as the through-plane sparsification grows?" -- and nothing
else. It reads ONLY nnUNet's own native-grid predictions and the native GT;
NO CNISP prediction is ever touched, and it does not depend on the
``compare`` phase.

Inputs
------
* ``{work_dir}/prediction/{exp}/sweep_manifest.json`` -- ``{steps: {XX:
  {sid: basename}}}``, the per-(step, source) index written by
  ``nnunet/predict_sparse_iso.py``.
* ``{work_dir}/prediction/{exp}/sparse_step_{XX}_native/{sid}.nii.gz`` --
  nnUNet's prediction on the step-XX sparsified CT, resampled onto the
  native CT grid (the thing we Dice). step_01 is the dense baseline.
* ``{work_dir}/input/{exp}/sparse_manifest.json`` (optional) -- supplies
  ``eff_res_mm`` per (source, step) for the eff_res column. Missing -> NaN.
* native-head GT (the ground-truth masks, which live under the CNISP
  aligned/metadata tree). This is the segmentation target, not a CNISP
  result; it is resolved with the shared ``resolve_gt`` helper so the GT
  scheme/offset handling matches the rest of the project.

The native Dice scorer, eff_res lookup, aggregation, table writers and
plotters all live in ``nnunet.lib`` (``lib.metrics`` + ``lib.viz``), shared
with the head-to-head comparison so the numbers can never drift apart on how
a mask is read or how Dice is scored. This script is just the orchestration
that wires those primitives together.

Outputs (under ``{work_dir}/prediction/{exp}/native_summary/``)
--------------------------------------------------------------
* ``nnunet_native_per_source__{exp}.csv``  -- WIDE, one row per
  (source_id, step_size): a column per structure (ON/Globe/Fat/Recti) +
  the 4-class ``mean`` + ``eff_res_mm``. Mirrors CNISP's ``test_results.csv``.
* ``nnunet_native_by_step__{exp}.csv``     -- aggregated by step_size:
  ``n_sources`` + ``mean +/- std`` per structure.
* ``nnunet_native_by_eff_res__{exp}.csv``  -- the same rows aggregated into
  the effective-resolution buckets used by ``build_method_summary`` (shared
  ``summary_bucket_edges_mm``), so this table/figure line up point-for-point
  with CNISP's ``*_dice_vs_eff_res`` plots. NOTE: step axis and eff_res axis
  are NOT interchangeable -- eff_res = base_spacing * step and base_spacing
  varies per scan, so one step spreads across several eff_res buckets.
* ``nnunet_native_dice_vs_step__{exp}.png`` -- overall mean Dice vs step
  (left) and the four per-class curves vs step (right).
* ``nnunet_native_dice_vs_eff_res__{exp}.png`` -- the same, on the eff_res
  axis (comparable to CNISP's per-method/paired summaries). step_01 (the
  dense baseline) is placed at its base through-plane spacing so the raw
  point appears at the low-eff_res end like CNISP's, rather than being
  dropped.

Outputs land under the prediction tree (like CNISP keeps its summaries
under its own run dir): ``{work_dir}/prediction/{exp}/native_summary/``.

Usage
-----
    python nnunet/build_nnunet_native_summary.py \\
        --config nnunet/configs.yaml --experiment thick

    # keep chk_* sources too (default drops them via viz_exclude_*):
    python nnunet/build_nnunet_native_summary.py \\
        --config nnunet/configs.yaml --experiment thick \\
        --exclude-source-prefixes ""
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

# Make ``nnunet.*`` importable when run as ``python nnunet/...``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nnunet.helpers.buckets import (  # noqa: E402
    DEFAULT_BUCKET_EDGES_MM,
    NNUNET_METHOD_LABEL,
)
from nnunet.helpers.config import load_yaml  # noqa: E402
from nnunet.helpers.paired_csv import resolve_source_prefix_filters  # noqa: E402
from nnunet.lib.metrics import (  # noqa: E402
    compute_nnunet_native_rows,
    eff_res_from_sparse_manifest,
    resolve_test_sources,
)
from nnunet.lib.viz import (  # noqa: E402
    aggregate_native_by_eff_res,
    aggregate_native_by_step,
    plot_native_dice_vs_eff_res,
    plot_native_dice_vs_step,
    write_native_by_eff_res_csv,
    write_native_by_step_csv,
    write_native_per_source_csv,
)


def build_nnunet_native_summary(
    cfg: Dict,
    work_dir: Path,
    experiment: str,
    out_dir: Path,
    include_prefixes: List[str],
    exclude_prefixes: List[str],
) -> List[Path]:
    """Collect nnUNet native Dice and write the per-step bundle. Returns paths."""
    cnisp_paths = load_yaml(Path(cfg["cnisp_paths_yaml"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    sources, _missing = resolve_test_sources(cnisp_paths)

    # Source-prefix filter (default: drop chk_* like the other viz scripts).
    inc = tuple(p for p in include_prefixes if p)
    exc = tuple(p for p in exclude_prefixes if p)
    if inc or exc:
        kept = []
        for s in sources:
            if inc and not s.source_id.startswith(inc):
                continue
            if exc and s.source_id.startswith(exc):
                continue
            kept.append(s)
        print(f"[nnunet_native_summary] source filter: include={list(inc)!r} "
              f"exclude={list(exc)!r} -> {len(kept)}/{len(sources)} sources.",
              file=sys.stderr)
        sources = kept
    if not sources:
        raise SystemExit(
            "All sources filtered out; relax include/exclude prefixes.")

    eff_res_idx = eff_res_from_sparse_manifest(work_dir, experiment)
    wide_rows, stats = compute_nnunet_native_rows(
        work_dir, experiment, sources, eff_res_idx)
    if not wide_rows:
        raise SystemExit(
            "No nnUNet native Dice rows produced -- check that "
            f"prediction/{experiment}/sparse_step_XX_native/ is populated.")

    bucket_edges = list(cfg.get("summary_bucket_edges_mm",
                                list(DEFAULT_BUCKET_EDGES_MM)))
    step_rows = aggregate_native_by_step(wide_rows)
    bucket_rows = aggregate_native_by_eff_res(wide_rows, bucket_edges)

    per_source_csv = out_dir / f"nnunet_native_per_source__{experiment}.csv"
    by_step_csv = out_dir / f"nnunet_native_by_step__{experiment}.csv"
    by_eff_csv = out_dir / f"nnunet_native_by_eff_res__{experiment}.csv"
    step_png = out_dir / f"nnunet_native_dice_vs_step__{experiment}.png"
    eff_png = out_dir / f"nnunet_native_dice_vs_eff_res__{experiment}.png"

    write_native_per_source_csv(wide_rows, per_source_csv)
    write_native_by_step_csv(step_rows, by_step_csv)
    write_native_by_eff_res_csv(bucket_rows, by_eff_csv)
    # step axis = the per-step view; eff_res axis = lines up with CNISP's
    # build_method_summary / paired plots (same bucket edges).
    plot_native_dice_vs_step(step_rows, NNUNET_METHOD_LABEL, step_png)
    plot_native_dice_vs_eff_res(bucket_rows, NNUNET_METHOD_LABEL, eff_png)

    n_sources = len({r["source_id"] for r in wide_rows})
    print(f"[nnunet_native_summary] {NNUNET_METHOD_LABEL} (experiment="
          f"{experiment}): {len(wide_rows)} (source,step) row(s) across "
          f"{n_sources} source(s), {len(step_rows)} step(s).")
    print(f"  sources Diced={stats['sources']} skipped_gt={stats['skipped_gt']} "
          f"skipped_pred={stats['skipped_pred']} "
          f"atlas_grid_mismatch={stats['skipped_atlas_mismatch']} "
          f"chk_resampled={stats['resampled_chk']}")
    outs = [per_source_csv, by_step_csv, by_eff_csv, step_png, eff_png]
    for p in outs:
        print(f"  {p}")
    return outs


def run(args) -> int:
    cfg = load_yaml(Path(args.config))
    work_dir = Path(args.work_dir or cfg["work_dir"])
    experiment = str(args.experiment)
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        out_dir = work_dir / "prediction" / experiment / "native_summary"

    include_prefixes, exclude_prefixes = resolve_source_prefix_filters(
        args.include_source_prefixes, args.exclude_source_prefixes, cfg)

    build_nnunet_native_summary(
        cfg=cfg,
        work_dir=work_dir,
        experiment=experiment,
        out_dir=out_dir,
        include_prefixes=include_prefixes,
        exclude_prefixes=exclude_prefixes,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="nnunet/configs.yaml")
    ap.add_argument("--experiment", choices=["thin", "thick", "real"],
                    default="thin",
                    help="Experiment layer: reads prediction/<experiment>/ "
                         "and writes prediction/<experiment>/native_summary/.")
    ap.add_argument("--work-dir", default=None,
                    help="Override work_dir from the config.")
    ap.add_argument("--out-dir", default=None,
                    help="Output directory. Default: "
                         "${work_dir}/prediction/<experiment>/native_summary/.")
    ap.add_argument("--include-source-prefixes", default=None,
                    help="Comma-separated source_id prefixes to keep. Default: "
                         "'viz_include_source_prefixes' from --config.")
    ap.add_argument("--exclude-source-prefixes", default=None,
                    help="Comma-separated source_id prefixes to drop. Default: "
                         "'viz_exclude_source_prefixes' from --config "
                         "(usually 'chk_'). Pass '' to keep ALL sources.")
    return ap


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
