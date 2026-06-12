#!/usr/bin/env python3
"""Per-method Dice summary by effective-resolution bucket.

Reads ``{work_dir}/comparison/paired_per_source__<run_tag>.csv`` (one
row per ``(source_id, method, step_size, structure, dice)``, written
by ``nnunet/compare_native.py``) and produces, for the requested method
label (e.g. ``nnUNet-sparse`` / ``CNISP-atlasGT`` / ``CNISP-nnUNetPred``),
a matched set of artifacts:

* ``{out_dir}/{method}_per_source.csv``          - long, filtered to method
* ``{out_dir}/{method}_summary_by_eff_res.csv``  - aggregated by
                                                    ``(eff_res_bucket, structure)``
* ``{out_dir}/{method}_summary_by_eff_res.txt``  - human-readable wide table
* ``{out_dir}/{method}_recon_summary.png``       - combined 3-subplot figure
* ``{out_dir}/{method}_overall_dice_vs_eff_res.png``
* ``{out_dir}/{method}_per_class_dice_vs_eff_res.png``
* ``{out_dir}/{method}_per_case_dice_distribution.png``
  The latter three are the same subplots in stand-alone format so each
  panel renders at full size (the combined PNG above cannot reproduce
  that without becoming too tall to read). Subplots:
    1. overall mean Dice vs eff_res     (errorbar over sources in bucket)
    2. per-class Dice vs eff_res        (4 lines: ON / Globe / Fat / Recti)
    3. per-case Dice distribution       (boxplot + scatter per bucket)

Why every method shares one driver
----------------------------------
``compare_native.py`` already emits
``comparison/paired_per_source__<run_tag>.csv`` with both methods'
rows interleaved -- same source set, same eff_res values, same bucket
edges. Driving the per-method viz off that file guarantees the CNISP
and nnUNet summaries never drift out of sync (same n_sources, same
axis), and the same plotting code renders any method just by changing
``--method``.

Notes
-----
* ``paired_per_source.csv`` only carries dense Dice -- ``compare_native``
  never computes "observed-only" Dice. That's why this viz drops the
  observed line CNISP's old ``recon_summary.png`` used to plot.
* eff_res values are read straight from the CSV (which inherited them
  from CNISP's ``sweep_results.pkl`` via ``compare_native``), so all
  methods share the same x-axis sample per source/step.

The aggregation / table writers / plotters live in ``nnunet.lib.viz``;
this script just reads the CSV, filters sources, and wires those together.

Usage
-----
    # nnUNet-sparse standalone bundle (rendered ONCE; the nnUNet sparse
    # predictions are independent of which CNISP run_tag is in flight,
    # and run_pipeline.sh picks paired_per_source__nnunet_pred.csv as
    # the canonical source because it is a strict superset of the
    # atlas_gt CSV's nnUNet-sparse rows).
    python nnunet/build_method_summary.py \\
        --config nnunet/configs.yaml \\
        --method nnUNet-sparse \\
        --paired-csv work_dir/comparison/paired_per_source__nnunet_pred.csv \\
        --out-dir    work_dir/comparison/viz/nnUNet-sparse

    # CNISP-atlasGT (one CNISP bundle per run_tag, since each CNISP
    # run uses a different latent-opt input and so has a different
    # Dice curve).
    python nnunet/build_method_summary.py \\
        --config nnunet/configs.yaml \\
        --method CNISP-atlasGT \\
        --paired-csv work_dir/comparison/paired_per_source__atlas_gt.csv \\
        --out-dir    cnisp_output_basedir/<model>/viz/atlas_gt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``nnunet.*`` importable when run as ``python nnunet/...``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nnunet.helpers.buckets import DEFAULT_BUCKET_EDGES_MM  # noqa: E402
from nnunet.helpers.config import load_yaml  # noqa: E402
from nnunet.helpers.paired_csv import (  # noqa: E402
    apply_source_filter,
    read_paired_csv,
    resolve_source_prefix_filters,
)
from nnunet.lib.viz import (  # noqa: E402
    aggregate_by_bucket,
    plot_method_summary,
    write_method_per_source_csv,
    write_method_summary_csv,
    write_method_summary_txt,
)


def run(args) -> int:
    cfg = load_yaml(Path(args.config))
    paired_csv = Path(args.paired_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bucket_edges = list(cfg.get(
        "summary_bucket_edges_mm",
        list(DEFAULT_BUCKET_EDGES_MM),
    ))

    include_prefixes, exclude_prefixes = resolve_source_prefix_filters(
        args.include_source_prefixes, args.exclude_source_prefixes, cfg,
    )

    rows = read_paired_csv(paired_csv, args.method)
    n_before = len(rows)
    rows = apply_source_filter(rows, include_prefixes, exclude_prefixes)
    if include_prefixes or exclude_prefixes:
        print(
            f"[build_method_summary] source filter: "
            f"include={include_prefixes!r} exclude={exclude_prefixes!r} "
            f"-> {len(rows)}/{n_before} rows kept.",
            file=sys.stderr,
        )
    if not rows:
        raise SystemExit(
            f"All rows filtered out for method={args.method!r}; "
            f"relax include/exclude prefixes or check source_id values."
        )
    bucket_order, bucket_struct, bucket_eff, bucket_step = aggregate_by_bucket(
        rows, bucket_edges,
    )

    per_src = out_dir / f"{args.method}_per_source.csv"
    summary_csv = out_dir / f"{args.method}_summary_by_eff_res.csv"
    summary_txt = out_dir / f"{args.method}_summary_by_eff_res.txt"
    summary_png = out_dir / f"{args.method}_recon_summary.png"
    standalone_paths = {
        "overall":   out_dir / f"{args.method}_overall_dice_vs_eff_res.png",
        "per_class": out_dir / f"{args.method}_per_class_dice_vs_eff_res.png",
        "per_case":  out_dir / f"{args.method}_per_case_dice_distribution.png",
    }

    write_method_per_source_csv(rows, per_src)
    write_method_summary_csv(bucket_order, bucket_struct, bucket_eff, summary_csv)
    write_method_summary_txt(args.method, bucket_order, bucket_struct,
                             bucket_eff, summary_txt)
    plot_method_summary(args.method, bucket_order, bucket_struct,
                        bucket_eff, bucket_step, summary_png,
                        standalone_paths=standalone_paths)

    print(f"[build_method_summary] {args.method}: {len(rows)} long rows -> "
          f"{out_dir}/")
    for p in (per_src, summary_csv, summary_txt, summary_png,
              standalone_paths["overall"],
              standalone_paths["per_class"],
              standalone_paths["per_case"]):
        print(f"  {p}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="nnunet/configs.yaml")
    ap.add_argument("--method", required=True,
                    help="Method label as written into paired_per_source"
                         "__*.csv (e.g. nnUNet-sparse, CNISP-atlasGT, "
                         "CNISP-nnUNetPred).")
    ap.add_argument("--paired-csv", required=True,
                    help="Path to the paired CSV for this CNISP run "
                         "(e.g. ${work_dir}/comparison/paired_per_source"
                         "__atlas_gt.csv).")
    ap.add_argument(
        "--out-dir", required=True,
        help="Where to write outputs. Pipeline conventions: "
             "${work_dir}/comparison/viz/<method>__<run_tag>/ for nnUNet "
             "rows, ${cnisp_output_basedir}/<model>/viz/<run_tag>/ for CNISP rows.",
    )
    ap.add_argument(
        "--include-source-prefixes", default=None,
        help="Comma-separated source_id prefixes to keep (e.g. 'atlas_'). "
             "Default: read 'viz_include_source_prefixes' from --config.",
    )
    ap.add_argument(
        "--exclude-source-prefixes", default=None,
        help="Comma-separated source_id prefixes to drop (e.g. 'chk_'). "
             "Default: read 'viz_exclude_source_prefixes' from --config "
             "(default 'chk_' there).",
    )
    return ap


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
