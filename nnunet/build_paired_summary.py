#!/usr/bin/env python3
"""Paired (head-to-head) Dice comparison plots for one CNISP run.

Sibling of ``nnunet/build_method_summary.py``. Where that script
emits a *per-method* by-eff_res bundle (one method's curves alone), this
one consumes the SAME ``paired_per_source__<run_tag>.csv`` and overlays
both methods on every panel so the comparison is visible at a glance.

Subplots (delta-focused layout, matches the layout the user picked in
the design checkpoint):

  1. Overall mean Dice vs effective resolution
       two lines on shared axes: nnUNet-sparse vs CNISP-<run>
  2. Per-class Dice vs effective resolution
       2x2 grid (ON / Globe / Fat / Recti); each panel overlays both
       methods so you can see whether ON drives the gap, etc.
  3. (CNISP - nnUNet) Dice delta vs eff_res bucket  (mean row)
       bar chart of the head-to-head difference within each shared
       bucket; positive bars => CNISP wins for that bucket.

Outputs (under ``--out-dir``):

* ``paired_overall_dice_vs_eff_res.png``       stand-alone panel 1
* ``paired_per_class_dice_vs_eff_res.png``     stand-alone panel 2 (2x2 grid)
* ``paired_delta_dice_vs_eff_res.png``         stand-alone panel 3
* ``paired_dice_vs_eff_res.png``               combined 3-row figure
* ``paired_summary_by_eff_res.csv``            machine-readable delta + per-
                                               method mean/std/n in each
                                               bucket (one row per
                                               (bucket, structure)).

The titles deliberately avoid the word "reconstruction": CNISP IS a
reconstruction model, but nnUNet-sparse is image-conditioned
segmentation, so we say "Dice vs effective resolution" -- a description
that's accurate for both.

The aggregation / table writer / plotters live in ``nnunet.lib.viz``;
this script just reads the CSV, filters sources, and wires those together.

Usage
-----
    python nnunet/build_paired_summary.py \\
        --config nnunet/configs.yaml \\
        --paired-csv work_dir/comparison/paired_per_source__atlas_gt.csv \\
        --cnisp-method CNISP-atlasGT \\
        --out-dir    work_dir/comparison/viz/paired__atlas_gt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``nnunet.*`` importable when run as ``python nnunet/...``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nnunet.helpers.buckets import (  # noqa: E402
    DEFAULT_BUCKET_EDGES_MM,
    NNUNET_METHOD_LABEL,
)
from nnunet.helpers.config import load_yaml  # noqa: E402
from nnunet.helpers.paired_csv import (  # noqa: E402
    apply_source_filter,
    read_paired_csv,
    resolve_source_prefix_filters,
)
from nnunet.lib.viz import (  # noqa: E402
    aggregate_paired,
    plot_paired,
    write_paired_csv,
)


def run(args) -> int:
    cfg = load_yaml(Path(args.config))
    bucket_edges = list(cfg.get(
        "summary_bucket_edges_mm",
        list(DEFAULT_BUCKET_EDGES_MM),
    ))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    include_prefixes, exclude_prefixes = resolve_source_prefix_filters(
        args.include_source_prefixes, args.exclude_source_prefixes, cfg,
    )

    methods = [args.nnunet_method, args.cnisp_method]
    rows = read_paired_csv(Path(args.paired_csv), methods)
    n_before = len(rows)
    rows = apply_source_filter(rows, include_prefixes, exclude_prefixes)
    if include_prefixes or exclude_prefixes:
        print(
            f"[build_paired_summary] source filter: "
            f"include={include_prefixes!r} exclude={exclude_prefixes!r} "
            f"-> {len(rows)}/{n_before} rows kept.",
            file=sys.stderr,
        )
    if not rows:
        raise SystemExit(
            f"All rows filtered out (include={include_prefixes!r}, "
            f"exclude={exclude_prefixes!r}). Relax the filter or check "
            f"source_id prefixes in {args.paired_csv}."
        )
    bucket_order, by_method_bucket, eff_by_bucket = aggregate_paired(
        rows, bucket_edges,
    )

    paths = plot_paired(
        args.cnisp_method, bucket_order, by_method_bucket,
        eff_by_bucket, out_dir,
    )
    csv_path = out_dir / "paired_summary_by_eff_res.csv"
    write_paired_csv(
        args.cnisp_method, bucket_order, by_method_bucket,
        eff_by_bucket, csv_path,
    )

    print(f"[build_paired_summary] {args.cnisp_method} vs {args.nnunet_method}: "
          f"{len(rows)} rows -> {out_dir}/")
    for k, p in paths.items():
        print(f"  [{k}] {p}")
    print(f"  [csv]      {csv_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="nnunet/configs.yaml")
    ap.add_argument(
        "--paired-csv", required=True,
        help="Path to paired_per_source__<run_tag>.csv (written by "
             "nnunet/compare_native.py).",
    )
    ap.add_argument(
        "--cnisp-method", required=True,
        help="Method label of the CNISP rows in the paired CSV "
             "(e.g. CNISP-atlasGT, CNISP-nnUNetPred). The nnUNet method "
             "label is always 'nnUNet-sparse'.",
    )
    ap.add_argument(
        "--nnunet-method", default=NNUNET_METHOD_LABEL,
        help=f"Override the nnUNet method label (default: {NNUNET_METHOD_LABEL}).",
    )
    ap.add_argument(
        "--out-dir", required=True,
        help="Where to write the paired plots and CSV. Pipeline "
             "convention: ${work_dir}/comparison/viz/paired__<run_tag>/.",
    )
    ap.add_argument(
        "--include-source-prefixes", default=None,
        help="Comma-separated source_id prefixes to keep (e.g. 'atlas_'). "
             "Default: read 'viz_include_source_prefixes' from --config "
             "(if absent, no include-side filtering -- keep everything).",
    )
    ap.add_argument(
        "--exclude-source-prefixes", default=None,
        help="Comma-separated source_id prefixes to drop (e.g. 'chk_'). "
             "Default: read 'viz_exclude_source_prefixes' from --config "
             "(default 'chk_' there, so the paired plots stay focused on "
             "human-labelled cases and avoid the chk_ deployment-mode "
             "identity-1.0 row at step=1).",
    )
    return ap


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
