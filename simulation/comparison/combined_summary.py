#!/usr/bin/env python3
"""One combined figure overlaying EVERY method for an experiment.

``paired_summary.py`` renders one figure per CNISP run_tag (nnUNet-sparse +
that run's CNISP curve + the nnUNet-C overlay). This driver instead pulls
ALL methods present for a single experiment onto ONE figure:

  * nnUNet-sparse                       (shared dense sweep, read once)
  * every CNISP run in cnisp_runs_to_compare (e.g. CNISP-v6.5-gt-atlasGT and
    CNISP-v6.5-gt) -- one curve each
  * nnUNet-C                            (control C corrector, read once)

so a 4-curve thick comparison lands in a single panel. The bottom delta
panel is the head-to-head ``<delta-method> - nnUNet-sparse`` (default
delta-method = nnUNet-C), i.e. how much the corrector beats the raw nnUNet
baseline within each effective-resolution bucket.

nnUNet-sparse and nnUNet-C are run-tag-independent, so they are read ONCE
from a canonical CSV (preferring nnunet_pred, else the first run_tag) to
avoid double-counting; each CNISP curve comes from its own run_tag CSV.

Outputs (under ``--out-dir``):

* ``combined_dice_vs_eff_res.png``            combined 3-row figure (the one
                                              you asked for: all curves +
                                              nnUNet-C - nnUNet delta)
* ``combined_overall_dice_vs_eff_res.png``    stand-alone overall panel
* ``combined_per_class_dice_vs_eff_res.png``  stand-alone 2x2 per-class panel
* ``combined_delta_dice_vs_eff_res.png``      stand-alone delta panel

Usage
-----
    python simulation/comparison/combined_summary.py \\
        --config nnunet/configs_v6_5_gt.yaml \\
        --comparison-dir comparison \\
        --experiment thick \\
        --out-dir comparison/viz/combined__thick
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Make ``nnunet.*`` importable (repo root is two levels up from
# simulation/comparison/).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nnunet.helpers.buckets import (  # noqa: E402
    DEFAULT_BUCKET_EDGES_MM,
    NNUNET_C_METHOD_LABEL,
    NNUNET_METHOD_LABEL,
    resolve_nnunet_c_runs,
)
from nnunet.helpers.config import load_yaml  # noqa: E402
from nnunet.helpers.paired_csv import (  # noqa: E402
    apply_source_filter,
    read_paired_csv,
    resolve_source_prefix_filters,
)
from nnunet.lib.viz import (  # noqa: E402
    aggregate_paired,
    canonical_run_tag,
    discover_experiments,
    draw_delta,
    draw_paired_overall,
    draw_paired_per_class,
    save_standalone,
)


def _collect_rows(
    comparison_dir: Path,
    exp: str,
    run_tags: List[str],
    run_to_method: Dict[str, str],
    nnunet_c_labels: List[str],
    include_pref: List[str],
    exclude_pref: List[str],
) -> List[Dict]:
    """Merge nnUNet-sparse (once) + each CNISP run + each nnUNet-C arm (once).

    The shared (run-tag-independent) nnUNet-sparse / nnUNet-C rows are read
    from a canonical CSV chosen ONLY among the run_tags passed here (i.e. the
    ones configured for this comparison), so a leftover ``nnunet_pred`` CSV
    from a different config never hijacks the baseline.
    """
    rows: List[Dict] = []
    canon = canonical_run_tag(run_tags) if run_tags else None
    if canon is None:
        return rows
    canon_csv = comparison_dir / f"paired_per_source__{canon}__{exp}.csv"

    # nnUNet-sparse + every nnUNet-C arm: run-tag-independent, read once.
    for shared_method in [NNUNET_METHOD_LABEL, *nnunet_c_labels]:
        try:
            r = read_paired_csv(canon_csv, shared_method)
        except SystemExit:
            continue  # method absent in this CSV; skip quietly
        rows.extend(apply_source_filter(r, include_pref, exclude_pref))

    # One CNISP curve per run_tag.
    for run_tag in run_tags:
        method = run_to_method.get(run_tag)
        if not method:
            continue
        csv_path = comparison_dir / f"paired_per_source__{run_tag}__{exp}.csv"
        try:
            r = read_paired_csv(csv_path, method)
        except (FileNotFoundError, SystemExit):
            continue
        rows.extend(apply_source_filter(r, include_pref, exclude_pref))
    return rows


def run(args) -> int:
    cfg = load_yaml(Path(args.config))
    comparison_dir = Path(args.comparison_dir)
    exp = str(args.experiment)
    out_dir = (Path(args.out_dir) if args.out_dir
               else comparison_dir / "viz" / f"combined__{exp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    edges = list(cfg.get("summary_bucket_edges_mm", list(DEFAULT_BUCKET_EDGES_MM)))
    include_pref, exclude_pref = resolve_source_prefix_filters(
        args.include_source_prefixes, args.exclude_source_prefixes, cfg)

    # Every nnUNet-C corrector arm (controls C and/or B), in config order.
    nnunet_c_labels = [lbl for lbl, _csv in resolve_nnunet_c_runs(cfg)]
    if not nnunet_c_labels:
        nnunet_c_labels = [cfg.get("nnunet_c_method_label",
                                   NNUNET_C_METHOD_LABEL)]
    # Delta panel deltas a single method vs nnUNet-sparse: CLI override, else
    # the first corrector arm.
    delta_label = args.delta_method or nnunet_c_labels[0]

    # run_tag -> CNISP method label (from config), ordered as configured.
    run_to_method: Dict[str, str] = {}
    for entry in cfg.get("cnisp_runs_to_compare", []) or []:
        rt = str(entry.get("run_tag", ""))
        ml = str(entry.get("method_label", ""))
        if rt and ml:
            run_to_method[rt] = ml

    discovered = discover_experiments(comparison_dir)
    run_tags = discovered.get(exp, [])
    if not run_tags:
        print(f"[combined_summary] no paired_per_source__*__{exp}.csv under "
              f"{comparison_dir}; run the `compare` phase first.",
              file=sys.stderr)
        return 2
    # Only draw the run_tags this config declares (in config order); any other
    # paired CSVs present on disk (e.g. a leftover nnunet_pred from a different
    # config) are intentionally ignored so the figure stays scoped + the
    # baseline canonical CSV is chosen among these run_tags only.
    ordered_run_tags = [rt for rt in run_to_method if rt in run_tags]
    if not ordered_run_tags:
        print(f"[combined_summary] none of the configured run_tags "
              f"{list(run_to_method)} have a __{exp} CSV under {comparison_dir} "
              f"(found: {run_tags}).", file=sys.stderr)
        return 2

    rows = _collect_rows(comparison_dir, exp, ordered_run_tags, run_to_method,
                         nnunet_c_labels, include_pref, exclude_pref)
    if not rows:
        print(f"[combined_summary] no rows after filtering for experiment="
              f"{exp}.", file=sys.stderr)
        return 2

    bucket_order, by_method_bucket, eff_by_bucket = aggregate_paired(rows, edges)

    # Overlay order: nnUNet-sparse, each CNISP curve (config order), then every
    # nnUNet-C arm (config order) last. Only methods that produced rows survive.
    present = {m for (m, _b) in by_method_bucket}
    methods: List[str] = []
    if NNUNET_METHOD_LABEL in present:
        methods.append(NNUNET_METHOD_LABEL)
    for rt in ordered_run_tags:
        m = run_to_method.get(rt)
        if m and m in present and m not in methods:
            methods.append(m)
    for lbl in nnunet_c_labels:
        if lbl in present and lbl not in methods:
            methods.append(lbl)

    print(f"[combined_summary] experiment={exp}  methods overlaid: {methods}")
    print(f"[combined_summary] delta panel: {delta_label} - "
          f"{NNUNET_METHOD_LABEL}")

    # ── Stand-alone panels ───────────────────────────────────────
    overall_path = out_dir / "combined_overall_dice_vs_eff_res.png"
    save_standalone(overall_path, (11, 5), lambda ax: draw_paired_overall(
        ax, methods, bucket_order, by_method_bucket, eff_by_bucket))

    per_class_path = out_dir / "combined_per_class_dice_vs_eff_res.png"
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    draw_paired_per_class(axes, methods, bucket_order, by_method_bucket,
                          eff_by_bucket)
    fig.suptitle("Per-class Dice vs effective resolution", fontsize=12,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(str(per_class_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    delta_path = out_dir / "combined_delta_dice_vs_eff_res.png"
    save_standalone(delta_path, (11, 5), lambda ax: draw_delta(
        ax, delta_label, bucket_order, by_method_bucket, eff_by_bucket))

    # ── Combined 3-row figure (the requested single image) ───────
    combined_path = out_dir / "combined_dice_vs_eff_res.png"
    fig = plt.figure(figsize=(12, 18))
    gs = fig.add_gridspec(3, 1, hspace=0.35, height_ratios=[1, 1.9, 1])

    ax0 = fig.add_subplot(gs[0])
    draw_paired_overall(ax0, methods, bucket_order, by_method_bucket,
                        eff_by_bucket)

    inner_gs = gs[1].subgridspec(2, 2, hspace=0.5, wspace=0.25)
    inner_axes = [
        [fig.add_subplot(inner_gs[0, 0]), fig.add_subplot(inner_gs[0, 1])],
        [fig.add_subplot(inner_gs[1, 0]), fig.add_subplot(inner_gs[1, 1])],
    ]
    draw_paired_per_class(inner_axes, methods, bucket_order, by_method_bucket,
                          eff_by_bucket)

    ax2 = fig.add_subplot(gs[2])
    draw_delta(ax2, delta_label, bucket_order, by_method_bucket,
               eff_by_bucket)

    fig.suptitle(
        f"{exp}: Dice vs effective resolution -- "
        + " / ".join(methods)
        + f"  (delta = {delta_label} - {NNUNET_METHOD_LABEL})",
        fontsize=12, fontweight="bold", y=0.92,
    )
    fig.savefig(str(combined_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[combined_summary] wrote -> {out_dir}/")
    for p in (combined_path, overall_path, per_class_path, delta_path):
        print(f"  {p}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="nnunet/configs.yaml")
    ap.add_argument("--comparison-dir", required=True,
                    help="repo-level comparison/ dir (holds paired_per_source"
                         "__<run_tag>__<exp>.csv).")
    ap.add_argument("--experiment", required=True,
                    choices=["thin", "thick", "real"],
                    help="Which experiment's CSVs to overlay onto one figure.")
    ap.add_argument("--out-dir", default=None,
                    help="Default: <comparison-dir>/viz/combined__<exp>.")
    ap.add_argument("--delta-method", default=None,
                    help="Method whose (method - nnUNet-sparse) delta fills the "
                         "bottom panel. Default: the nnunet_c_method_label "
                         f"config key, else {NNUNET_C_METHOD_LABEL!r}.")
    ap.add_argument("--include-source-prefixes", default=None)
    ap.add_argument("--exclude-source-prefixes", default=None)
    return ap


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
