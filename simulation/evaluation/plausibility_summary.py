#!/usr/bin/env python3
"""Driver: anatomical plausibility metrics + figures.

Reads the 5-arm mask_index.json, filters to the 4 plausibility-relevant arms
(nnUNet, CNISP, Cascade UNet, Proposed), computes per-case per-eye topology /
continuity / shape regularity metrics, runs paired statistical tests, and
renders Figures A-E.

Two-layer comparison:
  Layer 1 (prior channel):  nnUNet  vs  CNISP
  Layer 2 (cascade output): Cascade UNet  vs  Proposed

Usage:
    python simulation/evaluation/plausibility_summary.py \\
        --mask-index comparison/viz/evaluation__thick/mask_index.json \\
        --out comparison/viz/evaluation__thick/plausibility \\
        [--plausibility-csv ...]          \\
        [--min-cc-voxels 5]              \\
        [--do-shape-reg]                 \\
        [--layer1-nnunet-dir <path>]     \\
        [--layer1-cnisp-dir <path>]      \\
        [--qualitative-case <case>]      \\
        [--qualitative-step <int>]       \\
        [--ct-source <path_template>]    \\
        [--test-cases-map <json>]        \\
        [--common-samples / --no-common-samples]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simulation.evaluation.plausibility import build_plausibility_table
from simulation.evaluation.plausibility_aggregate import (
    PLAUSIBILITY_ARMS, assign_buckets, restrict_to_common_cases,
    topology_violation_rate, continuity_by_bucket, paired_tests,
)
from simulation.evaluation.plausibility_plots import (
    topology_violation_figure, continuity_figure,
    compactness_figure, qualitative_figure,
)


def _resolve_ct_path(
    case: str,
    step: int,
    test_cases_map: Optional[Dict],
    ct_source_template: Optional[str],
) -> Optional[str]:
    """Resolve degraded CT path for Figure E."""
    # Priority 1: test_cases_map source_image
    if test_cases_map:
        for entry in test_cases_map.get("cases", {}).values():
            sid = entry.get("source_id", "")
            s = entry.get("step")
            if sid == case and s is not None and int(s) == step:
                src = entry.get("source_image")
                if src and Path(src).exists():
                    return str(src)

    # Priority 2: explicit template
    if ct_source_template:
        p = ct_source_template.format(case=case, step=step)
        if Path(p).exists():
            return p

    # Priority 3: default pattern
    default = f"nnunet-c/data/images/{case}_step{step:02d}_0000.nii.gz"
    if Path(default).exists():
        return default

    return None


def run(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load mask index ──
    with open(args.mask_index) as f:
        full_index = json.load(f)

    # Filter to plausibility-relevant arms
    index = [e for e in full_index if e.get("arm") in PLAUSIBILITY_ARMS]
    if not index:
        print("[plausibility] no entries for plausibility arms in mask_index.",
              file=sys.stderr)
        return 2

    print(f"[plausibility] {len(index)} entries from mask_index "
          f"(arms: {sorted(set(e['arm'] for e in index))})")

    # ── Supplement Layer 1 paths if provided ──
    if args.layer1_nnunet_dir:
        _supplement_layer1(index, full_index, "nnUNet",
                           Path(args.layer1_nnunet_dir), "nnunet")
    if args.layer1_cnisp_dir:
        _supplement_layer1(index, full_index, "CNISP",
                           Path(args.layer1_cnisp_dir), "canonical")

    # ── Compute or load plausibility table ──
    import pandas as pd
    csv_path = out / "plausibility_long.csv"

    # Resume order: explicit --plausibility-csv, else auto-reuse the default-path
    # CSV if it already exists (the per-mask table is HOURS to build, so a plain
    # re-run must NOT silently recompute it). --recompute forces a rebuild.
    src_csv = None
    if args.plausibility_csv and Path(args.plausibility_csv).is_file():
        src_csv = args.plausibility_csv
    elif csv_path.is_file() and not args.recompute:
        src_csv = str(csv_path)
        print(f"[plausibility] reusing existing {csv_path} "
              f"(pass --recompute to rebuild it from the masks).")
    if src_csv:
        print(f"[plausibility] loading prebuilt CSV: {src_csv}")
        df = pd.read_csv(src_csv)
    else:
        df = build_plausibility_table(
            index,
            min_cc_voxels=args.min_cc_voxels,
            do_shape_reg=args.do_shape_reg,
            save_csv=str(csv_path),
            progress=True,
        )
        print(f"[plausibility] wrote {len(df)} rows -> {csv_path}")

    if df.empty:
        print("[plausibility] empty table -- nothing to plot.", file=sys.stderr)
        return 2

    # ── Assign eff_res buckets ──
    df = assign_buckets(df)

    # ── Common-sample restriction ──
    if args.common_samples:
        df = restrict_to_common_cases(df)

    # ── Paired statistical tests ──
    tests_csv = out / "plausibility_tests.csv"
    df_tests = paired_tests(df, save_csv=str(tests_csv))
    print(f"[plausibility] {len(df_tests)} test results -> {tests_csv}")

    # ── Figure A: Topology violation rate ──
    for metric in ("has_multi_cc", "has_holes"):
        viol = topology_violation_rate(df, metric=metric)
        fig_path = out / f"topology_violation_rate_{metric}.png"
        topology_violation_figure(viol, df_tests, fig_path, metric=metric)
        print(f"[plausibility] wrote {fig_path}")

    # ── Figure B: Cross-slice continuity ──
    fig_b = out / "cross_slice_continuity.png"
    continuity_figure(
        df, fig_b,
        metric="mean_centroid_jump_mm",
        ylabel="Mean centroid jump (mm)",
        title="Cross-slice continuity vs effective resolution",
    )
    print(f"[plausibility] wrote {fig_b}")

    # ── Figure C: Cross-slice area stability ──
    fig_c = out / "cross_slice_area_stability.png"
    continuity_figure(
        df, fig_c,
        metric="mean_area_rel_change",
        ylabel="Mean area relative change",
        title="Cross-slice area stability vs effective resolution",
    )
    print(f"[plausibility] wrote {fig_c}")

    # ── Figure D: Compactness (optional) ──
    if args.do_shape_reg and "compactness" in df.columns:
        fig_d = out / "compactness_distribution.png"
        compactness_figure(df, fig_d)
        print(f"[plausibility] wrote {fig_d}")

    # ── Figure E: Qualitative comparison ──
    _render_figure_e(args, df, full_index, out)

    return 0


def _supplement_layer1(
    index: List[Dict],
    full_index: List[Dict],
    arm: str,
    pred_dir: Path,
    scheme: str,
) -> None:
    """Add missing Layer 1 entries from an override directory."""
    existing = {(e["case"], e["step"]) for e in index if e["arm"] == arm}
    # Get case universe from the other arms
    all_cases = {(e["case"], e["step"]) for e in index}
    added = 0
    for case, step in all_cases - existing:
        # Try common naming patterns
        candidates = [
            pred_dir / f"{case}_step{step:02d}.nii.gz",
            pred_dir / f"{case}.nii.gz",
        ]
        for p in candidates:
            if p.exists():
                # Find eff_res from any existing entry for same (case, step)
                eff_res = None
                for e in index:
                    if e["case"] == case and e["step"] == step:
                        eff_res = e.get("eff_res")
                        break
                index.append({
                    "case": case, "arm": arm, "step": step,
                    "eff_res": eff_res,
                    "pred_path": str(p), "pred_scheme": scheme,
                    "offset_pred": 0,
                })
                added += 1
                break
    if added:
        print(f"[plausibility] supplemented {added} entries for arm={arm} "
              f"from {pred_dir}")


def _render_figure_e(args, df, full_index, out: Path) -> None:
    """Resolve CT + predictions for qualitative figure E."""
    case = args.qualitative_case
    step = args.qualitative_step

    if case is None:
        # Auto-pick: case with the most topology violations in nnUNet arm
        nnunet_rows = df[(df["arm"] == "nnUNet") & (df["has_multi_cc"] == True)]
        if nnunet_rows.empty:
            nnunet_rows = df[df["arm"] == "nnUNet"]
        if nnunet_rows.empty:
            return
        viol_counts = nnunet_rows.groupby(["case", "step"])["has_multi_cc"].sum()
        if viol_counts.empty:
            return
        best = viol_counts.idxmax()
        case, step = best

    if step is None:
        # Pick the largest step for this case
        steps = df[df["case"] == case]["step"].unique()
        step = int(max(steps)) if len(steps) > 0 else 3

    # Resolve CT
    test_cases_map = None
    if args.test_cases_map and Path(args.test_cases_map).is_file():
        with open(args.test_cases_map) as f:
            test_cases_map = json.load(f)

    ct_path = _resolve_ct_path(case, step, test_cases_map, args.ct_source)

    # Collect pred paths for this (case, step) from the index
    pred_paths: Dict[str, str] = {}
    pred_schemes: Dict[str, str] = {}
    pred_offsets: Dict[str, int] = {}
    gt_path = None
    gt_scheme = None
    gt_offset = 0

    for e in full_index:
        if e["case"] == case and e["step"] == step:
            arm = e["arm"]
            if arm in ("nnUNet", "CNISP", "Cascade UNet", "Proposed"):
                pred_paths[arm] = e["pred_path"]
                pred_schemes[arm] = e["pred_scheme"]
                pred_offsets[arm] = e.get("offset_pred", 0)
            if gt_path is None:
                gt_path = e.get("gt_path")
                gt_scheme = e.get("gt_scheme")
                gt_offset = e.get("offset_gt", 0)

    if not pred_paths:
        print(f"[plausibility] Figure E: no predictions for case={case} step={step}",
              file=sys.stderr)
        return

    fig_e = out / "qualitative_comparison.png"
    qualitative_figure(
        ct_path=ct_path,
        pred_paths=pred_paths,
        pred_schemes=pred_schemes,
        pred_offsets=pred_offsets,
        gt_path=gt_path,
        gt_scheme=gt_scheme,
        gt_offset=gt_offset,
        out_path=fig_e,
        view="coronal",
    )
    print(f"[plausibility] wrote {fig_e} (case={case}, step={step})")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--mask-index", required=True,
                    help="mask_index.json from build_mask_index.py.")
    ap.add_argument("--out", required=True,
                    help="output directory for figures + CSVs.")
    ap.add_argument("--plausibility-csv", default=None,
                    help="prebuilt plausibility_long.csv (skip recompute). If "
                         "omitted, an existing <out>/plausibility_long.csv is "
                         "reused automatically unless --recompute is given.")
    ap.add_argument("--recompute", action="store_true",
                    help="force rebuilding plausibility_long.csv from the masks "
                         "(hours) even if it already exists at the default path.")
    ap.add_argument("--min-cc-voxels", type=int, default=5,
                    help="minimum CC size to count as a topology violation "
                         "(default 5; filters CNISP rasterization artifacts).")
    ap.add_argument("--do-shape-reg", action="store_true",
                    help="compute shape regularity metrics (requires skimage).")
    ap.add_argument("--layer1-nnunet-dir", default=None,
                    help="override dir for Layer 1 nnUNet pred (if mask_index "
                         "lacks the nnUNet arm).")
    ap.add_argument("--layer1-cnisp-dir", default=None,
                    help="override dir for Layer 1 CNISP pred (if mask_index "
                         "lacks the CNISP arm).")
    ap.add_argument("--qualitative-case", default=None,
                    help="case key for Figure E (mask_index 'case' field). "
                         "Auto-picked if omitted.")
    ap.add_argument("--qualitative-step", type=int, default=None,
                    help="step for Figure E. Auto-picked if omitted.")
    ap.add_argument("--ct-source", default=None,
                    help="path template for degraded CT, e.g. "
                         "'data/images/{case}_step{step:02d}_0000.nii.gz'.")
    ap.add_argument("--test-cases-map", default=None,
                    help="corrector test_cases_map.json (resolves CT via "
                         "source_image field).")
    ap.add_argument("--common-samples", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="restrict to (case, step) common to all 4 arms "
                         "(default on). --no-common-samples uses full set.")
    return ap


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
