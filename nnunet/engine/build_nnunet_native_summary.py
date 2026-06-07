#!/usr/bin/env python3
"""nnUNet-only native-space Dice summary, indexed by sparsification step.

This is a SELF-CONTAINED collector for the nnUNet sparse-CT sweep: it reads
the nnUNet native-grid predictions straight out of the prediction tree and
Dices them against the native-head GT itself. It does NOT depend on the
``compare`` phase / any CNISP run -- the nnUNet curve is independent of
which CNISP latent-opt input is in flight, so there is no reason to wait
for the head-to-head comparison.

Inputs (all under ``{work_dir}`` + the CNISP GT roots)
------------------------------------------------------
* ``{work_dir}/prediction/{exp}/sweep_manifest.json`` -- ``{steps: {XX:
  {sid: basename}}}``, the per-(step, source) index written by
  ``nnunet/engine/predict_sparse_iso.py``.
* ``{work_dir}/prediction/{exp}/sparse_step_{XX}_native/{sid}.nii.gz`` --
  the plan-spacing (iso 0.5) prediction resampled onto the native CT grid
  (the Dice target). step_01 is a symlink to the dense baseline.
* ``{work_dir}/input/{exp}/sparse_manifest.json`` (optional) -- supplies
  ``eff_res_mm`` per (source, step) for the eff_res column. Missing -> NaN.
* native-head GT, resolved exactly as ``compare_native.py`` does
  (atlas manual GT for atlas_* sources; chk_pseudo for chk_* unless
  ``--deployment-chk-gt`` swaps it to Dataset835's dense pred).

Dice handling is shared verbatim with ``compare_native.py`` (same loaders,
same affine check, same chk_* world-aware pred resample, same per-side
label maps) so this summary can never disagree with the comparison.

Outputs (under ``{work_dir}/prediction/{exp}/native_summary/``)
--------------------------------------------------------------
* ``nnunet_native_per_source__{exp}.csv``  -- WIDE, one row per
  (source_id, step_size): a column per structure (ON/Globe/Fat/Recti) +
  the 4-class ``mean`` + ``eff_res_mm``. Mirrors CNISP's ``test_results.csv``.
* ``nnunet_native_by_step__{exp}.csv``     -- aggregated by step_size:
  ``n_sources`` + ``mean +/- std`` per structure.
* ``nnunet_native_dice_vs_step__{exp}.png`` -- overall mean Dice vs step
  (left) and the four per-class curves vs step (right).

Usage
-----
    python nnunet/engine/build_nnunet_native_summary.py \\
        --config nnunet/configs.yaml --experiment thick

    # keep chk_* sources too (default drops them via viz_exclude_*):
    python nnunet/engine/build_nnunet_native_summary.py \\
        --config nnunet/configs.yaml --experiment thick \\
        --exclude-source-prefixes ""
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Make ``nnunet.*`` importable when run as ``python nnunet/engine/...``.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Reuse the comparison machinery verbatim so the standalone nnUNet summary
# and the head-to-head comparison can never drift apart on Dice/GT handling.
from nnunet.compare_native import (  # noqa: E402
    _affines_consistent,
    _build_eff_res_index,
    _dice_for_source,
    _load_label_volume_with_affine,
    _override_chk_gt_for_deployment,
    _resample_pred_onto_gt,
)
from nnunet.helpers.buckets import (  # noqa: E402
    NNUNET_METHOD_LABEL,
    STRUCT_ORDER,
)
from nnunet.helpers.config import load_yaml  # noqa: E402
from nnunet.helpers.paired_csv import resolve_source_prefix_filters  # noqa: E402
from nnunet.resolve_gt import build_struct_to_value, resolve_sources  # noqa: E402


CLASS_COLORS = {
    "ON": "#d62728",
    "Globe": "#1f77b4",
    "Fat": "#2ca02c",
    "Recti": "#9467bd",
}
# Structure columns in display order, including the 4-class mean last.
COLS = STRUCT_ORDER + ["mean"]


# ── eff_res lookup (compare-independent) ──────────────────────────

def _eff_res_from_sparse_manifest(
    work_dir: Path, experiment: str,
) -> Dict[Tuple[str, int], float]:
    """(source_id, step) -> eff_res_mm read from the sparsify manifest.

    ``input/{exp}/sparse_manifest.json`` records ``by_step[XX][sid]`` with
    an ``eff_res_mm`` field (CNISP's row value). step_01 is absent there
    (it is the un-sparsified dense baseline), so its eff_res stays NaN.
    """
    p = work_dir / "input" / experiment / "sparse_manifest.json"
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            m = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    out: Dict[Tuple[str, int], float] = {}
    for step_tag, sid_map in m.get("by_step", {}).items():
        try:
            step = int(step_tag)
        except ValueError:
            continue
        for sid, info in sid_map.items():
            eff = info.get("eff_res_mm")
            if eff is not None:
                out[(sid, step)] = float(eff)
    return out


# ── nnUNet native Dice (self-contained) ──────────────────────────

def compute_nnunet_native_rows(
    work_dir: Path,
    experiment: str,
    sources: List,
    eff_res_idx: Dict[Tuple[str, int], float],
) -> Tuple[List[Dict], Dict[str, int]]:
    """Dice every nnUNet native-grid pred against its GT, one row per (src, step).

    Returns ``(wide_rows, stats)`` where each wide row has ``source_id``,
    ``gt_source``, ``step_size``, ``eff_res_mm`` and one float per
    :data:`COLS`. ``stats`` carries skip/resample counters for logging.
    """
    sweep_manifest = work_dir / "prediction" / experiment / "sweep_manifest.json"
    if not sweep_manifest.exists():
        raise SystemExit(
            f"nnUNet sweep manifest not found: {sweep_manifest}\n"
            f"  Run the `nnunet-predict-sweep` phase (experiment={experiment}) first."
        )
    with open(sweep_manifest) as f:
        nn_m = json.load(f)

    nn_pred_root = work_dir / "prediction" / experiment
    step_paths: Dict[int, Dict[str, Path]] = {}
    for step_tag, sid_map in nn_m.get("steps", {}).items():
        try:
            step = int(step_tag)
        except ValueError:
            continue
        native_dir = nn_pred_root / f"sparse_step_{step:02d}_native"
        step_paths[step] = {
            sid: native_dir / Path(raw).name for sid, raw in sid_map.items()
        }
    if not step_paths:
        raise SystemExit(
            f"nnUNet sweep manifest has no usable steps: {sweep_manifest}")

    # nnUNet masks are always written in the bare nnunet scheme with no
    # offset; the per-source GT keeps its own scheme/offset.
    nnunet_pred_struct_map = build_struct_to_value("nnunet", 0)

    wide_rows: List[Dict] = []
    stats = {"sources": 0, "skipped_gt": 0, "skipped_pred": 0,
             "skipped_atlas_mismatch": 0, "resampled_chk": 0}

    for src in sources:
        sid = src.source_id
        gt_path = src.gt_label_path
        if not gt_path.exists():
            stats["skipped_gt"] += 1
            print(f"  [skip] {sid}: GT not found at {gt_path}", file=sys.stderr)
            continue
        try:
            gt, gt_affine = _load_label_volume_with_affine(gt_path)
        except Exception as e:  # noqa: BLE001
            stats["skipped_gt"] += 1
            print(f"  [skip] {sid}: failed to read GT ({e})", file=sys.stderr)
            continue
        stats["sources"] += 1

        for step in sorted(step_paths):
            path_map = step_paths[step]
            if sid not in path_map:
                continue
            pred_path = path_map[sid]
            if not pred_path.exists():
                stats["skipped_pred"] += 1
                continue
            try:
                nn_pred, nn_aff = _load_label_volume_with_affine(pred_path)
            except Exception as e:  # noqa: BLE001
                stats["skipped_pred"] += 1
                print(f"  [skip nnUNet step{step:02d}] {sid}: load failed ({e})",
                      file=sys.stderr)
                continue

            grid_mismatch = (
                nn_pred.shape != gt.shape
                or not _affines_consistent(nn_aff, gt_affine)
            )
            if grid_mismatch:
                if not src.gt_source.startswith("chk_"):
                    # atlas pred must already sit on the GT (= raw CT) grid;
                    # a mismatch there is a real remap/orientation bug.
                    stats["skipped_atlas_mismatch"] += 1
                    print(f"  [skip nnUNet step{step:02d}] {sid}: pred grid "
                          f"{nn_pred.shape} != GT grid {gt.shape} (atlas "
                          f"should already be on the GT grid).", file=sys.stderr)
                    continue
                # chk_* pseudo-GT lives on a different grid; resample the PRED
                # onto the GT grid by world coords (GT is never resampled).
                nn_pred = _resample_pred_onto_gt(nn_pred, nn_aff, gt.shape, gt_affine)
                stats["resampled_chk"] += 1

            dices = _dice_for_source(
                nn_pred, gt,
                pred_scheme_map=nnunet_pred_struct_map,
                gt_scheme_map=src.gt_struct_to_value,
            )
            row: Dict = {
                "source_id": sid,
                "gt_source": src.gt_source,
                "step_size": step,
                "eff_res_mm": eff_res_idx.get((sid, step), float("nan")),
            }
            for c in COLS:
                row[c] = dices[c]
            wide_rows.append(row)

    wide_rows.sort(key=lambda r: (r["source_id"], r["step_size"]))
    return wide_rows, stats


# ── Aggregation + writers ────────────────────────────────────────

def aggregate_by_step(wide_rows: List[Dict]) -> List[Dict]:
    """Aggregate wide per-(source, step) rows by ``step_size``."""
    by_step: Dict[int, List[Dict]] = defaultdict(list)
    for r in wide_rows:
        by_step[r["step_size"]].append(r)

    out: List[Dict] = []
    for step in sorted(by_step):
        group = by_step[step]
        effs = [r["eff_res_mm"] for r in group if not math.isnan(r["eff_res_mm"])]
        agg: Dict = {
            "step_size": step,
            "n_sources": len(group),
            "eff_res_mm": float(np.mean(effs)) if effs else float("nan"),
        }
        for c in COLS:
            vals = [r[c] for r in group if not math.isnan(r[c])]
            if vals:
                arr = np.asarray(vals, dtype=np.float64)
                agg[f"{c}_mean"] = float(arr.mean())
                agg[f"{c}_std"] = float(arr.std())
            else:
                agg[f"{c}_mean"] = float("nan")
                agg[f"{c}_std"] = float("nan")
        out.append(agg)
    return out


def _fmt(v: float, nd: int = 6) -> str:
    return "" if (v is None or math.isnan(v)) else f"{v:.{nd}f}"


def write_per_source_csv(wide_rows: List[Dict], out_path: Path) -> None:
    fieldnames = ["source_id", "gt_source", "step_size", "eff_res_mm"] + COLS
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fieldnames)
        for r in wide_rows:
            w.writerow([
                r["source_id"], r["gt_source"], r["step_size"],
                _fmt(r["eff_res_mm"], 4),
                *[_fmt(r[c]) for c in COLS],
            ])


def write_by_step_csv(step_rows: List[Dict], out_path: Path) -> None:
    fieldnames = ["step_size", "n_sources", "eff_res_mm"]
    for c in COLS:
        fieldnames += [f"{c}_mean", f"{c}_std"]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fieldnames)
        for r in step_rows:
            row = [r["step_size"], r["n_sources"], _fmt(r["eff_res_mm"], 4)]
            for c in COLS:
                row += [_fmt(r[f"{c}_mean"], 4), _fmt(r[f"{c}_std"], 4)]
            w.writerow(row)


def plot_dice_vs_step(step_rows: List[Dict], method: str, out_path: Path) -> None:
    """Two panels: overall mean Dice vs step, and per-class Dice vs step."""
    steps = [r["step_size"] for r in step_rows]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))

    ys = [r["mean_mean"] for r in step_rows]
    es = [r["mean_std"] for r in step_rows]
    ax0.errorbar(steps, ys, yerr=es, fmt="o-", capsize=4, color="#444")
    for r in step_rows:
        if not math.isnan(r["mean_mean"]):
            ax0.annotate(f"{r['mean_mean']:.3f}\nn={r['n_sources']}",
                         (r["step_size"], r["mean_mean"]),
                         textcoords="offset points", xytext=(0, 10),
                         ha="center", fontsize=8, color="#444")
    ax0.set_xlabel("sparsification step (keep every Nth slice)")
    ax0.set_ylabel("mean Dice (4 foreground classes)")
    ax0.set_title(f"{method}: overall native Dice vs step")
    ax0.set_ylim(0, 1)
    ax0.grid(True, alpha=0.3)

    for c in STRUCT_ORDER:
        ys_c = [r[f"{c}_mean"] for r in step_rows]
        es_c = [r[f"{c}_std"] for r in step_rows]
        ax1.errorbar(steps, ys_c, yerr=es_c, fmt="o-", capsize=3,
                     color=CLASS_COLORS[c], label=c)
    ax1.set_xlabel("sparsification step (keep every Nth slice)")
    ax1.set_ylabel("Dice")
    ax1.set_title(f"{method}: per-class native Dice vs step")
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower left", fontsize=8, ncol=2)

    fig.suptitle(f"{method}: native-space Dice vs sparsification step",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Source resolution (mirrors compare_native.py) ─────────────────

def _resolve_sources_for_summary(cfg: Dict, cnisp_paths: Dict):
    """Resolve the test sources + native GT exactly like compare_native does."""
    meta_dir = Path(cnisp_paths["aligned_dir"]) / cnisp_paths.get(
        "metadata_dirname", "metadata")
    test_cases = Path(cnisp_paths["casefiles_dir"]) / "test_cases.txt"
    atlas_label_dir = cnisp_paths.get("atlas_label_dir")
    checklist_csv_str = cnisp_paths.get("checklist_csv")
    chk_pred_dir = (
        Path(checklist_csv_str).parent / "fold_0" / "predictions"
        if checklist_csv_str else None
    )
    sources, _missing = resolve_sources(
        test_cases_path=test_cases,
        meta_dir=meta_dir,
        detect_atlas_offset=True,
        resolve_ct=False,
        atlas_label_dir=Path(atlas_label_dir) if atlas_label_dir else None,
        chk_pred_dir=chk_pred_dir,
    )
    return sources


def build_nnunet_native_summary(
    cfg: Dict,
    work_dir: Path,
    experiment: str,
    out_dir: Path,
    include_prefixes: List[str],
    exclude_prefixes: List[str],
    deployment_chk_gt: bool,
) -> List[Path]:
    """Collect nnUNet native Dice and write the per-step bundle. Returns paths."""
    cnisp_paths = load_yaml(Path(cfg["cnisp_paths_yaml"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = _resolve_sources_for_summary(cfg, cnisp_paths)
    # chk_* GT swap (only relevant if chk_* survive the source filter).
    if deployment_chk_gt:
        deployment_dirname = cfg.get(
            "deployment_gt_dirname_for_chk", "prediction/native")
        sources = _override_chk_gt_for_deployment(
            sources, work_dir, deployment_dirname)

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

    eff_res_idx = _eff_res_from_sparse_manifest(work_dir, experiment)
    wide_rows, stats = compute_nnunet_native_rows(
        work_dir, experiment, sources, eff_res_idx)
    if not wide_rows:
        raise SystemExit(
            "No nnUNet native Dice rows produced -- check that "
            f"prediction/{experiment}/sparse_step_XX_native/ is populated.")

    step_rows = aggregate_by_step(wide_rows)

    per_source_csv = out_dir / f"nnunet_native_per_source__{experiment}.csv"
    by_step_csv = out_dir / f"nnunet_native_by_step__{experiment}.csv"
    png = out_dir / f"nnunet_native_dice_vs_step__{experiment}.png"

    write_per_source_csv(wide_rows, per_source_csv)
    write_by_step_csv(step_rows, by_step_csv)
    plot_dice_vs_step(step_rows, NNUNET_METHOD_LABEL, png)

    n_sources = len({r["source_id"] for r in wide_rows})
    print(f"[nnunet_native_summary] {NNUNET_METHOD_LABEL} (experiment="
          f"{experiment}): {len(wide_rows)} (source,step) row(s) across "
          f"{n_sources} source(s), {len(step_rows)} step(s).")
    print(f"  sources Diced={stats['sources']} skipped_gt={stats['skipped_gt']} "
          f"skipped_pred={stats['skipped_pred']} "
          f"atlas_grid_mismatch={stats['skipped_atlas_mismatch']} "
          f"chk_resampled={stats['resampled_chk']}")
    for p in (per_source_csv, by_step_csv, png):
        print(f"  {p}")
    return [per_source_csv, by_step_csv, png]


def main() -> int:
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
    ap.add_argument("--deployment-chk-gt", action="store_true",
                    help="Dice chk_* sources against Dataset835's dense pred "
                         "(deployment GT) instead of the legacy chk_pseudo GT. "
                         "No effect when chk_* are excluded (the default).")
    ap.add_argument("--include-source-prefixes", default=None,
                    help="Comma-separated source_id prefixes to keep. Default: "
                         "'viz_include_source_prefixes' from --config.")
    ap.add_argument("--exclude-source-prefixes", default=None,
                    help="Comma-separated source_id prefixes to drop. Default: "
                         "'viz_exclude_source_prefixes' from --config "
                         "(usually 'chk_'). Pass '' to keep ALL sources.")
    args = ap.parse_args()

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
        deployment_chk_gt=args.deployment_chk_gt,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
