#!/usr/bin/env python3
"""Per-source paired Dice: nnUNet vs CNISP, native head space.

Inputs
------
* ``{work_dir}/nnunet_input/{source_id}_0000.nii.gz``     - staged input CT
* ``{work_dir}/nnunet_pred_native/{source_id}.nii.gz``    - nnUNet pred
* ``output_basedir/{model}/native_space_step_{XX}/...``   - CNISP per-step
  predictions (produced by ``build_cnisp_native_sweep.py``)
* ``output_basedir/{model}/sweep_results.pkl``            - eff_res lookup
* Native-head GT (per ``resolve_gt.SourceInfo``)

Comparison
----------
* For each of the 31 sources, OD and OS are already merged on the CNISP
  side; nnUNet predicts the full head directly. Both live on the
  ORIGINAL CT's voxel grid -- no resampling.
* Dice computed per structure (ON, Globe, Fat, Recti) plus the
  unweighted mean across the four foreground structures.

Outputs (under ``{work_dir}``)
------------------------------
* ``paired_per_source.csv`` -- long: one row per (source, method,
  step_size, structure, dice).
* ``paired_summary.csv``    -- aggregated by (method, eff_res_bucket,
  structure): mean +/- std Dice and n_sources.
* ``paired_summary.txt``    -- human-readable header (asymmetry caveat
  + chk_* pseudo-GT note) and a wide table per structure.

Usage
-----
    python nnunet/compare_native.py --config nnunet/configs.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import yaml

# Ensure ``nnunet`` is importable when this file is run as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nnunet.resolve_gt import NNUNET_LABELS, resolve_sources  # noqa: E402


STRUCT_ORDER = ["ON", "Globe", "Fat", "Recti"]


# ── Generic helpers ──────────────────────────────────────────────


def _load_yaml(path: Path) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_label_volume(p: Path) -> np.ndarray:
    img = nib.load(str(p))
    arr = np.asarray(img.dataobj)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.rint(arr)
    return arr.astype(np.int32, copy=False)


def _binary_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)
    inter = int(np.logical_and(pred_bool, gt_bool).sum())
    denom = int(pred_bool.sum()) + int(gt_bool.sum())
    if denom == 0:
        # both empty -> perfect by convention
        return 1.0 if (not pred_bool.any() and not gt_bool.any()) else 0.0
    return 2.0 * inter / denom


def _assign_bucket(eff_res: Optional[float],
                   edges: List[float]) -> Tuple[int, str]:
    """Return (idx, label) for the given eff_res. None -> (-1, 'unknown')."""
    if eff_res is None or (isinstance(eff_res, float) and math.isnan(eff_res)):
        return -1, "unknown"
    for i, ub in enumerate(edges):
        if eff_res <= ub + 1e-6:
            lower = 0.0 if i == 0 else edges[i - 1]
            return i, f"({lower:.1f}, {ub:.1f}]"
    return len(edges), f"({edges[-1]:.1f}, inf]"


# ── Per-source Dice computation ──────────────────────────────────


def _dice_for_source(
    pred: np.ndarray,
    gt: np.ndarray,
    pred_scheme_map: Dict[str, int],
    gt_scheme_map: Dict[str, int],
) -> Dict[str, float]:
    """Compute per-structure Dice, with each side using its own label map."""
    out: Dict[str, float] = {}
    foreground = []
    for name in STRUCT_ORDER:
        pred_mask = pred == pred_scheme_map[name]
        gt_mask = gt == gt_scheme_map[name]
        d = _binary_dice(pred_mask, gt_mask)
        out[name] = d
        foreground.append(d)
    out["mean"] = float(np.mean(foreground)) if foreground else float("nan")
    return out


# ── Sweep -> (source, step) eff_res lookup ───────────────────────


def _build_eff_res_index(
    sweep_pkl: Path,
    meta_dir: Path,
) -> Dict[Tuple[str, int], float]:
    """source_id, step_size -> effective_resolution_mm (averaged over eyes)."""
    if not sweep_pkl.exists():
        print(f"  [warn] sweep_results.pkl not found: {sweep_pkl}; "
              f"eff_res will be inferred from CNISP step manifests only",
              file=sys.stderr)
        return {}
    with open(sweep_pkl, "rb") as f:
        sweep: List[dict] = pickle.load(f)

    accum: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    for r in sweep:
        cn = r.get("casename")
        if cn is None:
            continue
        if not (cn.endswith("_OD") or cn.endswith("_OS")):
            continue
        source_id = cn[:-3]
        accum[(source_id, int(r["step_size"]))].append(
            float(r["effective_resolution_mm"])
        )
    return {k: float(np.mean(v)) for k, v in accum.items()}


# ── Main ─────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="nnunet/configs.yaml")
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--strict-shape", action="store_true",
                    help="Fail if a prediction's shape differs from GT "
                         "(default: skip the source with a warning).")
    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config))
    cnisp_paths = _load_yaml(Path(cfg["cnisp_paths_yaml"]))

    model_name = args.model_name or cfg["cnisp_model_name"]
    work_dir = Path(args.work_dir or cfg["work_dir"])
    output_base = Path(cnisp_paths["output_basedir"]) / model_name
    meta_dir = Path(cnisp_paths["aligned_dir"]) / "metadata"
    casefiles_dir = Path(cnisp_paths["casefiles_dir"])
    test_cases = casefiles_dir / "test_cases.txt"

    nnunet_pred_dir = work_dir / "nnunet_pred_native"
    if not nnunet_pred_dir.exists():
        print(f"[compare_native] nnUNet pred dir not found: {nnunet_pred_dir}",
              file=sys.stderr)
        return 2

    bucket_edges = list(cfg.get("summary_bucket_edges_mm",
                                [1.0, 2.0, 3.0, 4.0, 5.0, 6.5, 8.5, 11.0, 13.0]))

    # ── Resolve the 31 sources ────────────────────────────────────
    sources, missing = resolve_sources(
        test_cases_path=test_cases,
        meta_dir=meta_dir,
        atlas_image_dir=Path(cfg["atlas_image_dir"]),
        pivot_csv=Path(cfg["pivot_csv"]),
        pivot_subject_column=cfg.get("pivot_subject_column", "subject"),
        pivot_image_path_columns=cfg.get("pivot_image_path_columns"),
        detect_atlas_offset=True,
        require_ct=False,            # comparison doesn't strictly need CT
    )
    if missing:
        # Non-fatal here; just print.
        print(f"[compare_native] note: {len(missing)} source(s) had "
              f"CT-resolution problems; comparison itself uses GT only.",
              file=sys.stderr)
        for m in missing[:5]:
            print(f"  - {m}", file=sys.stderr)
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more", file=sys.stderr)

    # ── nnUNet label scheme reminder ──────────────────────────────
    # We trust NNUNET_LABELS from resolve_gt (matches NNUNET_MAP_CT in
    # canonical_align.py). expected_nnunet_labels is surfaced in configs.yaml
    # just for documentation / future runtime checks.
    expected_labels = cfg.get("expected_nnunet_labels", {})
    if expected_labels:
        print(f"[compare_native] documented nnUNet labels (sanity): "
              f"{expected_labels}")

    # ── CNISP per-step manifest loader ────────────────────────────
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
            sid: Path(p) for sid, p in m.get("by_source_id", {}).items()
        }
    if not cnisp_step_paths:
        print(f"[compare_native] no CNISP step manifests under {output_base}.",
              file=sys.stderr)
        print(f"  Did you run build_cnisp_native_sweep.py?", file=sys.stderr)
        return 2
    print(f"[compare_native] CNISP steps available: {sorted(cnisp_step_paths)}")

    # ── eff_res lookup ────────────────────────────────────────────
    eff_res_idx = _build_eff_res_index(
        output_base / "sweep_results.pkl", meta_dir
    )

    # ── Iterate sources & emit per-source rows ────────────────────
    per_source_rows: List[Dict[str, str]] = []
    n_done = 0
    n_skipped_gt = 0
    n_skipped_nnunet = 0
    nnunet_struct_map = {n: NNUNET_LABELS[n] for n in STRUCT_ORDER}

    for src in sources:
        sid = src.source_id
        gt_path = src.gt_label_path
        if not gt_path.exists():
            n_skipped_gt += 1
            print(f"  [skip] {sid}: GT not found at {gt_path}", file=sys.stderr)
            continue
        try:
            gt = _load_label_volume(gt_path)
        except Exception as e:  # noqa: BLE001
            n_skipped_gt += 1
            print(f"  [skip] {sid}: failed to read GT ({e})", file=sys.stderr)
            continue

        # ── nnUNet ────────────────────────────────────────────────
        nnunet_pred_path = nnunet_pred_dir / f"{sid}.nii.gz"
        if not nnunet_pred_path.exists():
            n_skipped_nnunet += 1
            print(f"  [skip nnUNet] {sid}: no pred at {nnunet_pred_path}",
                  file=sys.stderr)
        else:
            try:
                nn_pred = _load_label_volume(nnunet_pred_path)
            except Exception as e:  # noqa: BLE001
                n_skipped_nnunet += 1
                print(f"  [skip nnUNet] {sid}: load failed ({e})",
                      file=sys.stderr)
                nn_pred = None

            if nn_pred is not None:
                if nn_pred.shape != gt.shape:
                    msg = (f"{sid}: nnUNet pred shape {nn_pred.shape} "
                           f"!= GT shape {gt.shape}")
                    if args.strict_shape:
                        print(f"  [error] {msg}", file=sys.stderr)
                        return 3
                    print(f"  [skip nnUNet] {msg}", file=sys.stderr)
                else:
                    dices = _dice_for_source(
                        nn_pred, gt,
                        pred_scheme_map=nnunet_struct_map,
                        gt_scheme_map=src.gt_struct_to_value,
                    )
                    for name in STRUCT_ORDER + ["mean"]:
                        per_source_rows.append({
                            "source_id": sid,
                            "gt_source": src.gt_source,
                            "method": "nnUNet",
                            "step_size": "",
                            "eff_res_mm": "",
                            "structure": name,
                            "dice": f"{dices[name]:.6f}",
                        })

        # ── CNISP per step ────────────────────────────────────────
        # CNISP's native_mapping.remap_canonical_to_original emits labels
        # in the SAME scheme as the source GT, so the same struct->value
        # map works for both.
        cnisp_pred_struct_map = src.gt_struct_to_value
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
                cn_pred = _load_label_volume(cnisp_path)
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

            dices = _dice_for_source(
                cn_pred, gt,
                pred_scheme_map=cnisp_pred_struct_map,
                gt_scheme_map=src.gt_struct_to_value,
            )
            eff_res = eff_res_idx.get((sid, step))
            for name in STRUCT_ORDER + ["mean"]:
                per_source_rows.append({
                    "source_id": sid,
                    "gt_source": src.gt_source,
                    "method": "CNISP",
                    "step_size": str(step),
                    "eff_res_mm": (f"{eff_res:.4f}" if eff_res is not None else ""),
                    "structure": name,
                    "dice": f"{dices[name]:.6f}",
                })

        n_done += 1

    print(f"\n[compare_native] processed {n_done} source(s); "
          f"skipped: gt={n_skipped_gt} nnUNet={n_skipped_nnunet}")

    # ── Write per-source CSV ──────────────────────────────────────
    work_dir.mkdir(parents=True, exist_ok=True)
    per_source_csv = work_dir / "paired_per_source.csv"
    with open(per_source_csv, "w", newline="") as f:
        fieldnames = ["source_id", "gt_source", "method", "step_size",
                      "eff_res_mm", "structure", "dice"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in per_source_rows:
            w.writerow(r)
    print(f"[compare_native] wrote {per_source_csv}")

    # ── Aggregate into summary CSV + TXT ──────────────────────────
    table_by_struct: Dict[str, Dict[str, Tuple[float, float, int]]] = {
        s: {} for s in STRUCT_ORDER + ["mean"]
    }

    # Group: (method, bucket_label) -> {structure: [dice]}
    grouped: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in per_source_rows:
        method = r["method"]
        if method == "nnUNet":
            col = "nnUNet"
        else:
            eff = float(r["eff_res_mm"]) if r["eff_res_mm"] else None
            _, label = _assign_bucket(eff, bucket_edges)
            col = f"CNISP {label}"
        grouped[(method, col)][r["structure"]].append(float(r["dice"]))

    # Stable column ordering: nnUNet first, then CNISP buckets in order.
    cnisp_col_order: List[str] = []
    seen = set()
    for r in per_source_rows:
        if r["method"] != "CNISP":
            continue
        if r["eff_res_mm"]:
            eff = float(r["eff_res_mm"])
            _, label = _assign_bucket(eff, bucket_edges)
        else:
            label = "unknown"
        col = f"CNISP {label}"
        if col not in seen:
            seen.add(col)
            cnisp_col_order.append(col)

    def _bucket_sort_key(col: str) -> float:
        # sort CNISP columns by lower bound (or 0 for the first);
        # unknown-bucket rows sink to the bottom.
        label = col[len("CNISP "):]
        if label == "unknown":
            return 1e9
        try:
            lo = label.split(",")[0].lstrip("(")
            return float(lo)
        except Exception:  # noqa: BLE001
            return 1e9

    cnisp_col_order.sort(key=_bucket_sort_key)
    all_cols = ["nnUNet"] + cnisp_col_order

    for col in all_cols:
        method = "nnUNet" if col == "nnUNet" else "CNISP"
        for struct in STRUCT_ORDER + ["mean"]:
            vals = grouped.get((method, col), {}).get(struct, [])
            if not vals:
                table_by_struct[struct][col] = (float("nan"), float("nan"), 0)
                continue
            arr = np.asarray(vals, dtype=np.float64)
            table_by_struct[struct][col] = (float(arr.mean()),
                                            float(arr.std()),
                                            int(len(arr)))

    summary_csv = work_dir / "paired_summary.csv"
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
    txt_path = work_dir / "paired_summary.txt"
    with open(txt_path, "w") as f:
        f.write("=" * 78 + "\n")
        f.write("nnUNet vs CNISP -- per-source full-head Dice (native space)\n")
        f.write("=" * 78 + "\n\n")
        f.write("Caveats\n")
        f.write("  - CNISP is GT-conditioned (sparse-slice latent optimization)\n")
        f.write("  - nnUNet is image-conditioned (CT only).\n")
        f.write("  - 6 chk_* sources use pseudo-GT (previous nnUNet's QA-kept\n")
        f.write("    predictions). Filter on gt_source=='atlas' in paired_per_source.csv\n")
        f.write("    for the manual-GT-only view.\n\n")

        f.write(f"Sources processed: {n_done}  "
                f"(skipped GT={n_skipped_gt}, skipped nnUNet={n_skipped_nnunet})\n\n")

        f.write("Mean Dice by eff_res bucket (n_sources in parentheses)\n")
        f.write("-" * 78 + "\n")
        col_w = 18
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

        # CNISP - nnUNet delta on mean row, when both available
        nn_cell = table_by_struct["mean"].get("nnUNet")
        if nn_cell is not None:
            nn_mean = nn_cell[0]
            if not math.isnan(nn_mean):
                f.write("CNISP mean Dice minus nnUNet mean (positive => CNISP wins)\n")
                f.write("-" * 78 + "\n")
                for col in cnisp_col_order:
                    cn_mean, _, _ = table_by_struct["mean"][col]
                    if math.isnan(cn_mean):
                        continue
                    f.write(f"  {col:<25} {cn_mean - nn_mean:+.3f}\n")
                f.write("\n")
    print(f"[compare_native] wrote {txt_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
