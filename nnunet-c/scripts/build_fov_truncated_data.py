#!/usr/bin/env python3
"""Build the FOV-truncation experiment's data/ tree (Part 2, "isolate FOV").

Produces truncation-only degraded CTs (NO slice thickening -- the only degradation
is the missing field of view, so the corrector must defer to the completed CNISP
prior exactly where ch0 has no evidence). Two truncation geometries:

  --mode slab  (type-1, default): blank a contiguous fraction of slices along the
      through-plane axis (superior/inferior/centred via --side). The classic
      "top/bottom of the FOV cut off" case.
  --mode box   (type-2): keep a single axis-aligned FOV BOX. The globe/anterior
      axis is NEVER cut (globe = "front"); the two orthogonal axes (up/down +
      left/right) are corner-clipped from one corner (--corner, one of SL/SR/IL/IR
      = superior|inferior x left|right, or random). Models a mis-centred
      acquisition where the patient is offset from the scan centre and part of the
      orbit pokes out of the scanner box on two faces. keep_fraction is the TOTAL
      retained orbit-volume fraction (split as sqrt across the two cut axes); the
      cut is anchored on the orbit bbox (from gt_candidate_pred) so it reliably
      bites into the eye, and each (case,step) records the per-structure retained
      fraction as a QC of the ">= half of every structure visible" property.

To reuse the ENTIRE existing corrector pipeline unchanged (`build_corrector_dataset
--layout cascade`, the stratified loader, by-step eval), each truncation level is
encoded as a **pseudo-step** `PP = round(keep_fraction*100)` (e.g. keep 0.5 ->
`_step50`). Downstream then stratifies by FOV severity instead of by thickness --
no plumbing changes; set the trainer's strata with `CORRECTOR_STRATA="50,65,80"`.

Input: an existing `corrector_data_manifest.json` (from build_corrector_data.py),
for each case's `source_image` + `gt_candidate_pred` + `step_axis`. Output (a
SEPARATE data root, so the thickness experiment is untouched):
    <out>/images/{case}_step{PP}_0000.nii.gz   (truncated CT, ch0)
    <out>/corrector_data_manifest.json          (same schema -> build_corrector_dataset)
    <out>/fov_truncation_manifest.json          (per (case,PP): source_shape + the
                                                 visible window -- slab: {trunc_axis,
                                                 visible_range}; box: {visible_box,
                                                 corner, cut_axes, retained_*} -- for
                                                 the region-restricted eval)
    <out>/{nnunet_pred,cnisp_pred}/             (empty; box runs fill them)

The truncation reuses `nnunet.sparsify_inputs._truncate_one_ct` (slab) /
`_truncate_one_ct_box` (box), next to `_sparsify_one_ct`; no duplicate code.

BOX FOLLOW-UP (see RUNBOOK_FOV.md): run the 835 stage-1 model on each truncated CT
to get a coarse seg, run CNISP 032 on that (`--steps 50,65,80`) to emit the
COMPLETED iso prior, then `build_corrector_dataset.py --layout cascade` on a FOV
config (data_root = this <out>, steps = the pseudo-steps, its own control-C id).

Usage:
    python nnunet-c/scripts/build_fov_truncated_data.py \
        --keep-fractions 0.5,0.65,0.8 --side end
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.config import add_repo_to_syspath, load_corrector_config  # noqa: E402

add_repo_to_syspath(__file__)

import numpy as np  # noqa: E402
import nibabel as nib  # noqa: E402

_DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "corrector.yaml"

# ── Box-mode (type-2) geometry ────────────────────────────────────────────────
# Type-2 truncation keeps a single axis-aligned BOX: the globe/anterior axis is
# never cut (globe = "front"), and the two orthogonal axes (up/down + left/right)
# are each clipped from one side, so the missing FOV is a CORNER of the volume --
# a mis-centred acquisition where the eye pokes out of the scanner box on two
# faces. --corner names the two faces the head EXITS (blanked), one letter from
# {S,I} (superior/inferior) and one from {L,R} (left/right).
_OPP = {"R": "L", "L": "R", "A": "P", "P": "A", "S": "I", "I": "S"}
_CORNERS = ["SL", "SR", "IL", "IR"]


def _anterior_axis(axcodes) -> int:
    """Voxel axis carrying the anterior/posterior (globe) direction -- kept intact."""
    for ax, c in enumerate(axcodes):
        if c in ("A", "P"):
            return ax
    raise ValueError(f"no A/P (anterior) axis in axcodes {axcodes}; can't place box")


def _axis_for_dir(axcodes, d: str) -> Tuple[int, bool]:
    """(axis, at_high) for physical direction ``d``: at_high=True when ``d`` sits at
    the HIGH voxel index of its axis (axcode==d), False when at the low index."""
    for ax, c in enumerate(axcodes):
        if c == d:
            return ax, True
        if c == _OPP[d]:
            return ax, False
    raise ValueError(f"direction {d!r} not found in axcodes {axcodes}")


def _bbox_corners(vlo: np.ndarray, vhi: np.ndarray) -> np.ndarray:
    """8 corners of the inclusive voxel bbox [vlo, vhi] as an (8,3) array."""
    import itertools
    return np.array([[vlo[a] if bit else vhi[a] for a, bit in enumerate(bits)]
                     for bits in itertools.product((0, 1), repeat=3)], dtype=float)


def _orbit_bbox_in_ct(gt_arr: np.ndarray, gt_affine: np.ndarray,
                      ct_affine: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Foreground (orbit) bbox of ``gt_arr`` mapped into the CT voxel grid.

    Robust to gt/CT living on different grids (orthogonal affines): map the GT
    foreground bbox's 8 corners through world coords into CT voxels, take the
    per-axis min/max. Returns ``(lo, hi)`` inclusive CT-voxel indices, or None."""
    fg = np.argwhere(gt_arr > 0)
    if fg.size == 0:
        return None
    corners = _bbox_corners(fg.min(0), fg.max(0))
    world = (gt_affine @ np.c_[corners, np.ones(len(corners))].T).T[:, :3]
    ct_vox = (np.linalg.inv(ct_affine) @ np.c_[world, np.ones(len(world))].T).T[:, :3]
    return np.floor(ct_vox.min(0)).astype(int), np.ceil(ct_vox.max(0)).astype(int)


def _box_keep_windows(ct_shape, ct_axcodes, orbit_lo, orbit_hi,
                      corner: str, keep_fraction: float):
    """Keep-window per cut axis so a fraction ``keep_fraction`` of the orbit extent
    survives on that axis, clipping inward from each ``corner`` face. Returns
    ``({axis: (lo, hi)}, cut_info)``."""
    windows: Dict[int, Tuple[int, int]] = {}
    cut_info: Dict[int, dict] = {}
    # keep_fraction is the TOTAL retained orbit-volume fraction of the corner box;
    # split evenly across the cut axes so the product matches (2 axes -> sqrt).
    per_frac = float(keep_fraction) ** (1.0 / max(1, len(corner)))
    for d in corner:                                   # e.g. "S" then "L"
        ax, at_high = _axis_for_dir(ct_axcodes, d)
        g_lo = max(0, int(orbit_lo[ax]))
        g_hi = min(int(ct_shape[ax]) - 1, int(orbit_hi[ax]))
        ext = max(1, g_hi - g_lo)
        keep_len = int(round(per_frac * ext))
        if at_high:                                    # head exits high side -> blank high
            cut = g_lo + keep_len
            windows[ax] = (0, min(int(ct_shape[ax]), cut))
        else:                                          # exits low side -> blank low
            cut = g_hi - keep_len
            windows[ax] = (max(0, cut), int(ct_shape[ax]))
        cut_info[ax] = {"dir": d, "at_high": bool(at_high),
                        "orbit": [g_lo, g_hi], "cut": int(cut)}
    return windows, cut_info


def _retained_fraction(gt_arr: np.ndarray, gt_axcodes, corner: str,
                       keep_fraction: float) -> Tuple[float, Dict[int, float]]:
    """QC: fraction of orbit foreground kept by the same corner box on the GT grid,
    overall and per label -- the ">= half of every structure visible" check."""
    fg = gt_arr > 0
    idx = np.argwhere(fg)
    if idx.size == 0:
        return 0.0, {}
    vlo, vhi = idx.min(0), idx.max(0)
    keep = np.ones(gt_arr.shape, dtype=bool)
    per_frac = float(keep_fraction) ** (1.0 / max(1, len(corner)))
    for d in corner:
        ax, at_high = _axis_for_dir(gt_axcodes, d)
        g_lo, g_hi = int(vlo[ax]), int(vhi[ax])
        keep_len = int(round(per_frac * max(1, g_hi - g_lo)))
        sl = [slice(None)] * gt_arr.ndim
        sl[ax] = (slice(0, g_lo + keep_len) if at_high
                  else slice(g_hi - keep_len, gt_arr.shape[ax]))
        m = np.zeros(gt_arr.shape, dtype=bool)
        m[tuple(sl)] = True
        keep &= m
    total = int(fg.sum())
    overall = round(int((fg & keep).sum()) / max(1, total), 3)
    per = {int(lab): round(int(((gt_arr == lab) & keep).sum())
                           / max(1, int((gt_arr == lab).sum())), 3)
           for lab in np.unique(gt_arr[fg])}
    return overall, per


def _pseudo_step(keep_fraction: float) -> int:
    pp = int(round(float(keep_fraction) * 100))
    if not (1 <= pp <= 99):
        raise ValueError(
            f"keep_fraction {keep_fraction} -> pseudo-step {pp}; must map into "
            f"1..99 (keep_fraction in (0.01, 0.99]); 1.0 = no truncation, skip it.")
    return pp


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(_DEFAULT_CONFIG))
    ap.add_argument("--src-manifest", default=None,
                    help="existing corrector_data_manifest.json (default: "
                         "<data_root>/corrector_data_manifest.json).")
    ap.add_argument("--out-data-root", default=None,
                    help="FOV data root to write (default: <data_root>_fov).")
    ap.add_argument("--keep-fractions", default="0.5,0.65,0.8",
                    help="comma list of RETAINED z-fractions, each -> pseudo-step "
                         "round(f*100). Default %(default)s -> steps 50,65,80.")
    ap.add_argument("--mode", choices=["slab", "box"], default="slab",
                    help="slab (type-1): blank a contiguous slab on the through-plane "
                         "axis. box (type-2): keep one axis-aligned FOV box -- the "
                         "globe/anterior axis is never cut, the two orthogonal axes "
                         "(up/down + left/right) are corner-clipped (mis-centred FOV).")
    ap.add_argument("--side", choices=["end", "start", "both", "random"],
                    default="end",
                    help="[slab mode] which end(s) to vis: end (superior cut-off), start "
                         "(inferior), both (centred limited FOV), or random per case.")
    ap.add_argument("--corner", choices=_CORNERS + ["random"], default="SL",
                    help="[box mode] the two faces the head EXITS (blanked), one of "
                         f"{_CORNERS} (S/I superior/inferior x L/R left/right) or random.")
    ap.add_argument("--pad-value", type=float, default=None,
                    help="fill HU for vised slices (default: each CT's own min = air).")
    ap.add_argument("--max-cases", type=int, default=0, help="cap number of cases (0=all).")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for --side/--corner random.")
    ap.add_argument("--force", action="store_true", help="re-truncate even if the image exists.")
    args = ap.parse_args()

    # Truncation helpers live next to _sparsify_one_ct (reused; imported lazily so
    # --help doesn't pull torch/simulation in).
    from nnunet.sparsify_inputs import _truncate_one_ct, _truncate_one_ct_box  # noqa: E402

    cfg = load_corrector_config(args.config, caller_file=__file__)
    cd = cfg["corrector_data"]
    res = cfg["_resolved"]
    data_root = Path(cd["data_root"])
    data_root = data_root if data_root.is_absolute() else (res["repo_root"] / data_root)

    src_manifest = (Path(args.src_manifest) if args.src_manifest
                    else data_root / "corrector_data_manifest.json")
    if not src_manifest.is_file():
        print(f"[fov] source manifest not found: {src_manifest}\n"
              f"      run build_corrector_data.py first.", file=sys.stderr)
        return 2
    src = json.load(open(src_manifest))

    out_root = (Path(args.out_data_root) if args.out_data_root
                else data_root.parent / f"{data_root.name}_fov")
    images_dir = out_root / cd.get("images_dirname", "images")
    images_dir.mkdir(parents=True, exist_ok=True)
    (out_root / cd.get("nnunet_pred_dirname", "nnunet_pred")).mkdir(parents=True, exist_ok=True)
    (out_root / cd.get("cnisp_pred_dirname", "cnisp_pred")).mkdir(parents=True, exist_ok=True)

    fracs = [float(x) for x in args.keep_fractions.split(",") if x.strip()]
    steps = [_pseudo_step(f) for f in fracs]
    rng = np.random.RandomState(int(args.seed))
    geom = f"corner={args.corner}" if args.mode == "box" else f"side={args.side}"
    print(f"[fov] src={src_manifest}")
    print(f"[fov] out={out_root}  keep_fractions={fracs} -> pseudo-steps {steps}  "
          f"mode={args.mode}  {geom}")

    manifest = {"experiment": "fov_truncation", "source_manifest": str(src_manifest),
                "keep_fractions": fracs, "pseudo_steps": steps, "mode": args.mode,
                "side": args.side, "corner": args.corner,
                "steps": steps, "cases": {}}
    sidecar: dict = {}
    n_cases = n_written = n_skipped = 0

    for case_id, sentry in sorted(src.get("cases", {}).items()):
        if args.max_cases and n_cases >= args.max_cases:
            break
        source_image = sentry.get("source_image", "")
        gt = sentry.get("gt_candidate_pred", "")
        axis = sentry.get("step_axis")
        if not source_image or not Path(source_image).exists() or axis is None:
            n_skipped += 1
            continue
        if not gt or not Path(gt).exists():
            # gt_candidate_pred is the corrector's label target; skip cases without it
            n_skipped += 1
            continue
        n_cases += 1

        entry = {"source_image": source_image, "gt_candidate_pred": gt,
                 "csv_z_spacing": sentry.get("csv_z_spacing", ""),
                 "through_plane_spacing": sentry.get("through_plane_spacing"),
                 "step_axis": int(axis), "steps": {}}
        sidecar[case_id] = {}

        # Box-mode (type-2) geometry is per-case, computed once: the anterior
        # (globe) axis to KEEP, and the orbit bbox that anchors the corner clip.
        box_ctx = None
        if args.mode == "box":
            ct_img = nib.load(str(source_image))                  # lazy header
            gt_img = nib.load(str(gt))
            gt_arr = np.asarray(gt_img.dataobj)
            bbox = _orbit_bbox_in_ct(gt_arr, gt_img.affine, ct_img.affine)
            if bbox is None:
                n_cases -= 1
                n_skipped += 1
                print(f"  {case_id}: empty gt_candidate_pred foreground; skip (box)")
                continue
            box_ctx = {"ct_shape": tuple(int(s) for s in ct_img.shape),
                       "ct_ax": nib.aff2axcodes(ct_img.affine),
                       "ant": _anterior_axis(nib.aff2axcodes(ct_img.affine)),
                       "orbit_lo": bbox[0], "orbit_hi": bbox[1],
                       "gt_arr": gt_arr, "gt_ax": nib.aff2axcodes(gt_img.affine)}

        summ = []
        for f, pp in zip(fracs, steps):
            out = images_dir / f"{case_id}_step{pp:02d}_0000.nii.gz"
            if args.mode == "box":
                corner = (str(rng.choice(_CORNERS)) if args.corner == "random"
                          else args.corner)
                windows, _cut = _box_keep_windows(
                    box_ctx["ct_shape"], box_ctx["ct_ax"],
                    box_ctx["orbit_lo"], box_ctx["orbit_hi"], corner, f)
                visible_box = [list(windows[ax]) if ax in windows
                               else [0, int(box_ctx["ct_shape"][ax])] for ax in range(3)]
                if not (out.exists() and not args.force):
                    arr, affine, _vb = _truncate_one_ct_box(
                        Path(source_image), windows, pad_value=args.pad_value)
                    nib.save(nib.Nifti1Image(arr.astype(np.float32), affine), str(out))
                    n_written += 1
                src_shape = list(box_ctx["ct_shape"])
                ret_all, ret_per = _retained_fraction(
                    box_ctx["gt_arr"], box_ctx["gt_ax"], corner, f)
                geom = {"mode": "box", "corner": corner,
                        "cut_axes": sorted(int(a) for a in windows),
                        "anterior_axis": int(box_ctx["ant"]),
                        "visible_box": visible_box,
                        "retained_fraction": ret_all,
                        "retained_per_structure": ret_per}
                entry["steps"][str(pp)] = {"kept": True, "keep_fraction": f,
                                           "image": str(out), **geom}
                sidecar[case_id][str(pp)] = {"source_shape": src_shape,
                                             "keep_fraction": f, **geom}
                summ.append(f"step{pp:02d}(keep{f:.2f},{corner},ret{ret_all:.2f})")
            else:
                side = (["end", "start", "both"][int(rng.randint(3))]
                        if args.side == "random" else args.side)
                arr, affine, vis = _truncate_one_ct(
                    Path(source_image), z_axis=int(axis), keep_fraction=f,
                    pad_value=args.pad_value, side=side)
                if not (out.exists() and not args.force):
                    nib.save(nib.Nifti1Image(arr.astype(np.float32), affine), str(out))
                    n_written += 1
                src_shape = [int(s) for s in arr.shape]
                entry["steps"][str(pp)] = {
                    "kept": True, "keep_fraction": f, "side": side,
                    "trunc_axis": int(axis), "visible_range": [int(vis[0]), int(vis[1])],
                    "image": str(out)}
                sidecar[case_id][str(pp)] = {
                    "trunc_axis": int(axis), "visible_range": [int(vis[0]), int(vis[1])],
                    "source_shape": src_shape, "keep_fraction": f, "side": side}
                summ.append(f"step{pp:02d}(keep{f:.2f},{side})")
        manifest["cases"][case_id] = entry
        tag = (f"box ant={box_ctx['ant']}" if args.mode == "box" else f"axis={axis}")
        print(f"  {case_id}: {tag} -> " + " ".join(summ))

    with open(out_root / "corrector_data_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    with open(out_root / "fov_truncation_manifest.json", "w") as fh:
        json.dump(sidecar, fh, indent=2)
    print(f"[fov] cases={n_cases} images_written={n_written} skipped={n_skipped}")
    print(f"[fov] manifest -> {out_root / 'corrector_data_manifest.json'}")
    print(f"[fov] sidecar  -> {out_root / 'fov_truncation_manifest.json'}")
    print("[fov] NEXT (box): 835 stage-1 predict on each truncated CT -> CNISP 032 "
          f"(--steps {','.join(str(s) for s in steps)}) -> build_corrector_dataset "
          "--layout cascade on a FOV config. See RUNBOOK_FOV.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
