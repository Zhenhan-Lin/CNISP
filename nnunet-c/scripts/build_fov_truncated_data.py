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
      NOTE: box uses a single GLOBAL corner box over BOTH orbits, so the eye nearer
      the corner loses more (combined ret != per-eye ret) -- use --min-retains for a
      guaranteed per-eye floor.
  --min-retains T1,T2,...  (Option 2, per-eye): the RECOMMENDED "both eyes stay
      evaluable" mode. Each orbit is split (OD/OS) and clipped INDEPENDENTLY to the
      DEEPEST corner cut that still holds ret_total>=T AND ret_ON>=T_on for THAT eye
      (binary-searched on real foreground), so BOTH eyes keep >= T of their
      foreground and >= T_on of their ON. T is a hard floor (not a geometric knob);
      pseudo-step = round(T*100). Writes a truncated-GT volume per case (gt_trunc/)
      and a per_eye sidecar (eye_bbox/kept_box + ret_total/ret_ON/per-structure +
      binding_constraint). Replaces --keep-fractions/--mode when given.

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


# ── Per-eye (Option 2) min-retain geometry ────────────────────────────────────
# Each orbit is clipped INDEPENDENTLY to a controlled retention floor so BOTH eyes
# stay evaluable: --min-retain T is a hard floor on per-eye foreground retention
# (union of ON+Globe+Recti+Fat) AND on ON retention, calibrated by binary search on
# the retained fraction k of each cut axis's extent. The globe/anterior axis is not cut.
_NN = {"ON": 1, "Recti": 2, "Globe": 3, "Fat": 4}   # fixed Dataset835 / nnUNet scheme


def _split_eyes(gt_arr, gt_affine):
    """Split ALL foreground into ``{"OD": mask, "OS": mask}`` by the L-R midline between
    the two globe-CC centroids (reuses ``canonical_align.separate_eyes``). One globe -> a
    single eye; empty -> {}."""
    from data_prep.canonical_align import separate_eyes    # scipy; lazy import
    eyes = separate_eyes(gt_arr, gt_affine, _NN["Globe"])
    if not eyes:
        return {}
    fg = gt_arr > 0
    axcodes = nib.aff2axcodes(gt_affine)
    lr_axis = next((i for i, c in enumerate(axcodes) if c in ("R", "L")), 0)
    if len(eyes) == 1:
        return {eyes[0]["eye"]: fg}
    mid = (float(eyes[0]["centroid_voxel"][lr_axis])
           + float(eyes[1]["centroid_voxel"][lr_axis])) / 2.0
    shp = [1] * gt_arr.ndim
    shp[lr_axis] = gt_arr.shape[lr_axis]
    low = np.arange(gt_arr.shape[lr_axis]).reshape(shp) < mid
    out = {}
    for e in eyes[:2]:
        out[e["eye"]] = fg & (low if float(e["centroid_voxel"][lr_axis]) < mid else ~low)
    return out


def _eye_bbox_of(mask):
    """Per-axis bbox of a boolean mask as half-open windows ``[(lo, hi)...]``, or None."""
    idx = np.argwhere(mask)
    if idx.size == 0:
        return None
    return [(int(idx[:, a].min()), int(idx[:, a].max()) + 1) for a in range(mask.ndim)]


def _kept_box_from_k(eye_bbox, cut_sides, k):
    """``eye_bbox`` shrunk to fraction ``k`` of its extent on each cut axis, from the
    truncated corner. ``cut_sides``: list of (axis, at_high). Anterior axis (absent from
    cut_sides) stays full. Returns per-axis half-open windows."""
    kb = [tuple(w) for w in eye_bbox]
    for ax, at_high in cut_sides:
        lo, hi = eye_bbox[ax]
        ext = hi - lo
        keep_len = max(1, min(ext, int(round(float(k) * ext))))
        kb[ax] = (lo, lo + keep_len) if at_high else (hi - keep_len, hi)
    return kb


def _ret_in_box(mask, kept_box):
    """Fraction of ``mask`` foreground inside ``kept_box`` (1.0 when mask is empty)."""
    tot = int(mask.sum())
    if tot == 0:
        return 1.0
    sl = tuple(slice(int(lo), int(hi)) for lo, hi in kept_box)
    return int(mask[sl].sum()) / tot


def _calibrate_eye_cut(eye_fg, struct_masks, on_mask, cut_sides, T, T_on, iters=26):
    """Binary-search the DEEPEST cut (smallest ``k``) whose kept_box still holds
    ret_total >= T AND ret_ON >= T_on (ret is monotone in k, so k=1 is always feasible).
    Returns the calibration dict (eye_bbox, kept_box, k, ret_total, ret_ON, per-structure,
    binding_constraint)."""
    eye_bbox = _eye_bbox_of(eye_fg)

    def _ok(k):
        kb = _kept_box_from_k(eye_bbox, cut_sides, k)
        return _ret_in_box(eye_fg, kb) >= T and _ret_in_box(on_mask, kb) >= T_on

    lo, hi = 0.0, 1.0
    if _ok(hi):
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if _ok(mid):
                hi = mid                 # feasible -> cut deeper (smaller k)
            else:
                lo = mid
    kb = _kept_box_from_k(eye_bbox, cut_sides, hi)
    ret_per = {n: round(_ret_in_box(m, kb), 4) for n, m in struct_masks.items()}
    return {"eye_bbox": [list(w) for w in eye_bbox], "kept_box": [list(w) for w in kb],
            "k": round(float(hi), 4),
            "ret_total": round(_ret_in_box(eye_fg, kb), 4),
            "ret_ON": round(_ret_in_box(on_mask, kb), 4),
            "ret_per_structure": ret_per,
            "binding_constraint": (min(ret_per, key=ret_per.get) if ret_per else None)}


def _map_box_gt_to_ct(box, gt_affine, ct_affine):
    """Map a per-axis half-open GT-voxel box to CT-voxel half-open windows (orthogonal
    affines: map the 8 corners through world coords, take per-axis min/max)."""
    lo = np.array([w[0] for w in box], dtype=float)
    hi = np.array([w[1] - 1 for w in box], dtype=float)      # inclusive corner
    corners = _bbox_corners(lo, hi)
    world = (gt_affine @ np.c_[corners, np.ones(len(corners))].T).T[:, :3]
    ctv = (np.linalg.inv(ct_affine) @ np.c_[world, np.ones(len(world))].T).T[:, :3]
    lo_ct = np.floor(ctv.min(0)).astype(int)
    hi_ct = np.ceil(ctv.max(0)).astype(int) + 1              # half-open
    return [(int(lo_ct[a]), int(hi_ct[a])) for a in range(3)]


def _pseudo_step(keep_fraction: float) -> int:
    pp = int(round(float(keep_fraction) * 100))
    if not (1 <= pp <= 99):
        raise ValueError(
            f"keep_fraction {keep_fraction} -> pseudo-step {pp}; must map into "
            f"1..99 (keep_fraction in (0.01, 0.99]); 1.0 = no truncation, skip it.")
    return pp


def _run_per_eye(args, src, out_root, images_dir, cd) -> int:
    """Option 2: per-eye min-retain truncation. Each orbit is split (OD/OS), clipped
    independently to the deepest corner cut still holding ret_total>=T and ret_ON>=T_on,
    and the union of the two corner notches is blanked in the CT (and mirrored in a
    truncated-GT volume for QC/viz). Writes the FOV data tree + per_eye sidecar."""
    from nnunet.sparsify_inputs import _truncate_one_ct_boxes, _blank_notches  # noqa: E402

    Ts = [float(x) for x in args.min_retains.split(",") if x.strip()]
    steps = [_pseudo_step(T) for T in Ts]
    T_on_fixed = float(args.min_retain_on) if args.min_retain_on is not None else None
    gt_dir = out_root / cd.get("gt_trunc_dirname", "gt_trunc")
    gt_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(int(args.seed))
    print(f"[fov] out={out_root}  min_retains={Ts} -> pseudo-steps {steps}  "
          f"mode=box_per_eye  corner={args.corner}  min_on_vox={args.min_on_vox}")

    manifest = {"experiment": "fov_truncation", "mode": "box_per_eye",
                "min_retains": Ts, "pseudo_steps": steps, "steps": steps,
                "corner": args.corner, "min_retain_on": args.min_retain_on, "cases": {}}
    sidecar: dict = {}
    n_cases = n_written = n_skipped = 0

    for case_id, sentry in sorted(src.get("cases", {}).items()):
        if args.max_cases and n_cases >= args.max_cases:
            break
        source_image = sentry.get("source_image", "")
        gt = sentry.get("gt_candidate_pred", "")
        if (not source_image or not Path(source_image).exists()
                or not gt or not Path(gt).exists()):
            n_skipped += 1
            continue
        gt_img = nib.load(str(gt))
        gt_arr = np.asarray(gt_img.dataobj)
        gt_affine = gt_img.affine
        gt_ax = nib.aff2axcodes(gt_affine)
        ct_img = nib.load(str(source_image))
        ct_affine = ct_img.affine
        ct_shape = tuple(int(s) for s in ct_img.shape)

        eyes = _split_eyes(gt_arr, gt_affine)
        if not eyes:
            n_skipped += 1
            print(f"  {case_id}: no globe / empty foreground; skip")
            continue
        # Per-eye masks + ON floor gate (independent of T) -- precompute once.
        eye_data, bad = {}, False
        for name, fg in eyes.items():
            if int(fg.sum()) == 0:
                continue
            on = fg & (gt_arr == _NN["ON"])
            if int(on.sum()) < args.min_on_vox:
                bad = True
                break
            eye_data[name] = {"fg": fg, "on": on,
                              "structs": {s: fg & (gt_arr == lab) for s, lab in _NN.items()}}
        if bad or not eye_data:
            n_skipped += 1
            print(f"  {case_id}: ON < {args.min_on_vox} vox in an eye (atypical); skip")
            continue
        n_cases += 1

        corner = str(rng.choice(_CORNERS)) if args.corner == "random" else args.corner
        cut_sides = [_axis_for_dir(gt_ax, d) for d in corner]
        entry = {"source_image": source_image, "gt_candidate_pred": gt,
                 "step_axis": sentry.get("step_axis"), "mode": "box_per_eye",
                 "corner": corner, "steps": {}}
        sidecar[case_id] = {}
        summ = []
        for T, pp in zip(Ts, steps):
            T_on = T_on_fixed if T_on_fixed is not None else T
            per_eye, regions_ct, regions_gt = {}, [], []
            for name, d in eye_data.items():
                cal = _calibrate_eye_cut(d["fg"], d["structs"], d["on"], cut_sides, T, T_on)
                eb_ct = _map_box_gt_to_ct(cal["eye_bbox"], gt_affine, ct_affine)
                kb_ct = _map_box_gt_to_ct(cal["kept_box"], gt_affine, ct_affine)
                regions_gt.append((cal["eye_bbox"], cal["kept_box"]))
                regions_ct.append((eb_ct, kb_ct))
                per_eye[name] = {"eye_bbox": [list(w) for w in eb_ct],
                                 "kept_box": [list(w) for w in kb_ct], "k": cal["k"],
                                 "ret_total": cal["ret_total"], "ret_ON": cal["ret_ON"],
                                 "ret_per_structure": cal["ret_per_structure"],
                                 "binding_constraint": cal["binding_constraint"]}
            out_ct = images_dir / f"{case_id}_step{pp:02d}_0000.nii.gz"
            if not (out_ct.exists() and not args.force):
                arr_ct, aff_ct = _truncate_one_ct_boxes(
                    Path(source_image), regions_ct, pad_value=args.pad_value)
                nib.save(nib.Nifti1Image(arr_ct.astype(np.float32), aff_ct), str(out_ct))
                n_written += 1
            gt_trunc = _blank_notches(gt_arr.copy(), regions_gt, pad_value=0)
            out_gt = gt_dir / f"{case_id}_step{pp:02d}.nii.gz"
            nib.save(nib.Nifti1Image(gt_trunc.astype(gt_arr.dtype), gt_affine), str(out_gt))

            geom = {"mode": "box_per_eye", "corner": corner,
                    "min_retain": T, "min_retain_on": T_on, "per_eye": per_eye}
            entry["steps"][str(pp)] = {"kept": True, "image": str(out_ct),
                                       "gt_trunc": str(out_gt), **geom}
            sidecar[case_id][str(pp)] = {"source_shape": list(ct_shape), **geom}
            worst = min(min(e["ret_total"], e["ret_ON"]) for e in per_eye.values())
            summ.append(f"step{pp:02d}(T{T:.2f},worst{worst:.2f})")
        manifest["cases"][case_id] = entry
        print(f"  {case_id}: per-eye {corner} eyes={list(eye_data)} -> " + " ".join(summ))

    with open(out_root / "corrector_data_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    with open(out_root / "fov_truncation_manifest.json", "w") as fh:
        json.dump(sidecar, fh, indent=2)
    print(f"[fov] cases={n_cases} images_written={n_written} skipped={n_skipped}")
    print(f"[fov] manifest -> {out_root / 'corrector_data_manifest.json'}")
    print(f"[fov] sidecar  -> {out_root / 'fov_truncation_manifest.json'}  (+ gt_trunc/, per_eye QC)")
    print("[fov] NEXT: 835 stage-1 on each truncated CT -> CNISP 032 "
          f"(--steps {','.join(str(s) for s in steps)}) -> build_corrector_dataset. See RUNBOOK_FOV.md.")
    return 0


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
    ap.add_argument("--min-retains", default=None,
                    help="[per-eye box, Option 2] comma list of per-eye retention FLOORS T "
                         "(e.g. 0.5,0.65,0.8). Each T -> pseudo-step round(T*100). Triggers "
                         "the per-eye path: each orbit is clipped INDEPENDENTLY to the "
                         "deepest cut still holding ret_total>=T AND ret_ON>=T_on for BOTH "
                         "eyes (guarantees both eyes stay >= half visible). Replaces "
                         "--keep-fractions/--mode when given.")
    ap.add_argument("--min-retain-on", type=float, default=None,
                    help="[per-eye] separate ON retention floor (default: = each T).")
    ap.add_argument("--min-on-vox", type=int, default=10,
                    help="[per-eye] skip a case if either eye's ON has < this many voxels "
                         "(atypical anatomy / bad split). Default %(default)s.")
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

    # Option 2 (per-eye min-retain) is a distinct path -- delegate and return.
    if args.min_retains:
        return _run_per_eye(args, src, out_root, images_dir, cd)

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
