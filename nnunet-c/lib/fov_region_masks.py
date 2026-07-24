"""
FOV-completion spatial region masks (implementation-plan §5 + §2.3), with the
review's required modifications:

  * plan-driven spacing (never hard-code 0.5): callers pass ``spacing_zyx`` from
    lib/plan_spacing.py; seam width may be given in voxels (default 12) or in mm
    (physical distance transform, review §2.4);
  * grid identity is ENFORCED, not assumed (review §3.2);
  * per-structure center pools -> class_locations_fov = {region: {struct: coords}}
    so small structures (ON, Recti) aren't drowned by Globe/Fat (review §3.3);
  * reproducible per (subject, condition, region, structure) subsample seeds
    (review §3.4).

Axis convention (review §2.2): arrays / spacing / visible_box are all z, y, x;
visible_box = [(z_lo,z_hi),(y_lo,y_hi),(x_lo,x_hi)], half-open, on the SAME grid
as ``gt_labels`` (the preprocessed/target GT grid — NOT the native CT grid; the
data-gen is responsible for projecting visible_box onto the target grid).

numpy + scipy.ndimage only.
"""

from __future__ import annotations

import zlib
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

_STRUCT_26 = ndimage.generate_binary_structure(3, 3)


# ── grid validation (review §3.2) ─────────────────────────────────────────────
def validate_grid(gt_labels: np.ndarray, visible_box, axis_order: str = "zyx") -> None:
    if gt_labels.ndim != 3:
        raise ValueError(f"gt_labels must be 3-D (z,y,x); got ndim={gt_labels.ndim}")
    if axis_order != "zyx":
        raise ValueError(f"axis_order must be 'zyx' for the target grid; got {axis_order!r}")
    if len(visible_box) != 3:
        raise ValueError(f"visible_box must have 3 axes; got {len(visible_box)}")
    for ax, ((lo, hi), size) in enumerate(zip(visible_box, gt_labels.shape)):
        if not (0 <= int(lo) <= int(hi) <= int(size)):
            raise ValueError(f"visible_box axis {ax} = [{lo},{hi}) out of bounds for "
                             f"size {size} (need 0<=lo<=hi<=size).")


def visible_box_to_mask(shape: Sequence[int], visible_box) -> np.ndarray:
    m = np.zeros(tuple(int(s) for s in shape), dtype=bool)
    sl = tuple(slice(int(lo), int(hi)) for lo, hi in visible_box)
    m[sl] = True
    return m


def compute_region_masks(
    gt_labels: np.ndarray,
    visible_box,
    spacing_zyx: Optional[Tuple[float, float, float]] = None,
    seam_width_voxels: int = 12,
    seam_width_mm: Optional[float] = None,
    fg_dilate_voxels: Optional[int] = None,
    axis_order: str = "zyx",
) -> Dict[str, np.ndarray]:
    """Return the six region masks (bool) for one case.

    Seam construction:
      * default -> voxel dilation band of half-width ``seam_width_voxels``;
      * if ``seam_width_mm`` given (requires ``spacing_zyx``) -> physical distance
        transform (anisotropy-correct, review §2.4).
    """
    gt = np.asarray(gt_labels)
    if gt.ndim == 4:
        gt = gt[..., 0]
    validate_grid(gt, visible_box, axis_order=axis_order)

    M_F = visible_box_to_mask(gt.shape, visible_box)
    M_miss = ~M_F
    M_fg = gt > 0
    M_miss_fg = M_miss & M_fg
    M_vis_fg = M_F & M_fg

    if not (M_miss.any() and M_F.any()):
        M_seam = np.zeros_like(M_fg)          # full-FOV anchor -> no seam
    elif seam_width_mm is not None:
        if spacing_zyx is None:
            raise ValueError("seam_width_mm requires spacing_zyx (from the plan).")
        s = tuple(float(v) for v in spacing_zyx)
        d_from_missing = ndimage.distance_transform_edt(M_miss, sampling=s)  # 0 on visible
        d_from_visible = ndimage.distance_transform_edt(M_F, sampling=s)     # 0 on missing
        band = (d_from_missing <= seam_width_mm) & (d_from_visible <= seam_width_mm)
        d_from_fg = ndimage.distance_transform_edt(~M_fg, sampling=s)
        M_seam = band & (d_from_fg <= seam_width_mm)
    else:
        fg_dil = seam_width_voxels if fg_dilate_voxels is None else fg_dilate_voxels
        band = _dilate(M_F, seam_width_voxels) & _dilate(M_miss, seam_width_voxels)
        M_seam = band & _dilate(M_fg, fg_dil)

    return {"M_F": M_F, "M_miss": M_miss, "M_fg": M_fg,
            "M_miss_fg": M_miss_fg, "M_vis_fg": M_vis_fg, "M_seam": M_seam}


def _dilate(mask: np.ndarray, iters: int) -> np.ndarray:
    if iters <= 0:
        return mask.astype(bool, copy=True)
    return ndimage.binary_dilation(mask, structure=_STRUCT_26, iterations=int(iters))


# ── per-structure center pools -> class_locations_fov (review §3.3) ───────────
def _stable_seed(*parts) -> int:
    """Deterministic seed from (base, subject, condition, region, structure)
    independent of Python's hash randomization (review §3.4)."""
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return int(zlib.crc32(key)) & 0xFFFFFFFF


def region_center_pools(
    masks: Dict[str, np.ndarray],
    gt_labels: np.ndarray,
    struct_values: Dict[str, int],
    max_per_class: int = 5000,
    base_seed: int = 0,
    seed_tag: str = "",
) -> Dict[str, Dict[int, np.ndarray]]:
    """Per-structure voxel-coordinate pools:
        {"missing": {struct_value: (N,3) int32}, "seam": {...}, "visible": {...}}

    Each pool = argwhere(region_mask & (gt == struct_value)), reproducibly
    subsampled to ``max_per_class``. ``seed_tag`` should encode subject+condition
    so reruns give identical pools.
    """
    gt = np.asarray(gt_labels)
    if gt.ndim == 4:
        gt = gt[..., 0]
    region_to_key = {"missing": "M_miss_fg", "seam": "M_seam", "visible": "M_vis_fg"}
    out: Dict[str, Dict[int, np.ndarray]] = {}
    for region, mkey in region_to_key.items():
        rmask = masks[mkey]
        out[region] = {}
        for name, val in struct_values.items():
            coords = np.argwhere(rmask & (gt == int(val))).astype(np.int32)
            if len(coords) > max_per_class:
                rng = np.random.default_rng(_stable_seed(base_seed, seed_tag, region, name))
                coords = coords[rng.choice(len(coords), max_per_class, replace=False)]
            out[region][int(val)] = coords
    return out


def anatomical_missing_fraction(
    gt_labels: np.ndarray,
    missing_mask: np.ndarray,
    struct_values: Dict[str, int],
) -> Dict[str, float]:
    """Fraction of each structure's foreground inside the missing region + union
    (plan §2.3). Structures with no foreground report 0.0."""
    gt = np.asarray(gt_labels)
    if gt.ndim == 4:
        gt = gt[..., 0]
    miss = np.asarray(missing_mask, dtype=bool)
    out: Dict[str, float] = {}
    for name, val in struct_values.items():
        Yk = gt == int(val)
        tot = int(Yk.sum())
        out[name] = round(float((Yk & miss).sum()) / tot, 6) if tot else 0.0
    fg = gt > 0
    tot_u = int(fg.sum())
    out["union"] = round(float((fg & miss).sum()) / tot_u, 6) if tot_u else 0.0
    return out


# ── self-test ────────────────────────────────────────────────────────────────
def _selftest() -> int:
    S = 60
    gt = np.zeros((S, S, S), np.int16)
    zz, yy, xx = np.ogrid[:S, :S, :S]
    gt[(xx - 30) ** 2 + (yy - 30) ** 2 + (zz - 30) ** 2 < 12 ** 2] = 3   # Globe
    gt[10:50, 28:33, 28:33] = 1                                          # ON
    struct_values = {"ON": 1, "Globe": 3}
    visible_box = [[25, S], [0, S], [0, S]]                             # cut axis-0 at 25

    # grid validation rejects an out-of-bounds box
    try:
        compute_region_masks(gt, [[0, S + 5], [0, S], [0, S]])
        raise AssertionError("grid validation should have failed")
    except ValueError:
        pass

    m = compute_region_masks(gt, visible_box, seam_width_voxels=6)
    fg = gt > 0
    assert np.array_equal(m["M_vis_fg"] | m["M_miss_fg"], fg)
    assert not (m["M_vis_fg"] & m["M_miss_fg"]).any()
    assert np.argwhere(m["M_miss_fg"])[:, 0].max() < 25
    assert np.argwhere(m["M_vis_fg"])[:, 0].min() >= 25

    # physical seam == voxel seam when spacing is isotropic 1.0 (both ~ 6-vox band)
    m_mm = compute_region_masks(gt, visible_box, spacing_zyx=(1.0, 1.0, 1.0),
                                seam_width_mm=6.0)
    sx = np.argwhere(m_mm["M_seam"])[:, 0]
    assert sx.size > 0 and sx.min() >= 25 - 7 and sx.max() <= 25 + 7, (sx.min(), sx.max())

    # per-structure pools: reproducible + separated by class
    pools = region_center_pools(m, gt, struct_values, max_per_class=1000,
                                base_seed=7, seed_tag="subjA_axial_rm50")
    pools2 = region_center_pools(m, gt, struct_values, max_per_class=1000,
                                 base_seed=7, seed_tag="subjA_axial_rm50")
    assert set(pools) == {"missing", "seam", "visible"}
    assert set(pools["missing"]) == {1, 3}                     # keyed by struct value
    assert len(pools["missing"][3]) > 0                        # globe present in missing
    for reg in pools:
        for v in pools[reg]:
            assert np.array_equal(pools[reg][v], pools2[reg][v])   # reproducible

    frac = anatomical_missing_fraction(gt, m["M_miss"], struct_values)
    assert abs(frac["ON"] - 15.0 / 40.0) < 0.02
    m_full = compute_region_masks(gt, [[0, S], [0, S], [0, S]], seam_width_voxels=6)
    assert not m_full["M_miss_fg"].any() and not m_full["M_seam"].any()

    print(f"missing_fg {int(m['M_miss_fg'].sum())} | vis_fg {int(m['M_vis_fg'].sum())} "
          f"| seam(vox) {int(m['M_seam'].sum())} | seam(mm) {int(m_mm['M_seam'].sum())}")
    print(f"pools missing: ON={len(pools['missing'][1])} Globe={len(pools['missing'][3])}")
    print(f"ON missing frac {frac['ON']} | union {frac['union']}")
    print("FOV REGION-MASK SELF-TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
