"""
FOV-completion spatial region masks (implementation-plan §5 + §2.3), aligned with
nnU-Net's channel-first segmentation contract per the re-audit:

  * labels are normalized with ``as_zyx_label`` -> a preprocessed nnU-Net seg
    ``(1, z, y, x)`` becomes ``seg[0]``; a raw 3-D label ``(z, y, x)`` passes
    through; channel-last / multi-channel arrays are REJECTED (they are not part
    of the corrector contract). The old ``gt[..., 0]`` slice was wrong for
    channel-first data (it dropped the last spatial axis);
  * plan-driven spacing (never hard-code 0.5): callers pass ``spacing_zyx`` from
    lib/plan_spacing.py; seam width may be voxels (default 12) or mm (physical
    distance transform);
  * grid identity, integer box indices, non-empty FOV, and positive finite
    spacing are ENFORCED; region-mask / missing-mask shapes are validated;
  * per-structure center pools -> class_locations_fov = {region: {struct: coords}}
    with reproducible per (subject, condition, region, structure) seeds.

Axis convention: arrays / spacing / visible_box are all z, y, x; visible_box =
[(z_lo,z_hi),(y_lo,y_hi),(x_lo,x_hi)], half-open, on the SAME (preprocessed/target)
grid as the label.

numpy + scipy.ndimage only.
"""

from __future__ import annotations

import zlib
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

_STRUCT_26 = ndimage.generate_binary_structure(3, 3)


# ── label contract (re-audit §4, §5.1) ────────────────────────────────────────
def as_zyx_label(labels: np.ndarray, *, source: str = "auto") -> np.ndarray:
    """Normalize a corrector label to a 3-D ``(z, y, x)`` array.

    Supported contracts:
      * raw / normalized 3-D label   -> ``(z, y, x)``  (returned as-is)
      * nnU-Net preprocessed seg     -> ``(1, z, y, x)`` -> ``seg[0]``

    Channel-last ``(z, y, x, 1)`` and multi-channel ``(C>1, ...)`` are rejected —
    they are not part of the nnunet-c training contract. ``seg[0]`` is a view, so
    the caller keeps zero-copy access.
    """
    arr = np.asarray(labels)
    if arr.ndim == 3:
        return arr
    if arr.ndim != 4:
        raise ValueError("expected a 3-D label or channel-first nnU-Net seg "
                         f"(1,z,y,x); got shape {arr.shape}.")
    if arr.shape[0] != 1:
        raise ValueError("a 4-D nnU-Net segmentation must have one leading channel; "
                         f"got shape {arr.shape} (channel-last / multi-channel is "
                         "not part of the corrector contract).")
    if source not in {"auto", "nnunet_preprocessed"}:
        raise ValueError(f"unsupported 4-D label source {source!r}.")
    return arr[0]


# ── grid validation (§3.2 + re-audit §10.1) ───────────────────────────────────
def validate_grid(gt_zyx: np.ndarray, visible_box, axis_order: str = "zyx") -> None:
    if gt_zyx.ndim != 3:
        raise ValueError(f"label must be 3-D (z,y,x) after normalization; got {gt_zyx.shape}")
    if axis_order != "zyx":
        raise ValueError(f"axis_order must be 'zyx'; got {axis_order!r}")
    if len(visible_box) != 3:
        raise ValueError(f"visible_box must have 3 axes; got {len(visible_box)}")
    for ax, ((lo, hi), size) in enumerate(zip(visible_box, gt_zyx.shape)):
        if int(lo) != lo or int(hi) != hi:
            raise ValueError(f"visible_box axis {ax} = [{lo},{hi}) must be integer voxel indices.")
        if not (0 <= int(lo) <= int(hi) <= int(size)):
            raise ValueError(f"visible_box axis {ax} = [{lo},{hi}) out of bounds for size {size}.")


def _check_spacing(spacing_zyx) -> Tuple[float, float, float]:
    s = tuple(float(v) for v in spacing_zyx)
    if len(s) != 3 or any((not np.isfinite(v)) or v <= 0 for v in s):
        raise ValueError(f"spacing_zyx must be 3 positive finite values; got {spacing_zyx}")
    return s


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
    fg_values: Optional[Sequence[int]] = None,
    axis_order: str = "zyx",
) -> Dict[str, np.ndarray]:
    """Return the six region masks (bool). See module docstring.

    ``fg_values``: explicit foreground label values (e.g. {1,2,3,4}); when given
    the foreground is ``isin(gt, fg_values)`` rather than ``gt > 0`` — safer if a
    special/ignore value like -1 ever appears (re-audit §6.2).
    """
    gt = as_zyx_label(gt_labels)
    validate_grid(gt, visible_box, axis_order=axis_order)

    M_F = visible_box_to_mask(gt.shape, visible_box)
    if not M_F.any():
        raise ValueError("empty acquired FOV: visible_box selects no voxels.")
    M_miss = ~M_F
    M_fg = np.isin(gt, list(fg_values)) if fg_values is not None else (gt > 0)
    M_miss_fg = M_miss & M_fg
    M_vis_fg = M_F & M_fg

    if not M_miss.any():
        M_seam = np.zeros_like(M_fg)          # full-FOV anchor -> no seam
    elif seam_width_mm is not None:
        s = _check_spacing(spacing_zyx)
        d_from_missing = ndimage.distance_transform_edt(M_miss, sampling=s)  # 0 on visible
        d_from_visible = ndimage.distance_transform_edt(M_F, sampling=s)     # 0 on missing
        band = (d_from_missing <= seam_width_mm) & (d_from_visible <= seam_width_mm)
        d_from_fg = ndimage.distance_transform_edt(~M_fg, sampling=s)
        M_seam = band & (d_from_fg <= seam_width_mm)
    else:
        if spacing_zyx is not None:
            _check_spacing(spacing_zyx)
        fg_dil = seam_width_voxels if fg_dilate_voxels is None else fg_dilate_voxels
        band = _dilate(M_F, seam_width_voxels) & _dilate(M_miss, seam_width_voxels)
        M_seam = band & _dilate(M_fg, fg_dil)

    return {"M_F": M_F, "M_miss": M_miss, "M_fg": M_fg,
            "M_miss_fg": M_miss_fg, "M_vis_fg": M_vis_fg, "M_seam": M_seam}


def _dilate(mask: np.ndarray, iters: int) -> np.ndarray:
    if iters <= 0:
        return mask.astype(bool, copy=True)
    return ndimage.binary_dilation(mask, structure=_STRUCT_26, iterations=int(iters))


# ── per-structure center pools -> class_locations_fov (§3.3) ──────────────────
def _stable_seed(*parts) -> int:
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
    """Per-structure coordinate pools: {"missing":{struct_value:(N,3)},"seam":..,"visible":..}."""
    gt = as_zyx_label(gt_labels)
    for key in ("M_miss_fg", "M_seam", "M_vis_fg"):
        if masks[key].shape != gt.shape:
            raise ValueError(f"{key} shape {masks[key].shape} != label shape {gt.shape}.")
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
    gt = as_zyx_label(gt_labels)
    miss = np.asarray(missing_mask, dtype=bool)
    if miss.shape != gt.shape:
        raise ValueError(f"missing_mask shape {miss.shape} != label shape {gt.shape}.")
    out: Dict[str, float] = {}
    for name, val in struct_values.items():
        Yk = gt == int(val)
        tot = int(Yk.sum())
        out[name] = round(float((Yk & miss).sum()) / tot, 6) if tot else 0.0
    fg = np.isin(gt, [int(v) for v in struct_values.values()])
    tot_u = int(fg.sum())
    out["union"] = round(float((fg & miss).sum()) / tot_u, 6) if tot_u else 0.0
    return out


# ── self-test ────────────────────────────────────────────────────────────────
def _test_label_contract():
    # §9.1 raw 3-D passes through
    assert as_zyx_label(np.zeros((20, 30, 40), np.int16)).shape == (20, 30, 40)
    # §9.2 channel-first -> seg[0], zero-copy
    seg = np.zeros((1, 20, 30, 40), np.int16)
    gt = as_zyx_label(seg, source="nnunet_preprocessed")
    assert gt.shape == (20, 30, 40) and np.shares_memory(gt, seg)
    # §9.3 reject multi-channel
    for bad in (np.zeros((2, 20, 30, 40), np.int16),          # multi-channel
                np.zeros((20, 30, 40, 1), np.int16),          # channel-last
                np.zeros((20, 30), np.int16)):                # 2-D
        try:
            as_zyx_label(bad)
            raise AssertionError(f"should reject {bad.shape}")
        except ValueError:
            pass
    # §9.5 the old [...,0] bug: last x index survives seg[0]
    seg2 = np.zeros((1, 20, 30, 40), np.int16)
    seg2[0, 5, 6, 39] = 2
    assert as_zyx_label(seg2)[5, 6, 39] == 2
    print("label-contract tests OK (channel-first seg[0]; rejects channel-last/multi-channel)")


def _selftest() -> int:
    _test_label_contract()

    S = 60
    gt = np.zeros((S, S, S), np.int16)
    zz, yy, xx = np.ogrid[:S, :S, :S]
    gt[(xx - 30) ** 2 + (yy - 30) ** 2 + (zz - 30) ** 2 < 12 ** 2] = 3   # Globe
    gt[10:50, 28:33, 28:33] = 1                                          # ON
    gt[0:3, 0:3, 0:3] = -1                                               # ignore label
    struct_values = {"ON": 1, "Globe": 3}
    visible_box = [[25, S], [0, S], [0, S]]

    # channel-first preprocessed seg gives identical masks to the 3-D label (the bug fix)
    seg = gt[None]                                            # (1,z,y,x)
    m3 = compute_region_masks(gt, visible_box, seam_width_voxels=6, fg_values=[1, 3])
    m4 = compute_region_masks(seg, visible_box, seam_width_voxels=6, fg_values=[1, 3])
    for k in m3:
        assert np.array_equal(m3[k], m4[k]), f"channel-first != 3-D for {k}"

    m = m3
    fg = np.isin(gt, [1, 3])
    assert np.array_equal(m["M_vis_fg"] | m["M_miss_fg"], fg)
    assert not (m["M_vis_fg"] & m["M_miss_fg"]).any()
    assert not m["M_fg"][0, 0, 0]                             # -1 excluded from fg

    # guards
    for bad_call in (
        lambda: compute_region_masks(gt, [[0, S + 1], [0, S], [0, S]]),          # OOB
        lambda: compute_region_masks(gt, [[0.5, S], [0, S], [0, S]]),            # non-integer
        lambda: compute_region_masks(gt, [[10, 10], [10, 10], [10, 10]]),       # empty FOV
        lambda: compute_region_masks(gt, visible_box, spacing_zyx=(0.0, 1, 1)), # bad spacing
    ):
        try:
            bad_call()
            raise AssertionError("guard should have raised")
        except ValueError:
            pass

    pools = region_center_pools(m, seg, struct_values, max_per_class=1000,
                                base_seed=7, seed_tag="subjA_axial_rm50")
    assert set(pools["missing"]) == {1, 3} and len(pools["missing"][3]) > 0
    # mask/label shape mismatch is caught
    try:
        region_center_pools({k: v[:, :, :30] for k, v in m.items()}, gt, struct_values)
        raise AssertionError("shape-mismatch guard should raise")
    except ValueError:
        pass

    frac = anatomical_missing_fraction(seg, m["M_miss"], struct_values)
    assert abs(frac["ON"] - 15.0 / 40.0) < 0.02
    m_full = compute_region_masks(gt, [[0, S], [0, S], [0, S]], seam_width_voxels=6, fg_values=[1, 3])
    assert not m_full["M_miss_fg"].any() and not m_full["M_seam"].any()

    print(f"missing_fg {int(m['M_miss_fg'].sum())} | vis_fg {int(m['M_vis_fg'].sum())} "
          f"| seam {int(m['M_seam'].sum())} | ON miss {frac['ON']}")
    print("FOV REGION-MASK SELF-TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
