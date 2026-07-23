"""
FOV co-framed pair alignment.

Implements the intended FOV-truncation patch construction, where obs
(nnUNet prediction on the truncated scan) and GT (the full-res prediction /
target) end up in the SAME physical frame so they correspond voxel-for-voxel:

  1. separate_eyes() on the OBS prediction -> the two eyes.
  2. Per eye, take the centroid over the WHOLE visible eye of the OBS (all
     foreground, midplane-clipped to ipsilateral; keep_all -- no largest-CC
     reduction).
  3. Crop a fixed patch_size_mm cube at that OBS centroid from BOTH the obs and
     the GT (the SAME crop box -> co-framed). Out-of-source voxels are padded.
  4. valid_mask = the acquired region inside the patch = within source bounds
     AND within the sidecar visible_box. Everything else (crop padding + the
     truncated part of the FOV) is invalid -- that is exactly the region the
     latent fit must ignore.
  5. Midplane-clip to the ipsilateral side so contralateral bleed never enters.
  6. CHECK: the GT foreground that belongs to this eye is fully contained in the
     patch (gt_captured_frac ~ 1.0); a low value means the 80 mm box is too
     small / mis-centred to hold the whole target.
  7. remap -> RAS reorient -> OS->OD flip, applied identically to obs, GT and
     the valid mask, so all three stay registered.

The obs prediction and the GT prediction MUST live on the same source grid
(same shape + affine). For the FOV pipeline they do: the truncated pred is
mapped back into the full source grid (fov_crop_pred), and the GT candidate is
the full-res pred on that same grid.

numpy + nibabel only (no torch), so data_prep stays engine-independent.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
from typing import List, Optional

import numpy as np
import nibabel as nib

_ROOT = str(_Path(__file__).resolve().parents[1])
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from data_prep.canonical_align import (
    detect_label_scheme, separate_eyes, extract_single_eye_lcc,
    remap_to_canonical, reorient_to_ras, flip_os_to_od,
    validate_diagonal_affine,
)


# ── fixed-size padded crop (truncation shows up as padding) ────────────────────
def _pad_or_crop(vol: np.ndarray, lo_vox, shape_vox, fill=0) -> np.ndarray:
    lo = np.asarray(lo_vox, dtype=np.int64)
    shp = np.asarray(shape_vox, dtype=np.int64)
    hi = lo + shp
    vs = np.asarray(vol.shape, dtype=np.int64)
    src_lo = np.maximum(lo, 0)
    src_hi = np.minimum(hi, vs)
    dst_lo = src_lo - lo
    dst_hi = dst_lo + (src_hi - src_lo)
    out = np.full(tuple(int(s) for s in shp), fill, dtype=vol.dtype)
    if np.all(src_hi > src_lo):
        out[dst_lo[0]:dst_hi[0], dst_lo[1]:dst_hi[1], dst_lo[2]:dst_hi[2]] = \
            vol[src_lo[0]:src_hi[0], src_lo[1]:src_hi[1], src_lo[2]:src_hi[2]]
    return out


def _voxel_sizes(affine):
    return np.sqrt(np.sum(np.asarray(affine)[:3, :3] ** 2, axis=0))


def _ipsilateral_mask(shape, this_globe_vox, other_globe_vox):
    """True on THIS eye's side of the midplane between the two globes."""
    m = np.ones(shape, dtype=bool)
    if other_globe_vox is None:
        return m
    diff = np.asarray(other_globe_vox) - np.asarray(this_globe_vox)
    sep = int(np.argmax(np.abs(diff)))
    mid = int(round((other_globe_vox[sep] + this_globe_vox[sep]) / 2.0))
    idx = [slice(None)] * 3
    if diff[sep] > 0:
        idx[sep] = slice(mid, None)      # other eye is at higher index -> drop it
    else:
        idx[sep] = slice(0, mid)
    m[tuple(idx)] = False
    return m


def align_fov_pair(
    obs_seg_path: str,
    gt_seg_path: str,
    patch_size_mm: float = 80.0,
    visible_box: Optional[list] = None,
    search_size_mm: Optional[float] = None,
) -> List[dict]:
    """Return one co-framed dict per eye. See the module docstring for the steps.

    ``visible_box`` : per-source-axis ``[lo, hi)`` voxel window (the sidecar
    ``visible_box``) marking the acquired FOV in the OBS/GT source grid. When
    None the valid mask falls back to "within source bounds" only.
    """
    if search_size_mm is None:
        search_size_mm = patch_size_mm * 1.5

    obs_img = nib.load(str(obs_seg_path))
    gt_img = nib.load(str(gt_seg_path))
    obs = np.rint(np.asanyarray(obs_img.dataobj)).astype(np.int64)
    gt = np.rint(np.asanyarray(gt_img.dataobj)).astype(np.int64)
    if obs.ndim == 4:
        obs = obs[..., 0]
    if gt.ndim == 4:
        gt = gt[..., 0]
    affine = np.asarray(obs_img.affine, dtype=np.float64)

    if obs.shape != gt.shape:
        raise ValueError(f"obs shape {obs.shape} != gt shape {gt.shape}; the FOV "
                         f"pair must be on the same source grid.")
    if not np.allclose(affine, np.asarray(gt_img.affine, float), atol=1e-3):
        raise ValueError("obs and gt affines differ; not the same source grid.")

    o_scheme, o_map = detect_label_scheme(obs)
    if not o_map:
        raise ValueError(f"{obs_seg_path}: unrecognized OBS label scheme")
    g_scheme, g_map = detect_label_scheme(gt)
    if not g_map:
        raise ValueError(f"{gt_seg_path}: unrecognized GT label scheme")
    obs_globe = next(l for l, n in o_map.items() if n == "Globe")

    eyes = separate_eyes(obs, affine, obs_globe)
    if not eyes:
        raise ValueError(f"{obs_seg_path}: no globe CC in OBS (globe fully "
                         f"truncated?) -- cannot anchor the eye.")

    vox = _voxel_sizes(affine)
    half_vox = np.round((patch_size_mm / 2.0) / vox).astype(np.int64)
    shape_vox = (2 * half_vox).astype(np.int64)
    obs_fg_labels = np.asarray(list(o_map.keys()), dtype=obs.dtype)
    gt_fg_labels = np.asarray(list(g_map.keys()), dtype=gt.dtype)

    # visible_box -> boolean acquired-region mask on the source grid.
    vb_mask = None
    if visible_box is not None:
        vb_mask = np.zeros(obs.shape, dtype=bool)
        sl = tuple(slice(int(lo), int(hi)) for lo, hi in visible_box)
        vb_mask[sl] = True

    globe_vox = [e["centroid_voxel"] for e in eyes]
    out = []
    for i, eye in enumerate(eyes):
        other = globe_vox[1 - i] if len(eyes) == 2 else None

        # (2) whole visible-eye centroid on the OBS, midplane-clipped ipsilateral.
        fg_bbox = _obs_eye_fg_bbox(obs, obs_fg_labels, eye["centroid_voxel"],
                                   other, search_size_mm, vox)
        centroid_vox = fg_bbox["centroid"]

        # (3) fixed cube at the OBS centroid, cropped from BOTH volumes.
        lo = (np.round(centroid_vox).astype(np.int64) - half_vox)
        obs_patch = _pad_or_crop(obs, lo, shape_vox)
        gt_patch = _pad_or_crop(gt, lo, shape_vox)

        # (4) valid = within source bounds AND within visible_box (acquired).
        in_bounds = _pad_or_crop(np.ones(obs.shape, dtype=bool), lo, shape_vox,
                                 fill=False)
        valid = in_bounds.copy()
        if vb_mask is not None:
            valid &= _pad_or_crop(vb_mask, lo, shape_vox, fill=False)

        # (5) midplane clip -> ipsilateral only.
        ipsi = _pad_or_crop(_ipsilateral_mask(obs.shape, eye["centroid_voxel"],
                                              other), lo, shape_vox, fill=False)
        obs_patch = np.where(ipsi, obs_patch, 0)
        gt_patch = np.where(ipsi, gt_patch, 0)
        valid &= ipsi

        # (6) CHECK: is this eye's GT foreground fully inside the patch?
        gt_eye_total = int(np.isin(gt, gt_fg_labels).astype(bool)
                           [_ipsilateral_mask(gt.shape, eye["centroid_voxel"], other)
                            & _search_bbox_mask(gt.shape, eye["centroid_voxel"],
                                                search_size_mm, vox)].sum())
        gt_in_patch = int(np.isin(gt_patch, gt_fg_labels).sum())
        gt_captured = (gt_in_patch / gt_eye_total) if gt_eye_total else 1.0

        # (7) remap -> RAS -> OS->OD flip, applied identically to all three.
        eye_label = eye.get("eye", f"cc{i}")
        casename = f"{_stem(obs_seg_path)}_{eye_label}"
        pa = affine.copy()
        pa[:3, 3] += affine[:3, :3] @ lo.astype(np.float64)

        obs_c = remap_to_canonical(obs_patch, o_map)
        gt_c = remap_to_canonical(gt_patch, g_map)
        obs_c, pa_ras, ornt = reorient_to_ras(obs_c, pa)
        gt_c, _, _ = reorient_to_ras(gt_c, pa)
        valid_c, _, _ = reorient_to_ras(valid.astype(np.int16), pa)
        validate_diagonal_affine(pa_ras, casename)

        was_flipped = (eye_label == "OS")
        if was_flipped:
            obs_c, pa_flip = flip_os_to_od(obs_c, pa_ras)
            gt_c = np.flip(gt_c, axis=0).copy()
            valid_c = np.flip(valid_c, axis=0).copy()
            pa_ras = pa_flip

        spacing = np.sqrt(np.sum(pa_ras[:3, :3] ** 2, axis=0))
        out.append({
            "eye": eye_label,
            "casename": casename,
            "obs_patch": obs_c.astype(np.int16),
            "gt_patch": gt_c.astype(np.int16),
            "valid_mask": (valid_c > 0),
            "affine": pa_ras,
            "spacing": spacing,
            "was_flipped": was_flipped,
            "original_ornt": ornt,
            "obs_centroid_vox": [float(x) for x in centroid_vox],
            "obs_fg_vox": int((obs_c > 0).sum()),
            "gt_fg_in_patch": gt_in_patch,
            "gt_fg_eye_total": gt_eye_total,
            "gt_captured_frac": round(float(gt_captured), 4),
            "valid_frac": round(float((valid_c > 0).mean()), 4),
        })
    return out


def _search_bbox_mask(shape, globe_vox, search_size_mm, vox):
    half = np.round((search_size_mm / 2.0) / vox).astype(int)
    c = np.round(globe_vox).astype(int)
    m = np.zeros(shape, dtype=bool)
    sl = tuple(slice(int(max(0, c[a] - half[a])), int(min(shape[a], c[a] + half[a])))
               for a in range(3))
    m[sl] = True
    return m


def _obs_eye_fg_bbox(obs, fg_labels, this_globe_vox, other_globe_vox,
                     search_size_mm, vox) -> dict:
    """Whole visible-eye centroid on the OBS: mean over ALL ipsilateral fg in the
    midplane-clipped search bbox (keep_all -- no largest-CC reduction)."""
    bbox = _search_bbox_mask(obs.shape, this_globe_vox, search_size_mm, vox)
    ipsi = _ipsilateral_mask(obs.shape, this_globe_vox, other_globe_vox)
    fg = np.isin(obs, fg_labels) & bbox & ipsi
    _, centroid, count, total = extract_single_eye_lcc(fg, keep_all=True)
    if centroid is None:
        centroid = np.asarray(this_globe_vox, dtype=float)
    return {"centroid": np.asarray(centroid, dtype=float),
            "fg_count": int(count), "total_fg": int(total)}


def _stem(p: str) -> str:
    from pathlib import Path
    s = Path(p).name
    return s.replace(".nii.gz", "").replace(".nii", "")


def _selftest() -> int:
    """Synthetic proof of the co-framed pair (no data). Builds a 2-eye source
    (nnUNet canonical {1:ON,2:Recti,3:Globe,4:Fat}), truncates one eye, and
    checks: obs & GT share the frame (every obs fg voxel is a gt fg voxel), obs
    fg lies inside the valid mask, truncation shrinks the valid region, and the
    GT-completeness check reports full capture."""
    import tempfile
    from pathlib import Path

    S = 100

    def paint_eye(vol, cx):
        zz, yy, xx = np.ogrid[:S, :S, :S]
        vol[cx - 8:cx + 8, 42:58, 42:58] = 4                       # Fat
        vol[(xx - cx) ** 2 + (yy - 50) ** 2 + (zz - 50) ** 2 < 7 ** 2] = 3  # Globe
        vol[cx - 12:cx - 8, 48:52, 48:52] = 1                      # ON
        vol[cx - 2:cx + 2, 40:42, 48:52] = 2                       # Recti

    gt = np.zeros((S, S, S), np.int16)
    paint_eye(gt, 30)                                              # OD
    paint_eye(gt, 70)                                              # OS
    obs = gt.copy()
    obs[:25, :, :] = 0                                             # truncate OD anterior
    vb = [[25, S], [0, S], [0, S]]
    aff = np.diag([0.5, 0.5, 0.5, 1.0]).astype(float)

    with tempfile.TemporaryDirectory() as d:
        op = str(Path(d) / "case_step50.nii.gz")
        gp = str(Path(d) / "case.nii.gz")
        op2 = str(Path(d) / "case2_step99.nii.gz")                 # intact obs = gt
        nib.save(nib.Nifti1Image(obs, aff), op)
        nib.save(nib.Nifti1Image(gt, aff), gp)
        nib.save(nib.Nifti1Image(gt, aff), op2)
        trunc = align_fov_pair(op, gp, patch_size_mm=40.0, visible_box=vb)
        intact = align_fov_pair(op2, gp, patch_size_mm=40.0, visible_box=None)

    for tag, rr in [("trunc", trunc), ("intact", intact)]:
        for r in rr:
            o = r["obs_patch"] > 0
            g = r["gt_patch"] > 0
            v = r["valid_mask"]
            leak = int((o & ~g).sum())
            outside = int((o & ~v).sum())
            print(f"{tag:6s} eye={r['eye']}: obs_fg={int(o.sum())} gt_fg={int(g.sum())} "
                  f"valid_frac={r['valid_frac']} gt_captured={r['gt_captured_frac']} "
                  f"obs-not-in-gt={leak} obs-outside-valid={outside}")
            assert leak == 0, "co-framing broken: obs fg must be a subset of gt fg"
            assert outside == 0, "obs fg must lie inside the acquired (valid) region"
            assert r["gt_captured_frac"] >= 0.99, "patch clipped the GT target"

    od_tr = [r for r in trunc if r["eye"] == "OD"][0]["valid_frac"]
    od_in = [r for r in intact if r["eye"] == "OD"][0]["valid_frac"]
    assert od_tr < od_in, "truncation must shrink the valid region"
    print(f"\nvalid_frac OD: truncated={od_tr} < intact={od_in}")
    print("FOV CO-FRAMED PAIR SELF-TEST PASSED")
    return 0


if __name__ == "__main__":
    import sys
    _root = str(__import__("pathlib").Path(__file__).resolve().parents[1])
    if _root not in sys.path:
        sys.path.insert(0, _root)
    sys.exit(_selftest())
