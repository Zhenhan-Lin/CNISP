"""Per-structure segmentation metrics from label masks (computation layer).

The lowest layer of ``simulation.evaluation`` (analogous to ``nnunet.lib.metrics``
for the comparison subsystem): turn (prediction, GT) label-mask pairs into a tidy
long-format table of per-structure numbers -- volume, Dice, and the surface
metrics ASSD / HD95 / Surface-Dice(NSD). Nothing here plots or aggregates.

Label schemes (values differ per source; the scheme MUST be given per mask, not
guessed -- nnU-Net and CNISP-canonical are BOTH {1,2,3,4} but map to different
structures):
    nnU-Net       {1:ON, 2:Recti, 3:Globe, 4:Fat}
    labelfusion   {1:ON, 3:Recti, 5:Globe, 7:Fat}   (atlas GT, often -1000 offset)
    canonical     {1:ON, 2:Globe, 3:Fat, 4:Recti}   (CNISP-only decode output)

Depends on numpy + scipy.ndimage + nibabel (+ pandas for the table writer).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from scipy import ndimage

# Foreground structures in DISPLAY-name order, used everywhere downstream.
STRUCTURES: List[str] = ["Globe", "Optic nerve", "Recti", "Fat"]

# The five evaluated pipelines (arms), in A-E display order:
#   A. nnUNet       image-conditioned nnUNet on the sparse CT (baseline)
#   B. Cascade UNet nnU->nnU self-correction (control B, nnUNet-prelabel corrector)
#   C. CNISP        CNISP shape prior with the nnUNet sparse pred as input
#   D. Proposed     nnU->CNISP->nnU corrector (control C, CNISP-prelabel corrector)
#   E. Oracle       CNISP shape prior with the GT as input (ceiling)
METHODS: List[str] = ["nnUNet", "Cascade UNet", "CNISP", "Proposed", "Oracle"]

# structure -> integer label, per source scheme (keys are the DISPLAY names).
SCHEMES: Dict[str, Dict[str, int]] = {
    "nnunet":      {"Optic nerve": 1, "Recti": 2, "Globe": 3, "Fat": 4},
    "labelfusion": {"Optic nerve": 1, "Recti": 3, "Globe": 5, "Fat": 7},
    "canonical":   {"Optic nerve": 1, "Globe": 2, "Fat": 3, "Recti": 4},
}

DEFAULT_TAU_MM: float = 1.0   # Surface-Dice tolerance (~ expert inter-rater margin)

PathLike = Union[str, Path]


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Binary Dice (both-empty -> 1.0), matching the rest of the pipeline."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = np.count_nonzero(pred & gt)
    total = np.count_nonzero(pred) + np.count_nonzero(gt)
    if total == 0:
        return 1.0
    return 2.0 * inter / total


def _surface(mask: np.ndarray) -> np.ndarray:
    """Boundary voxels of a binary mask (mask minus its erosion)."""
    er = ndimage.binary_erosion(mask, iterations=1, border_value=0)
    return mask & ~er


def surface_metrics(pred: np.ndarray, gt: np.ndarray, spacing,
                    tau: float = DEFAULT_TAU_MM) -> Dict[str, float]:
    """Symmetric surface distances (mm) via Euclidean distance transform.

    Returns ASSD (mean symmetric surface distance), HD95 (95th pct of symmetric
    surface distances), and NSD = Surface Dice at tolerance ``tau``. ``pred`` and
    ``gt`` must live on the SAME grid. Either mask empty -> NaN (undefined).
    """
    pred, gt = pred.astype(bool), gt.astype(bool)
    if pred.sum() == 0 or gt.sum() == 0:
        return dict(assd=np.nan, hd95=np.nan, nsd=np.nan)
    ps, gs = _surface(pred), _surface(gt)
    dt_to_gt = ndimage.distance_transform_edt(~gs, sampling=spacing)
    dt_to_pred = ndimage.distance_transform_edt(~ps, sampling=spacing)
    d_p2g = dt_to_gt[ps]      # pred-surface -> nearest GT-surface distances
    d_g2p = dt_to_pred[gs]    # GT-surface -> nearest pred-surface distances
    both = np.concatenate([d_p2g, d_g2p])
    return dict(
        assd=float(both.mean()),
        hd95=float(np.percentile(both, 95)),
        nsd=float((np.count_nonzero(d_p2g <= tau) + np.count_nonzero(d_g2p <= tau)) /
                  (d_p2g.size + d_g2p.size)),
    )


def load_labelmap(path: PathLike, offset: int = 0):
    """Load a NIfTI label map -> (int array, spacing[mm]) in array-axis order.

    Atlas maps stored with a -1000 offset: pass ``offset=1000``.
    """
    import nibabel as nib
    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    if offset:
        data = np.clip(data + offset, 0, None)
    spacing = np.array(img.header.get_zooms()[:3], dtype=float)
    return data.astype(np.int32), spacing


def binary_structures(data: np.ndarray, scheme: str) -> Dict[str, np.ndarray]:
    """{display_structure: binary mask} for the given label scheme."""
    lut = SCHEMES[scheme]
    return {s: (data == lut[s]) for s in STRUCTURES}


def compute_case_metrics(pred_path: PathLike, gt_path: PathLike,
                         pred_scheme: str, gt_scheme: str,
                         tau: float = DEFAULT_TAU_MM,
                         offset_pred: int = 0, offset_gt: int = 0) -> List[Dict]:
    """Per-structure metric dicts for one (prediction, GT) pair.

    The prediction is resampled onto the GT voxel grid by WORLD coordinates
    (nearest / order 0, so labels stay discrete) whenever the two grids differ
    -- the SAME convention ``eval_corrector.py`` / ``compare_native.py`` use, and
    the reason the corrector masks (exported on the iso-0.5 head grid) can be
    Diced against the original-resolution GT. GT is never resampled. Every metric
    (Dice, surface, volume) is then computed on the shared GT grid, so ``vol_pred``
    and ``vol_gt`` use the same voxel volume.
    """
    import nibabel as nib
    from nibabel.processing import resample_from_to

    gimg = nib.load(str(gt_path))
    gdat = np.asarray(gimg.dataobj)
    if offset_gt:
        gdat = np.clip(gdat + offset_gt, 0, None)
    gdat = gdat.astype(np.int32)
    gspc = np.array(gimg.header.get_zooms()[:3], dtype=float)

    pimg = nib.load(str(pred_path))
    pdat = np.asarray(pimg.dataobj)
    if offset_pred:
        pdat = np.clip(pdat + offset_pred, 0, None)
    pdat = pdat.astype(np.int32)

    same_grid = (pdat.shape == gdat.shape
                 and np.allclose(np.asarray(pimg.affine, dtype=float),
                                 np.asarray(gimg.affine, dtype=float), atol=1e-3))
    if not same_grid:
        pimg_off = nib.Nifti1Image(pdat.astype(np.int16),
                                   np.asarray(pimg.affine, dtype=float))
        pres = resample_from_to(
            pimg_off, (tuple(int(x) for x in gdat.shape),
                       np.asarray(gimg.affine, dtype=float)),
            order=0, mode="constant", cval=0)
        pdat = np.asarray(pres.dataobj).astype(np.int32)

    pm = binary_structures(pdat, pred_scheme)
    gm = binary_structures(gdat, gt_scheme)
    vv = float(np.prod(gspc))   # both masks now live on the GT grid
    out = []
    for s in STRUCTURES:
        sm = surface_metrics(pm[s], gm[s], gspc, tau)
        out.append(dict(structure=s,
                        dice=compute_dice(pm[s], gm[s]),
                        vol_pred=float(pm[s].sum()) * vv,
                        vol_gt=float(gm[s].sum()) * vv, **sm))
    return out


def build_metrics_table(index: List[Dict], tau: float = DEFAULT_TAU_MM,
                        save_csv: Optional[PathLike] = None):
    """Turn a MASK_INDEX into a tidy long-format DataFrame (one row per structure).

    ``index``: list of dicts, one per (case, arm, step, mode) mask::

        {case, arm, step, mode, eff_res, pred_path, gt_path,
         pred_scheme, gt_scheme, [offset_pred=0], [offset_gt=0]}

    Writes ``save_csv`` when given and returns the DataFrame. This is the shared
    interface consumed by every ``*_summary`` driver (like the paired CSV is the
    interface for the comparison summaries).
    """
    import pandas as pd
    recs = []
    for it in index:
        rows = compute_case_metrics(it["pred_path"], it["gt_path"],
                                    it["pred_scheme"], it["gt_scheme"], tau,
                                    it.get("offset_pred", 0), it.get("offset_gt", 0))
        for r in rows:
            vg = r["vol_gt"]
            recs.append({**{k: it[k] for k in ("case", "arm", "step", "mode", "eff_res")},
                         **r,
                         "signed_pct": (100.0 * (r["vol_pred"] - vg) / vg) if vg > 0 else np.nan})
    df = pd.DataFrame.from_records(recs)
    if save_csv:
        Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(str(save_csv), index=False)
    return df


def load_metrics_df(metrics_csv: Optional[PathLike] = None,
                    mask_index: Optional[List[Dict]] = None,
                    tau: float = DEFAULT_TAU_MM):
    """Resolve a metrics DataFrame from a prebuilt CSV or a MASK_INDEX.

    Returns the DataFrame, or ``None`` when neither source is given (the caller
    then falls back to the synthetic illustrative layout).
    """
    import pandas as pd
    if metrics_csv and Path(metrics_csv).is_file():
        return pd.read_csv(str(metrics_csv))
    if mask_index:
        return build_metrics_table(mask_index, tau=tau, save_csv=None)
    return None


__all__ = [
    "STRUCTURES", "METHODS", "SCHEMES", "DEFAULT_TAU_MM",
    "compute_dice", "surface_metrics", "load_labelmap", "binary_structures",
    "compute_case_metrics", "build_metrics_table", "load_metrics_df",
]
