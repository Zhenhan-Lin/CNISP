"""Convert MRI-convention atlas labels into the CT label convention.

The FLAIR / T2 atlases use the SAME raw value sets as CT but assign them to
DIFFERENT structures:

    CT  order:  ON, Recti, Globe, Fat
    MRI order:  ON, Globe, Fat, Recti

(in either the consecutive {1,2,3,4} or odd {1,3,5,7} form, optionally with the
atlas -1000 offset). This module rewrites an MRI label volume into the CT
convention ONCE, up front, so the canonical-align step
(``data_prep/canonical_align.py``) only ever sees CT-convention masks and makes
no modality decision. After conversion the file is indistinguishable from a
native CT atlas label, so every CT-assuming consumer (canonical_align,
resolve_gt, the corrector) stays correct.

Per structure, value-by-form:
                consecutive        odd
    structure   CT   MRI           CT   MRI
    ON           1    1             1    1
    Globe        3    2             5    3
    Fat          4    3             7    5
    Recti        2    4             3    7
"""

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

# structure -> raw value, per form, per convention.
_FORMS = {
    "consecutive": {
        "ct":  {"ON": 1, "Recti": 2, "Globe": 3, "Fat": 4},
        "mri": {"ON": 1, "Globe": 2, "Fat": 3, "Recti": 4},
    },
    "odd": {
        "ct":  {"ON": 1, "Recti": 3, "Globe": 5, "Fat": 7},
        "mri": {"ON": 1, "Globe": 3, "Fat": 5, "Recti": 7},
    },
}


def _detect_form(shifted_values: set) -> str:
    """Decide consecutive vs odd from the (offset-removed) positive values."""
    if shifted_values & {5, 7}:
        return "odd"
    if shifted_values & {2, 4}:
        return "consecutive"
    raise ValueError(
        f"relabel_mri: cannot determine label form from values "
        f"{sorted(shifted_values)} (need a 5/7 for odd or a 2/4 for "
        f"consecutive). Refusing to guess."
    )


def convert_mri_array_to_ct(
    arr: np.ndarray, drop_labels: Iterable[int] = (),
) -> Tuple[np.ndarray, str]:
    """Return ``(ct_convention_array, form)`` for one MRI label volume.

    ``drop_labels`` are raw values set to 0 (background) BEFORE anything else --
    used for the T2 atlas's mislabeled raw ``2`` (the T2 masks are the odd
    {1,3,5,7} form, so a ``2`` is always spurious there).
    """
    out_dtype = arr.dtype if np.issubdtype(arr.dtype, np.integer) else np.int32
    arr = arr.astype(np.int32, copy=True)

    for d in drop_labels:
        arr[arr == int(d)] = 0

    labels = set(np.unique(arr).tolist()) - {0}
    if not labels:
        return arr.astype(out_dtype, copy=False), "empty"

    offset = min(labels)
    offset = offset if offset < 0 else 0
    shifted = {v - offset for v in labels} - {0}

    form = _detect_form(shifted)
    mri = _FORMS[form]["mri"]
    ct = _FORMS[form]["ct"]

    # Capture masks from the ORIGINAL array first so moving a structure onto a
    # value still occupied by another structure can't clobber it.
    masks = {name: (arr == offset + mri[name]) for name in mri}
    out = arr.copy()
    for name, m in masks.items():
        out[m] = offset + ct[name]
    return out.astype(out_dtype, copy=False), form


def convert_mri_file_to_ct(
    src_path, dst_path, drop_labels: Iterable[int] = (),
) -> str:
    """Load an MRI label NIfTI, convert to CT convention, save to ``dst_path``.

    Returns the detected ``form``. Affine/header are preserved so the converted
    file is a drop-in native-space replacement.
    """
    import nibabel as nib  # local import: only needed on conversion hosts

    src_path = Path(src_path)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    img = nib.load(str(src_path))
    arr = np.asarray(img.dataobj).astype(np.int32, copy=False)
    out, form = convert_mri_array_to_ct(arr, drop_labels=drop_labels)

    out_img = nib.Nifti1Image(out, img.affine, img.header)
    nib.save(out_img, str(dst_path))
    return form
