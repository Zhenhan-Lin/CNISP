"""Label-scheme resolution + remapping for the corrector experiment.

Thin wrapper over ``nnunet/data_prep/resolve_gt.py`` (the single source of truth
for per-source GT paths and label schemes). Two GT schemes exist:

  * labelfusion (atlas): {ON:1, Recti:3, Globe:5, Fat:7}, possibly -1000 offset
  * nnunet (chk_*):      {ON:1, Recti:2, Globe:3, Fat:4}

``resolve_gt`` returns ``gt_struct_to_value`` already accounting for the offset,
so the SAME map drives extraction from both the GT label and the CNISP native
mask (which is remapped back to the source's original scheme).

Requires the repo root on sys.path (see lib.config.add_repo_to_syspath).
Depends on numpy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

# resolve_gt is the authority; NNUNET_LABELS is the fixed nnUNet target scheme.
from nnunet.data_prep.resolve_gt import (  # noqa: E402
    NNUNET_LABELS,
    SourceInfo,
    build_struct_to_value,
    resolve_sources,
)


def resolve_source_infos(cfg: Dict, source_ids: List[str]) -> Dict[str, SourceInfo]:
    """Resolve GT path + label scheme for each source_id via resolve_gt.

    Builds a tiny in-memory casefile from the source_ids (expanding each to its
    eyes by reading the alignment metadata dir), then calls resolve_sources in
    GT-only mode (no CT/pivot lookup needed; ch0 comes from the degraded input).
    """
    res = cfg["_resolved"]
    meta_dir: Path = res["metadata_dir"]
    wanted = set(source_ids)

    # Expand source_ids -> casenames by globbing the metadata sidecars.
    grouped: Dict[str, List[str]] = {}
    for sid in source_ids:
        eyes = sorted(p.stem for p in meta_dir.glob(f"{sid}_O*.json"))
        if not eyes:
            raise FileNotFoundError(
                f"no alignment metadata for source {sid!r} under {meta_dir} "
                f"(expected {sid}_OD.json / {sid}_OS.json)"
            )
        grouped[sid] = eyes

    # Write a temp casefile resolve_sources can parse.
    tmp_casefile = res["staging_root"] / "_resolve_cases.tmp.txt"
    tmp_casefile.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_casefile, "w") as f:
        for sid in sorted(grouped):
            for cn in grouped[sid]:
                f.write(cn + "\n")

    sources, missing = resolve_sources(
        test_cases_path=tmp_casefile,
        meta_dir=meta_dir,
        detect_atlas_offset=True,   # needed for correct gt_struct_to_value
        resolve_ct=False,           # ch0 = degraded input, not original CT
        require_ct=False,
    )
    if missing:
        raise RuntimeError(
            "resolve_gt could not resolve: " + "; ".join(missing)
        )
    out = {s.source_id: s for s in sources if s.source_id in wanted}
    found = set(out)
    if found != wanted:
        raise RuntimeError(
            f"resolve_gt returned {sorted(found)} but expected {sorted(wanted)}"
        )
    return out


def nnunet_value(structure: str) -> int:
    """nnUNet target label value for a structure name (ON/Recti/Globe/Fat)."""
    return NNUNET_LABELS[structure]


def remap_to_nnunet(
    arr: np.ndarray, struct_to_value: Dict[str, int], structures: List[str]
) -> np.ndarray:
    """Remap a label array (in its source scheme) to the nnUNet {1,2,3,4} scheme.

    Everything not matching a known structure value becomes background (0).
    """
    out = np.zeros_like(arr, dtype=np.uint8)
    for name in structures:
        src_val = struct_to_value[name]
        out[arr == src_val] = NNUNET_LABELS[name]
    return out


def detect_scheme_and_offset(arr: np.ndarray):
    """Infer (scheme, offset) from a native label array's value set.

    Background = the most frequent value. Foreground "bases" = fg values minus
    background. labelfusion -> bases subset of {1,3,5,7} (and contains 5 or 7);
    nnunet -> bases subset of {1,2,3,4}. The offset is the background value
    (e.g. -1000 for atlas-offset volumes, 0 otherwise).
    """
    vals, counts = np.unique(arr, return_counts=True)
    bg = int(vals[int(np.argmax(counts))])
    bases = sorted({int(v) - bg for v in vals if int(v) != bg})
    bset = set(bases)
    if bset <= {1, 3, 5, 7} and (bset & {5, 7}):
        return "labelfusion", bg
    if bset <= {1, 2, 3, 4}:
        return "nnunet", bg
    raise ValueError(
        f"cannot infer label scheme from foreground bases {bases} (bg={bg}); "
        f"pass scheme explicitly"
    )


def remap_native_to_nnunet(
    arr: np.ndarray,
    structures: List[str],
    scheme: str = "auto",
    offset: int = None,
):
    """Remap a NATIVE-space mask (any scheme/offset) to nnUNet {1,2,3,4} BY NAME.

    Returns (remapped_uint8, scheme, offset). Remapping by structure NAME (not a
    value shift) is mandatory because CNISP's canonical ordering differs from the
    nnUNet ordering (e.g. canonical 2=Globe but nnUNet 2=Recti).
    """
    if scheme == "auto":
        scheme, off = detect_scheme_and_offset(arr)
        if offset is None:
            offset = off
    elif offset is None:
        mn = int(arr.min())
        offset = mn if mn < 0 else 0
    stv = build_struct_to_value(scheme, offset)  # name -> source value
    out = remap_to_nnunet(arr, stv, structures)
    return out, scheme, offset
