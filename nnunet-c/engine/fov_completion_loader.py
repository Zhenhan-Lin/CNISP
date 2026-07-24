"""
FOV-completion data loader — a THIN extension of the proven corrector loader
(re-audit §7-8). It changes only:

  1. which cases are selected      -> get_indices() delegates to FOVCompletionBatchPlanner
  2. where each patch is centered   -> region + structure aware center, from
                                       properties['class_locations_fov']

Everything else — case/seg/seg_prev loading, cropping, seg_prev concatenation,
MoveSegAsOneHotToDataTransform (one-hot prior into channels 1-4), target
extraction, deep supervision — is INHERITED unchanged from
StepStratifiednnUNetDataLoader / the stock nnU-Net loader. The FOV loader never
touches channel assembly or label handling.

Center injection reuses the stock ``get_bbox`` clamp: for a region slot we hand
``super().get_bbox`` a class_locations dict containing only the chosen structure's
region coordinates (+ overwrite_class), so nnU-Net does the bounds math and we
only steer WHICH pool it samples. The anchor/random slot uses stock behavior.

The pure region/structure selection (``select_region_structure``) is unit-tested
here; the loader class needs the installed nnU-Net (masi-55).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# Region fallback order (plan §7.1-7.3): requested region first, then degrade.
_FALLBACK = {
    "missing": ("missing", "seam", "visible", "random"),
    "seam": ("seam", "missing", "visible", "random"),
    "visible": ("visible", "seam", "missing", "random"),
    "random": ("random",),
}


def select_region_structure(
    class_locations_fov: Dict[str, Dict[int, np.ndarray]],
    requested_region: str,
    rng: np.random.Generator,
) -> Tuple[Optional[int], Optional[np.ndarray], str]:
    """Pick (struct_value, coords, actual_region) for a requested region.

    Structure is chosen ~uniformly over the structures that HAVE eligible centers
    in the region (re-audit §3.3), so small structures (ON/Recti) aren't drowned
    by Globe/Fat. Falls through the region order; ``("random")`` -> (None, None,
    "random") so the caller uses stock foreground/random behavior.
    """
    for region in _FALLBACK.get(requested_region, ("random",)):
        if region == "random":
            return None, None, "random"
        pools = class_locations_fov.get(region, {}) or {}
        available = [int(s) for s, c in pools.items() if len(c) > 0]
        if not available:
            continue
        struct = available[int(rng.integers(len(available)))]
        return struct, np.asarray(pools[struct]), region
    return None, None, "random"


# ── the loader class (needs the installed nnU-Net) ────────────────────────────
try:                                            # installed into nnunetv2.training.nnUNetTrainer.*
    from corrector_stratified_loader import StepStratifiednnUNetDataLoader as _Base
except Exception:                               # noqa: BLE001
    try:
        from engine.corrector_stratified_loader import StepStratifiednnUNetDataLoader as _Base
    except Exception:                           # noqa: BLE001
        _Base = object                          # allows import for the pure self-test


class FOVCompletionStratifiedDataLoader(_Base):  # type: ignore[misc,valid-type]
    """Thin FOV subclass. Requires: ``planner`` (FOVCompletionBatchPlanner) and
    ``preprocessed_folder`` (to read each case's class_locations_fov). All other
    args are forwarded to the corrector/stock loader unchanged."""

    def __init__(self, *args, planner, preprocessed_folder, **kwargs):
        # NOTE: do not pass ``strata`` — FOV strata come from the planner, not steps.
        kwargs.pop("strata", None)
        super().__init__(*args, **kwargs)        # type: ignore[misc]
        self.planner = planner
        self.preprocessed_folder = Path(preprocessed_folder)
        self._clf_cache: Dict[str, dict] = {}
        self._slots = None                       # per selected key, in order
        self._bbox_call = 0
        self._fov_rng = np.random.default_rng(getattr(planner, "effective_seed", 0) + 7)

    # (1) case selection ------------------------------------------------------
    def get_indices(self):
        self._slots = self.planner.make_plan()
        self._bbox_call = 0
        return [s.case_id for s in self._slots]

    # (2) region/structure-aware patch center ---------------------------------
    def _class_locations_fov(self, case_id: str) -> dict:
        if case_id not in self._clf_cache:
            p = self.preprocessed_folder / f"{case_id}.pkl"
            clf = {}
            if p.exists():
                with open(p, "rb") as f:
                    clf = (pickle.load(f) or {}).get("class_locations_fov", {}) or {}
            # keys may be str after JSON round-trip; normalize struct keys to int
            clf = {r: {int(k): np.asarray(v) for k, v in (d or {}).items()}
                   for r, d in clf.items()}
            self._clf_cache[case_id] = clf
        return self._clf_cache[case_id]

    def get_bbox(self, data_shape, force_fg, class_locations, overwrite_class=None,
                 *args, **kwargs):
        # NOTE: signature mirrors nnU-Net v2.5's get_bbox; if your installed
        # version differs, only this wrapper needs adjusting.
        slot = None
        if self._slots is not None and self._bbox_call < len(self._slots):
            slot = self._slots[self._bbox_call]
        self._bbox_call += 1

        if slot is not None and not slot.is_anchor and class_locations is not None:
            clf = self._class_locations_fov(slot.case_id)
            struct, coords, _region = select_region_structure(clf, slot.region, self._fov_rng)
            if struct is not None and coords is not None and len(coords):
                # hand stock get_bbox a region-restricted, single-structure pool so
                # it does the bounds clamp; we only steer the sampled coordinate.
                region_cl = {struct: coords}
                return super().get_bbox(data_shape, True, region_cl,
                                        overwrite_class=struct, *args, **kwargs)
        # anchor / random / fallback -> stock behavior
        return super().get_bbox(data_shape, force_fg, class_locations, overwrite_class,
                                *args, **kwargs)


# ── self-test (pure selection logic; no nnU-Net) ──────────────────────────────
def _selftest() -> int:
    rng = np.random.default_rng(0)

    def pool(n):
        return np.arange(n * 3).reshape(n, 3).astype(np.int32)

    # all regions populated, per-structure
    clf = {"missing": {1: pool(5), 2: pool(0), 3: pool(50), 4: pool(20)},
           "seam": {1: pool(3), 3: pool(10)},
           "visible": {1: pool(4), 2: pool(4), 3: pool(4), 4: pool(4)}}

    # requested missing -> a missing structure with coords, never the empty struct 2
    chosen = [select_region_structure(clf, "missing", rng)[0] for _ in range(2000)]
    assert set(chosen) <= {1, 3, 4} and 2 not in chosen, set(chosen)
    # ~uniform over the 3 available structures (Globe not dominating despite 50 vs 5)
    import collections
    freq = collections.Counter(chosen)
    for s in (1, 3, 4):
        assert 0.25 < freq[s] / 2000 < 0.42, freq
    print("missing struct freq:", {k: round(v / 2000, 3) for k, v in sorted(freq.items())})

    # empty missing -> fall back to seam
    clf2 = {"missing": {1: pool(0)}, "seam": {3: pool(9)}, "visible": {1: pool(2)}}
    s, c, reg = select_region_structure(clf2, "missing", rng)
    assert reg == "seam" and s == 3 and len(c) == 9

    # all empty -> random
    s, c, reg = select_region_structure({"missing": {}, "seam": {}, "visible": {}}, "missing", rng)
    assert reg == "random" and s is None and c is None

    print("FOV COMPLETION LOADER SELF-TEST PASSED (region/structure selection)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
