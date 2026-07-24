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

import os
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

    def __init__(self, *args, planner, preprocessed_folder, require_clf: bool = True,
                 log_every: int = 200, **kwargs):
        # NOTE: do not pass ``strata`` — FOV strata come from the planner, not steps.
        kwargs.pop("strata", None)
        super().__init__(*args, **kwargs)        # type: ignore[misc]
        self.planner = planner
        self.preprocessed_folder = Path(preprocessed_folder)
        self.require_clf = bool(require_clf)     # §10: fail-closed by default
        self._log_every = int(log_every)
        self._clf_cache: Dict[str, dict] = {}
        self._slots = None                       # per selected key, in order
        self._bbox_call = 0
        # per-process RNG init is deferred to get_indices (§6): a value here is only
        # a placeholder; forked workers reseed with a worker-specific id.
        self._fov_pid: Optional[int] = None
        self._fov_rng = np.random.default_rng(getattr(planner, "effective_seed", 0) + 7)
        # §10 diagnostics: requested vs actual region, random-fallback, struct counts.
        self._n_batches = 0
        self._region_requested: Dict[str, int] = {"missing": 0, "seam": 0, "visible": 0, "random": 0}
        self._region_actual: Dict[str, int] = {"missing": 0, "seam": 0, "visible": 0, "random": 0}
        self._struct_counts: Dict[int, int] = {}

    # (0) per-process reseed (§6) ---------------------------------------------
    def _ensure_process_rng(self) -> None:
        """The NonDet augmenter FORKS a copy of this loader into each worker process;
        without a per-process reseed every worker would replay the planner's identical
        stream (review §6). On the first call inside a new process, derive a
        worker-specific id and reseed the planner + the structure-selection RNG. PID
        is logged so the (otherwise non-deterministic) worker seed is recorded."""
        pid = os.getpid()
        if self._fov_pid == pid:
            return
        self._fov_pid = pid
        # worker id from PID (recorded below). batchgenerators' NonDet augmenter is
        # already non-deterministic across workers; we only need DISTINCT streams.
        worker_id = pid % 100_003
        eff = self.planner.reseed(worker_id=worker_id)
        self._fov_rng = np.random.default_rng(eff + 7)
        print(f"[fov-loader] pid={pid} worker_id={worker_id} planner_effective_seed={eff}",
              flush=True)

    # (1) case selection ------------------------------------------------------
    def get_indices(self):
        self._ensure_process_rng()
        self._slots = self.planner.make_plan()
        self._bbox_call = 0
        self._n_batches += 1
        if self._log_every and self._n_batches % self._log_every == 0:
            self._log_fov_stats()
        return [s.case_id for s in self._slots]

    def fov_stats(self) -> dict:
        req_tot = max(1, sum(self._region_requested.values()))
        return {
            "pid": self._fov_pid,
            "n_batches": self._n_batches,
            "region_requested": dict(self._region_requested),
            "region_actual": dict(self._region_actual),
            "random_fallback_rate": round(self._region_actual["random"] / req_tot, 4),
            "struct_counts": {int(k): v for k, v in sorted(self._struct_counts.items())},
        }

    def _log_fov_stats(self) -> None:
        print(f"[fov-loader] stats {self.fov_stats()}", flush=True)

    # (2) region/structure-aware patch center ---------------------------------
    def _class_locations_fov(self, case_id: str) -> dict:
        if case_id not in self._clf_cache:
            p = self.preprocessed_folder / f"{case_id}.pkl"
            if not p.exists():
                if self.require_clf:
                    raise FileNotFoundError(
                        f"[fov-loader] properties file missing for non-anchor case "
                        f"{case_id!r}: {p}. Run write_class_locations_fov.py (run script "
                        f"step 4b) before training, or set require_clf=False to fall back "
                        f"to stock sampling (NOT recommended — disables FOV supervision).")
                self._clf_cache[case_id] = {}
                return self._clf_cache[case_id]
            with open(p, "rb") as f:
                props = pickle.load(f) or {}
            if "class_locations_fov" not in props:
                if self.require_clf:
                    raise RuntimeError(
                        f"[fov-loader] {p} has no 'class_locations_fov' for non-anchor "
                        f"case {case_id!r}; the post-pass did not run for this case.")
                self._clf_cache[case_id] = {}
                return self._clf_cache[case_id]
            raw = props.get("class_locations_fov") or {}
            # keys may be str after JSON round-trip; normalize struct keys to int
            clf = {r: {int(k): np.asarray(v) for k, v in (d or {}).items()}
                   for r, d in raw.items()}
            if self.require_clf and not any(
                    len(c) for region in clf.values() for c in region.values()):
                raise RuntimeError(
                    f"[fov-loader] class_locations_fov for non-anchor case {case_id!r} is "
                    f"entirely empty (no missing/seam/visible centers); projection likely "
                    f"failed for this case.")
            self._clf_cache[case_id] = clf
        return self._clf_cache[case_id]

    @staticmethod
    def _to_nnunet_class_locations(coords_zyx: np.ndarray) -> np.ndarray:
        """Stock nnU-Net class_locations are (N, dims+1) = [seg_channel, z, y, x] and
        get_bbox reads selected_voxel[i+1] for each spatial axis (review §5). Our
        internal pools are (N,3)=[z,y,x]; prepend a zero seg-channel column."""
        c = np.asarray(coords_zyx, dtype=np.int32)
        if c.ndim != 2 or c.shape[1] != 3:
            raise ValueError(f"expected FOV centers with shape (N,3); got {c.shape}.")
        return np.concatenate([np.zeros((len(c), 1), dtype=c.dtype), c], axis=1)

    def get_bbox(self, data_shape, force_fg, class_locations, overwrite_class=None,
                 *args, **kwargs):
        # NOTE: signature mirrors nnU-Net v2.5's get_bbox; if your installed
        # version differs, only this wrapper needs adjusting.
        slot = None
        if self._slots is not None and self._bbox_call < len(self._slots):
            slot = self._slots[self._bbox_call]
        self._bbox_call += 1

        if slot is not None and not slot.is_anchor and class_locations is not None:
            self._region_requested[slot.region] = self._region_requested.get(slot.region, 0) + 1
            clf = self._class_locations_fov(slot.case_id)
            struct, coords, region = select_region_structure(clf, slot.region, self._fov_rng)
            self._region_actual[region] = self._region_actual.get(region, 0) + 1
            if struct is not None and coords is not None and len(coords):
                self._struct_counts[int(struct)] = self._struct_counts.get(int(struct), 0) + 1
                # hand stock get_bbox a region-restricted, single-structure pool (in
                # the (N,4) class_locations format) so it does the bounds clamp; we
                # only steer WHICH coordinate is sampled. overwrite_class passed
                # POSITIONALLY (like the fallback) to avoid a kw/positional clash.
                region_cl = {int(struct): self._to_nnunet_class_locations(coords)}
                return super().get_bbox(data_shape, True, region_cl, int(struct),
                                        *args, **kwargs)
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

    # (N,3) -> (N,4) nnU-Net class_locations conversion (§5): prepend a zero
    # seg-channel column so stock get_bbox's selected_voxel[i+1] reads z,y,x.
    conv = FOVCompletionStratifiedDataLoader._to_nnunet_class_locations
    zyx = np.array([[10, 20, 30], [1, 2, 3]], dtype=np.int32)
    out = conv(zyx)
    assert out.shape == (2, 4) and np.array_equal(out[:, 0], [0, 0])
    assert np.array_equal(out[:, 1:], zyx)              # spatial coords at index 1..3
    for bad in (np.zeros((5, 4), np.int32), np.zeros((5,), np.int32)):
        try:
            conv(bad)
            raise AssertionError("should reject non-(N,3)")
        except ValueError:
            pass
    print("class_locations conversion OK: (N,3)[z,y,x] -> (N,4)[c,z,y,x]")

    print("FOV COMPLETION LOADER SELF-TEST PASSED (region/structure selection + coords)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
