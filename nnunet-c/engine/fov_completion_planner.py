"""
FOV-completion batch planner (implementation-plan §10-11) + the review's required
modifications:

  * true 50/30/20 region ratio via a deterministic quota scheduler, not the
    42/33/25 fixed-pattern cycle (review §4.2);
  * rank/worker-aware seeding so DDP ranks and data-loader workers don't replay
    the same stream (review §4.3);
  * serializable planner state for exact resume (iteration + RNG + quota queue)
    (review §4.4);
  * subject-uniqueness / fallback statistics, and never fail on a temporary
    shortage of distinct subjects (review §4.5, §5.1).

Pure sampling logic — NO nnU-Net import — so it is unit-testable in isolation.
``severity`` is the plan's "% removed" label {20,35,50}; the keep-fraction mapping
lives in data-gen, never here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

CropType = str
RegionType = str


@dataclass(frozen=True)
class FOVCondition:
    crop_type: CropType
    severity: int


@dataclass
class BatchSlot:
    slot_index: int
    case_id: str
    subject_id: str
    region: RegionType
    condition: Optional[FOVCondition]
    is_anchor: bool = False
    subject_reused: bool = False   # True if a distinct subject wasn't available


class FOVCaseIndex:
    REQUIRED_CONDITIONS: Tuple[FOVCondition, ...] = (
        FOVCondition("axial", 20), FOVCondition("axial", 35), FOVCondition("axial", 50),
        FOVCondition("corner", 20), FOVCondition("corner", 35), FOVCondition("corner", 50),
    )

    def __init__(self, manifest_records: Sequence[dict]):
        self.by_condition: Dict[FOVCondition, List[str]] = {
            c: [] for c in self.REQUIRED_CONDITIONS}
        self.full_fov_cases: List[str] = []
        self.full_fov_set: Set[str] = set()          # O(1) metadata-based anchor test
        self.subject_by_case: Dict[str, str] = {}
        for rec in manifest_records:
            case_id = rec["case_id"]
            if case_id in self.subject_by_case:      # re-audit §10.2: no duplicates
                raise ValueError(f"FOVCaseIndex: duplicate case_id {case_id!r}.")
            self.subject_by_case[case_id] = str(rec["subject_id"])
            if rec.get("is_full_fov"):
                self.full_fov_cases.append(case_id)
                self.full_fov_set.add(case_id)
                continue
            cond = FOVCondition(rec["crop_type"], int(rec["severity"]))
            if cond not in self.by_condition:        # re-audit §10.2: reject, don't drop
                raise ValueError(f"FOVCaseIndex: unsupported condition {cond} for case "
                                 f"{case_id!r} (allowed: axial/corner x 20/35/50).")
            self.by_condition[cond].append(case_id)
        self._validate()

    def _validate(self):
        if not self.full_fov_cases:
            raise ValueError("FOVCaseIndex: no full-FOV anchor cases found.")
        empty = [c for c, cs in self.by_condition.items() if not cs]
        if empty:
            raise ValueError(f"FOVCaseIndex: empty FOV condition strata: {empty}")

    def _sample(self, pool: Sequence[str], rng, exclude_subjects) -> Tuple[str, bool]:
        """Return (case_id, subject_reused). Prefers a subject not in
        ``exclude_subjects``; falls back to any (flagged) rather than failing."""
        exclude = exclude_subjects or set()
        preferred = [c for c in pool if self.subject_by_case[c] not in exclude]
        if preferred:
            return preferred[int(rng.integers(len(preferred)))], False
        return list(pool)[int(rng.integers(len(pool)))], True

    def sample_case(self, condition, rng, exclude_subjects=None) -> Tuple[str, bool]:
        return self._sample(self.by_condition[condition], rng, exclude_subjects)

    def sample_full_fov(self, rng, exclude_subjects=None) -> Tuple[str, bool]:
        return self._sample(self.full_fov_cases, rng, exclude_subjects)


class RegionQuotaScheduler:
    """Deterministic 50/30/20 region scheduler (review §4.2). Every 10 batches =
    30 condition slots -> 15 missing / 9 seam / 6 visible, shuffled locally."""

    QUOTA = ("missing",) * 15 + ("seam",) * 9 + ("visible",) * 6   # per 10 batches

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.queue: List[str] = []

    def _refill(self):
        q = list(self.QUOTA)
        self.rng.shuffle(q)
        self.queue = q

    def next_pattern(self) -> Tuple[RegionType, RegionType, RegionType]:
        if len(self.queue) < 3:
            # keep leftovers, then top up (preserves exact long-run proportions)
            leftover = self.queue
            self._refill()
            self.queue = leftover + self.queue
        pattern = tuple(self.queue[:3])
        del self.queue[:3]
        return pattern  # type: ignore[return-value]


@dataclass
class PlannerStats:
    n_plans: int = 0
    anchor_full: int = 0
    unique_subjects_sum: int = 0
    subject_reuse_events: int = 0
    region_counts: Dict[str, int] = field(default_factory=lambda: {"missing": 0, "seam": 0, "visible": 0})
    condition_counts: Dict[Tuple[str, int], int] = field(default_factory=dict)

    def summary(self) -> dict:
        n = max(1, self.n_plans)
        return {
            "n_plans": self.n_plans,
            "anchor_full_rate": round(self.anchor_full / n, 4),
            "mean_unique_subjects": round(self.unique_subjects_sum / n, 4),
            "subject_reuse_events": self.subject_reuse_events,
            "region_frac": {k: round(v / max(1, sum(self.region_counts.values())), 4)
                            for k, v in self.region_counts.items()},
            "condition_counts": {f"{ct}_{sv}": c for (ct, sv), c in sorted(self.condition_counts.items())},
        }


class FOVCompletionBatchPlanner:
    CONDITION_PATTERNS: Tuple[Tuple[FOVCondition, ...], ...] = (
        (FOVCondition("axial", 20), FOVCondition("corner", 35), FOVCondition("axial", 50)),
        (FOVCondition("corner", 20), FOVCondition("axial", 35), FOVCondition("corner", 50)),
        (FOVCondition("corner", 50), FOVCondition("axial", 20), FOVCondition("corner", 35)),
        (FOVCondition("axial", 50), FOVCondition("corner", 20), FOVCondition("axial", 35)),
    )

    def __init__(self, case_index: FOVCaseIndex, full_fov_anchor_probability: float = 0.5,
                 base_seed: int = 12345, global_rank: int = 0, worker_id: int = 0):
        self.case_index = case_index
        if not (0.0 <= float(full_fov_anchor_probability) <= 1.0):
            raise ValueError(f"full_fov_anchor_probability must be in [0,1]; got "
                             f"{full_fov_anchor_probability}")
        self.full_fov_anchor_probability = float(full_fov_anchor_probability)
        # rank/worker-aware seed (review §4.3); apply ONCE (loader must not re-offset)
        self.effective_seed = int(base_seed) + 100_003 * int(global_rank) + 1_009 * int(worker_id)
        self.rng = np.random.default_rng(self.effective_seed)
        self.region_scheduler = RegionQuotaScheduler(self.rng)
        self.iteration = 0
        self.stats = PlannerStats()

    def make_plan(self) -> List[BatchSlot]:
        cond_pattern = self.CONDITION_PATTERNS[self.iteration % len(self.CONDITION_PATTERNS)]
        region_pattern = self.region_scheduler.next_pattern()
        self.iteration += 1

        used: Set[str] = set()
        anchor_case, anchor_reused = self._sample_anchor(used)
        anchor_subject = self.case_index.subject_by_case[anchor_case]
        used.add(anchor_subject)
        slots = [BatchSlot(0, anchor_case, anchor_subject, "random", None,
                           is_anchor=True, subject_reused=anchor_reused)]

        for i, (cond, region) in enumerate(zip(cond_pattern, region_pattern), start=1):
            case_id, reused = self.case_index.sample_case(cond, self.rng, exclude_subjects=used)
            subject_id = self.case_index.subject_by_case[case_id]
            used.add(subject_id)
            slots.append(BatchSlot(i, case_id, subject_id, region, cond,
                                   is_anchor=False, subject_reused=reused))

        self._record(slots)
        return slots

    def _sample_anchor(self, used: Set[str]) -> Tuple[str, bool]:
        if self.rng.random() < self.full_fov_anchor_probability:
            return self.case_index.sample_full_fov(self.rng, exclude_subjects=used)
        conds = list(self.case_index.by_condition.keys())
        cond = conds[int(self.rng.integers(len(conds)))]
        return self.case_index.sample_case(cond, self.rng, exclude_subjects=used)

    def _record(self, slots: List[BatchSlot]):
        st = self.stats
        st.n_plans += 1
        if self._is_full(slots[0].case_id):
            st.anchor_full += 1
        st.unique_subjects_sum += len({s.subject_id for s in slots})
        st.subject_reuse_events += sum(1 for s in slots if s.subject_reused)
        for s in slots[1:]:
            st.region_counts[s.region] = st.region_counts.get(s.region, 0) + 1
            key = (s.condition.crop_type, s.condition.severity)
            st.condition_counts[key] = st.condition_counts.get(key, 0) + 1

    def _is_full(self, case_id: str) -> bool:
        # re-audit §10.2: decide anchor-full from index metadata, not the filename.
        return case_id in self.case_index.full_fov_set

    # ── resume (review §4.4) ──────────────────────────────────────────────────
    def get_state(self) -> dict:
        return {"iteration": self.iteration,
                "rng_state": self.rng.bit_generator.state,
                "region_queue": list(self.region_scheduler.queue),
                "effective_seed": self.effective_seed}

    def set_state(self, state: dict) -> None:
        self.iteration = int(state["iteration"])
        self.rng.bit_generator.state = state["rng_state"]
        self.region_scheduler.queue = list(state["region_queue"])


# ── self-test ────────────────────────────────────────────────────────────────
def _synthetic_records(n_subjects: int = 8) -> List[dict]:
    recs = []
    for s in range(n_subjects):
        sid = f"{s:03d}"
        recs.append({"case_id": f"corr_{sid}_full", "subject_id": sid, "is_full_fov": True})
        for ct in ("axial", "corner"):
            for sev in (20, 35, 50):
                recs.append({"case_id": f"corr_{sid}_{ct}_rm{sev}", "subject_id": sid,
                             "crop_type": ct, "severity": sev, "is_full_fov": False})
    return recs


def _selftest() -> int:
    idx = FOVCaseIndex(_synthetic_records(8))

    # guards (re-audit §10.2): duplicate case_id, unsupported condition, bad prob
    for bad in (
        lambda: FOVCaseIndex(_synthetic_records(2)
                             + [{"case_id": "corr_000_full", "subject_id": "000", "is_full_fov": True}]),
        lambda: FOVCaseIndex([{"case_id": "x", "subject_id": "0", "crop_type": "axial",
                               "severity": 99, "is_full_fov": False}] + _synthetic_records(1)),
        lambda: FOVCompletionBatchPlanner(idx, 1.5),
    ):
        try:
            bad()
            raise AssertionError("guard should have raised")
        except ValueError:
            pass

    planner = FOVCompletionBatchPlanner(idx, 0.5, base_seed=0)

    N = 3000
    for _ in range(N):
        plan = planner.make_plan()
        assert len(plan) == 4 and plan[0].is_anchor
        assert sorted(s.condition.severity for s in plan[1:]) == [20, 35, 50]

    s = planner.stats.summary()
    print("planner stats:", s)
    # exact 50/30/20 from the quota scheduler
    assert abs(s["region_frac"]["missing"] - 0.50) < 0.01, s["region_frac"]
    assert abs(s["region_frac"]["seam"] - 0.30) < 0.01, s["region_frac"]
    assert abs(s["region_frac"]["visible"] - 0.20) < 0.01, s["region_frac"]
    assert 0.45 <= s["anchor_full_rate"] <= 0.55
    assert s["mean_unique_subjects"] > 3.9

    # rank/worker seeding -> different streams
    p0 = FOVCompletionBatchPlanner(idx, 0.5, base_seed=0, global_rank=0, worker_id=0)
    p1 = FOVCompletionBatchPlanner(idx, 0.5, base_seed=0, global_rank=1, worker_id=0)
    seq0 = [tuple(sl.case_id for sl in p0.make_plan()) for _ in range(50)]
    seq1 = [tuple(sl.case_id for sl in p1.make_plan()) for _ in range(50)]
    assert seq0 != seq1, "different ranks must not replay the same stream"

    # resume: state snapshot -> identical continuation
    pa = FOVCompletionBatchPlanner(idx, 0.5, base_seed=42)
    for _ in range(37):
        pa.make_plan()
    snap = pa.get_state()
    cont_a = [tuple(sl.case_id for sl in pa.make_plan()) for _ in range(20)]
    pb = FOVCompletionBatchPlanner(idx, 0.5, base_seed=999)   # different seed
    pb.set_state(snap)
    cont_b = [tuple(sl.case_id for sl in pb.make_plan()) for _ in range(20)]
    assert cont_a == cont_b, "set_state must reproduce the continuation exactly"

    print("FOV COMPLETION PLANNER SELF-TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
