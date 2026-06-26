"""Merge Dataset835's plan into a freshly-planned 855/845 plan (potholes 1 & 3).

855/845 are new 5-channel datasets, so nnUNet recomputes their fingerprint:
different CT intensity stats (pothole 1) and possibly a different target spacing
(pothole 3) than the 835 weights we finetune from. We CANNOT use raw
`move_plans_between_datasets` (it copies the 1-channel plan wholesale and would
leave 1-entry per-channel lists -> preprocess crash). Instead we keep the target
plan's valid 5-channel per-channel lists and OVERRIDE only the fields that must
equal 835:

  * foreground_intensity_properties_per_channel["0"]  (ch0 CT normalization)
  * configurations[cfg].spacing + top-level original_median_spacing_after_transp
  * configurations[cfg].architecture + patch_size  (so weights transfer)
  * transpose_forward/backward                       (spacing semantics)
  * normalization_schemes[0] == CTNormalization      (sanity)

This pairs with pothole-2 (a-ii): with the plan spacing == 835, the builder's
pre-resampled data matches and nnUNet's resample is a no-op.

Pure JSON manipulation; no nnunetv2 import. Caller dumps before/after.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_plan(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


def save_plan(plan: Dict, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(plan, f, indent=2, sort_keys=False)


def merge_finetune_plan(
    ref_plan: Dict, target_plan: Dict, configuration: str
) -> Tuple[Dict, List[str]]:
    """Return (merged_plan, overrides) = target_plan with 835 fields injected."""
    merged = copy.deepcopy(target_plan)
    overrides: List[str] = []

    def _set_top(key: str):
        if key in ref_plan:
            merged[key] = copy.deepcopy(ref_plan[key])
            overrides.append(f"top.{key}")

    # ── ch0 CT intensity stats (pothole 1) ──────────────────────────
    ref_fip = ref_plan.get("foreground_intensity_properties_per_channel", {})
    if "0" in ref_fip:
        merged.setdefault("foreground_intensity_properties_per_channel", {})
        merged["foreground_intensity_properties_per_channel"]["0"] = \
            copy.deepcopy(ref_fip["0"])
        overrides.append("top.foreground_intensity_properties_per_channel['0']")

    # ── spacing semantics (pothole 3) ───────────────────────────────
    _set_top("original_median_spacing_after_transp")
    _set_top("transpose_forward")
    _set_top("transpose_backward")

    # ── configuration-level: spacing + architecture (potholes 1 & 3) ─
    ref_cfgs = ref_plan.get("configurations", {})
    if configuration not in ref_cfgs:
        raise KeyError(
            f"reference plan has no configuration {configuration!r} "
            f"(has {sorted(ref_cfgs)})"
        )
    if configuration not in merged.get("configurations", {}):
        raise KeyError(
            f"target plan has no configuration {configuration!r} "
            f"(has {sorted(merged.get('configurations', {}))})"
        )
    rc = ref_cfgs[configuration]
    mc = merged["configurations"][configuration]
    # NOTE: do NOT copy per-channel lists (use_mask_for_norm / normalization_schemes)
    # from the reference -- 835 is 1-channel so they have a single entry and would
    # break the 5-channel preprocessor (_normalize loops over all channels and
    # indexes use_mask_for_norm[c] -> IndexError). Keep the target's 5-entry lists.
    for key in ("spacing", "patch_size", "architecture",
                "median_image_size_in_voxels"):
        if key in rc:
            mc[key] = copy.deepcopy(rc[key])
            overrides.append(f"configurations.{configuration}.{key}")

    # ── normalization schemes: keep target's 5 entries; pin ch0 ─────
    nschemes = mc.get("normalization_schemes")
    if isinstance(nschemes, list) and nschemes:
        if nschemes[0] != "CTNormalization":
            nschemes[0] = "CTNormalization"
            overrides.append(
                f"configurations.{configuration}.normalization_schemes[0]=CTNormalization"
            )

    return merged, overrides


def merge_plan_files(
    ref_plan_path: Path,
    target_plan_path: Path,
    configuration: str,
    out_path: Path,
    dump_dir: Path = None,
) -> Dict:
    """Load both plans, merge, write out, and dump before/after for diffing."""
    ref = load_plan(ref_plan_path)
    target = load_plan(target_plan_path)
    merged, overrides = merge_finetune_plan(ref, target, configuration)

    dump_dir = Path(dump_dir or Path(out_path).parent)
    save_plan(target, dump_dir / "plan_before.json")
    save_plan(merged, dump_dir / "plan_after.json")
    save_plan(merged, out_path)

    report = {
        "ref_plan": str(ref_plan_path),
        "target_plan": str(target_plan_path),
        "out_plan": str(out_path),
        "configuration": configuration,
        "overrides": overrides,
        "plan_before": str(dump_dir / "plan_before.json"),
        "plan_after": str(dump_dir / "plan_after.json"),
    }
    return report
