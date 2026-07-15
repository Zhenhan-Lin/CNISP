#!/usr/bin/env python3
"""Build the FOV-truncation experiment's data/ tree (Part 2, "isolate FOV").

Produces truncation-only degraded CTs: the NATIVE CT is FOV-truncated along its
through-plane axis (a contiguous z-fraction vised with air), with NO slice
thickening -- so the only degradation is the missing field of view (the corrector
must learn to defer to the completed CNISP prior exactly where ch0 has no evidence).

To reuse the ENTIRE existing corrector pipeline unchanged (`build_corrector_dataset
--layout cascade`, the stratified loader, by-step eval), each truncation level is
encoded as a **pseudo-step** `PP = round(keep_fraction*100)` (e.g. keep 0.5 ->
`_step50`). Downstream then stratifies by FOV severity instead of by thickness --
no plumbing changes; set the trainer's strata with `CORRECTOR_STRATA="50,65,80"`.

Input: an existing `corrector_data_manifest.json` (from build_corrector_data.py),
for each case's `source_image` + `gt_candidate_pred` + `step_axis`. Output (a
SEPARATE data root, so the thickness experiment is untouched):
    <out>/images/{case}_step{PP}_0000.nii.gz   (truncated CT, ch0)
    <out>/corrector_data_manifest.json          (same schema -> build_corrector_dataset)
    <out>/fov_truncation_manifest.json          (per (case,PP): trunc axis + vised
                                                 slice range + source shape, for the
                                                 region-restricted eval)
    <out>/{nnunet_pred,cnisp_pred}/             (empty; box runs fill them)

The truncation itself reuses `nnunet.sparsify_inputs._truncate_one_ct` (next to
`_sparsify_one_ct`; no duplicate degradation code).

BOX FOLLOW-UP (see RUNBOOK_FOV.md): run the 835 stage-1 model on each truncated CT
to get a coarse seg, run CNISP 032 on that (`--steps 50,65,80`) to emit the
COMPLETED iso prior, then `build_corrector_dataset.py --layout cascade` on a FOV
config (data_root = this <out>, steps = the pseudo-steps, its own control-C id).

Usage:
    python nnunet-c/scripts/build_fov_truncated_data.py \
        --keep-fractions 0.5,0.65,0.8 --side end
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.config import add_repo_to_syspath, load_corrector_config  # noqa: E402

add_repo_to_syspath(__file__)

import numpy as np  # noqa: E402
import nibabel as nib  # noqa: E402

_DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "corrector.yaml"


def _pseudo_step(keep_fraction: float) -> int:
    pp = int(round(float(keep_fraction) * 100))
    if not (1 <= pp <= 99):
        raise ValueError(
            f"keep_fraction {keep_fraction} -> pseudo-step {pp}; must map into "
            f"1..99 (keep_fraction in (0.01, 0.99]); 1.0 = no truncation, skip it.")
    return pp


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(_DEFAULT_CONFIG))
    ap.add_argument("--src-manifest", default=None,
                    help="existing corrector_data_manifest.json (default: "
                         "<data_root>/corrector_data_manifest.json).")
    ap.add_argument("--out-data-root", default=None,
                    help="FOV data root to write (default: <data_root>_fov).")
    ap.add_argument("--keep-fractions", default="0.5,0.65,0.8",
                    help="comma list of RETAINED z-fractions, each -> pseudo-step "
                         "round(f*100). Default %(default)s -> steps 50,65,80.")
    ap.add_argument("--side", choices=["end", "start", "both", "random"],
                    default="end",
                    help="which end(s) to vis: end (superior cut-off), start "
                         "(inferior), both (centred limited FOV), or random per case.")
    ap.add_argument("--pad-value", type=float, default=None,
                    help="fill HU for vised slices (default: each CT's own min = air).")
    ap.add_argument("--max-cases", type=int, default=0, help="cap number of cases (0=all).")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for --side random.")
    ap.add_argument("--force", action="store_true", help="re-truncate even if the image exists.")
    args = ap.parse_args()

    # _truncate_one_ct lives next to _sparsify_one_ct (reused; imported lazily so
    # --help doesn't pull torch/simulation in).
    from nnunet.sparsify_inputs import _truncate_one_ct  # noqa: E402

    cfg = load_corrector_config(args.config, caller_file=__file__)
    cd = cfg["corrector_data"]
    res = cfg["_resolved"]
    data_root = Path(cd["data_root"])
    data_root = data_root if data_root.is_absolute() else (res["repo_root"] / data_root)

    src_manifest = (Path(args.src_manifest) if args.src_manifest
                    else data_root / "corrector_data_manifest.json")
    if not src_manifest.is_file():
        print(f"[fov] source manifest not found: {src_manifest}\n"
              f"      run build_corrector_data.py first.", file=sys.stderr)
        return 2
    src = json.load(open(src_manifest))

    out_root = (Path(args.out_data_root) if args.out_data_root
                else data_root.parent / f"{data_root.name}_fov")
    images_dir = out_root / cd.get("images_dirname", "images")
    images_dir.mkdir(parents=True, exist_ok=True)
    (out_root / cd.get("nnunet_pred_dirname", "nnunet_pred")).mkdir(parents=True, exist_ok=True)
    (out_root / cd.get("cnisp_pred_dirname", "cnisp_pred")).mkdir(parents=True, exist_ok=True)

    fracs = [float(x) for x in args.keep_fractions.split(",") if x.strip()]
    steps = [_pseudo_step(f) for f in fracs]
    rng = np.random.RandomState(int(args.seed))
    print(f"[fov] src={src_manifest}")
    print(f"[fov] out={out_root}  keep_fractions={fracs} -> pseudo-steps {steps}  side={args.side}")

    manifest = {"experiment": "fov_truncation", "source_manifest": str(src_manifest),
                "keep_fractions": fracs, "pseudo_steps": steps, "side": args.side,
                "steps": steps, "cases": {}}
    sidecar: dict = {}
    n_cases = n_written = n_skipped = 0

    for case_id, sentry in sorted(src.get("cases", {}).items()):
        if args.max_cases and n_cases >= args.max_cases:
            break
        source_image = sentry.get("source_image", "")
        gt = sentry.get("gt_candidate_pred", "")
        axis = sentry.get("step_axis")
        if not source_image or not Path(source_image).exists() or axis is None:
            n_skipped += 1
            continue
        if not gt or not Path(gt).exists():
            # gt_candidate_pred is the corrector's label target; skip cases without it
            n_skipped += 1
            continue
        n_cases += 1

        entry = {"source_image": source_image, "gt_candidate_pred": gt,
                 "csv_z_spacing": sentry.get("csv_z_spacing", ""),
                 "through_plane_spacing": sentry.get("through_plane_spacing"),
                 "step_axis": int(axis), "steps": {}}
        sidecar[case_id] = {}
        for f, pp in zip(fracs, steps):
            side = (["end", "start", "both"][int(rng.randint(3))]
                    if args.side == "random" else args.side)
            out = images_dir / f"{case_id}_step{pp:02d}_0000.nii.gz"
            if out.exists() and not args.force:
                # reuse; still need the vis range for the sidecar -> recompute cheaply
                arr, affine, vis = _truncate_one_ct(
                    Path(source_image), z_axis=int(axis), keep_fraction=f,
                    pad_value=args.pad_value, side=side)
                src_shape = [int(s) for s in arr.shape]
            else:
                arr, affine, vis = _truncate_one_ct(
                    Path(source_image), z_axis=int(axis), keep_fraction=f,
                    pad_value=args.pad_value, side=side)
                nib.save(nib.Nifti1Image(arr.astype(np.float32), affine), str(out))
                n_written += 1
                src_shape = [int(s) for s in arr.shape]
            entry["steps"][str(pp)] = {
                "kept": True, "keep_fraction": f, "side": side,
                "trunc_axis": int(axis), "visible_range": [int(vis[0]), int(vis[1])],
                "image": str(out),
            }
            sidecar[case_id][str(pp)] = {
                "trunc_axis": int(axis), "visible_range": [int(vis[0]), int(vis[1])],
                "source_shape": src_shape, "keep_fraction": f, "side": side,
            }
        manifest["cases"][case_id] = entry
        print(f"  {case_id}: axis={axis} -> " +
              " ".join(f"step{pp:02d}(keep{f:.2f})" for f, pp in zip(fracs, steps)))

    with open(out_root / "corrector_data_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    with open(out_root / "fov_truncation_manifest.json", "w") as fh:
        json.dump(sidecar, fh, indent=2)
    print(f"[fov] cases={n_cases} images_written={n_written} skipped={n_skipped}")
    print(f"[fov] manifest -> {out_root / 'corrector_data_manifest.json'}")
    print(f"[fov] sidecar  -> {out_root / 'fov_truncation_manifest.json'}")
    print("[fov] NEXT (box): 835 stage-1 predict on each truncated CT -> CNISP 032 "
          f"(--steps {','.join(str(s) for s in steps)}) -> build_corrector_dataset "
          "--layout cascade on a FOV config. See RUNBOOK_FOV.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
