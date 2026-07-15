#!/usr/bin/env python3
"""Build the corrector's 5-channel nnUNet raw dataset from the data/ tree.

Consumes the self-contained corrector data layout (NOT the work_dir sweep):
    ch0  = data/images/{case}_step{XX}_0000.nii.gz      (degraded CT, pinned)
    ch1..ch4 = control prelabel split into per-class binaries:
        control B -> data/nnunet_pred/{case}_step{XX}.nii.gz   (835 pred, {1,2,3,4})
        control C -> data/cnisp_pred/{case}_step{XX}.nii.gz    (CNISP,   {1,2,3,4})
    label = full-res Dataset835 prediction (manifest gt_candidate_pred), the
            pseudo-GT target (keep=False images have no manual GT), {0..4}.
Everything is resampled to the 835 plan-spacing grid (pothole-2 a-ii) so nnUNet's
preprocess resample is a no-op and the binary channels stay {0,1}.

Each (case_id, step) -> one nnUNet case `corr_{case_id}_step{XX}`. Only samples
whose ch0 + prelabel (+ optionally cnisp, to mirror C) + gt all exist are built,
so a capped CNISP run (e.g. --max-samples 300) yields a matching dataset.

Usage:
    python nnunet-c/scripts/build_corrector_dataset.py --control C
    python nnunet-c/scripts/build_corrector_dataset.py --control B --require-cnisp
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.config import add_repo_to_syspath, load_corrector_config, get_control  # noqa: E402

add_repo_to_syspath(__file__)

import numpy as np  # noqa: E402
import nibabel as nib  # noqa: E402

from engine.convert import convert_case, STRUCTS  # noqa: E402  (the SINGLE converter)
from engine.build_dataset import _raw_root, _dataset_dir, _write_dataset_json  # noqa: E402
from lib.resample import build_reference_grid, resolve_target_spacing  # noqa: E402
from lib import prelabel as _pre  # noqa: E402
from lib.labels import remap_native_to_nnunet  # noqa: E402

_DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "corrector.yaml"


def _assemble_one(task):
    """Assemble ONE (case, step) into a 5ch nnUNet case.

    Top-level (picklable) so it can run in a ProcessPoolExecutor worker. Each
    (case, step) is independent -- it only reads its own ct/prelabel/gt and
    writes its own imagesTr/labelsTr files -- so the assembly parallelises
    cleanly across cases. Returns (cid, summary).
    """
    (cid, ct_path, prelabel_path, ref_grid, experiment,
     images_out, gt, labels_out, degraded_marker) = task
    summary = convert_case(
        case_id=cid,
        ct_path=ct_path,
        prelabel_path=prelabel_path,
        ref_grid=ref_grid, experiment=experiment,
        images_dir=images_out, gt_path=gt, labels_dir=labels_out,
        degraded_marker=degraded_marker,
    )
    return cid, summary


def _assemble_one_cascade(task):
    """Assemble ONE (case,step) into the native-cascade (Route A) layout.

    Writes TWO 1-channel nnUNet cases sharing the SAME ref_grid + the SAME CT:
      * MAIN dataset  (control C = 845): ch0 CT (order 3) + GT label (order 0)
      * PRIOR dataset (parallel, e.g. 846): ch0 CT (order 3) + the CNISP prior AS
        the label ({1,2,3,4}, order 0)
    Because the CT (hence nnUNet's nonzero crop) is identical, preprocessing BOTH
    with the same finetune plan yields byte-aligned grids -> the prior's
    ``{id}_seg.b2nd`` is a valid ``seg_prev`` once relocated into the main
    dataset's ``predicted_next_stage/<cfg>/{id}.b2nd`` (see relocate_prevseg.py).

    The prior is fed through ``assemble_case`` as the GT label; since it is already
    in the nnUNet {1,2,3,4} scheme, ``remap_to_nnunet(.., NNUNET_LABELS)`` is the
    identity. Top-level/picklable so it runs in a ProcessPoolExecutor worker.
    """
    from lib import channels as _ch
    from lib.labels import NNUNET_LABELS
    (cid, ct_path, prior_path, gt, ref_grid, experiment,
     images_main, labels_main, images_prior, labels_prior, degraded_marker) = task
    main = _ch.assemble_case(
        case_id=cid, ct_path=ct_path, gt_path=gt, target_spacing=None,
        n_channels=1, structures=STRUCTS, gt_struct_to_value=dict(NNUNET_LABELS),
        images_dir=images_main, labels_dir=labels_main, experiment=experiment,
        prelabel_path=None, ref_grid=ref_grid, degraded_marker=degraded_marker,
    )
    prior = _ch.assemble_case(
        case_id=cid, ct_path=ct_path, gt_path=prior_path, target_spacing=None,
        n_channels=1, structures=STRUCTS, gt_struct_to_value=dict(NNUNET_LABELS),
        images_dir=images_prior, labels_dir=labels_prior, experiment=experiment,
        prelabel_path=None, ref_grid=ref_grid, degraded_marker=degraded_marker,
    )
    return cid, {"main": main, "prior": prior}


def _stage_iso_prelabel_nn(cfg, case_id: str, step: int, stage_dir: Path) -> Path:
    """TRAIN iso prelabel for (case,step) -> remapped BY NAME to nnUNet {1,2,3,4}.

    Mirrors build_corrector_testset._nn_prelabel: the CNISP iso mask is in
    canonical scheme, so remap_native_to_nnunet(scheme='auto') aligns structures
    to channels (a value split would swap Globe/Recti). Returns the staged path.
    """
    src = _pre._c_train_iso_prelabel_path(cfg, case_id, step)
    img = nib.load(str(src))
    arr = np.asanyarray(img.dataobj)
    nn, _scheme, _off = remap_native_to_nnunet(arr, STRUCTS, scheme="auto")
    stage_dir.mkdir(parents=True, exist_ok=True)
    out = stage_dir / f"corr_{case_id}_step{step:02d}.nii.gz"
    nib.save(nib.Nifti1Image(nn.astype(np.uint8), img.affine), str(out))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(_DEFAULT_CONFIG))
    ap.add_argument("--control", required=True, choices=["B", "C", "b", "c"])
    ap.add_argument("--require-cnisp", action="store_true",
                    help="also require data/cnisp_pred to exist for each sample "
                         "(use for control B so it matches C's capped case set).")
    ap.add_argument("--max-samples", type=int, default=0,
                    help="cap the built dataset to the first N (case,step) in "
                         "sorted (case_id, step) order (0=all). When >0, candidacy "
                         "requires BOTH nnunet_pred AND cnisp_pred (a control-"
                         "INDEPENDENT basis) so --control B and --control C select "
                         "the IDENTICAL N samples -- the only difference being which "
                         "prelabel fills ch1..4.")
    ap.add_argument("--steps", default=None,
                    help="comma-separated step sizes to INCLUDE, e.g. 3,6,9. "
                         "Only (case,step) whose step is in this set are built; "
                         "everything else in the manifest is skipped. DEFAULT "
                         "(unset): use corrector_data.steps from the config (so a "
                         "rollback config with steps: [3,6,9] drops step 12 at "
                         "build time without re-running data-gen). Pass 'all' to "
                         "build every step present in the manifest.")
    ap.add_argument("--raw-root", default=None, help="override $nnUNet_raw")
    ap.add_argument("--prelabel-grid", choices=["iso", "native"], default="iso",
                    help="control C ch1..4 source: 'iso' (default) = CNISP "
                         "iso-DIRECT decode (data/cnisp_pred_train_iso, remapped "
                         "BY NAME to {1,2,3,4}) -- the SAME decode path as the test "
                         "build, so train/test ch1..4 match. 'native' = legacy "
                         "data/cnisp_pred native mask resampled to the iso grid. "
                         "Ignored for control B (ch1..4 = nnUNet pred).")
    ap.add_argument("--iso-mm", type=float, default=None,
                    help="iso spacing (mm) for the assembly ref grid, built from "
                         "each step's degraded CT FOV. DEFAULT (unset): resolve "
                         "from the 835 iso plan (reference_plan_json, e.g. "
                         "0.4765625) via resolve_target_spacing -- i.e. the SAME "
                         "spacing the network's plan uses and the first training "
                         "used. MUST match the test build "
                         "(build_corrector_testset --iso-mm).")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel worker processes for the 5ch assembly. Each "
                         "(case,step) is independent, so this scales the (slow) "
                         "order-3 CT resampling across cores. Default 1 "
                         "(sequential; unchanged behaviour). Try 8-16 on a big "
                         "box; watch RAM (each worker holds a full-res volume).")
    ap.add_argument("--layout", choices=["stacked", "cascade"], default="stacked",
                    help="'stacked' (default) = the legacy 5-channel image (ch0 CT "
                         "+ ch1..4 binary prior), unchanged. 'cascade' (Route A) = "
                         "1-channel CT image + GT for the MAIN dataset, and a "
                         "PARALLEL prior dataset (same CT + the CNISP prior AS the "
                         "label); nnUNet then loads the prior as a per-case seg_prev "
                         "and folds it in with MoveSegAsOneHot AFTER intensity aug. "
                         "cascade requires control C (CNISP) + --prelabel-grid iso.")
    ap.add_argument("--prior-dataset-id", type=int, default=None,
                    help="(cascade only) nnUNet dataset id for the PARALLEL prior "
                         "dataset. Default = control dataset_id + 1. Its preprocessed "
                         "{id}_seg.b2nd become the seg_prev (relocate_prevseg.py).")
    ap.add_argument("--prior-dataset-name", default=None,
                    help="(cascade only) name for the parallel prior dataset. "
                         "Default = <control dataset_name>_prior.")
    args = ap.parse_args()

    cfg = load_corrector_config(args.config, caller_file=__file__)
    control = get_control(cfg, args.control)
    if control.get("external"):
        raise RuntimeError(f"control {args.control.upper()} is external (Dataset"
                           f"{control['dataset_id']}); nothing to build.")
    if int(control["n_channels"]) != 5:
        raise RuntimeError("this builder is for the 5-channel controls (B/C).")

    # control C ch1..4: iso-direct decode (default, matches test) vs legacy native.
    is_cnisp = control["prelabel_source"] == "cnisp"
    use_iso = is_cnisp and args.prelabel_grid == "iso"

    cd = cfg["corrector_data"]
    res = cfg["_resolved"]
    data_root = Path(cd["data_root"])
    data_root = data_root if data_root.is_absolute() else (res["repo_root"] / data_root)
    images_dirname = cd.get("images_dirname", "images")
    images_dir = data_root / images_dirname
    pre_dirname = (cd.get("nnunet_pred_dirname", "nnunet_pred")
                   if control["prelabel_source"] == "nnunet"
                   else cd.get("cnisp_pred_dirname", "cnisp_pred"))
    prelabel_dir = data_root / pre_dirname
    cnisp_dir = data_root / cd.get("cnisp_pred_dirname", "cnisp_pred")
    manifest_path = data_root / "corrector_data_manifest.json"
    if not manifest_path.is_file():
        print(f"[build] {manifest_path} missing -- run build_corrector_data.py first.",
              file=sys.stderr)
        return 2
    manifest = json.load(open(manifest_path))

    raw = _raw_root(args.raw_root)
    ds_dir = _dataset_dir(raw, control)
    images_out = ds_dir / "imagesTr"
    labels_out = ds_dir / "labelsTr"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    # Assembly spacing = the 835 iso plan spacing (resolve_target_spacing reads
    # nnUNetPlans_iso05 -> e.g. [0.4765625]*3), unless --iso-mm overrides it. This
    # is what the FIRST training used and what the finetune plan resamples to, so
    # nnUNet's preprocess resample is a near no-op. MUST match the test builder.
    iso_spacing = ([float(args.iso_mm)] * 3 if args.iso_mm is not None
                   else [float(x) for x in resolve_target_spacing(cfg)])

    stage_dir = ds_dir / "_prelabel_nn_train"   # staged remapped iso prelabels (C+iso)
    pre_desc = (f"CNISP iso-direct decode ({_pre._cnisp_train_iso_root(cfg)}, "
                f"remapped by name)" if use_iso else str(prelabel_dir))
    print(f"[build] control={args.control.upper()} -> {ds_dir}")
    print(f"[build] ch0={images_dir}  prelabel={pre_desc}")
    print(f"[build] grid = iso {iso_spacing} from each step's degraded-CT FOV "
          f"(from {'--iso-mm' if args.iso_mm is not None else '835 iso plan'}; "
          f"matches build_corrector_testset)")

    # ── Phase 1: deterministic candidate selection ───────────────────
    # Build the ordered list of buildable (case_id, step) FIRST, then optionally
    # cap it, then assemble only the selected ones (so a capped run never does
    # convert_case work for samples it will not keep). When --max-samples is set
    # we use a control-INDEPENDENT candidacy (ct + nnunet_pred + cnisp_pred + gt)
    # so --control B and --control C pick the IDENTICAL first-N set; otherwise we
    # keep the per-control predicate (ct + this control's prelabel [+ cnisp when
    # --require-cnisp] + gt).
    nnunet_dir = data_root / cd.get("nnunet_pred_dirname", "nnunet_pred")

    # Step filter (drops e.g. step 12 for a rollback WITHOUT re-running data-gen):
    # --steps > config corrector_data.steps > all steps in the manifest.
    if args.steps is not None and args.steps.strip().lower() == "all":
        steps_filter = None
    elif args.steps is not None:
        steps_filter = {int(s) for s in args.steps.split(",") if s.strip()}
    elif cd.get("steps"):
        steps_filter = {int(s) for s in cd["steps"]}
    else:
        steps_filter = None
    print(f"[build] steps filter = "
          f"{sorted(steps_filter) if steps_filter is not None else 'all (manifest)'}")

    cap = max(0, int(args.max_samples))
    need_cnisp = bool(args.require_cnisp) or cap > 0
    need_nnunet = cap > 0

    candidates, skipped = [], 0   # candidates: list of (case_id, step, gt_path)
    for case_id, entry in sorted(manifest["cases"].items()):
        gt = entry.get("gt_candidate_pred", "")
        if not gt or not Path(gt).exists():
            skipped += 1
            continue
        for step in sorted(int(s) for s in entry.get("steps", {})):
            if steps_filter is not None and step not in steps_filter:
                continue
            sinfo = entry["steps"][str(step)]
            if not sinfo.get("kept"):
                continue
            stem = f"{case_id}_step{step:02d}"
            ct = images_dir / f"{stem}_0000.nii.gz"
            if not ct.exists():
                skipped += 1
                continue
            # ch1..4 availability: iso prelabel (C+iso, via manifest) or native file.
            if use_iso:
                try:
                    _pre._c_train_iso_prelabel_path(cfg, case_id, step)
                except (FileNotFoundError, KeyError):
                    skipped += 1
                    continue
            elif not (prelabel_dir / f"{stem}.nii.gz").exists():
                skipped += 1
                continue
            if need_cnisp and not (cnisp_dir / f"{stem}.nii.gz").exists():
                skipped += 1
                continue
            if need_nnunet and not (nnunet_dir / f"{stem}.nii.gz").exists():
                skipped += 1
                continue
            candidates.append((case_id, step, Path(gt)))

    # candidates are already globally sorted by (case_id, step): the outer loop
    # iterates cases in sorted order and the inner loop iterates steps ascending.
    if cap > 0:
        if len(candidates) < cap:
            print(f"[build] WARNING: only {len(candidates)} candidate(s) available "
                  f"< --max-samples {cap}; building all of them.", file=sys.stderr)
        candidates = candidates[:cap]

    # ── Route A: native-cascade layout (main 1-ch dataset + parallel prior) ──
    if args.layout == "cascade":
        # C (CNISP) and B (nnUNet pred) both supported -- same seg_prev machinery,
        # only the prior label source differs. C requires the iso-direct decode.
        if is_cnisp and not use_iso:
            raise RuntimeError(
                "--layout cascade for control C (CNISP) requires --prelabel-grid iso "
                "(the iso-direct decode remapped by name to {1,2,3,4}).")
        prior_id = (int(args.prior_dataset_id) if args.prior_dataset_id is not None
                    else int(control["dataset_id"]) + 1)
        prior_name = args.prior_dataset_name or f"{control['dataset_name']}_prior"
        prior_control = dict(control)
        prior_control.update(dataset_id=prior_id, dataset_name=prior_name, n_channels=1)
        ds846 = _dataset_dir(raw, prior_control)
        images846 = ds846 / "imagesTr"
        labels846 = ds846 / "labelsTr"
        images846.mkdir(parents=True, exist_ok=True)
        labels846.mkdir(parents=True, exist_ok=True)
        print(f"[build] LAYOUT=cascade  main(1ch CT + GT)          -> {ds_dir}")
        print(f"[build]                 prior(1ch CT + {'CNISP' if is_cnisp else 'nnUNet'} lbl) -> {ds846}")
        print(f"[build]                 prior dataset = Dataset{prior_id:03d}_{prior_name}")

        tasks = []
        for case_id, step, gt in candidates:
            stem = f"{case_id}_step{step:02d}"
            ct_path = images_dir / f"{stem}_0000.nii.gz"
            ref_grid = build_reference_grid(nib.load(str(ct_path)), iso_spacing)
            cid = f"corr_{stem}"
            # prior label: C -> CNISP iso-direct decode (staged, remapped by name);
            # B -> the nnUNet pred ({1,2,3,4} already). Same seg_prev downstream.
            prior_path = (_stage_iso_prelabel_nn(cfg, case_id, step, stage_dir)
                          if use_iso else prelabel_dir / f"{stem}.nii.gz")
            tasks.append((
                cid, ct_path, prior_path, gt, ref_grid, cfg["experiment"],
                images_out, labels_out, images846, labels846, f"/{images_dirname}/",
            ))

        main_sums, prior_sums = [], []
        workers = max(1, int(args.workers))
        if workers > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            print(f"[build] assembling {len(tasks)} case(s) x2 (main+prior), {workers} workers")
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_assemble_one_cascade, t) for t in tasks]
                for fut in as_completed(futs):
                    cid, s = fut.result()
                    main_sums.append(s["main"]); prior_sums.append(s["prior"])
                    print(f"  {cid}: main lbls={s['main']['label_values']} "
                          f"prior lbls={s['prior']['label_values']}")
        else:
            for t in tasks:
                cid, s = _assemble_one_cascade(t)
                main_sums.append(s["main"]); prior_sums.append(s["prior"])
                print(f"  {cid}: main lbls={s['main']['label_values']} "
                      f"prior lbls={s['prior']['label_values']}")

        n = len(main_sums)
        _write_dataset_json(ds_dir, control, cfg, num_training=n, image_channels=1)
        _write_dataset_json(ds846, prior_control, cfg, num_training=n, image_channels=1)
        with open(ds_dir / "corrector_build_manifest.json", "w") as f:
            json.dump({"control": args.control.upper(), "layout": "cascade", "n": n,
                       "prior_dataset": f"Dataset{prior_id:03d}_{prior_name}",
                       "cases": main_sums}, f, indent=2)
        with open(ds846 / "corrector_build_manifest.json", "w") as f:
            json.dump({"role": "cnisp_prior_prev_stage", "for_dataset": ds_dir.name,
                       "n": n, "cases": prior_sums}, f, indent=2)
        print(f"[build] wrote {n} case(s) x2; skipped {skipped} (missing files).")
        print(f"[build] main  -> {ds_dir}")
        print(f"[build] prior -> {ds846}")
        print("[build] NEXT: preprocess BOTH datasets (finetune plan), then "
              "relocate_prevseg.py  (run_train.sh handles this under cascade).")
        return 0

    # ── Phase 2: assemble only the selected (case, step) ─────────────
    # Grid = the iso-mm grid built from THIS step's degraded-CT FOV -- IDENTICAL
    # to how the test set is assembled (build_corrector_testset --prelabel-grid
    # iso), so train and test share the same geometry (no GT grid at train, which
    # test can never use). ch0 = degraded CT resampled onto it, ch1-4 = prelabel
    # mask resampled onto it, label = GT resampled onto it (order 0) -- GT is used
    # ONLY for supervision, not as the geometry reference. nnUNet then resamples
    # this iso grid -> the plan spacing at preprocess (near no-op). ``_assemble_one``
    # is identical work either way, so --workers 1 reproduces sequential behaviour.
    assembled = []
    tasks = []
    for case_id, step, gt in candidates:
        stem = f"{case_id}_step{step:02d}"
        ct_path = images_dir / f"{stem}_0000.nii.gz"
        # iso grid from this step's degraded CT FOV (lazy: reads shape+affine only)
        ref_grid = build_reference_grid(nib.load(str(ct_path)), iso_spacing)
        cid = f"corr_{stem}"
        # ch1..4: iso-direct decode (staged, remapped by name) or legacy native.
        if use_iso:
            prelabel_path = _stage_iso_prelabel_nn(cfg, case_id, step, stage_dir)
        else:
            prelabel_path = prelabel_dir / f"{stem}.nii.gz"
        tasks.append((
            cid,
            ct_path,
            prelabel_path,
            ref_grid, cfg["experiment"],
            images_out, gt, labels_out, f"/{images_dirname}/",
        ))

    workers = max(1, int(args.workers))
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        print(f"[build] assembling {len(tasks)} case(s) with {workers} workers")
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_assemble_one, t) for t in tasks]
            for fut in as_completed(futs):
                cid, summary = fut.result()
                assembled.append(summary)
                print(f"  {cid}: shape={summary['shape']} "
                      f"labels={summary['label_values']}")
    else:
        for t in tasks:
            cid, summary = _assemble_one(t)
            assembled.append(summary)
            print(f"  {cid}: shape={summary['shape']} "
                  f"labels={summary['label_values']}")

    _write_dataset_json(ds_dir, control, cfg, num_training=len(assembled))
    with open(ds_dir / "corrector_build_manifest.json", "w") as f:
        json.dump({"control": args.control.upper(), "n": len(assembled),
                   "cases": assembled}, f, indent=2)
    print(f"[build] wrote {len(assembled)} case(s); skipped {skipped} (missing files).")
    print(f"[build] dataset -> {ds_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
