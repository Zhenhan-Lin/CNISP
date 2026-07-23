#!/usr/bin/env python3
"""
032: CNISP inference for the nnUNet-C corrector (lean, single-output).

CNISP's predictions for the corrector serve ONLY the corrector, so this script
deliberately does NOT write the usual reconstructions/<model>/runs/... pile
(sweep_results.pkl, test_results.csv, per-step step_XX/pred, latents, viz,
original-scheme native_space masks, ...). It computes the latent fit + dense
decode + native inversion IN MEMORY (via run_sweep, which writes nothing) and
saves ONLY the final merged native mask per (source, step), remapped to the
nnUNet scheme {1,2,3,4}, into nnunet-c/data/cnisp_pred/.

Remap is BY STRUCTURE NAME from CNISP's canonical labels (fixed):
    canonical {1:ON, 2:Globe, 3:Fat, 4:Recti}  ->  nnUNet {ON:1, Recti:2, Globe:3, Fat:4}
i.e. {1->1, 2->3, 3->4, 4->2}. (A value shift would swap Globe/Recti.)

Output: data/cnisp_pred/<stem>_step{XX}.nii.gz  (+ a small manifest.json).

Usage:
    python orbital_shape_prior_st1/scripts/032_cnisp_infer_corrector.py \
        -m orbital_ad_v6_5_gt -t configs/train_v6_5_gt.yaml -c configs/test_corrector.yaml \
        --checkpoint best --test-label-source nnunet_pred --experiment thick \
        --test-casefile corrector_train_cases.txt

NOTE: the cases must already be CNISP-canonical-aligned (metadata under
aligned_dir). For nnunet_pred mode the per-step Dataset835 sparse patches must
exist (Stage 1). This script does inference + native remap only.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import nibabel as nib
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]   # orbital_shape_prior_st1/
REPO_ROOT = PROJECT_ROOT.parent
for _p in (str(PROJECT_ROOT), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from engine.dataset import load_casenames                       # noqa: E402
from engine.train import create_model                           # noqa: E402
from engine.infer import (                                      # noqa: E402
    load_model_checkpoint, optimize_latent,
    _load_labels_dense_per_case, _build_label_obs_loader, _meta_path_for_case,
    _observed_meta_path_for, _AUTOCAST_DTYPE,
    device as INFER_DEVICE,
)
from engine.test_label_sources import build_run_layout, step_input_patch_path  # noqa: E402
# ROLLBACK/ABLATION toggle: CNISP_BUGGY_NATIVE_MAPPING=1 swaps in the PRE-FIX
# (pre-8540137) native mapping to regenerate the OLD buggy CNISP prelabels.
if os.environ.get("CNISP_BUGGY_NATIVE_MAPPING") == "1":     # noqa: E402
    from engine.native_mapping_buggy import (               # noqa: E402
        invert_alignment_single_eye, _extract_sub_crop_info,
        map_iso_results_to_native,
    )
    print("[native_mapping] CNISP_BUGGY_NATIVE_MAPPING=1 -> using PRE-FIX BUGGY native mapping")
else:
    from engine.native_mapping import (                     # noqa: E402
        invert_alignment_single_eye, _extract_sub_crop_info,
        map_iso_results_to_native,
    )
from diagnostics.resolution_sweep import (                     # noqa: E402
    eval_case_at_resolution, adaptive_steps_for_case,
)
from data_prep.sparsify import resolve_slice_step_axes          # noqa: E402

# Fixed canonical -> nnUNet remap (by structure name). DO NOT value-shift.
CANON2NNUNET = {0: 0, 1: 1, 2: 3, 3: 4, 4: 2}


def _source_of(casename: str) -> str:
    """`chk_X_OD` / `10058_..._CT_0_OD` -> source id (strip the _OD/_OS eye)."""
    return casename[:-3] if casename.endswith(("_OD", "_OS")) else casename


def _resolve_cfg(path_arg: str) -> Path:
    p = Path(path_arg)
    for cand in (p, REPO_ROOT / p, PROJECT_ROOT / p, PROJECT_ROOT / "configs" / p.name):
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"config not found: {path_arg}")


def _load_params(*yamls: Path) -> dict:
    params: dict = {}
    for y in yamls:
        with open(y) as f:
            params.update(yaml.safe_load(f) or {})
    return params


def _canon_to_nnunet(arr: np.ndarray) -> np.ndarray:
    out = np.zeros(arr.shape, dtype=np.uint8)
    for canon, nn in CANON2NNUNET.items():
        if nn:
            out[arr == canon] = nn
    return out


def _maybe_load_delta(params: dict, model_state: dict):
    """Load + freeze Delta if the (denoise) config asks for it; else None."""
    den = params.get("denoise") or {}
    if not (bool(den.get("enabled", False)) and bool(den.get("use_delta", True))):
        return None
    delta_state = model_state.get("delta")
    if delta_state is None:
        raise RuntimeError("denoise.use_delta=true but checkpoint has no 'delta'.")
    from models.denoise import LatentDenoiser
    delta = LatentDenoiser(
        latent_dim=params["latent_dim"],
        hidden_dim=den.get("delta_hidden_dim") or None,
        num_hidden_layers=int(den.get("delta_num_hidden_layers", 2)),
    )
    delta.load_state_dict(delta_state)
    delta = delta.to(INFER_DEVICE).eval()
    for p in delta.parameters():
        p.requires_grad_(False)
    return delta


def _save_final_masks(results: List[Dict], layout, out_dir: Path,
                      net=None, emit_iso_dir=None, emit_iso_mm: float = 0.4765625) -> Dict:
    """Group results by (source, step), merge eyes (canonical), remap, save one
    {1,2,3,4} native mask per (source, step). Also save each eye's optimized
    latent under out_dir/latent/ for reference / cheap re-decode (e.g. at iso).

    When ``emit_iso_dir`` + ``net`` are given, ALSO decode each eye's latent at
    ``emit_iso_mm`` and write a full-head iso prelabel per (source, step) under
    ``emit_iso_dir/native_space_step_XX/`` (+ by_source_id manifest), via the
    SAME ``map_iso_results_to_native`` path (incl. ``observed_meta_path_for``)
    that ``engine/infer.py`` uses for the TEST iso prelabels -- so the corrector's
    TRAIN ch1..4 are generated identically to its TEST ch1..4.

    Returns a manifest dict."""
    meta_for = _meta_path_for_case(layout)
    # Observed input-patch metadata resolver: re-frame the reconstruction from
    # the dense target crop to the observed nnUNet-pred crop it was fit to,
    # otherwise the OS mask is mirrored/misplaced (worse at high step). None
    # for atlas_gt / real_pair.
    obs_meta_for = _observed_meta_path_for(layout)

    def _observed_meta(result: dict):
        if obs_meta_for is None:
            return None
        mp = obs_meta_for(result["casename"], int(result.get("step_size", 1)),
                          int(result.get("slice_start_id", 0)))
        if mp is None or not Path(mp).exists():
            return None
        return json.load(open(mp))

    out_dir.mkdir(parents=True, exist_ok=True)
    latent_dir = out_dir / "latent"
    latent_dir.mkdir(parents=True, exist_ok=True)

    # group by (source_id, step) over start==0 rows only
    groups: Dict[Tuple[str, int], List[Tuple[dict, dict]]] = defaultdict(list)
    for r in results:
        if int(r.get("slice_start_id", 0)) != 0:
            continue
        cn = r["casename"]
        mp = Path(meta_for(cn))
        if not mp.exists():
            print(f"  WARN: metadata missing for {cn} ({mp}); skipping")
            continue
        meta = json.load(open(mp))
        groups[(meta["source_id"], int(r["step_size"]))].append((r, meta))

    manifest: Dict[str, list] = defaultdict(list)
    n = 0
    for (source_id, step), items in sorted(groups.items()):
        ref_meta = items[0][1]
        merged = np.zeros(ref_meta["original_shape"], dtype=np.int16)
        for r, meta in items:
            # Save this eye's optimized latent for reference / cheap re-decode.
            lat = r.get("latent")
            if lat is not None:
                np.save(str(latent_dir / f"{r['casename']}_step{step:02d}.npy"),
                        np.asarray(lat))
            lo, sh = _extract_sub_crop_info(r, r["casename"])
            full = invert_alignment_single_eye(
                r["pred_class_map"], meta,
                sub_crop_lo_vox_dense=lo, sub_crop_shape_vox_dense=sh,
                casename=r["casename"],
                observed_meta=_observed_meta(r),
            )
            fg = full > 0
            merged[fg] = full[fg]                      # canonical-label union
        remapped = _canon_to_nnunet(merged)            # -> {1,2,3,4}
        stem = Path(ref_meta["original_nifti_path"]).name
        stem = stem.replace(".nii.gz", "").replace(".nii", "")
        dst = out_dir / f"{stem}_step{step:02d}.nii.gz"
        nib.save(nib.Nifti1Image(remapped, np.array(ref_meta["original_affine"])),
                 str(dst))
        vals = sorted(int(v) for v in np.unique(remapped))
        print(f"  {source_id} step={step:02d}: {len(items)} eye(s) -> {dst.name} "
              f"values={vals}")
        manifest[source_id].append({"step": step, "file": dst.name, "values": vals})
        n += 1
    print(f"[032] wrote {n} final mask(s) -> {out_dir}")

    # ── Optional iso-mm prelabel emit (corrector TRAIN ch1..4) ────────────
    # Mirror engine/infer.py's iso emit EXACTLY (same map_iso_results_to_native
    # + observed_meta_path_for), so train iso prelabels match the test ones.
    if emit_iso_dir is not None and net is not None:
        iso_root = Path(emit_iso_dir)
        n_iso = 0
        for (source_id, step), items in sorted(groups.items()):
            ref_meta = items[0][1]
            stem = Path(ref_meta["original_nifti_path"]).name
            stem = stem.replace(".nii.gz", "").replace(".nii", "")
            results_step = []
            for r, meta in items:
                lat = r.get("latent")
                if lat is None or np.asarray(lat).size <= 1:
                    continue
                lo, sh = _extract_sub_crop_info(r, r["casename"])
                lat_t = torch.from_numpy(
                    np.asarray(lat, dtype=np.float32)
                ).reshape(1, -1).to(INFER_DEVICE)
                tgt = torch.round(
                    net.image_size.detach().cpu().float() / float(emit_iso_mm)
                ).long()
                with torch.no_grad():
                    iso_map = net.predict_dense(
                        lat_t, tgt.to(INFER_DEVICE),
                        torch.full((3,), float(emit_iso_mm),
                                   dtype=torch.float32).to(INFER_DEVICE),
                        autocast_dtype=_AUTOCAST_DTYPE,
                    )
                r["pred_class_map_iso"] = iso_map.numpy().astype(np.int16)
                r["sub_crop_lo_vox_dense"] = lo
                r["sub_crop_shape_vox_dense"] = sh
                results_step.append(r)
            if not results_step:
                continue
            step_dir = iso_root / f"native_space_step_{step:02d}"
            suffix = f"_cnisp_iso_step{step:02d}"
            meta_dir_fallback = Path(meta_for(items[0][0]["casename"])).parent
            map_iso_results_to_native(
                results_step, meta_dir_fallback, step_dir,
                suffix=suffix, iso_mm=float(emit_iso_mm),
                meta_path_for_casename=meta_for,
                observed_meta_path_for=obs_meta_for,
            )
            # Per-SOURCE manifest file (race-free across concurrent shard workers,
            # unlike a single shared manifest.json). build_corrector_dataset reads
            # manifest_by_source/<source_id>.json.
            mbs_dir = step_dir / "manifest_by_source"
            mbs_dir.mkdir(parents=True, exist_ok=True)
            json.dump({"file": f"{stem}{suffix}.nii.gz",
                       "iso_mm": float(emit_iso_mm), "step_size": step},
                      open(mbs_dir / f"{source_id}.json", "w"), indent=2)
            n_iso += 1
        if n_iso:
            print(f"[032] emitted {n_iso} iso-{emit_iso_mm}mm prelabel(s) -> {iso_root}")

    return dict(manifest)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-p", "--paths", default="configs/paths.yaml")
    ap.add_argument("-t", "--train_config", default="configs/train_v6_5_gt.yaml")
    ap.add_argument("-c", "--config", default="configs/test_corrector.yaml")
    ap.add_argument("-m", "--model_name", default="orbital_ad_v6_5_gt")
    ap.add_argument("--checkpoint", default="best", choices=["best", "latest"])
    ap.add_argument("--test-label-source", default="nnunet_pred",
                    choices=["atlas_gt", "nnunet_pred", "real_pair"])
    ap.add_argument("--experiment", default="thick", choices=["thin", "thick", "real", "fov"])
    ap.add_argument("--test-casefile", default=None,
                    help="casefile under casefiles_dir (default: test yaml's value)")
    ap.add_argument("--out-dir", default=None,
                    help="output dir (default: nnunet-c/data/cnisp_pred)")
    ap.add_argument("--aligned-dir",
                    default=str(REPO_ROOT / "nnunet-c" / "data" / "aligned_patch"),
                    help="aligned_dir for the corrector patches (labels_dataset835*/ "
                         "+ metadata_dataset835/). Default: nnunet-c/data/aligned_patch. "
                         "Pass the CNISP aligned_dir to use the shared tree instead.")
    ap.add_argument("--steps", default="3,6,9,12",
                    help="explicit step sizes matching the degraded data "
                         "(default 3,6,9,12). Use 'adaptive' to fall back to the "
                         "test yaml's adaptive_step_sweep.")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="split sources across N shard slots (for multi-GPU / "
                         "CPU concurrency). Sharding is by SOURCE so OD+OS stay "
                         "together for the merge.")
    ap.add_argument("--shard-id", default="0",
                    help="this worker's shard slot(s) in [0,num_shards); "
                         "comma-separated for weighted assignment, e.g. '0,2'.")
    ap.add_argument("--skip-existing", dest="skip_existing", action="store_true",
                    default=True,
                    help="skip (source,step) whose output mask already exists "
                         "(default ON; makes a crashed worker's re-run resume).")
    ap.add_argument("--no-skip-existing", dest="skip_existing",
                    action="store_false",
                    help="recompute even if the output mask exists.")
    ap.add_argument("--remap-from-latent", dest="remap_from_latent",
                    action="store_true", default=False,
                    help="RESUME mode: reuse each (source,step,eye)'s saved "
                         "latent under <out_dir>/latent/ and re-run ONLY the "
                         "dense decode + native mapping (no latent optimization, "
                         "no re-infer). Use this to regenerate masks after a "
                         "mapping-side fix. Overwrites existing masks (ignores "
                         "--skip-existing); a (source,step,eye) without a saved "
                         "latent is skipped.")
    ap.add_argument("--emit-iso-prelabel-dir", dest="emit_iso_prelabel_dir",
                    default=None,
                    help="ALSO emit a full-head iso prelabel per (source,step) "
                         "into <dir>/native_space_step_XX/ (+ by_source_id "
                         "manifest), decoded from the same latent at --emit-iso-mm "
                         "via the SAME path infer.py uses for TEST iso prelabels "
                         "(incl. observed-meta reframe). Use it to make the "
                         "corrector's TRAIN ch1..4 identical to its TEST ch1..4.")
    ap.add_argument("--emit-iso-mm", dest="emit_iso_mm", type=float,
                    default=0.4765625,
                    help="iso spacing (mm) for --emit-iso-prelabel-dir. Default "
                         "0.4765625 = the 835 iso plan (nnUNetPlans_iso05) spacing, "
                         "matching the train/test builder + network plan grid.")
    ap.add_argument("--max-samples", type=int, default=0,
                    help="cap the number of (source,step) samples to run, "
                         "GLOBALLY (0=all). Selected as the first N by sorted "
                         "(source_id, step) among those with an nnUNet obs "
                         "patch. Applied BEFORE sharding so multi-worker runs "
                         "stay consistent. (explicit --steps only)")
    args = ap.parse_args()

    params = _load_params(_resolve_cfg(args.paths),
                          _resolve_cfg(args.train_config),
                          _resolve_cfg(args.config))
    params["model_name"] = args.model_name
    params["checkpoint"] = args.checkpoint
    params["test_label_source"] = args.test_label_source
    params["experiment"] = args.experiment
    params["run_tag"] = params.get("run_tag", "corrector_gt")
    if args.test_casefile:
        params["test_casefile"] = args.test_casefile
    # keep sweep_mode consistent with the experiment (ceiling curve degrades GT)
    if args.experiment in ("thin", "thick"):
        params["sweep_mode"] = args.experiment

    out_dir = (Path(args.out_dir) if args.out_dir
               else REPO_ROOT / "nnunet-c" / "data" / "cnisp_pred")

    # Read/write the corrector's aligned patches from data/aligned_patch (not the
    # shared CNISP aligned_dir), unless overridden.
    if args.aligned_dir:
        params["aligned_dir"] = args.aligned_dir

    layout = build_run_layout(params)   # used for metadata/label resolution only
    print(f"[032] aligned_dir = {params['aligned_dir']}")
    model_dir = Path(params["model_basedir"]) / params["model_name"]

    print("=" * 64)
    print("032 CNISP infer for corrector (lean; only final {1,2,3,4} masks)")
    print(f"  model        : {args.model_name} ({args.checkpoint})")
    print(f"  label source : {args.test_label_source}  experiment={args.experiment}")
    print(f"  casefile     : {params['test_casefile']}")
    print(f"  out dir      : {out_dir}")
    print(f"  device       : {INFER_DEVICE}")
    print("=" * 64)

    # ── model (+ optional Delta) ─────────────────────────────────────
    model_state, _ = load_model_checkpoint(model_dir, args.checkpoint, verbose=True)
    net = create_model(params, torch.ones(3)).to(INFER_DEVICE).eval()
    net.load_state_dict(model_state["net"], strict=True)
    delta = _maybe_load_delta(params, model_state)
    optimize_fn = functools.partial(optimize_latent, delta=delta)

    # ── cases + dense targets + (deployment) input loader ────────────
    casefiles_dir = Path(params["casefiles_dir"])
    casenames_all = load_casenames(casefiles_dir / params["test_casefile"])

    # ── GLOBAL --max-samples cap (first N (source,step) with an obs patch) ─
    # Computed from the RAW casefile (deterministic, identical across workers)
    # so the cap is consistent before sharding. Restricts which sources we even
    # bother loading dense targets for.
    allowed_pairs = None
    explicit_steps = args.steps.strip().lower() != "adaptive"
    if args.max_samples and args.max_samples > 0 and explicit_steps:
        steps_list = [int(s) for s in args.steps.split(",") if s.strip()]
        by_src: Dict[str, List[str]] = {}
        for cn in casenames_all:
            by_src.setdefault(_source_of(cn), []).append(cn)
        is_atlas_gt = (layout.test_label_source == "atlas_gt")
        allowed_pairs, allowed_src = set(), set()
        for src in sorted(by_src):
            for step in steps_list:
                # "has nnUNet pred" = the per-step obs patch exists for an eye
                # (atlas_gt has no obs patch -> count all by availability of GT).
                ok = is_atlas_gt or any(
                    step_input_patch_path(layout, cn, step, 0).exists()
                    for cn in by_src[src]
                )
                if not ok:
                    continue
                allowed_pairs.add((src, step))
                allowed_src.add(src)
                if len(allowed_pairs) >= args.max_samples:
                    break
            if len(allowed_pairs) >= args.max_samples:
                break
        casenames_all = [cn for cn in casenames_all if _source_of(cn) in allowed_src]
        print(f"[032] --max-samples {args.max_samples}: selected "
              f"{len(allowed_pairs)} (source,step) over {len(allowed_src)} source(s)")

    labels_dense, spacings_dense, casenames = _load_labels_dense_per_case(
        layout, casenames_all,
    )
    if not casenames:
        print("[032] no resolvable cases; nothing to do", file=sys.stderr)
        return 1

    # ── shard by SOURCE (keep OD+OS together) for multi-GPU/CPU concurrency ─
    if args.num_shards > 1:
        try:
            shard_ids = [int(x) for x in str(args.shard_id).split(",") if x.strip()]
        except ValueError:
            print(f"[032] bad --shard-id {args.shard_id!r}", file=sys.stderr)
            return 2
        for sid in shard_ids:
            if not (0 <= sid < args.num_shards):
                print(f"[032] shard-id {sid} out of range [0,{args.num_shards})",
                      file=sys.stderr)
                return 2
        src_order = []
        for cn in casenames:
            s = _source_of(cn)
            if s not in src_order:
                src_order.append(s)
        sorted_src = sorted(src_order)
        keep = set()
        for sid in shard_ids:
            keep |= set(sorted_src[sid::args.num_shards])
        idx = [i for i, cn in enumerate(casenames) if _source_of(cn) in keep]
        casenames = [casenames[i] for i in idx]
        labels_dense = [labels_dense[i] for i in idx]
        spacings_dense = [spacings_dense[i] for i in idx]
        print(f"[032] shard {shard_ids}/{args.num_shards}: "
              f"{len(keep)} source(s), {len(casenames)} case(s)")
        if not casenames:
            print("[032] empty shard; nothing to do")
            return 0

    label_obs_loader = _build_label_obs_loader(layout)
    step_axes = resolve_slice_step_axes(params["slice_step_axis"], spacings_dense)

    # ── per-case step lists: explicit list OR CNISP's ADAPTIVE sweep ──
    # BOTH modes go through the SAME per-source incremental + skip-existing loop,
    # so a re-run never recomputes a (source,step) whose mask already exists
    # (CNISP latent fit is the slow part; degraded CTs + nnUNet obs patches are
    # reused, never regenerated). real_pair is not used by the corrector.
    from collections import OrderedDict
    adaptive = args.steps.strip().lower() == "adaptive"
    if adaptive:
        _sw = dict(params.get("adaptive_step_sweep", {}))
        _inc = float(_sw.get("target_eff_res_increment_mm", 1.0))
        _maxn = int(_sw.get("max_num_steps_per_case", 5))
        _maxeff = float(_sw.get("max_eff_resolution_mm", 12.0))

        def steps_for(ci: int):
            sp = float(spacings_dense[ci][step_axes[ci]])
            return adaptive_steps_for_case(sp, _inc, _maxn, _maxeff)
        print(f"[032] ADAPTIVE sweep (per-case steps from spacing)  "
              f"skip_existing={args.skip_existing}")
    else:
        _explicit = [int(s) for s in args.steps.split(",") if s.strip()]

        def steps_for(ci: int):
            return _explicit
        print(f"[032] explicit steps = {_explicit}  skip_existing={args.skip_existing}")

    meta_for = _meta_path_for_case(layout)
    _stem_cache: Dict[str, str] = {}

    def _source_stem(casename: str):
        s = _source_of(casename)
        if s not in _stem_cache:
            mp = Path(meta_for(casename))
            stem = None
            if mp.exists():
                on = json.load(open(mp)).get("original_nifti_path", "")
                stem = Path(on).name.replace(".nii.gz", "").replace(".nii", "")
            _stem_cache[s] = stem
        return _stem_cache[s]

    # group eyes by source so both eyes of a (source, step) merge together
    src_cases: "OrderedDict[str, list]" = OrderedDict()
    for ci, cn in enumerate(casenames):
        src_cases.setdefault(_source_of(cn), []).append((ci, cn))

    manifest: Dict[str, list] = {}
    n_src = len(src_cases)
    for si, (src, eyecases) in enumerate(src_cases.items(), 1):
        src_results = []
        for ci, cn in eyecases:
            for step in steps_for(ci):
                if allowed_pairs is not None and (src, step) not in allowed_pairs:
                    continue  # outside the global --max-samples selection
                # Skip-if-done only applies to fresh inference. In remap mode we
                # WANT to overwrite the (buggy) masks with corrected mappings.
                if args.skip_existing and not args.remap_from_latent:
                    stem = _source_stem(cn)
                    if stem and (out_dir / f"{stem}_step{step:02d}.nii.gz").exists():
                        print(f"  {cn} step={step:02d}: SKIP (output exists)")
                        continue

                # Resume-from-latent: reuse the saved latent, skip optimization.
                lat_override = None
                if args.remap_from_latent:
                    latp = out_dir / "latent" / f"{cn}_step{step:02d}.npy"
                    if not latp.exists():
                        print(f"  {cn} step={step:02d}: SKIP remap "
                              f"(no saved latent at {latp.name})")
                        continue
                    lat_override = np.load(str(latp)).astype(np.float32)

                override = (label_obs_loader(cn, step, 0)
                            if label_obs_loader is not None else None)
                if label_obs_loader is not None and override is None:
                    print(f"  {cn} step={step:02d}: SKIP (no input patch)")
                    continue
                r = eval_case_at_resolution(
                    net=net, optimize_fn=optimize_fn,
                    label_dense=labels_dense[ci], spacing_dense=spacings_dense[ci],
                    step_size=step, step_axis=step_axes[ci],
                    params=params, device=INFER_DEVICE,
                    use_thick_slices=params.get("use_thick_slices", False),
                    label_obs_override=override,
                    mode=params.get("sweep_mode", "thin"),
                    modality=params.get("sweep_modality", "ct"),
                    num_classes=params.get("num_classes", 5),
                    start=0,
                    latent_override=lat_override,
                )
                r["casename"] = cn
                src_results.append(r)
                # Crash-safe: persist THIS (case, step) latent immediately, not
                # only after the whole source finishes in _save_final_masks. The
                # latent is the expensive test-optimization artifact; a crash
                # mid-source used to lose every eye/step computed so far.
                _lat = r.get("latent")
                if _lat is not None and np.asarray(_lat).size > 1:
                    _ld = out_dir / "latent"
                    _ld.mkdir(parents=True, exist_ok=True)
                    np.save(str(_ld / f"{cn}_step{step:02d}.npy"),
                            np.asarray(_lat))
                print(f"  [{si}/{n_src}] {cn} step={step:02d}: "
                      f"dense={r['dice']['mean']:.3f} obs={r['dice_observed']['mean']:.3f}")
        if src_results:
            manifest.update(_save_final_masks(
                src_results, layout, out_dir,
                net=(net if args.emit_iso_prelabel_dir else None),
                emit_iso_dir=args.emit_iso_prelabel_dir,
                emit_iso_mm=args.emit_iso_mm,
            ))

    # Shard-aware manifest name so concurrent workers don't clobber each other.
    shard_tag = str(args.shard_id).replace(",", "-")
    mf_name = ("cnisp_pred_manifest.json" if args.num_shards <= 1
               else f"cnisp_pred_manifest.shard{shard_tag}of{args.num_shards}.json")
    with open(out_dir / mf_name, "w") as f:
        json.dump({"model": args.model_name, "experiment": args.experiment,
                   "label_source": args.test_label_source,
                   "shard_ids": shard_tag, "num_shards": args.num_shards,
                   "sources": manifest}, f, indent=2)
    print(f"[032] manifest -> {out_dir/mf_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
