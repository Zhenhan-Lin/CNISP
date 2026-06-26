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
    device as INFER_DEVICE,
)
from engine.test_label_sources import build_run_layout          # noqa: E402
from engine.native_mapping import (                             # noqa: E402
    invert_alignment_single_eye, _extract_sub_crop_info,
)
from diagnostics.resolution_sweep import run_sweep              # noqa: E402
from data_prep.sparsify import resolve_slice_step_axes          # noqa: E402

# Fixed canonical -> nnUNet remap (by structure name). DO NOT value-shift.
CANON2NNUNET = {0: 0, 1: 1, 2: 3, 3: 4, 4: 2}


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


def _save_final_masks(results: List[Dict], layout, out_dir: Path) -> Dict:
    """Group results by (source, step), merge eyes (canonical), remap, save one
    {1,2,3,4} native mask per (source, step). Returns a manifest dict."""
    meta_for = _meta_path_for_case(layout)
    out_dir.mkdir(parents=True, exist_ok=True)

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
            lo, sh = _extract_sub_crop_info(r, r["casename"])
            full = invert_alignment_single_eye(
                r["pred_class_map"], meta,
                sub_crop_lo_vox_dense=lo, sub_crop_shape_vox_dense=sh,
                casename=r["casename"],
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
    ap.add_argument("--experiment", default="thick", choices=["thin", "thick", "real"])
    ap.add_argument("--test-casefile", default=None,
                    help="casefile under casefiles_dir (default: test yaml's value)")
    ap.add_argument("--out-dir", default=None,
                    help="output dir (default: nnunet-c/data/cnisp_pred)")
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

    layout = build_run_layout(params)   # used for metadata/label resolution only
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
    labels_dense, spacings_dense, casenames = _load_labels_dense_per_case(
        layout, casenames_all,
    )
    if not casenames:
        print("[032] no resolvable cases; nothing to do", file=sys.stderr)
        return 1
    label_obs_loader = _build_label_obs_loader(layout)
    step_axes = resolve_slice_step_axes(params["slice_step_axis"], spacings_dense)
    sweep_cfg = dict(params.get("adaptive_step_sweep", {}))

    # ── run sweep IN MEMORY (output_dir=None -> writes nothing) ───────
    results = run_sweep(
        net=net, optimize_fn=optimize_fn, casenames=casenames,
        labels_dense=labels_dense, spacings_dense=spacings_dense,
        step_axis=step_axes, params=params, device=INFER_DEVICE,
        sweep_cfg=sweep_cfg, output_dir=None,
        label_obs_override_loader=label_obs_loader,
        real_pair=(layout.test_label_source == "real_pair"),
        on_case_done=None,
    )

    # ── save ONLY the final merged {1,2,3,4} native masks ────────────
    print("\n[032] writing final corrector masks ...")
    manifest = _save_final_masks(results, layout, out_dir)
    with open(out_dir / "cnisp_pred_manifest.json", "w") as f:
        json.dump({"model": args.model_name, "experiment": args.experiment,
                   "label_source": args.test_label_source, "sources": manifest}, f,
                  indent=2)
    print(f"[032] manifest -> {out_dir/'cnisp_pred_manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
