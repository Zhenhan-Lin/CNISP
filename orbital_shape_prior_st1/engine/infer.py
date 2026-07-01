"""
Test-time inference: latent optimization + dense reconstruction.

For each test case:
    1. Load model from best_checkpoint.pth (default) or latest periodic checkpoint
    2. Initialize z ~ N(0, 1e-4)
    3. Freeze MLP, optimize z against sparse observations
    4. Dense-sample on full grid → multi-class label map
    5. Export NIfTI + compute metrics

Two test_label_source modes share this code path (see test_default.yaml):

  atlas_gt     latent-opt input = sparsen_volume(canonical GT patch).
               Dense Dice target = the same GT patch. Ceiling curve.

  nnunet_pred  latent-opt input = per-step canonical-aligned Dataset835
               sparse-CT pred (one ``label_obs`` patch per (case, step),
               loaded from aligned_dir/labels_dataset835_step_XX/).
               Dense Dice target = atlas manual GT for atlas cases or
               Dataset835 dense pred canonical-aligned for chk_* cases.
               Output lives in a sibling runs/<run_tag>/ subdir so the
               two curves never overwrite each other's predictions.
"""

import functools
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch

from models.losses import MultiClassDiceMetric, MultiClassShapeLoss
from engine.dataset import load_casenames
from diagnostics.resolution_sweep import (
    DEFAULT_BUCKET_EDGES_MM,
    print_sweep_summary,
    run_sweep,
    save_sweep_csvs,
    step_dir_name,
)
from engine.train import create_model
from engine.io_utils import load_latest_checkpoint
from engine.native_mapping import map_results_to_native
from engine.test_label_sources import (
    RunLayout,
    build_run_layout,
    dense_target_paths,
    load_patch_as_label_tensor,
    step_input_patch_path,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _dump_pickle_atomic(obj, path: Path) -> None:
    """Pickle ``obj`` to ``path`` atomically (temp file + os.replace).

    A direct ``pickle.dump`` to the final path leaves a TRUNCATED file if
    the write fails midway (e.g. disk full), which then poisons every
    downstream reader. Writing to a sibling temp file and renaming means the
    canonical path is only ever the previous valid file or the new complete
    one -- never a partial dump.
    """
    import os

    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "wb") as f:
            pickle.dump(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _autocast_dtype() -> torch.dtype:
    """
    Pick the best mixed-precision dtype for the current GPU.

    bfloat16: matches fp32 exponent range, no GradScaler needed (Ampere+).
    float16:  half range, requires GradScaler (Volta/Turing fallback).
    fp32:     CPU fallback.
    """
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


_AUTOCAST_DTYPE = _autocast_dtype()

# ── Checkpoint loading ────────────────────────────────────────────

def load_model_checkpoint(model_dir: Path, which: str = "best", verbose: bool = True):
    """
    Load model checkpoint.

    Args:
        model_dir: directory containing checkpoints
        which: "best" (default) or "latest"
            - "best": loads best_checkpoint.pth (best val dice during training)
            - "latest": loads the most recent periodic checkpoint

    Returns:
        (model_state_dict, metadata_dict)
    """
    if which == "best":
        best_path = model_dir / "best_checkpoint.pth"
        if best_path.exists():
            if verbose:
                state = torch.load(best_path, map_location="cpu")
                dice = state.get("best_val_dice", "?")
                epoch = state.get("num_epochs_trained", "?")
                print(f"Loading best checkpoint: {best_path}")
                print(f"  Best val dice: {dice}, epoch: {epoch}")
                return state["model_state"], state
            state = torch.load(best_path, map_location="cpu")
            return state["model_state"], state
        else:
            print(f"WARNING: best_checkpoint.pth not found in {model_dir}")
            print(f"  Falling back to latest periodic checkpoint.")
            which = "latest"

    if which == "latest":
        model_state, optim_state, n_steps, n_epochs = load_latest_checkpoint(
            model_dir, "checkpoint", "pth", verbose
        )
        return model_state, {"num_epochs_trained": n_epochs, "num_steps_trained": n_steps}

    raise ValueError(f"Unknown checkpoint type: {which}. Use 'best' or 'latest'.")


# ── Latent optimization ──────────────────────────────────────────

def optimize_latent(net, labels_sparse, coords, latent_dim, lr,
                    lat_reg_lambda, num_iters, max_num_const_dsc,
                    device, verbose=True, soft=False, label_smoothing=0.1,
                    delta=None):
    """
    Test-time latent optimization for one case.

    Denoise framework (``delta`` not None)
    --------------------------------------
    The latent ``alpha_nn_test`` is fit to the nnUNet observation exactly as
    before (net + Delta both frozen; only the latent is a Parameter). AFTER
    convergence the trained Delta is applied once:

        alpha_hat = alpha_nn_test + Delta(alpha_nn_test)

    and ``alpha_hat`` is returned, so every downstream consumer (dense decode,
    iso, native map, the saved latents/<case>.npy cache) reproduces the
    corrected prediction without needing Delta again at replay time.

    Single forward+backward per iteration (no chunking).
    Memory budget assumption: the caller has filtered patch sizes so that
    a full backward graph fits in VRAM after bf16/fp16 autocast. For an
    80mm patch, bf16 on a 48 GB GPU comfortably handles up to ~15M voxels
    (single forward); beyond that the caller is responsible for downsampling
    or skipping the case.

    Soft latent-fit (``soft=True``)
    -------------------------------
    Replaces the hard one-hot CE target with a label-smoothed soft target
    (``label_smoothing`` mass spread over the off classes). The latent is
    then no longer forced to reproduce every observed voxel exactly, which
    reduces overcommitment to a noisy nnUNet observation on a degraded
    image. Dice + latent L2 are unchanged. ``soft=False`` (default) keeps
    the original hard-label optimisation bit-for-bit.
    """
    latent = torch.nn.Parameter(
        torch.normal(0.0, 1e-4, [1, latent_dim], device=device),
        requires_grad=True,
    )
    eps = float(label_smoothing) if soft else 0.0
    criterion = MultiClassShapeLoss(label_smoothing=eps).to(device)
    metric = MultiClassDiceMetric(net.num_classes).to(device)
    optimizer = torch.optim.Adam([latent], lr=lr)

    net.eval()

    use_amp = device.type == "cuda" and _AUTOCAST_DTYPE != torch.float32
    # fp16 needs loss scaling; bf16 does not.
    scaler = (torch.cuda.amp.GradScaler()
              if use_amp and _AUTOCAST_DTYPE == torch.float16
              else None)

    if verbose:
        n_vox = int(torch.tensor(labels_sparse.shape).prod().item())
        dt = ("bf16" if _AUTOCAST_DTYPE == torch.bfloat16
              else "fp16" if _AUTOCAST_DTYPE == torch.float16
              else "fp32")
        fit_mode = (f"soft(ls={eps:.2f})" if soft else "hard")
        print(f"  optimize_latent: {n_vox} voxels, dtype={dt}, "
              f"iters={num_iters}, fit={fit_mode}")

    prev_dsc, n_const = 0.0, 0
    t0 = time.time()

    for i in range(num_iters):
        optimizer.zero_grad()

        with torch.autocast(device_type=device.type,
                            dtype=_AUTOCAST_DTYPE,
                            enabled=use_amp):
            logits = net(latent, coords)
            loss = criterion(logits, labels_sparse)
            if lat_reg_lambda > 0:
                lat_reg = torch.mean(torch.sum(latent ** 2, dim=1))
                loss = loss + min(1.0, i / 100.0) * lat_reg_lambda * lat_reg

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        loss_val = loss.item()

        if (i + 1) % 10 == 0:
            with torch.no_grad():
                dsc = metric(logits, labels_sparse)["mean"]

            if verbose and (i + 1) % 100 == 0:
                print(f"  step {i+1:04d}/{num_iters}: loss={loss_val:.4f} "
                      f"dice={dsc:.3f} |z|²={torch.sum(latent**2).item():.2f} "
                      f"({time.time()-t0:.1f}s)")
                t0 = time.time()
            if round(dsc, 3) == round(prev_dsc, 3):
                n_const += 1
            else:
                n_const = 0
            if 0 < max_num_const_dsc <= n_const:
                if verbose:
                    print(f"  converged at step {i+1}")
                break
            prev_dsc = dsc

    latent_final = latent.detach()
    if delta is not None:
        # Apply the frozen Delta correction once: navigate the noisy test-fit
        # latent toward the GT-decoding latent before dense reconstruction.
        delta.eval()
        with torch.no_grad():
            resid = delta(latent_final)
            latent_final = latent_final + resid
        if verbose:
            print(f"  delta applied: |alpha_nn|={torch.norm(latent.detach()).item():.3f} "
                  f"|delta|={torch.norm(resid).item():.3f} "
                  f"|alpha_hat|={torch.norm(latent_final).item():.3f}")
    return latent_final


# ── Average shape ────────────────────────────────────────────────

def generate_average_shape(
    net,
    latent_dim: int,
    spacing: torch.Tensor,
    output_path: str,
    device: torch.device = None,
):
    """
    Generate the average shape by querying the MLP with z=0.

    The zero vector is the "center" of the latent space (L2 regularization
    pulls all training latents toward it), so f(x, z=0) represents the
    shape prior's learned mean anatomy.

    Args:
        net: trained MultiClassAutoDecoder (eval mode)
        latent_dim: dimensionality of latent vector
        spacing: [3] voxel spacing in mm for the output grid
        output_path: where to save the NIfTI file
        device: GPU/CPU (defaults to net's device)

    Returns:
        avg_map: [D1, D2, D3] integer class map (numpy)
    """
    if device is None:
        device = next(net.parameters()).device

    target_shape = (net.image_size.cpu() / spacing.cpu()).round().long()
    z_zero = torch.zeros(1, latent_dim, device=device)

    net.eval()
    avg_map = net.predict_dense(
        z_zero, target_shape.to(device), spacing.to(device),
        autocast_dtype=_AUTOCAST_DTYPE,
    )

    aff = np.diag([*spacing.cpu().numpy(), 1.0])
    nib.save(
        nib.Nifti1Image(avg_map.numpy().astype(np.uint8), aff),
        str(output_path),
    )

    n_classes = len(np.unique(avg_map.numpy()))
    print(f"  Average shape saved: {output_path} "
          f"(shape={list(avg_map.shape)}, {n_classes} classes)")
    return avg_map

# ── Visualization: observed vs reconstructed ─────────────────────

def create_obs_vs_recon_map(
    pred_map: np.ndarray,
    slice_step_size: int,
    slice_start_id: int,
    slice_axis: int,
    num_fg_classes: int = 4,
    recon_offset: int = 10,
) -> np.ndarray:
    """
    Create a label map where observed and reconstructed slices have
    different label values, for visualization.

    Observed slices: labels 1,2,3,4 (original)
    Reconstructed slices: labels 11,12,13,14 (offset by recon_offset)

    Args:
        pred_map: [D1, D2, D3] dense reconstruction (integer labels)
        slice_step_size: keep every Nth slice
        slice_start_id: starting slice index for sparsification
        slice_axis: which axis was sparsified (0/1/2)
        num_fg_classes: number of foreground classes (4 for orbital)
        recon_offset: offset to add to reconstructed labels

    Returns:
        viz_map: [D1, D2, D3] with observed labels {1..4} and
                 reconstructed labels {11..14}
    """
    dense_size = pred_map.shape[slice_axis]

    # Which slice indices in the dense volume were observed?
    observed_ids = set(
        range(slice_start_id, dense_size, slice_step_size)
    )

    viz_map = np.zeros_like(pred_map)

    for idx in range(dense_size):
        # Extract this slice from pred_map
        slc = [slice(None)] * 3
        slc[slice_axis] = idx
        pred_slice = pred_map[tuple(slc)]

        if idx in observed_ids:
            # Observed: keep original labels
            viz_map[tuple(slc)] = pred_slice
        else:
            # Reconstructed: offset foreground labels
            out_slice = np.zeros_like(pred_slice)
            for c in range(1, num_fg_classes + 1):
                out_slice[pred_slice == c] = c + recon_offset
            viz_map[tuple(slc)] = out_slice

    return viz_map

# ── Full test set inference ──────────────────────────────────────

def _pick_primary_per_case(all_results: List[Dict],
                           primary_eff_res_mm: float) -> List[Dict]:
    """For each case pick the result whose eff_res is closest to the target.

    Only start==0 results are eligible: the start-offset fan-out exists for
    coarse acquisitions (eff_res>threshold) and the native-space primary pick
    targets ~3 mm, so the canonical start is the deterministic representative.
    """
    by_case = defaultdict(list)
    for r in all_results:
        if int(r.get("slice_start_id", 0)) != 0:
            continue
        by_case[r["casename"]].append(r)
    picked = []
    for _, results in by_case.items():
        best = min(
            results,
            key=lambda r: abs(r["effective_resolution_mm"] - primary_eff_res_mm),
        )
        picked.append(best)
    return picked


def _load_labels_dense_per_case(
    layout: RunLayout, casenames: List[str]
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[str]]:
    """Load the dense Dice-target patch + spacing for each test case.

    ``layout.test_label_source`` decides which directory each case
    reads from:

      atlas_gt     - everything from ``layout.labels_dir`` (ceiling).
      nnunet_pred  - atlas cases from ``layout.labels_dir`` (manual GT
                     stays put), chk_* cases from
                     ``layout.labels_dataset835_dir`` (Dataset835 dense
                     pred canonical-aligned). chk_* cases without a
                     Dataset835 patch on disk are dropped from the
                     test set for this run with a printed warning --
                     the deployment story for that case is undefined
                     until ``cnisp-prep-dataset835-gt`` runs.
    """
    labels: List[torch.Tensor] = []
    spacings: List[torch.Tensor] = []
    surviving: List[str] = []
    n_atlas = n_chk = n_dropped = 0
    for cn in casenames:
        label_path, _meta_path = dense_target_paths(layout, cn)
        if not label_path.exists():
            print(f"  [drop case] {cn}: dense target missing at {label_path}")
            n_dropped += 1
            continue
        vol, spacing, _offset = load_patch_as_label_tensor(label_path)
        labels.append(vol)
        spacings.append(spacing)
        surviving.append(cn)
        if cn.startswith("atlas_"):
            n_atlas += 1
        else:
            n_chk += 1
    print(f"  Dense Dice targets resolved: atlas={n_atlas} chk_*={n_chk} "
          f"dropped={n_dropped}  (test_label_source={layout.test_label_source})")
    return labels, spacings, surviving


def _build_label_obs_loader(
    layout: RunLayout,
) -> Optional[Callable[[str, int],
                       Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]]:
    """Return a per-(case, step) override loader, or None for atlas_gt.

    Under ``nnunet_pred`` we read each (case, step) override from
    ``aligned_dir/labels_dataset835_step_XX/<casename>.nii.gz``. When
    the file is missing -- e.g. nnUNet failed globe localisation at
    that sparsity, so the canonical-align step refused to write -- the
    loader returns ``None`` and ``run_sweep`` skips that row.
    """
    if layout.test_label_source == "real_pair":
        # Single real low-res input patch per case (step ignored): the real
        # anisotropy is fixed by the acquisition, not swept.
        def _rp_loader(casename: str, step: int, start: int = 0):
            p = step_input_patch_path(layout, casename, step)
            if not p.exists():
                return None
            vol, spacing, offset = load_patch_as_label_tensor(p)
            return vol, spacing, offset
        return _rp_loader

    if layout.test_label_source != "nnunet_pred":
        return None

    def _loader(casename: str, step: int, start: int = 0):
        # step=1 is the dense baseline; under the deployment curve the
        # "step_01" sparse patch is just Dataset835's dense canonical-
        # aligned pred (same content as labels_dataset835/ for chk_*,
        # and a fresh canonical_align of the atlas dense pred). We
        # serve it via the same lookup so the latent-opt input grid
        # follows the input modality consistently across steps.
        p = step_input_patch_path(layout, casename, step, start)
        if not p.exists():
            return None
        vol, spacing, offset = load_patch_as_label_tensor(p)
        if step > 1:
            # load_patch_as_label_tensor uses offset = spacing/2.  On the
            # through-plane (step) axis spacing = step * dense_spacing, so
            # offset becomes step*dense_spacing/2.  The ceiling curve
            # (compute_sparse_affine) puts sparse voxel 0's centre at dense
            # voxel ``start``'s centre, i.e. dense_spacing*(start+0.5) =
            # spacing*(2*start+1)/(2*step).  For start=0 this reduces to
            # spacing/(2*step) (the original formula); for the high-eff_res
            # start-offset fan-out it shifts by the started slice.
            step_axis = int(torch.argmax(spacing))
            offset[step_axis] = spacing[step_axis] * (2 * start + 1) / (2.0 * step)
        return vol, spacing, offset

    return _loader


def _meta_path_for_case(layout: RunLayout) -> Callable[[str], Path]:
    """Resolver for native_mapping: pick the metadata tree per case.

    For Option C nnunet_pred mode, chk_* cases live in
    ``metadata_dataset835/`` (the sidecar of the Dataset835 dense
    canonical-aligned target). atlas cases always read from the
    existing ``metadata/`` tree.
    """
    def _resolve(casename: str) -> Path:
        if layout.test_label_source == "real_pair":
            return layout.metadata_realpair_gt_dir / f"{casename}.json"
        if (layout.test_label_source == "nnunet_pred"
                and not casename.startswith("atlas_")):
            return layout.metadata_dataset835_dir / f"{casename}.json"
        return layout.metadata_dir / f"{casename}.json"
    return _resolve


def _observed_meta_path_for(
    layout: RunLayout,
) -> Optional[Callable[[str, int, int], Optional[Path]]]:
    """Resolver for the OBSERVED input patch's per-step alignment metadata.

    Only meaningful under ``nnunet_pred``: the latent-opt input is a
    per-(case, step) Dataset835 sparse patch, canonical-aligned FRESH on its
    own globe centroid (a different crop than the dense target patch). CNISP's
    native/iso inversion needs THAT crop's metadata to re-frame the
    reconstruction back onto the nnUNet pred; without it the OS mask is
    mirrored/misplaced (see engine.native_mapping._deployment_index_shift).

    The metadata dirs mirror the sparse-label step dirs with
    "labels_dataset835" -> "metadata_dataset835" (written by
    nnunet/build_dataset835_sparse_patches.py). Returns ``None`` for
    atlas_gt / real_pair (input already shares the target frame, or is
    handled by post-hoc registration).
    """
    if layout.test_label_source != "nnunet_pred":
        return None
    labels_prefix = layout.labels_dataset835_step_prefix  # Path (.../labels_..._step_)
    meta_prefix = labels_prefix.with_name(
        labels_prefix.name.replace("labels_dataset835", "metadata_dataset835", 1)
    )

    def _resolve(casename: str, step: int, start: int = 0) -> Path:
        ostr = "" if int(start) == 0 else f"_o{int(start)}"
        step_dir = Path(f"{meta_prefix.as_posix()}{int(step):02d}{ostr}")
        return step_dir / f"{casename}.json"

    return _resolve


def _sub_crop_sidecar_dict(result: dict) -> dict:
    """Build the sub-patch sidecar dict for one sweep result.

    Single source of truth shared by the incremental per-case save (crash
    safety) and the end-of-run export loop, so both write byte-identical
    sidecars. Records where the 64 mm prediction sits inside the 80 mm disk
    patch -- mandatory for cache reload AND for re-decoding/inverting a saved
    latent later (e.g. to query an iso-0.5 corrector input from the .npy).
    """
    return {
        "casename": result["casename"],
        "step_size": int(result["step_size"]),
        "slice_start_id": int(result.get("slice_start_id", 0)),
        "step_axis": int(result["step_axis"]),
        "sub_crop_lo_vox_dense": list(map(int, result["sub_crop_lo_vox_dense"])),
        "sub_crop_shape_vox_dense": list(
            map(int, result["sub_crop_shape_vox_dense"])
        ),
        "sub_origin_mm_in_disk": (
            list(map(float, result["sub_origin_mm_in_disk"]))
            if result.get("sub_origin_mm_in_disk") is not None
            else None
        ),
        "disk_patch_dense_shape": (
            list(map(int, result["disk_patch_dense_shape"]))
            if result.get("disk_patch_dense_shape") is not None
            else None
        ),
        "visible_lcc_voxel_count": int(result.get("visible_lcc_voxel_count", 0)),
        "visible_total_fg_count": int(result.get("visible_total_fg_count", 0)),
    }


def infer_test_set(params):
    """
    Run controlled reconstruction on the test set with a per-case adaptive
    resolution sweep.

    Config (test yaml):
        adaptive_step_sweep:
          target_eff_res_increment_mm: 1.0
          max_num_steps_per_case: 5
          max_eff_resolution_mm: 12.0
          primary_eff_res_mm: 3.0
          summary_bucket_edges_mm: [1.0, 2.0, 3.0, 4.0, 5.0, 6.5, 8.5, 11.0, 13.0]
        slice_step_axis: auto    # or int 0/1/2 (legacy uniform RAS axis)
        # Option C switches:
        test_label_source: atlas_gt          # or nnunet_pred (deployment)
        run_tag:           atlas_gt          # output subdir name

    Output structure (rooted at ``output_basedir/<model>/runs/<run_tag>/``):
        runs/<run_tag>/
        ├── test_results.csv          (per (case, step), with eff_res bucket)
        ├── average_shape_z0.nii.gz
        ├── step_01/
        │   ├── pred/
        │   ├── latents/              one .npy per case (for cache resume)
        │   ├── obs_vs_recon/
        │   ├── iso_space/
        │   └── metadata.json         per-case spacing + eff_res
        ├── step_03/
        │   └── ...
        ├── native_space/             primary picks per source (closest to
        │                             target eff_res) — full-head, OD+OS merged
        ├── native_space_step_01/     every step mapped to native space
        │   ├── manifest.json         {source_id: full-head .nii.gz path}
        │   └── *_cnisp_step01.nii.gz
        ├── native_space_step_03/     ...
        └── native_sweep_manifest.json   top-level index over per-step manifests

    Resume artefacts
    ----------------
    Test-time inference freezes the prior MLP (see ``optimize_latent``;
    only the per-(case, step) latent vector is a torch.nn.Parameter, the
    Adam optimiser is built over ``[latent]`` alone, and ``net.eval()``
    stays in effect through the loop). The triple needed to replay any
    downstream stage (native mapping, compare_native, visualisation)
    without redoing latent optimisation is therefore:

      1. The prior checkpoint at ``model_basedir/model_name/`` (single
         file, shared across all cases and steps).
      2. ``step_XX/latents/<casename>.npy`` -- the optimised latent z.
      3. ``aligned_dir/<metadata_dirname>/<casename>.json`` -- the
         canonical-align metadata (orientation, crop slices, original
         affine, label scheme).

    ``diagnostics/resolution_sweep._try_load_cached`` already consumes
    ``step_XX/pred/<casename>_pred.nii.gz`` + the matching latent sidecar
    to skip latent opt on subsequent runs, so a plain ``cnisp-infer``
    rerun is the canonical "resume downstream" entry point.
    """

    layout = build_run_layout(params)
    model_dir = Path(params["model_basedir"]) / params["model_name"]
    output_dir = layout.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep the degradation applied by the sweep consistent with the
    # experiment folder: experiment is the authoritative knob. For the
    # ceiling curve (atlas_gt) the sweep degrades canonical GT, so
    # sweep_mode must equal the experiment (thin/thick). real_pair uses a
    # real low-res observation (no synthetic degradation), so leave it.
    if layout.experiment in ("thin", "thick"):
        if str(params.get("sweep_mode", "thin")) != layout.experiment:
            print(f"  [infer] syncing sweep_mode -> '{layout.experiment}' "
                  f"to match experiment folder")
        params["sweep_mode"] = layout.experiment

    print(f"Device: {device}")
    print(f"Run layout:")
    print(f"  experiment              = {layout.experiment}")
    print(f"  test_label_source       = {layout.test_label_source}")
    print(f"  run_tag                 = {layout.run_tag}")
    print(f"  output_dir              = {output_dir}")

    # ── Load model ────────────────────────────────────────────────
    which_ckpt = params.get("checkpoint", "best")
    model_state, ckpt_meta = load_model_checkpoint(model_dir, which_ckpt, verbose=True)

    net = create_model(params, torch.ones(3))
    net.load_state_dict(model_state["net"], strict=True)
    net = net.to(device).eval()

    # ── Denoise framework: load + freeze Delta (gated by config) ──────
    # Problem 4: a config that asks for Delta but a checkpoint without it is a
    # silent-collapse hazard -- fail loudly instead of running plain CNISP.
    den = params.get("denoise") or {}
    use_delta = bool(den.get("enabled", False)) and bool(den.get("use_delta", True))
    delta = None
    if use_delta:
        delta_state = model_state.get("delta", None)
        if delta_state is None:
            raise RuntimeError(
                "denoise.use_delta=true but the loaded checkpoint contains no "
                "'delta' weights. Re-train with denoise.use_delta or set "
                "denoise.use_delta:false / denoise.enabled:false to run plain "
                "CNISP."
            )
        from models.denoise import LatentDenoiser  # local import; avoid cycles
        delta = LatentDenoiser(
            latent_dim=params["latent_dim"],
            hidden_dim=den.get("delta_hidden_dim") or None,
            num_hidden_layers=int(den.get("delta_num_hidden_layers", 2)),
        )
        delta.load_state_dict(delta_state)
        delta = delta.to(device).eval()
        for p in delta.parameters():
            p.requires_grad_(False)
        print(f"  [denoise] Delta loaded + frozen "
              f"(hidden={den.get('delta_hidden_dim') or params['latent_dim']}, "
              f"layers={int(den.get('delta_num_hidden_layers', 2))})")

    # Bind Delta into the latent optimiser used by the sweep (no-op when None).
    optimize_fn = functools.partial(optimize_latent, delta=delta)

    # ── Load test data (dense Dice targets) ───────────────────────
    casefiles_dir = Path(params["casefiles_dir"])
    casenames_all = load_casenames(casefiles_dir / params["test_casefile"])
    labels_dense, spacings_dense, casenames = _load_labels_dense_per_case(
        layout, casenames_all,
    )
    if not casenames:
        raise SystemExit(
            f"No test cases have a resolvable dense target under "
            f"test_label_source={layout.test_label_source}. Did you run "
            f"the cnisp-prep-dataset835-gt phase?"
        )

    # ── Per-(case, step) latent-opt input override (deployment only) ─
    label_obs_loader = _build_label_obs_loader(layout)
    if label_obs_loader is not None:
        if layout.test_label_source == "real_pair":
            print(f"  label_obs override : enabled (real low-res input patches "
                  f"in {layout.labels_realpair_input_dir})")
        else:
            print(f"  label_obs override : enabled (Dataset835 sparse patches in "
                  f"{layout.labels_dataset835_step_prefix.as_posix()}XX/)")

    # ── Sweep configuration (per-case adaptive) ───────────────────
    # slice_step_axis can be an int (uniform RAS axis across all cases,
    # legacy) or "auto" (per-case argmax(patch_spacing), so each scan is
    # sparsified along its own natural through-plane direction). The
    # resolver returns one int per case regardless; runs serialize that
    # full list to disk so downstream code (compare_native, native
    # mapping, nnUNet sparsify) can mirror the same per-case choices.
    from data_prep.sparsify import resolve_slice_step_axes  # local import; avoid cycles
    step_axes = resolve_slice_step_axes(
        params["slice_step_axis"], spacings_dense,
    )
    sweep_cfg = dict(params.get("adaptive_step_sweep", {}))
    primary_eff_res = float(sweep_cfg.get("primary_eff_res_mm", 3.0))
    bucket_edges = tuple(sweep_cfg.get(
        "summary_bucket_edges_mm", DEFAULT_BUCKET_EDGES_MM
    ))

    print(f"\nTest cases: {len(casenames)}")
    print(f"Sweep cfg: {sweep_cfg}")
    print(f"slice_step_axis: {params['slice_step_axis']!r} "
          f"-> per-case axes: {step_axes}")
    print(f"Primary eff_res target: {primary_eff_res} mm")

    # ── Average shape ─────────────────────────────────────────────
    median_spacing = torch.stack(spacings_dense).median(dim=0)[0]
    print("\nGenerating average shape (z=0)...")
    generate_average_shape(
        net, params["latent_dim"], median_spacing,
        output_dir / "average_shape_z0.nii.gz", device,
    )

    # ── Mask-saving whitelist ─────────────────────────────────────
    # save_mask_source_ids restricts which sources get full .nii.gz masks
    # written (per-step pred/obs_vs_recon/iso AND native-space). Empty/None
    # -> save all (back-compat). Latents, *_sub_crop.json, metadata.json,
    # sweep_results.pkl and test_results.csv are written for ALL sources so
    # the eff_res aggregate (which reads canonical Dice from the pkl/csv) and
    # cache bookkeeping stay complete regardless of which masks are on disk.
    # Computed BEFORE the sweep so the per-case incremental native remap
    # (below) can reuse the same whitelist + metadata resolver.
    _save_ids_cfg = params.get("save_mask_source_ids") or None
    save_id_set = set(_save_ids_cfg) if _save_ids_cfg else None

    def _keep_mask(casename: str) -> bool:
        if save_id_set is None:
            return True
        # casename ends with _OD / _OS (3 chars) -> source_id.
        return casename[:-3] in save_id_set

    if save_id_set is not None:
        print(f"  [mask-whitelist] saving native/per-step masks for "
              f"{len(save_id_set)} source(s) only; latents/json/pkl/csv "
              f"kept for all.")

    meta_path_for = _meta_path_for_case(layout)
    # Observed input-patch metadata resolver (deployment re-framing; None for
    # atlas_gt / real_pair). Passed to every native/iso mapping call so the
    # reconstruction is inverted through the crop it was actually fit to.
    obs_meta_for = _observed_meta_path_for(layout)
    export_preds = params.get("export_predictions", True)

    # ── Optional per-case incremental native remap ────────────────
    # When incremental_native_remap is on (default), each case's native-grid
    # masks (native_space_step_XX[_oN]/) are written the moment that case
    # finishes its sweep -- via the run_sweep on_case_done hook -- instead of
    # only after EVERY case completes. The end-of-run loop then just (re)builds
    # the per-step manifests without re-mapping. Set it false to restore the
    # strictly-batched behaviour.
    incremental_remap = bool(params.get("incremental_native_remap", True))

    # The incremental remap must fire once PER SOURCE (both eyes merged), not
    # per eye: native masks are saved per source_id, so a per-eye emit makes
    # the second eye's single-eye render overwrite the first (the OD/OS merge
    # in map_results_to_native only sees the eyes passed in one call). We
    # therefore buffer each source's eye results and only map once ALL of that
    # source's expected eyes (from the swept casenames) have completed.
    _expected_eyes: Dict[str, set] = defaultdict(set)
    for _cn in casenames:
        _expected_eyes[_cn[:-3]].add(_cn)
    _pending_results: Dict[str, List[dict]] = defaultdict(list)
    _seen_eyes: Dict[str, set] = defaultdict(set)

    def _flush_source_native(source_id: str, case_results: List[dict]) -> None:
        """Map+merge ONE source's accumulated (OD+OS) results to native space."""
        groups: Dict[Tuple[int, int], List[dict]] = defaultdict(list)
        for r in case_results:
            groups[(int(r["step_size"]),
                    int(r.get("slice_start_id", 0)))].append(r)
        for (step, start) in sorted(groups):
            _ostr = "" if start == 0 else f"_o{start}"
            step_native_dir = (
                output_dir / f"native_space_{step_dir_name(step, start)}"
            )
            paths = map_results_to_native(
                groups[(step, start)], layout.metadata_dir, step_native_dir,
                suffix=f"_cnisp_step{step:02d}{_ostr}",
                meta_path_for_casename=meta_path_for,
                save_source_ids=save_id_set,
                observed_meta_path_for=obs_meta_for,
            )
            if paths:
                print(f"    [native] {source_id} step={step:02d}{_ostr}: "
                      f"{len(paths)} mask(s) -> {step_native_dir.name}/")

    def _emit_case_native(casename: str, case_results: List[dict]) -> None:
        if not export_preds:
            return
        source_id = casename[:-3]
        _pending_results[source_id].extend(case_results)
        _seen_eyes[source_id].add(casename)
        # Only map once every expected eye for this source has finished, so
        # OD+OS land in the same native mask instead of overwriting.
        if _seen_eyes[source_id] >= _expected_eyes.get(source_id, {casename}):
            _flush_source_native(source_id, _pending_results.pop(source_id))
            _seen_eyes.pop(source_id, None)

    # ── Incremental (crash-safe) per-case latent + sidecar save ───
    # Write each finished (case, step)'s latent + sub_crop sidecar the MOMENT
    # the case completes, not only after the WHOLE sweep. The latent is the
    # expensive, irreproducible artifact (test-time optimisation); a crash
    # mid-sweep used to lose every latent computed so far. The sidecar is cheap
    # and is what lets the latent be re-decoded (e.g. at iso-0.5) + inverted to
    # native later. The end-of-run export loop below still rewrites these
    # (idempotent), so this only ADDS durability -- it never changes outputs.
    def _save_case_artifacts_incremental(casename: str,
                                         case_results: List[dict]) -> None:
        if not export_preds:
            return
        for result in case_results:
            step = int(result["step_size"])
            start = int(result.get("slice_start_id", 0))
            step_dir = output_dir / step_dir_name(step, start)
            pred_dir = step_dir / "pred"
            pred_dir.mkdir(parents=True, exist_ok=True)
            with open(pred_dir / f"{result['casename']}_sub_crop.json", "w") as f:
                json.dump(_sub_crop_sidecar_dict(result), f, indent=2)
            # Skip cache hits without a real latent (placeholder size<=1).
            if result.get("latent_missing", False) or result["latent"].size <= 1:
                continue
            lat_dir = step_dir / "latents"
            lat_dir.mkdir(parents=True, exist_ok=True)
            np.save(str(lat_dir / f"{result['casename']}.npy"),
                    np.asarray(result["latent"], dtype=np.float32))

    def _on_case_done(casename: str, case_results: List[dict]) -> None:
        # Durable artifacts first (latent + sidecar), then the optional
        # incremental native remap (masks are also written per-source there).
        _save_case_artifacts_incremental(casename, case_results)
        if incremental_remap:
            _emit_case_native(casename, case_results)

    # ── Run sweep ─────────────────────────────────────────────────
    all_results = run_sweep(
        net=net,
        optimize_fn=optimize_fn,
        casenames=casenames,
        labels_dense=labels_dense,
        spacings_dense=spacings_dense,
        step_axis=step_axes,
        params=params,
        device=device,
        sweep_cfg=sweep_cfg,
        output_dir=output_dir,
        label_obs_override_loader=label_obs_loader,
        real_pair=(layout.test_label_source == "real_pair"),
        on_case_done=_on_case_done,
    )

    # Flush any source whose eye set never completed (e.g. an eye produced
    # zero sweep rows, so on_case_done never fired for it). Emit with whatever
    # eyes we have so their native masks still get written before the per-step
    # manifests are (re)built below.
    if incremental_remap and _pending_results:
        for source_id, buffered in sorted(_pending_results.items()):
            if not buffered:
                continue
            print(f"  [native] flushing incomplete source {source_id} "
                  f"({len(_seen_eyes.get(source_id, set()))}/"
                  f"{len(_expected_eyes.get(source_id, set()))} eye(s) seen)")
            _flush_source_native(source_id, buffered)
        _pending_results.clear()
        _seen_eyes.clear()

    # ── Export predictions per step subdirectory ──────────────────
    if params.get("export_predictions", True):
        step_metadata = defaultdict(list)

        for result in all_results:
            step = result["step_size"]
            start = int(result.get("slice_start_id", 0))
            step_dir = output_dir / step_dir_name(step, start)
            pred_dir = step_dir / "pred"
            lat_dir = step_dir / "latents"
            ovr_dir = step_dir / "obs_vs_recon"
            iso_dir = step_dir / "iso_space"
            for d in [pred_dir, lat_dir, ovr_dir, iso_dir]:
                d.mkdir(parents=True, exist_ok=True)

            sp = result["spacing"]
            aff = np.diag([*sp, 1.0])
            casename = result["casename"]

            case_axis = int(result["step_axis"])
            step_metadata[(step, start)].append({
                "casename": casename,
                "spacing_xyz_mm": [float(s) for s in sp],
                "effective_through_plane_mm": result["effective_resolution_mm"],
                "step_size": step,
                "slice_start_id": start,
                "step_axis": case_axis,
                "n_observed_slices": result["n_observed_slices"],
                "n_total_slices": result["n_total_slices"],
                "dice_dense_mean": round(result["dice"]["mean"], 4),
                "dice_observed_mean": round(result["dice_observed"]["mean"], 4),
            })

            keep_mask = _keep_mask(casename)

            # pred/  (re-saved when on the whitelist; tolerant of cache hits)
            if keep_mask:
                nib.save(
                    nib.Nifti1Image(result["pred_class_map"].astype(np.uint8), aff),
                    str(pred_dir / f"{casename}_pred.nii.gz"),
                )
            # Sub-patch sidecar: tells the cache-reload path
            # (diagnostics.resolution_sweep._try_load_cached) where this
            # 64 mm prediction lives inside the 80 mm canonical disk
            # patch, and tells native_mapping how to compose
            #   pred → disk → full volume
            # without re-running sparsify or LCC. Mandatory under the
            # inner-crop pipeline.
            with open(pred_dir / f"{casename}_sub_crop.json", "w") as f:
                json.dump(_sub_crop_sidecar_dict(result), f, indent=2)

            # latents/  (sidecar so cache resume keeps iso reconstruction
            # working AND so any downstream replay -- native mapping,
            # compare_native, viz -- can skip latent opt by feeding this
            # back through net.predict_dense with the frozen prior).
            if not result.get("latent_missing", False) and result["latent"].size > 1:
                np.save(str(lat_dir / f"{casename}.npy"),
                        np.asarray(result["latent"], dtype=np.float32))

            # obs_vs_recon/  (whitelist only)
            if keep_mask:
                obs_vs_recon = create_obs_vs_recon_map(
                    result["pred_class_map"],
                    slice_step_size=step,
                    slice_start_id=start,
                    slice_axis=case_axis,
                )
                nib.save(
                    nib.Nifti1Image(obs_vs_recon.astype(np.uint8), aff),
                    str(ovr_dir / f"{casename}_obs_vs_recon.nii.gz"),
                )

            # iso_space/  (whitelist only; skip if anisotropy is negligible
            # OR latent unavailable)
            if not keep_mask:
                continue
            spacing_t = torch.from_numpy(sp)
            iso_sp = spacing_t.min().repeat(3)
            if torch.allclose(iso_sp, spacing_t, atol=0.01):
                continue
            if result.get("latent_missing", False) or result["latent"].size <= 1:
                # Cache hit without a saved latent: don't fabricate an iso
                # reconstruction (predict_dense would either fail with a
                # shape mismatch or return garbage from a placeholder).
                continue
            latent_t = torch.from_numpy(
                np.asarray(result["latent"], dtype=np.float32)
            ).unsqueeze(0).to(device)
            iso_target = (net.image_size.cpu() / iso_sp).round().long()
            with torch.no_grad():
                iso_map = net.predict_dense(
                    latent_t, iso_target.to(device), iso_sp.to(device),
                    autocast_dtype=_AUTOCAST_DTYPE,
                )
            iso_aff = np.diag([*iso_sp.numpy(), 1.0])
            nib.save(
                nib.Nifti1Image(iso_map.numpy().astype(np.uint8), iso_aff),
                str(iso_dir / f"{casename}_pred_iso.nii.gz"),
            )

        for (step, start), cases_meta in step_metadata.items():
            step_dir = output_dir / step_dir_name(step, start)
            meta_path = step_dir / "metadata.json"
            # step_axis may differ per case under slice_step_axis: auto;
            # surface the full mapping here as well as the top-level value
            # (set only if all cases agree) for legacy readers.
            unique_axes = sorted({int(c["step_axis"]) for c in cases_meta})
            with open(meta_path, "w") as f:
                json.dump({
                    "step_size": step,
                    "slice_start_id": start,
                    "step_axis": (unique_axes[0]
                                  if len(unique_axes) == 1 else None),
                    "step_axes_unique": unique_axes,
                    "n_cases": len(cases_meta),
                    "cases": cases_meta,
                }, f, indent=2)
            print(f"  Metadata saved: {meta_path}")

    # ── Class names for summary ───────────────────────────────────
    from data_prep.canonical_align import CANONICAL_LABEL_NAMES
    num_fg = net.num_classes - 1
    class_names = [CANONICAL_LABEL_NAMES.get(c, f"class_{c}")
                   for c in range(1, num_fg + 1)]

    # ── Print & save ──────────────────────────────────────────────
    ckpt_info = (f"Checkpoint: {which_ckpt} "
                 f"(epoch {ckpt_meta.get('num_epochs_trained', '?')})")
    print_sweep_summary(all_results, class_names,
                        bucket_edges=bucket_edges, ckpt_info=ckpt_info)
    save_sweep_csvs(all_results, class_names, output_dir,
                    bucket_edges=bucket_edges)

    # ── Map primary-eff_res predictions to native space ───────────
    # (meta_path_for was resolved before the sweep for the incremental hook.)
    primary_results = _pick_primary_per_case(all_results, primary_eff_res)
    if primary_results and params.get("export_predictions", True):
        native_dir = output_dir / "native_space"
        chosen = [(r["casename"], r["step_size"],
                   round(r["effective_resolution_mm"], 2))
                  for r in primary_results]
        print(f"\nMapping primary predictions to native space "
              f"(target eff_res {primary_eff_res} mm):")
        for cn, st, er in chosen:
            print(f"  {cn}: step={st}, eff_res={er} mm")
        native_paths = map_results_to_native(
            primary_results, layout.metadata_dir, native_dir,
            meta_path_for_casename=meta_path_for,
            save_source_ids=save_id_set,
            observed_meta_path_for=obs_meta_for,
        )
        print(f"Native-space predictions: {native_dir} ({len(native_paths)} volumes)")

    # ── Map EVERY sweep step to native space ──────────────────────
    # Mirrors what nnunet/build_cnisp_native_sweep.py does as a backfill
    # for already-run experiments; here it is folded into the inference
    # loop so a single run produces every artifact the cross-model
    # comparison (see nnunet/compare_native.py) consumes.
    if all_results and params.get("export_predictions", True):
        # Group by (step, start) so the high-eff_res start fan-out lands in
        # parallel native_space_step_XX[_oN]/ dirs. start=0 keeps the legacy
        # native_space_step_XX/ name + step-keyed manifest entry.
        by_step: Dict[Tuple[int, int], List[dict]] = defaultdict(list)
        for r in all_results:
            by_step[(int(r["step_size"]),
                     int(r.get("slice_start_id", 0)))].append(r)
        sweep_manifest: Dict[str, Dict[str, str]] = {}
        print(f"\nMapping all sweep steps to native space "
              f"({len(by_step)} (step,start) values):")
        for (step, start) in sorted(by_step):
            _sd = step_dir_name(step, start)
            step_native_dir = output_dir / f"native_space_{_sd}"
            _ostr = "" if start == 0 else f"_o{start}"
            suffix = f"_cnisp_step{step:02d}{_ostr}"
            if incremental_remap:
                # Masks were already written per-case by _emit_case_native;
                # this loop only (re)builds the manifest below.
                step_paths: List[Path] = []
            else:
                step_paths = map_results_to_native(
                    by_step[(step, start)], layout.metadata_dir, step_native_dir,
                    suffix=suffix,
                    meta_path_for_casename=meta_path_for,
                    save_source_ids=save_id_set,
                    observed_meta_path_for=obs_meta_for,
                )

            # ``by_source_id`` stores **basename only** -- consumers
            # (compare_native.py, visualize.py audit) anchor it against
            # the manifest's own directory. This makes the manifest
            # location-independent: moving runs/<run_tag>/ around keeps
            # the manifest valid, no path rewrite needed.
            per_step_manifest: Dict[str, str] = {}
            seen = set()
            for r in by_step[(step, start)]:
                mp = meta_path_for(r["casename"])
                if not mp.exists():
                    continue
                with open(mp) as f:
                    m = json.load(f)
                sid = str(m["source_id"])
                if sid in seen:
                    continue
                # Only list sources whose native mask was actually written
                # (the whitelist), so downstream readers don't point at a
                # missing file.
                if save_id_set is not None and sid not in save_id_set:
                    continue
                seen.add(sid)
                stem = (Path(m["original_nifti_path"]).name
                        .replace(".nii.gz", "").replace(".nii", ""))
                per_step_manifest[sid] = f"{stem}{suffix}.nii.gz"

            with open(step_native_dir / "manifest.json", "w") as f:
                json.dump({
                    "model_name": params["model_name"],
                    "run_tag": layout.run_tag,
                    "test_label_source": layout.test_label_source,
                    "step_size": step,
                    "slice_start_id": start,
                    "suffix": suffix,
                    "n_sources": len(per_step_manifest),
                    "by_source_id": per_step_manifest,
                }, f, indent=2)
            # Manifest key: bare step for start=0 (back-compat), step_oN else.
            _mkey = str(step) if start == 0 else f"{step}_o{start}"
            sweep_manifest[_mkey] = per_step_manifest
            _n_masks = (len(per_step_manifest) if incremental_remap
                        else len(step_paths))
            print(f"  {_sd}: {step_native_dir} ({_n_masks} sources"
                  f"{' [written per-case]' if incremental_remap else ''})")

        with open(output_dir / "native_sweep_manifest.json", "w") as f:
            json.dump({
                "model_name": params["model_name"],
                "run_tag": layout.run_tag,
                "test_label_source": layout.test_label_source,
                "primary_eff_res_mm": primary_eff_res,
                # Whitelist of source_ids whose native masks were saved
                # (null = all). The backfill (build_cnisp_native_sweep.py)
                # reads this so it applies the same filter on legacy runs.
                "save_mask_source_ids": (sorted(save_id_set)
                                         if save_id_set is not None else None),
                # Recorded so downstream (nnUNet sparsify, native mapping)
                # can mirror per-case axes; sweep_results.pkl rows hold
                # the authoritative per-(case, step) value.
                "slice_step_axis_cfg": params["slice_step_axis"],
                "slice_step_axes_per_case": {
                    cn: int(ax) for cn, ax in zip(casenames, step_axes)
                },
                "steps": sweep_manifest,
            }, f, indent=2)

    # ── Optional iso-0.5 prelabel emit (ADDITIVE; nnUNet-C corrector ch1..4) ──
    # Decode each fitted latent on a FIXED iso grid (default 0.5 mm) and place it
    # back into a full-head iso volume via the iso link map_iso_results_to_native
    # (iso_sp pinned to iso_mm, head grid = FOV + iso_mm, NOT the GT/native grid).
    # This is a PURE EXTRA output: native masks, CSVs, pickles and Dice above are
    # byte-identical. Off unless params['emit_iso_prelabel']['enabled'] is set
    # (03_infer.py --emit-iso-prelabel-dir). The fitted latent is already the
    # Delta-corrected alpha_hat, so decoding it reproduces the prediction.
    iso_cfg = params.get("emit_iso_prelabel") or {}
    if (iso_cfg.get("enabled") and all_results
            and params.get("export_predictions", True)):
        from engine.native_mapping import map_iso_results_to_native
        iso_mm = float(iso_cfg.get("iso_mm", 0.5))
        iso_out = Path(iso_cfg["out_dir"])
        iso_out.mkdir(parents=True, exist_ok=True)
        print(f"\n[iso-prelabel] emit iso-{iso_mm}mm full-head masks -> {iso_out}")
        by_step_iso: Dict[int, List[dict]] = defaultdict(list)
        for r in all_results:
            # corrector consumes the canonical start=0 pick only
            if int(r.get("slice_start_id", 0)) != 0:
                continue
            if r.get("latent_missing", False) or r["latent"].size <= 1:
                continue
            lat = torch.from_numpy(
                np.asarray(r["latent"], dtype=np.float32)
            ).unsqueeze(0).to(device)
            tgt = torch.round(
                net.image_size.detach().cpu().float() / iso_mm
            ).long()
            with torch.no_grad():
                iso_map = net.predict_dense(
                    lat, tgt.to(device),
                    torch.full((3,), iso_mm, dtype=torch.float32).to(device),
                    autocast_dtype=_AUTOCAST_DTYPE,
                )
            r["pred_class_map_iso"] = iso_map.numpy().astype(np.int16)
            by_step_iso[int(r["step_size"])].append(r)
        n_iso = 0
        for step, results_step in sorted(by_step_iso.items()):
            # Mirror the native_space layout so nnUNet-C consumes it uniformly:
            #   <iso_out>/native_space_step_XX/<stem>_cnisp_iso_stepXX.nii.gz
            #   <iso_out>/native_space_step_XX/manifest.json {by_source_id}
            step_dir = iso_out / f"native_space_step_{step:02d}"
            suffix = f"_cnisp_iso_step{step:02d}"
            paths = map_iso_results_to_native(
                results_step, layout.metadata_dir, step_dir,
                suffix=suffix, iso_mm=iso_mm,
                meta_path_for_casename=meta_path_for,
                observed_meta_path_for=obs_meta_for,
            )
            n_iso += len(paths)
            # by_source_id manifest (basename only; mirrors the native path).
            per_step: Dict[str, str] = {}
            seen: set = set()
            for r in results_step:
                mp = meta_path_for(r["casename"])
                if not mp.exists():
                    continue
                with open(mp) as f:
                    m = json.load(f)
                sid = str(m["source_id"])
                if sid in seen:
                    continue
                seen.add(sid)
                stem = (Path(m["original_nifti_path"]).name
                        .replace(".nii.gz", "").replace(".nii", ""))
                per_step[sid] = f"{stem}{suffix}.nii.gz"
            with open(step_dir / "manifest.json", "w") as f:
                json.dump({
                    "iso_mm": iso_mm,
                    "step_size": step,
                    "suffix": suffix,
                    "run_tag": layout.run_tag,
                    "test_label_source": layout.test_label_source,
                    "experiment": layout.experiment,
                    "by_source_id": per_step,
                }, f, indent=2)
        print(f"[iso-prelabel] wrote {n_iso} iso-{iso_mm}mm head mask(s) under "
              f"{iso_out}/native_space_step_XX/ (scheme=original; downstream "
              f"remaps to {{1,2,3,4}} by name)")

    # ── Pickle layout ────────────────────────────────────────────
    # inference_results.pkl : per-case primary picks (one row per case),
    #     consumed by map_to_native.py and downstream visualization
    # sweep_results.pkl     : full per-(case, step) sweep, used by
    #     scripts/04_visualization.py and by nnunet/compare_native.py
    #
    # Write atomically (temp file + os.replace) so a disk-full / crashed
    # dump can never leave a TRUNCATED canonical pickle behind -- the old
    # valid file (or nothing) survives instead, and downstream consumers
    # won't hit "UnpicklingError: pickle data was truncated".
    _dump_pickle_atomic(primary_results, output_dir / "inference_results.pkl")
    _dump_pickle_atomic(all_results, output_dir / "sweep_results.pkl")
    print(f"\nPickled {len(primary_results)} primary results "
          f"and {len(all_results)} sweep rows under {output_dir}")

    return {"primary": primary_results, "all": all_results}