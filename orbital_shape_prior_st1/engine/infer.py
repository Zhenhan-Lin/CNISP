"""
Test-time inference: latent optimization + dense reconstruction.

For each test case:
    1. Load model from best_checkpoint.pth (default) or latest periodic checkpoint
    2. Initialize z ~ N(0, 1e-4)
    3. Freeze MLP, optimize z against sparse observations
    4. Dense-sample on full grid → multi-class label map
    5. Export NIfTI + compute metrics
"""

import time
from pathlib import Path
from typing import Dict, List

import nibabel as nib
import numpy as np
import torch
import json
from collections import defaultdict

from models.multiclass_ad import MultiClassAutoDecoder
from models.losses import MultiClassShapeLoss, MultiClassDiceMetric
from engine.dataset import PhaseType, create_data_loader, load_casenames, load_orbital_volumes
from diagnostics.resolution_sweep import run_sweep, print_sweep_summary, save_sweep_csvs
from data_prep.sparsify import sparsen_volume
from engine.train import create_model
from engine.io_utils import load_latest_checkpoint
from engine.native_mapping import map_results_to_native


MAX_POINTS_PER_CHUNK = 3_000_000

def _flatten_spatial(tensor):
    """Flatten spatial dims: [1, D1, D2, D3, ...] → [1, N, 1, 1, ...]"""
    shape = tensor.shape
    if tensor.dim() == 5:
        # coords: [1, D1, D2, D3, 3] → [1, N, 1, 1, 3]
        return tensor.reshape(1, -1, 1, 1, shape[-1])
    elif tensor.dim() == 4:
        # labels: [1, D1, D2, D3] → [1, N, 1, 1]
        return tensor.reshape(1, -1, 1, 1)
    return tensor

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
                    device, verbose=True):
    """
    Test-time latent optimization for one case.
 
    Supports chunked forward pass for large volumes (e.g. step=1 dense):
    splits coordinates into chunks, accumulates gradients on the shared
    latent, so memory usage is bounded regardless of volume size.
    """
    latent = torch.nn.Parameter(
        torch.normal(0.0, 1e-4, [1, latent_dim], device=device),
        requires_grad=True,
    )
    criterion = MultiClassShapeLoss().to(device)
    metric = MultiClassDiceMetric(net.num_classes).to(device)
    optimizer = torch.optim.Adam([latent], lr=lr)
 
    net.eval()
 
    # Flatten spatial dims for uniform chunking
    coords_flat = _flatten_spatial(coords)     # [1, N, 1, 1, 3]
    labels_flat = _flatten_spatial(labels_sparse)  # [1, N, 1, 1]
    total_points = coords_flat.shape[1]
    use_chunks = total_points > MAX_POINTS_PER_CHUNK
 
    if use_chunks:
        n_chunks = (total_points + MAX_POINTS_PER_CHUNK - 1) // MAX_POINTS_PER_CHUNK
        if verbose:
            print(f"  Chunked optimization: {total_points} points → "
                  f"{n_chunks} chunks of ≤{MAX_POINTS_PER_CHUNK}")
 
    prev_dsc, n_const = 0.0, 0
    t0 = time.time()
 
    for i in range(num_iters):
        optimizer.zero_grad()
 
        if use_chunks:
            # Chunked forward: accumulate gradients on latent
            total_loss = 0.0
            for c_start in range(0, total_points, MAX_POINTS_PER_CHUNK):
                c_end = min(c_start + MAX_POINTS_PER_CHUNK, total_points)
                c_coords = coords_flat[:, c_start:c_end]
                c_labels = labels_flat[:, c_start:c_end]
                c_logits = net(latent, c_coords)
                c_loss = criterion(c_logits, c_labels) * (c_end - c_start) / total_points
                c_loss.backward()
                total_loss += c_loss.item()
 
            if lat_reg_lambda > 0:
                lat_reg = torch.mean(torch.sum(latent ** 2, dim=1))
                reg_loss = min(1.0, i / 100.0) * lat_reg_lambda * lat_reg
                reg_loss.backward()
                total_loss += reg_loss.item()
 
            optimizer.step()
            loss_val = total_loss
        else:
            # Single forward (fits in memory)
            logits = net(latent, coords)
            loss = criterion(logits, labels_sparse)
            if lat_reg_lambda > 0:
                lat_reg = torch.mean(torch.sum(latent ** 2, dim=1))
                loss = loss + min(1.0, i / 100.0) * lat_reg_lambda * lat_reg
            loss.backward()
            optimizer.step()
            loss_val = loss.item()
 
        if (i + 1) % 10 == 0:
            with torch.no_grad():
                if use_chunks:
                    # Chunked metric computation
                    all_logits = []
                    for c_start in range(0, total_points, MAX_POINTS_PER_CHUNK):
                        c_end = min(c_start + MAX_POINTS_PER_CHUNK, total_points)
                        c_logits = net(latent, coords_flat[:, c_start:c_end])
                        all_logits.append(c_logits)
                    full_logits = torch.cat(all_logits, dim=1)
                    dsc = metric(full_logits, labels_flat)["mean"]
                else:
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
 
    return latent.detach()


# ── Dice computation ─────────────────────────────────────────────

def compute_hard_dice(pred_map, gt_map, num_classes):
    """Compute per-class and mean hard Dice between integer label maps."""
    per_class = []
    for c in range(1, num_classes):  # skip BG
        p = (pred_map == c)
        g = (gt_map == c)
        inter = np.sum(p & g)
        total = np.sum(p) + np.sum(g)
        dice = 2.0 * inter / (total + 1e-5)
        per_class.append(float(dice))
    return {"mean": float(np.mean(per_class)), "per_class": per_class}

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
    avg_map = net.predict_dense(z_zero, target_shape.to(device), spacing.to(device))

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
    sparse_shape: tuple,
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
        sparse_shape: shape of the sparsified volume (to infer which slices were kept)
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

# ── Single-case inference ────────────────────────────────────────

def infer_single_case(net, batch, params, device):
    labels_sparse = batch["labels"].to(device)
    coords = batch["coords"].to(device)
    casename = batch["casenames"]
    if isinstance(casename, (list, tuple)):
        casename = casename[0]

    print(f"\nCase: {casename}")

    latent = optimize_latent(
        net, labels_sparse, coords,
        latent_dim=params["latent_dim"],
        lr=params.get("latent_lr", 1e-2),
        lat_reg_lambda=params["lat_reg_lambda"],
        num_iters=params.get("latent_num_iters", 1200),
        max_num_const_dsc=params.get("max_num_const_train_dsc", -1),
        device=device,
    )

    # Dense reconstruction at NATIVE spacing (for dice computation)
    if "labels_hr" in batch:
        gt = batch["labels_hr"]
        target_shape = torch.tensor(
            gt.shape[1:] if gt.dim() == 4 else gt.shape
        )
        spacing = batch.get("spacings_hr", batch["spacings"])[0]
    else:
        target_shape = torch.tensor(labels_sparse.shape[1:])
        spacing = batch["spacings"][0]

    pred_map = net.predict_dense(latent, target_shape.to(device), spacing.to(device))
    pred_np = pred_map.numpy()

    result = {
        "casename": casename,
        "pred_class_map": pred_np,
        "spacing": spacing.numpy(),
        "latent": latent.cpu().squeeze(0).numpy(),
    }

    # Dice against full-resolution GT (same grid, same spacing)
    if "labels_hr" in batch:
        gt_np = gt.squeeze(0).numpy() if gt.dim() == 4 else gt.numpy()
        result["gt_class_map"] = gt_np
        dice = compute_hard_dice(pred_np, gt_np, net.num_classes)
        result["dice"] = dice
        print(f"  dice={dice['mean']:.3f} per-class={[f'{d:.3f}' for d in dice['per_class']]}")

    # Create observed-vs-reconstructed visualization map
    result["pred_obs_vs_recon"] = create_obs_vs_recon_map(
        pred_np,
        sparse_shape=tuple(labels_sparse.shape[1:]),
        slice_step_size=params["slice_step_size"],
        slice_start_id=0,  # determined by seed, but 0 is approximate
        slice_axis=params["slice_step_axis"],
    )

    # Isotropic reconstruction for visualization (separate from dice)
    iso_sp = spacing.min().repeat(3)
    if not torch.allclose(iso_sp, spacing, atol=0.01):
        iso_target = (net.image_size.cpu() / iso_sp).round().long()
        iso_map = net.predict_dense(latent, iso_target.to(device), iso_sp.to(device))
        result["pred_class_map_iso"] = iso_map.numpy()
        result["spacing_iso"] = iso_sp.numpy()

    return result


# ── Full test set inference ──────────────────────────────────────
# Replace everything from this line to the end of infer.py.
#
# Also add this import at the top of infer.py:
#   from engine.dataset import ..., load_casenames, load_orbital_volumes
 
def infer_test_set(params):
    """
    Run controlled reconstruction on the test set.
 
    Supports resolution sweep: evaluates at multiple effective through-plane
    resolutions when test_step_sizes has multiple values.
 
    Config (test yaml):
        test_step_sizes: [1, 3, 5, 7, 9]   # resolution sweep
        test_step_sizes: [4]                 # single resolution
        # Omit → defaults to [slice_step_size]
 
    Output structure:
        output_dir/
        ├── test_results.csv
        ├── test_summary.csv
        ├── average_shape_z0.nii.gz
        ├── step_01/
        │   ├── pred/
        │   ├── obs_vs_recon/
        │   ├── iso_space/
        │   └── metadata.json      per-case spacing + effective resolution
        ├── step_03/
        │   └── ...
        └── native_space/   (primary step only)
    """
 
    model_dir = Path(params["model_basedir"]) / params["model_name"]
    output_dir = Path(params["output_basedir"]) / params["model_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
 
    # ── Load model ────────────────────────────────────────────────
    which_ckpt = params.get("checkpoint", "best")
    model_state, ckpt_meta = load_model_checkpoint(model_dir, which_ckpt, verbose=True)
 
    net = create_model(params, torch.ones(3))
    net.load_state_dict(model_state["net"], strict=True)
    net = net.to(device).eval()
 
    # ── Load test data (dense) ────────────────────────────────────
    labels_dir = Path(params["aligned_dir"]) / params.get("labels_dirname", "labels")
    casefiles_dir = Path(params["casefiles_dir"])
    casenames = load_casenames(casefiles_dir / params["test_casefile"])
    labels_dense, spacings_dense = load_orbital_volumes(labels_dir, casenames)
 
    # ── Step sizes ────────────────────────────────────────────────
    step_axis = params["slice_step_axis"]
    step_sizes = params.get("test_step_sizes", [params["slice_step_size"]])
    if isinstance(step_sizes, int):
        step_sizes = [step_sizes]
    primary_step = params["slice_step_size"]
 
    effective_res = [float(spacings_dense[0][step_axis]) * s for s in step_sizes]
    print(f"\nTest cases: {len(casenames)}")
    print(f"Step sizes: {step_sizes}")
    print(f"Effective resolutions (mm): {[f'{r:.1f}' for r in effective_res]}")
 
    # ── Average shape ─────────────────────────────────────────────
    median_spacing = torch.stack(spacings_dense).median(dim=0)[0]
    print("\nGenerating average shape (z=0)...")
    generate_average_shape(
        net, params["latent_dim"], median_spacing,
        output_dir / "average_shape_z0.nii.gz", device,
    )
 
    # ── Run sweep ─────────────────────────────────────────────────
    all_results = run_sweep(
        net=net,
        optimize_fn=optimize_latent,
        casenames=casenames,
        labels_dense=labels_dense,
        spacings_dense=spacings_dense,
        step_sizes=step_sizes,
        step_axis=step_axis,
        params=params,
        device=device,
        output_dir=output_dir
    )
 
    # ── Export predictions per step subdirectory ──────────────────
    #   step_XX/
    #   ├── pred/                  dense prediction at native spacing
    #   ├── obs_vs_recon/          observed vs reconstructed visualization
    #   ├── iso_space/             isotropic resampled prediction
    #   └── metadata.json          per-case spacing + resolution info
    if params.get("export_predictions", True):
        step_metadata = defaultdict(list)
 
        for result in all_results:
            step = result["step_size"]
            step_dir = output_dir / f"step_{step:02d}"
            pred_dir = step_dir / "pred"
            ovr_dir = step_dir / "obs_vs_recon"
            iso_dir = step_dir / "iso_space"
            for d in [pred_dir, ovr_dir, iso_dir]:
                d.mkdir(parents=True, exist_ok=True)
 
            sp = result["spacing"]
            aff = np.diag([*sp, 1.0])
            casename = result["casename"]
 
            # Collect per-case metadata for this step
            step_metadata[step].append({
                "casename": casename,
                "spacing_xyz_mm": [float(s) for s in sp],
                "effective_through_plane_mm": result["effective_resolution_mm"],
                "step_size": step,
                "step_axis": step_axis,
                "n_observed_slices": result["n_observed_slices"],
                "n_total_slices": result["n_total_slices"],
                "dice_dense_mean": round(result["dice"]["mean"], 4),
                "dice_observed_mean": round(result["dice_observed"]["mean"], 4),
            })
 
            # pred/
            nib.save(
                nib.Nifti1Image(result["pred_class_map"].astype(np.uint8), aff),
                str(pred_dir / f"{casename}_pred.nii.gz"),
            )
 
            # obs_vs_recon/
            obs_vs_recon = create_obs_vs_recon_map(
                result["pred_class_map"],
                sparse_shape=tuple(
                    result["gt_class_map"].shape[:step_axis]
                    + (result["n_observed_slices"],)
                    + result["gt_class_map"].shape[step_axis+1:]
                ),
                slice_step_size=step if step > 1 else 1,
                slice_start_id=0,
                slice_axis=step_axis,
            )
            nib.save(
                nib.Nifti1Image(obs_vs_recon.astype(np.uint8), aff),
                str(ovr_dir / f"{casename}_obs_vs_recon.nii.gz"),
            )
 
            # iso_space/
            spacing_t = torch.from_numpy(sp)
            iso_sp = spacing_t.min().repeat(3)
            if not torch.allclose(iso_sp, spacing_t, atol=0.01):
                latent_t = torch.from_numpy(
                    result["latent"]
                ).unsqueeze(0).to(device)
                iso_target = (net.image_size.cpu() / iso_sp).round().long()
                with torch.no_grad():
                    iso_map = net.predict_dense(
                        latent_t, iso_target.to(device), iso_sp.to(device),
                    )
                iso_aff = np.diag([*iso_sp.numpy(), 1.0])
                nib.save(
                    nib.Nifti1Image(iso_map.numpy().astype(np.uint8), iso_aff),
                    str(iso_dir / f"{casename}_pred_iso.nii.gz"),
                )
 
        # Save metadata JSON per step directory
        for step, cases_meta in step_metadata.items():
            step_dir = output_dir / f"step_{step:02d}"
            meta_path = step_dir / "metadata.json"
            with open(meta_path, "w") as f:
                json.dump({
                    "step_size": step,
                    "step_axis": step_axis,
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
    print_sweep_summary(all_results, step_sizes, class_names, ckpt_info)
    save_sweep_csvs(all_results, step_sizes, class_names, output_dir)
 
    # ── Map primary-step predictions to native space ──────────────
    primary_results = [r for r in all_results if r["step_size"] == primary_step]
    if primary_results and params.get("export_predictions", True):
        meta_dir = Path(params["aligned_dir"]) / "metadata"
        native_dir = output_dir / "native_space"
        print(f"\nMapping predictions to native space...")
        native_paths = map_results_to_native(primary_results, meta_dir, native_dir)
        print(f"Native-space predictions: {native_dir} ({len(native_paths)} volumes)")
 
    return all_results