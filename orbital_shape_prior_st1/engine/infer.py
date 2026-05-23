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
                    device, verbose=True):
    """
    Test-time latent optimization for one case.

    Single forward+backward per iteration (no chunking).
    Memory budget assumption: the caller has filtered patch sizes so that
    a full backward graph fits in VRAM after bf16/fp16 autocast. For an
    80mm patch, bf16 on a 48 GB GPU comfortably handles up to ~15M voxels
    (single forward); beyond that the caller is responsible for downsampling
    or skipping the case.
    """
    latent = torch.nn.Parameter(
        torch.normal(0.0, 1e-4, [1, latent_dim], device=device),
        requires_grad=True,
    )
    criterion = MultiClassShapeLoss().to(device)
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
        print(f"  optimize_latent: {n_vox} voxels, dtype={dt}, "
              f"iters={num_iters}")

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

    return latent.detach()


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
    """For each case pick the result whose eff_res is closest to the target."""
    by_case = defaultdict(list)
    for r in all_results:
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
    if layout.test_label_source != "nnunet_pred":
        return None

    def _loader(casename: str, step: int):
        # step=1 is the dense baseline; under the deployment curve the
        # "step_01" sparse patch is just Dataset835's dense canonical-
        # aligned pred (same content as labels_dataset835/ for chk_*,
        # and a fresh canonical_align of the atlas dense pred). We
        # serve it via the same lookup so the latent-opt input grid
        # follows the input modality consistently across steps.
        p = step_input_patch_path(layout, casename, step)
        if not p.exists():
            return None
        vol, spacing, offset = load_patch_as_label_tensor(p)
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
        if (layout.test_label_source == "nnunet_pred"
                and not casename.startswith("atlas_")):
            return layout.metadata_dataset835_dir / f"{casename}.json"
        return layout.metadata_dir / f"{casename}.json"
    return _resolve


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
        slice_step_axis: 2
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
    """

    layout = build_run_layout(params)
    model_dir = Path(params["model_basedir"]) / params["model_name"]
    output_dir = layout.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Run layout:")
    print(f"  test_label_source = {layout.test_label_source}")
    print(f"  run_tag           = {layout.run_tag}")
    print(f"  output_dir        = {output_dir}")

    # ── Load model ────────────────────────────────────────────────
    which_ckpt = params.get("checkpoint", "best")
    model_state, ckpt_meta = load_model_checkpoint(model_dir, which_ckpt, verbose=True)

    net = create_model(params, torch.ones(3))
    net.load_state_dict(model_state["net"], strict=True)
    net = net.to(device).eval()

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
        print(f"  label_obs override : enabled (Dataset835 sparse patches in "
              f"{layout.labels_dataset835_step_prefix.as_posix()}XX/)")

    # ── Sweep configuration (per-case adaptive) ───────────────────
    step_axis = params["slice_step_axis"]
    sweep_cfg = dict(params.get("adaptive_step_sweep", {}))
    primary_eff_res = float(sweep_cfg.get("primary_eff_res_mm", 3.0))
    bucket_edges = tuple(sweep_cfg.get(
        "summary_bucket_edges_mm", DEFAULT_BUCKET_EDGES_MM
    ))

    print(f"\nTest cases: {len(casenames)}")
    print(f"Sweep cfg: {sweep_cfg}")
    print(f"Primary eff_res target: {primary_eff_res} mm")

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
        step_axis=step_axis,
        params=params,
        device=device,
        sweep_cfg=sweep_cfg,
        output_dir=output_dir,
        label_obs_override_loader=label_obs_loader,
    )

    # ── Export predictions per step subdirectory ──────────────────
    if params.get("export_predictions", True):
        step_metadata = defaultdict(list)

        for result in all_results:
            step = result["step_size"]
            step_dir = output_dir / f"step_{step:02d}"
            pred_dir = step_dir / "pred"
            lat_dir = step_dir / "latents"
            ovr_dir = step_dir / "obs_vs_recon"
            iso_dir = step_dir / "iso_space"
            for d in [pred_dir, lat_dir, ovr_dir, iso_dir]:
                d.mkdir(parents=True, exist_ok=True)

            sp = result["spacing"]
            aff = np.diag([*sp, 1.0])
            casename = result["casename"]

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

            # pred/  (always re-saved; tolerant of cache hits)
            nib.save(
                nib.Nifti1Image(result["pred_class_map"].astype(np.uint8), aff),
                str(pred_dir / f"{casename}_pred.nii.gz"),
            )

            # latents/  (sidecar so cache resume keeps iso reconstruction working)
            if not result.get("latent_missing", False) and result["latent"].size > 1:
                np.save(str(lat_dir / f"{casename}.npy"),
                        np.asarray(result["latent"], dtype=np.float32))

            # obs_vs_recon/
            obs_vs_recon = create_obs_vs_recon_map(
                result["pred_class_map"],
                slice_step_size=step,
                slice_start_id=0,
                slice_axis=step_axis,
            )
            nib.save(
                nib.Nifti1Image(obs_vs_recon.astype(np.uint8), aff),
                str(ovr_dir / f"{casename}_obs_vs_recon.nii.gz"),
            )

            # iso_space/  (skip if anisotropy is negligible OR latent unavailable)
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
    print_sweep_summary(all_results, class_names,
                        bucket_edges=bucket_edges, ckpt_info=ckpt_info)
    save_sweep_csvs(all_results, class_names, output_dir,
                    bucket_edges=bucket_edges)

    # ── Map primary-eff_res predictions to native space ───────────
    primary_results = _pick_primary_per_case(all_results, primary_eff_res)
    meta_path_for = _meta_path_for_case(layout)
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
        )
        print(f"Native-space predictions: {native_dir} ({len(native_paths)} volumes)")

    # ── Map EVERY sweep step to native space ──────────────────────
    # Mirrors what nnunet/engine/build_cnisp_native_sweep.py does as a backfill
    # for already-run experiments; here it is folded into the inference
    # loop so a single run produces every artifact the cross-model
    # comparison (see nnunet/compare_native.py) consumes.
    if all_results and params.get("export_predictions", True):
        by_step: Dict[int, List[dict]] = defaultdict(list)
        for r in all_results:
            by_step[int(r["step_size"])].append(r)
        sweep_manifest: Dict[str, Dict[str, str]] = {}
        print(f"\nMapping all sweep steps to native space "
              f"({len(by_step)} step values):")
        for step in sorted(by_step):
            step_native_dir = output_dir / f"native_space_step_{step:02d}"
            suffix = f"_cnisp_step{step:02d}"
            step_paths = map_results_to_native(
                by_step[step], layout.metadata_dir, step_native_dir,
                suffix=suffix,
                meta_path_for_casename=meta_path_for,
            )

            per_step_manifest: Dict[str, str] = {}
            seen = set()
            for r in by_step[step]:
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
                per_step_manifest[sid] = str(
                    step_native_dir / f"{stem}{suffix}.nii.gz"
                )

            with open(step_native_dir / "manifest.json", "w") as f:
                json.dump({
                    "model_name": params["model_name"],
                    "run_tag": layout.run_tag,
                    "test_label_source": layout.test_label_source,
                    "step_size": step,
                    "suffix": suffix,
                    "n_sources": len(per_step_manifest),
                    "by_source_id": per_step_manifest,
                }, f, indent=2)
            sweep_manifest[str(step)] = per_step_manifest
            print(f"  step_{step:02d}: {step_native_dir} "
                  f"({len(step_paths)} sources)")

        with open(output_dir / "native_sweep_manifest.json", "w") as f:
            json.dump({
                "model_name": params["model_name"],
                "run_tag": layout.run_tag,
                "test_label_source": layout.test_label_source,
                "primary_eff_res_mm": primary_eff_res,
                "steps": sweep_manifest,
            }, f, indent=2)

    # ── Pickle layout ────────────────────────────────────────────
    # inference_results.pkl : per-case primary picks (one row per case),
    #     consumed by map_to_native.py and downstream visualization
    # sweep_results.pkl     : full per-(case, step) sweep, used by
    #     scripts/04_visualization.py and by nnunet/compare_native.py
    with open(output_dir / "inference_results.pkl", "wb") as f:
        pickle.dump(primary_results, f)
    with open(output_dir / "sweep_results.pkl", "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nPickled {len(primary_results)} primary results "
          f"and {len(all_results)} sweep rows under {output_dir}")

    return {"primary": primary_results, "all": all_results}