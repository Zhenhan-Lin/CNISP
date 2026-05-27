"""
I/O utilities adapted from Amiranashvili et al.
"""

import datetime as _datetime
import sys
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


REFINED_SAMPLE_FORMAT_VERSION = 1
"""On-disk schema version for ``save_refined_sample`` / ``load_refined_sample``.

Bump when adding/removing/renaming a top-level key so consumers can detect
incompatible snapshots and refuse to load (rather than silently degrade).
"""


class Logger(TextIOWrapper):
    """Dual logger: stdout + file."""
    def __init__(self, filepath: Path, mode: str):
        super().__init__(sys.__stdout__.buffer)
        self.file = open(filepath, mode)

    def __del__(self):
        self.file.close()

    def write(self, data):
        self.file.write(data)
        sys.__stdout__.write(data)

    def flush(self):
        self.file.flush()
        sys.__stdout__.flush()


class RollingCheckpointWriter:
    """Write checkpoints with automatic deletion of old ones."""

    def __init__(self, base_dir: Path, base_name: str,
                 max_ckpts: int, ext: str = "pth"):
        self.base_dir = base_dir
        self.base_name = base_name
        self.max_ckpts = max_ckpts
        self.ext = ext

    def write_checkpoint(self, model_state: dict, optim_state: dict,
                         n_steps: int, n_epochs: int):
        state = {
            "model_state": model_state,
            "optimizer_state": optim_state,
            "num_steps_trained": n_steps,
            "num_epochs_trained": n_epochs,
        }
        path = self.base_dir / f"{self.base_name}_{n_steps}.{self.ext}"
        torch.save(state, path)

        # Prune old checkpoints
        paths = sorted(
            self.base_dir.glob(f"{self.base_name}_*.{self.ext}"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
        for p in paths[:-self.max_ckpts]:
            p.unlink()


def load_latest_checkpoint(
    base_dir: Path, base_name: str, ext: str = "pth", verbose: bool = False
) -> Tuple[dict, dict, int, int]:
    """Load the most recent checkpoint."""
    paths = sorted(
        base_dir.glob(f"{base_name}_*.{ext}"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    if not paths:
        raise FileNotFoundError(f"No checkpoints in {base_dir}")

    latest = paths[-1]
    if verbose:
        print(f"Loading checkpoint: {latest}")
    state = torch.load(latest, map_location="cpu")
    return (state["model_state"], state["optimizer_state"],
            state["num_steps_trained"], state["num_epochs_trained"])


# ── Refined-sample export / load ─────────────────────────────────
# The CNISP "model" at test time is (frozen prior MLP, per-(case, step)
# optimised latent z). Latent optimisation is the slow part of inference:
# rebuilding the dense pred from a cached latent is seconds, while
# re-optimising z is minutes per case. We therefore checkpoint each
# refined sample to a portable, stable location so any downstream change
# (native mapping, compare_native, viz) can be replayed without redoing
# latent optimisation -- and so the cache survives a ``rm -rf runs/...``
# cleanup of the per-run output directory.
#
# Files are written one per (model, run_tag, casename, step_size) so
# atlas_gt and nnunet_pred runs of the same model don't overwrite each
# other and so each sample is independently loadable.


def refined_sample_path(
    export_dir: Path,
    model_name: str,
    run_tag: str,
    casename: str,
    step_size: int,
) -> Path:
    """Canonical path for one refined sample on disk.

    Filename pattern: ``<model_name>-<run_tag>-<casename>_step<NN>.pt``.

    The triple ``model-run-case`` namespace makes it safe to share a
    single export directory across runs (different ``run_tag``) and
    models -- which is the whole point of the user-supplied stable
    location like ``/fs5/p_masi/linz18/local_projects/cnisp/``.
    """
    fname = f"{model_name}-{run_tag}-{casename}_step{int(step_size):02d}.pt"
    return Path(export_dir) / fname


def save_refined_sample(
    export_dir: Path,
    *,
    model_name: str,
    run_tag: str,
    test_label_source: str,
    casename: str,
    step_size: int,
    step_axis: int,
    effective_resolution_mm: float,
    latent: np.ndarray,
    pred_class_map: np.ndarray,
    spacing_mm: np.ndarray,
    dice_dense_mean: float,
    dice_observed_mean: float,
    n_observed_slices: int,
    n_total_slices: int,
    prior_checkpoint_path: Optional[Path] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """Write one refined-sample snapshot to disk; return its path.

    Saves a single self-describing ``.pt`` file containing the
    refined latent, the dense canonical-frame prediction, the spacing
    and patch shape required to call ``net.predict_dense`` for replay,
    and per-row provenance / cached Dice scores for sanity checks.

    Returns ``None`` if the export was skipped (empty/invalid latent,
    or the export directory could not be created). Skipping is
    treated as non-fatal so a missing HPC mount does not break
    inference on a local machine.

    Parameters
    ----------
    export_dir : Path
        Directory the refined snapshots live in. Created if missing;
        if creation fails (permissions, missing mount) we warn and
        return ``None`` rather than crash the inference run.
    latent : ndarray
        Optimised latent z. Empty / size-1 latents (sentinel for a
        cache hit without a sidecar ``latents/<case>.npy``) are
        rejected with a warning, since they can't reconstruct.
    pred_class_map : ndarray
        Canonical-frame dense reconstruction. Stored as ``uint8`` --
        canonical labels are in ``{0..4}`` so no precision is lost.
    spacing_mm : ndarray, shape (3,)
        Per-axis voxel spacing in millimetres.
    prior_checkpoint_path : Path, optional
        Absolute path to the prior MLP checkpoint these latents were
        optimised against. Recorded so replay scripts can locate the
        matching weights without guessing.
    extra : dict, optional
        Additional metadata to embed (e.g. eff_res bucket, sweep
        config snapshot). Caller-controlled namespace; nothing in the
        loader depends on it.
    """
    if latent is None or np.asarray(latent).size <= 1:
        print(f"  [refined-export skip] {casename} step={step_size}: "
              f"latent is empty (cache hit without saved latent); "
              f"can't reconstruct, skipping refined snapshot.")
        return None

    export_dir = Path(export_dir)
    try:
        export_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"  [refined-export skip] cannot create {export_dir}: {e}. "
              f"Set refined_latent_export_dir to a writable path or null "
              f"to silence. Continuing inference without portable snapshots.")
        return None

    out_path = refined_sample_path(
        export_dir, model_name, run_tag, casename, step_size,
    )

    payload: Dict[str, Any] = {
        "format_version": REFINED_SAMPLE_FORMAT_VERSION,
        "saved_at": _datetime.datetime.utcnow().isoformat() + "Z",

        # ── Identity / provenance ─────────────────────────────────
        "model_name": str(model_name),
        "run_tag": str(run_tag),
        "test_label_source": str(test_label_source),
        "casename": str(casename),
        "step_size": int(step_size),
        "step_axis": int(step_axis),
        "effective_resolution_mm": float(effective_resolution_mm),
        "prior_checkpoint_path": (
            str(prior_checkpoint_path) if prior_checkpoint_path else None
        ),

        # ── The refined "weight" plus the artifact it produced ────
        # latent: small, needed for any predict_dense replay (e.g. iso
        # reconstruction at a new spacing).
        # pred_class_map: the canonical-frame dense reconstruction;
        # cached so downstream stages (native mapping, compare_native)
        # never need to call the prior MLP at all.
        "latent": torch.from_numpy(
            np.asarray(latent, dtype=np.float32).reshape(-1)
        ),
        "pred_class_map": torch.from_numpy(
            np.asarray(pred_class_map, dtype=np.uint8)
        ),

        # ── Geometry for replay ───────────────────────────────────
        "spacing_mm": torch.from_numpy(
            np.asarray(spacing_mm, dtype=np.float32).reshape(3)
        ),
        "patch_shape": list(int(s) for s in np.asarray(pred_class_map).shape),

        # ── Sanity / Dice cache ───────────────────────────────────
        "dice_dense_mean": float(dice_dense_mean),
        "dice_observed_mean": float(dice_observed_mean),
        "n_observed_slices": int(n_observed_slices),
        "n_total_slices": int(n_total_slices),
    }
    if extra:
        # Keep caller-extras under a namespaced key to avoid collisions
        # with future schema additions.
        payload["extra"] = dict(extra)

    torch.save(payload, str(out_path))
    return out_path


def load_refined_sample(path: Path) -> Dict[str, Any]:
    """Load a refined sample written by ``save_refined_sample``.

    Returns the raw payload dict. Tensors come back on CPU; the
    caller is responsible for moving them to the desired device.

    Raises a ``ValueError`` for schema mismatches so callers can
    decide whether to refuse the snapshot or fall back to a fresh
    inference run.
    """
    payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(
            f"{path}: refined snapshot is not a dict "
            f"(type={type(payload).__name__}); refusing to load."
        )
    ver = int(payload.get("format_version", -1))
    if ver != REFINED_SAMPLE_FORMAT_VERSION:
        raise ValueError(
            f"{path}: refined snapshot format_version={ver} but loader "
            f"expects {REFINED_SAMPLE_FORMAT_VERSION}. Regenerate the "
            f"snapshot or migrate the loader."
        )
    return payload
