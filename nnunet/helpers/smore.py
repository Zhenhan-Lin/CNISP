"""SMORE plumbing shared by the nnUNet-side preprocessing scripts.

Self-contained helpers around the IACL ``run-smore`` CLI
(https://gitlab.com/iacl/smore). Two backends are supported:

* ``local``      -- expects ``run-smore`` (i.e. ``pip install
                    git+https://gitlab.com/iacl/smore@v4.0.5``) on PATH.
* ``container``  -- invokes the Singularity image directly via
                    ``singularity run --nv``.

The output layout that ``run-smore`` produces is fixed by the upstream
``smore/main.py``::

    out_dir = Path(args.out_dir).resolve() / subj_id
    out_fpath = out_dir / f"{subj_id}{args.suffix}.{ext}"

where ``subj_id, ext = in_fpath.name.split('.', maxsplit=1)``. The
``build_smore_test_images.py`` driver pre-symlinks the source CT as
``<case_root>/_src/<source_id>.nii.gz`` and passes ``out_dir=<case_root>``
so SMORE writes to ``<case_root>/<source_id>/<source_id>_smore.nii.gz``.

The lock primitives (``_acquire_dir_lock`` / ``_release_dir_lock``) are
intentionally based on ``mkdir`` -- which is POSIX-atomic across the
same filesystem -- so multiple workers (potentially on multiple hosts
sharing the output filesystem) can race for the same case without
clobbering each other.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib


# ── Directory-lock primitives ────────────────────────────────────────

def _acquire_dir_lock(claim_dir: Path) -> bool:
    """Atomically attempt to claim ``claim_dir`` for the current worker.

    Returns ``True`` on successful claim, ``False`` if another worker
    already owns the lock. ``mkdir`` is atomic on POSIX filesystems, so
    this works across both processes and machines that share the
    output filesystem.
    """
    claim_dir = Path(claim_dir)
    try:
        claim_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        return False
    except OSError:
        return False
    try:
        (claim_dir / "owner.txt").write_text(
            f"host={socket.gethostname()}\n"
            f"pid={os.getpid()}\n"
            f"time={datetime.now().isoformat(timespec='seconds')}\n"
        )
    except OSError:
        pass
    return True


def _release_dir_lock(claim_dir: Path) -> None:
    """Best-effort release of a lock previously acquired by this worker."""
    claim_dir = Path(claim_dir)
    if not claim_dir.exists():
        return
    shutil.rmtree(claim_dir, ignore_errors=True)


# ── Output-path predictor ────────────────────────────────────────────

def _smore_expected_out_fpath(in_fpath: Path, out_root: Path,
                              suffix: str) -> Path:
    """Predict where ``run-smore`` will write the SR volume.

    Mirrors the path math in upstream ``smore/main.py``: a per-input
    subdirectory named after the input filename stem, then
    ``<stem><suffix>.<ext>`` inside it.
    """
    name = Path(in_fpath).name
    if "." not in name:
        raise ValueError(
            f"SMORE input must have an extension; got: {in_fpath}"
        )
    subj_id, ext = name.split(".", maxsplit=1)
    out_dir = Path(out_root).resolve() / subj_id
    return out_dir / f"{subj_id}{suffix}.{ext}"


# ── Compatibility check from the NIfTI header alone ──────────────────

def _is_smore_compatible_from_nifti_header(
    in_fpath: Path,
    min_slice_separation: float = 1.2,
    inplane_atol: float = 1e-2,
    isotropic_eps: float = 1e-3,
    require_unique_worst_axis: bool = True,
) -> Tuple[bool, str, Dict]:
    """Decide whether a NIfTI volume is anisotropic-enough for SMORE.

    SMORE assumes the input has two ~equal in-plane spacings and one
    coarser through-plane spacing. This is a cheap header-only check so
    we can decide per-case whether to call ``run-smore`` or pass the
    original CT through.

    Returns ``(compatible, message, info)`` where ``info`` contains
    ``zooms`` and ``worst_axis``.
    """
    in_fpath = Path(in_fpath)
    info: Dict = {"path": str(in_fpath)}
    try:
        img = nib.load(str(in_fpath))
        zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    except Exception as e:  # noqa: BLE001
        return False, f"unreadable header: {type(e).__name__}: {e}", info

    info["zooms"] = zooms
    if len(zooms) != 3:
        return False, f"expected 3D zooms, got {zooms}", info
    if any(z <= 0 for z in zooms):
        return False, f"non-positive spacing in zooms={zooms}", info

    span = max(zooms) - min(zooms)
    if span <= float(isotropic_eps):
        return False, (
            f"already isotropic (max-min={span:.6f} <= "
            f"{isotropic_eps}); zooms={zooms}"
        ), info

    worst_idx = max(range(3), key=lambda i: zooms[i])
    worst_val = zooms[worst_idx]
    others = [zooms[i] for i in range(3) if i != worst_idx]
    info["worst_axis"] = worst_idx
    info["worst_spacing"] = worst_val
    info["inplane_spacings"] = others

    if require_unique_worst_axis:
        second_largest = max(others)
        if (worst_val - second_largest) <= float(inplane_atol):
            return False, (
                f"no unique worst axis (worst={worst_val:.4f}, "
                f"others={others}); zooms={zooms}"
            ), info

    if abs(others[0] - others[1]) > float(inplane_atol):
        return False, (
            f"in-plane spacings differ beyond atol={inplane_atol}: "
            f"{others}; zooms={zooms}"
        ), info

    if worst_val < float(min_slice_separation):
        return False, (
            f"through-plane spacing {worst_val:.4f} < "
            f"min_slice_separation={min_slice_separation}; "
            f"zooms={zooms}"
        ), info

    return True, (
        f"OK: zooms={zooms} worst_axis={worst_idx} "
        f"worst_spacing={worst_val:.4f}"
    ), info


# ── ``run-smore`` invocations ────────────────────────────────────────

def _build_run_smore_argv(
    *,
    in_fpath: Path,
    out_root: Path,
    suffix: str,
    gpu_id: int,
    patch_sampling: str,
    slice_thickness: Optional[float],
    blur_kernel_fpath: Optional[str],
) -> List[str]:
    """Flags shared by both backends (everything after the executable)."""
    argv = [
        "--in-fpath", str(Path(in_fpath)),
        "--out-dir", str(Path(out_root)),
        "--gpu-id", str(int(gpu_id)),
        "--patch-sampling", str(patch_sampling),
        "--suffix", str(suffix),
    ]
    if slice_thickness is not None:
        argv.extend(["--slice-thickness", str(float(slice_thickness))])
    if blur_kernel_fpath:
        argv.extend(["--blur-kernel-fpath", str(blur_kernel_fpath)])
    return argv


def _stream_subprocess_to_log(cmd: List[str], log_path: Path,
                              banner: str) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as logf:
        header = (
            f"\n----- {banner} at "
            f"{datetime.now().isoformat(timespec='seconds')} -----\n"
            f"cmd: {' '.join(cmd)}\n"
        ).encode("utf-8")
        logf.write(header)
        logf.flush()
        proc = subprocess.run(
            cmd, stdout=logf, stderr=subprocess.STDOUT, check=False,
        )
        footer = (
            f"----- exit_code={proc.returncode} at "
            f"{datetime.now().isoformat(timespec='seconds')} -----\n"
        ).encode("utf-8")
        logf.write(footer)
    return proc.returncode


def _run_smore_local_run_smore(
    in_fpath: Path,
    out_root: Path,
    suffix: str,
    gpu_id: int,
    patch_sampling: str = "gradient",
    slice_thickness: Optional[float] = None,
    blur_kernel_fpath: Optional[str] = None,
) -> Path:
    """Invoke the host ``run-smore`` CLI for one volume.

    The per-case ``run_smore.log`` lives directly under ``out_root`` so
    parallel workers (each with their own ``out_root = case_root``)
    never race on a shared log.
    """
    in_fpath = Path(in_fpath)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    cmd = ["run-smore", *_build_run_smore_argv(
        in_fpath=in_fpath, out_root=out_root, suffix=suffix,
        gpu_id=gpu_id, patch_sampling=patch_sampling,
        slice_thickness=slice_thickness,
        blur_kernel_fpath=blur_kernel_fpath,
    )]

    log_path = out_root / "run_smore.log"
    rc = _stream_subprocess_to_log(cmd, log_path, "run-smore")
    if rc != 0:
        raise RuntimeError(
            f"run-smore failed (rc={rc}); see {log_path}"
        )

    out_fpath = _smore_expected_out_fpath(in_fpath, out_root, suffix)
    if not out_fpath.exists():
        raise RuntimeError(
            f"run-smore reported success but expected output is missing: "
            f"{out_fpath} (see {log_path})"
        )
    return out_fpath


def _run_smore_singularity_run_smore(
    sif_path: Path,
    bind_roots: List[Path],
    in_fpath: Path,
    out_root: Path,
    suffix: str,
    gpu_id: int,
    patch_sampling: str = "gradient",
    slice_thickness: Optional[float] = None,
    blur_kernel_fpath: Optional[str] = None,
) -> Path:
    """Invoke the SMORE Singularity image for one volume.

    ``bind_roots`` is de-duplicated and each entry is mounted with
    ``-B <root>`` so paths used inside the container resolve the same as
    on the host. We rely on ``--nv`` for GPU pass-through.
    """
    in_fpath = Path(in_fpath)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    seen: set = set()
    bind_args: List[str] = []
    for b in bind_roots:
        bp = str(Path(b))
        if bp and bp not in seen:
            seen.add(bp)
            bind_args.extend(["-B", bp])

    cmd = [
        "singularity", "run", "--nv", *bind_args, str(Path(sif_path)),
        *_build_run_smore_argv(
            in_fpath=in_fpath, out_root=out_root, suffix=suffix,
            gpu_id=gpu_id, patch_sampling=patch_sampling,
            slice_thickness=slice_thickness,
            blur_kernel_fpath=blur_kernel_fpath,
        ),
    ]

    log_path = out_root / "run_smore.log"
    rc = _stream_subprocess_to_log(cmd, log_path, "singularity run-smore")
    if rc != 0:
        raise RuntimeError(
            f"singularity run-smore failed (rc={rc}); see {log_path}"
        )

    out_fpath = _smore_expected_out_fpath(in_fpath, out_root, suffix)
    if not out_fpath.exists():
        raise RuntimeError(
            f"singularity run-smore reported success but expected output "
            f"is missing: {out_fpath} (see {log_path})"
        )
    return out_fpath


__all__ = [
    "_acquire_dir_lock",
    "_release_dir_lock",
    "_smore_expected_out_fpath",
    "_is_smore_compatible_from_nifti_header",
    "_run_smore_local_run_smore",
    "_run_smore_singularity_run_smore",
]
