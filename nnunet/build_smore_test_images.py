#!/usr/bin/env python3
"""Super-resolve the 31 CNISP test source CTs with SMORE.

Wraps the IACL ``run-smore`` CLI (compatibility check, local +
container backends, multi-GPU concurrency, multi-machine-safe
claim/lock semantics via :mod:`nnunet._smore_helpers`) for a list of CT
paths instead of a CSV-driven nnUNet dataset.

Output layout::

    /fs5/p_masi/linz18/data/smore_resolved_images/
      <source_id>/
        _src/<source_id>.nii.gz                 # symlink to original CT
        run_smore.log                            # per-case SMORE stdout/stderr
        <source_id>/                             # SMORE writes here
            <source_id>_smore.nii.gz             # SR output (canonical path)
            weights/best_weights.pt
        <source_id>_smore.nii.gz -> <source_id>/<source_id>_smore.nii.gz
          (convenience symlink so downstream uses one stable path)

    build_smore_test_images.<host>.<pid>.tsv     # per-host run log

The pre-symlink with ``{source_id}.nii.gz`` is what makes SMORE's
auto-derived ``subj_id`` equal ``source_id``. Per-case ``out_root``
isolates each case's ``run_smore.log`` (otherwise parallel workers
would race on a shared log file).

Usage
-----
    # local backend (host run-smore in PATH)
    python nnunet/build_smore_test_images.py --config nnunet/configs.yaml \
        --smore-gpu-ids 0,1 --smore-per-gpu-concurrency 1

    # container backend
    python nnunet/build_smore_test_images.py --config nnunet/configs.yaml \
        --smore-backend container \
        --smore-sif /path/to/smore.sif \
        --smore-gpu-ids 0
"""

from __future__ import annotations

import argparse
import csv
import os
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Lock
from typing import Dict, List, Optional

import yaml


# ── Wire up imports to reuse the existing SMORE helpers ───────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from nnunet.helpers.smore import (  # noqa: E402
    _acquire_dir_lock,
    _is_smore_compatible_from_nifti_header,
    _release_dir_lock,
    _run_smore_local_run_smore,
    _run_smore_singularity_run_smore,
    _smore_expected_out_fpath,
)
from nnunet.resolve_gt import fail_on_missing, resolve_sources  # noqa: E402


def _load_yaml(path: Path) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _safe_symlink(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src)


def _top_level_root(p: Path) -> Path:
    """Mirrors the helper in nnunetv2_build_datasets2.main()."""
    p = Path(p)
    try:
        parts = p.resolve().parts
    except Exception:  # noqa: BLE001
        parts = p.parts
    if len(parts) >= 2:
        return Path("/") / parts[1]
    return Path("/")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="nnunet/configs.yaml")
    ap.add_argument("--smore-out-root", default=None,
                    help="Override smore_out_root from config")
    ap.add_argument("--cases", default=None,
                    help="Optional path to a casename file (one casename "
                         "or source_id per line). Default: CNISP's test_cases.txt.")

    # SMORE backend / runtime knobs (mirror nnunetv2_build_datasets2.py)
    ap.add_argument("--smore-backend", choices=["local", "container"],
                    default="local")
    ap.add_argument("--smore-sif", default="",
                    help="Path to SMORE SIF (required when backend=container)")
    ap.add_argument("--smore-bind-roots", default="",
                    help="Extra comma-separated bind roots for the container "
                         "backend (in addition to auto-detected ones).")
    ap.add_argument("--smore-gpu-ids", default="",
                    help="Comma-separated GPU ids (e.g. '0,1').")
    ap.add_argument("--smore-gpu-id", type=int, default=0,
                    help="Single-GPU fallback when --smore-gpu-ids is empty.")
    ap.add_argument("--smore-per-gpu-concurrency", type=int, default=1)
    ap.add_argument("--smore-patch-sampling", default="gradient")
    ap.add_argument("--smore-slice-thickness", type=float, default=None)
    ap.add_argument("--smore-blur-kernel-fpath", type=str, default=None)
    ap.add_argument("--smore-suffix", default="_smore")
    ap.add_argument("--smore-on-incompatible",
                    choices=["original", "skip"], default="original",
                    help="What to do if a case fails the SMORE compatibility "
                         "check. 'original' copies the source CT through with "
                         "the SMORE suffix so downstream code stays uniform.")
    # Compat check thresholds (defaults match the existing script).
    ap.add_argument("--smore-min-slice-separation", type=float, default=1.2)
    ap.add_argument("--smore-inplane-atol", type=float, default=1e-2)
    ap.add_argument("--smore-isotropic-eps", type=float, default=1e-3)
    ap.add_argument("--smore-require-unique-worst-axis", action="store_true",
                    default=True)

    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config))
    cnisp_paths = _load_yaml(Path(cfg["cnisp_paths_yaml"]))

    smore_out_root = Path(args.smore_out_root or cfg["smore_out_root"])
    casefiles_dir = Path(cnisp_paths["casefiles_dir"])
    test_cases = (Path(args.cases) if args.cases
                  else casefiles_dir / "test_cases.txt")
    meta_dir = Path(cnisp_paths["aligned_dir"]) / "metadata"

    smore_out_root.mkdir(parents=True, exist_ok=True)

    sources, missing = resolve_sources(
        test_cases_path=test_cases,
        meta_dir=meta_dir,
        atlas_image_dir=Path(cfg["atlas_image_dir"]),
        pivot_csv=Path(cfg["pivot_csv"]),
        pivot_subject_column=cfg.get("pivot_subject_column", "subject"),
        pivot_image_path_columns=cfg.get("pivot_image_path_columns"),
        detect_atlas_offset=False,
        require_ct=True,
    )
    fail_on_missing(missing, "build_smore_test_images")

    print(f"[build_smore_test_images] {len(sources)} source(s); "
          f"out_root={smore_out_root}  backend={args.smore_backend}")

    # ── GPU slots ─────────────────────────────────────────────────
    if args.smore_gpu_ids.strip():
        gpu_ids = [int(x.strip())
                   for x in args.smore_gpu_ids.split(",") if x.strip()]
    else:
        gpu_ids = [int(args.smore_gpu_id)]
    if not gpu_ids:
        print("[ERROR] no GPU id(s) configured for SMORE.", file=sys.stderr)
        return 2
    per_gpu = int(args.smore_per_gpu_concurrency)
    if per_gpu < 1:
        print("[ERROR] --smore-per-gpu-concurrency must be >= 1",
              file=sys.stderr)
        return 2

    slots = [g for g in gpu_ids for _ in range(per_gpu)]

    # ── Container backend prep ────────────────────────────────────
    smore_sif: Optional[Path] = None
    bind_roots_user: List[Path] = []
    if args.smore_backend == "container":
        if not args.smore_sif:
            print("[ERROR] --smore-backend container requires --smore-sif",
                  file=sys.stderr)
            return 2
        smore_sif = Path(args.smore_sif)
        if args.smore_bind_roots:
            bind_roots_user = [Path(x.strip())
                               for x in args.smore_bind_roots.split(",")
                               if x.strip()]
        # Minimum required binds added inside the worker so each case can
        # also pull in its own source path's volume.
        bind_roots_user.append(_top_level_root(smore_out_root))

    # ── Stage per-source claim work ───────────────────────────────
    suffix = str(args.smore_suffix)
    jobs: List[Dict] = []
    skipped_already_done: List[str] = []
    incompatible_records: List[Dict] = []

    for src in sources:
        sid = src.source_id
        ct_path = src.ct_image_path
        if ct_path is None:
            continue
        case_root = smore_out_root / sid
        case_root.mkdir(parents=True, exist_ok=True)

        # Pre-symlink so SMORE's auto-derived subj_id equals source_id.
        staged_in = case_root / "_src" / f"{sid}.nii.gz"
        _safe_symlink(ct_path, staged_in)

        # SMORE writes to <case_root>/<sid>/<sid>_smore.nii.gz (the extra
        # subdir is SMORE's own convention because we use per-case
        # out_root). We additionally maintain a top-level convenience
        # symlink at <case_root>/<sid>_smore.nii.gz pointing at it.
        expected_sr = _smore_expected_out_fpath(staged_in, case_root, suffix)
        # = case_root / <sid> / <sid>_smore.nii.gz
        top_level_link = case_root / f"{sid}{suffix}.nii.gz"

        if expected_sr.exists():
            # ensure the convenience link exists even on resume
            if not top_level_link.exists() and not top_level_link.is_symlink():
                rel = Path(sid) / expected_sr.name
                top_level_link.symlink_to(rel)
            skipped_already_done.append(sid)
            continue

        compatible, compat_msg, _ = _is_smore_compatible_from_nifti_header(
            staged_in,
            min_slice_separation=float(args.smore_min_slice_separation),
            inplane_atol=float(args.smore_inplane_atol),
            isotropic_eps=float(args.smore_isotropic_eps),
            require_unique_worst_axis=bool(args.smore_require_unique_worst_axis),
        )
        if not compatible:
            if args.smore_on_incompatible == "skip":
                incompatible_records.append({
                    "source_id": sid,
                    "ct_image_path": str(ct_path),
                    "status": "skip_incompatible_smore",
                    "message": compat_msg,
                    "smore_path": "",
                })
            else:
                # Pass the original through under the canonical _smore
                # name (symlink only). Phase 2 downstream can read it
                # via the same path as SMORE'd cases.
                expected_sr.parent.mkdir(parents=True, exist_ok=True)
                _safe_symlink(ct_path, expected_sr)
                if not top_level_link.exists() and not top_level_link.is_symlink():
                    rel = Path(sid) / expected_sr.name
                    top_level_link.symlink_to(rel)
                incompatible_records.append({
                    "source_id": sid,
                    "ct_image_path": str(ct_path),
                    "status": "ok_passthrough_incompatible_smore",
                    "message": compat_msg,
                    "smore_path": str(top_level_link),
                })
            continue

        jobs.append({
            "source_id": sid,
            "ct_image_path": str(ct_path),
            "staged_in": str(staged_in),
            "case_root": str(case_root),
            "top_level_link": str(top_level_link),
            "compat_msg": compat_msg,
        })

    print(f"[build_smore_test_images] eligible jobs={len(jobs)}  "
          f"already_done={len(skipped_already_done)}  "
          f"incompatible={len(incompatible_records)}")

    # ── Worker ────────────────────────────────────────────────────
    slot_q: Queue = Queue()
    for s in slots:
        slot_q.put(s)
    progress_lock = Lock()
    progress = {"done": 0, "avg_sec": None}
    total_jobs = len(jobs)
    log_rows: List[Dict] = []
    log_lock = Lock()

    def _ensure_top_level_link(top_level_link: Path, expected_sr: Path,
                               sid: str) -> None:
        if top_level_link.exists() or top_level_link.is_symlink():
            return
        rel = Path(sid) / expected_sr.name
        try:
            top_level_link.symlink_to(rel)
        except FileExistsError:  # benign race
            pass

    def _run_one(job: Dict) -> Dict:
        gpu_id = slot_q.get()
        claim_dir: Optional[Path] = None
        acquired = False
        case_root = Path(str(job["case_root"]))
        staged_in = Path(str(job["staged_in"]))
        sid = str(job["source_id"])
        top_level_link = Path(str(job["top_level_link"]))
        try:
            claim_dir = case_root / ".smore_claim"
            if not _acquire_dir_lock(claim_dir):
                expected_sr = _smore_expected_out_fpath(
                    staged_in, case_root, suffix
                )
                if expected_sr.exists():
                    _ensure_top_level_link(top_level_link, expected_sr, sid)
                    return {
                        "source_id": sid,
                        "ct_image_path": str(job["ct_image_path"]),
                        "status": "ok_smore_cached_late",
                        "message": "completed elsewhere",
                        "smore_path": str(top_level_link),
                        "gpu_id": gpu_id,
                        "duration_sec": 0.0,
                    }
                return {
                    "source_id": sid,
                    "ct_image_path": str(job["ct_image_path"]),
                    "status": "defer_claimed_elsewhere",
                    "message": "claimed by another worker",
                    "smore_path": "",
                    "gpu_id": gpu_id,
                    "duration_sec": 0.0,
                }
            acquired = True
            t0 = time.time()
            print(f"[SMORE][START] {sid} gpu={gpu_id} "
                  f"at={datetime.now().strftime('%m-%d %H:%M:%S')}")

            if args.smore_backend == "container":
                assert smore_sif is not None
                bind_roots = list(dict.fromkeys([
                    *bind_roots_user,
                    _top_level_root(Path(job["ct_image_path"])),
                    _top_level_root(case_root),
                ]))
                sr_path = _run_smore_singularity_run_smore(
                    sif_path=smore_sif,
                    bind_roots=bind_roots,
                    in_fpath=staged_in,
                    out_root=case_root,
                    suffix=suffix,
                    gpu_id=int(gpu_id),
                    patch_sampling=str(args.smore_patch_sampling),
                    slice_thickness=args.smore_slice_thickness,
                    blur_kernel_fpath=args.smore_blur_kernel_fpath,
                )
            else:
                sr_path = _run_smore_local_run_smore(
                    in_fpath=staged_in,
                    out_root=case_root,
                    suffix=suffix,
                    gpu_id=int(gpu_id),
                    patch_sampling=str(args.smore_patch_sampling),
                    slice_thickness=args.smore_slice_thickness,
                    blur_kernel_fpath=args.smore_blur_kernel_fpath,
                )

            _ensure_top_level_link(top_level_link, sr_path, sid)
            return {
                "source_id": sid,
                "ct_image_path": str(job["ct_image_path"]),
                "status": "ok_smore",
                "message": str(job["compat_msg"]),
                "smore_path": str(top_level_link),
                "gpu_id": gpu_id,
                "duration_sec": float(time.time() - t0),
            }
        finally:
            if acquired and claim_dir is not None:
                try:
                    _release_dir_lock(claim_dir)
                except Exception:  # noqa: BLE001
                    pass
            slot_q.put(gpu_id)

    max_workers = len(slots)
    if total_jobs > 0:
        print(f"[build_smore_test_images] running {total_jobs} job(s) "
              f"with {max_workers} workers "
              f"({len(gpu_ids)} GPUs x {per_gpu} each)")
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_run_one, j): j for j in jobs}
            for fut in as_completed(futs):
                j = futs[fut]
                sid = str(j["source_id"])
                try:
                    res = fut.result()
                except Exception as e:  # noqa: BLE001
                    res = {
                        "source_id": sid,
                        "ct_image_path": str(j["ct_image_path"]),
                        "status": "error",
                        "message": f"{type(e).__name__}: {e}",
                        "smore_path": "",
                        "gpu_id": -1,
                        "duration_sec": 0.0,
                    }
                with progress_lock:
                    progress["done"] += 1
                    done = progress["done"]
                    dur = float(res.get("duration_sec", 0.0))
                    if progress["avg_sec"] is None:
                        progress["avg_sec"] = dur
                    elif done > 0:
                        avg = float(progress["avg_sec"])
                        progress["avg_sec"] = (avg * (done - 1) + dur) / done
                print(f"[SMORE][DONE {progress['done']}/{total_jobs}] "
                      f"{sid} status={res['status']} gpu={res.get('gpu_id')} "
                      f"dur={int(dur)}s")
                with log_lock:
                    log_rows.append(res)

    # already-done + incompatible go into the log too
    for sid in skipped_already_done:
        log_rows.append({
            "source_id": sid,
            "ct_image_path": "",
            "status": "ok_smore_already_done",
            "message": "expected SR output already existed at start",
            "smore_path": str(smore_out_root / sid / f"{sid}{suffix}.nii.gz"),
            "gpu_id": -1,
            "duration_sec": 0.0,
        })
    log_rows.extend(incompatible_records)

    # ── Write per-host log + per-source pointer file ──────────────
    host = socket.gethostname()
    log_path = smore_out_root / f"build_smore_test_images.{host}.{os.getpid()}.tsv"
    with open(log_path, "w", newline="") as f:
        fields = ["source_id", "ct_image_path", "status", "message",
                  "smore_path", "gpu_id", "duration_sec"]
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for r in log_rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"[build_smore_test_images] log: {log_path}")

    # ── Quick summary ─────────────────────────────────────────────
    statuses: Dict[str, int] = {}
    for r in log_rows:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    print("[build_smore_test_images] status summary:")
    for k, v in sorted(statuses.items()):
        print(f"  {k:<35s} {v}")

    # Non-zero exit if any error occurred so a driver can detect it.
    err = sum(v for k, v in statuses.items() if k == "error")
    return 1 if err > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
