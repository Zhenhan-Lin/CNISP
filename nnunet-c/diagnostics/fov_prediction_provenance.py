#!/usr/bin/env python3
"""Prediction provenance for FOV validation/test runs (revised-plan §12, P2-9).

For every prediction directory we record exactly which checkpoint + inputs produced
it, so a stale ``--continue_prediction`` can never silently mix predictions from two
different checkpoints. ``--continue_prediction`` is allowed ONLY when the stored
provenance matches the current run; otherwise the caller must re-predict from clean.

Provenance JSON (written as ``<pred_dir>/prediction_provenance.json``):
    {checkpoint_path, checkpoint_sha256, trainer, plans_file, fold,
     case_map_sha256, completion_manifest_sha256, git_commit}

Pure hashing + match logic is unit-tested (``--self-test``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

PROV_NAME = "prediction_provenance.json"
# fields that must match for --continue_prediction to be safe (git_commit is advisory).
_MATCH_KEYS = ("checkpoint_sha256", "trainer", "plans_file", "fold",
               "case_map_sha256", "completion_manifest_sha256")


def sha256_file(path: Optional[str]) -> Optional[str]:
    if not path or not Path(path).is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(repo_dir: Optional[str] = None) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_dir or Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:                                            # noqa: BLE001
        return None


def build_provenance(checkpoint_path: str, trainer: str, plans_file: str, fold: int,
                     case_map_path: Optional[str] = None,
                     completion_manifest: Optional[str] = None) -> Dict[str, object]:
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "trainer": str(trainer),
        "plans_file": str(plans_file),
        "fold": int(fold),
        "case_map_sha256": sha256_file(case_map_path),
        "completion_manifest_sha256": sha256_file(completion_manifest),
        "git_commit": git_commit(),
    }


def write_provenance(pred_dir: str, prov: Dict[str, object]) -> str:
    p = Path(pred_dir)
    p.mkdir(parents=True, exist_ok=True)
    out = p / PROV_NAME
    out.write_text(json.dumps(prov, indent=2))
    return str(out)


def read_provenance(pred_dir: str) -> Optional[Dict[str, object]]:
    p = Path(pred_dir) / PROV_NAME
    return json.loads(p.read_text()) if p.is_file() else None


def provenance_matches(pred_dir: str, expected: Dict[str, object]) -> bool:
    """True iff the stored provenance matches on every match-key (git_commit ignored)."""
    stored = read_provenance(pred_dir)
    if stored is None:
        return False
    return all(stored.get(k) == expected.get(k) for k in _MATCH_KEYS)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    w = sub.add_parser("write")
    w.add_argument("--pred-dir", required=True)
    w.add_argument("--checkpoint", required=True)
    w.add_argument("--trainer", required=True)
    w.add_argument("--plans-file", required=True)
    w.add_argument("--fold", type=int, required=True)
    w.add_argument("--case-map", default=None)
    w.add_argument("--completion-manifest", default=None)
    ck = sub.add_parser("check")   # exit 0 = match (continue OK), 1 = mismatch (re-predict)
    ck.add_argument("--pred-dir", required=True)
    ck.add_argument("--checkpoint", required=True)
    ck.add_argument("--trainer", required=True)
    ck.add_argument("--plans-file", required=True)
    ck.add_argument("--fold", type=int, required=True)
    ck.add_argument("--case-map", default=None)
    ck.add_argument("--completion-manifest", default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if getattr(args, "self_test", False) or args.cmd is None and "--self-test" in sys.argv:
        return _selftest()
    prov = build_provenance(args.checkpoint, args.trainer, args.plans_file, args.fold,
                            args.case_map, args.completion_manifest)
    if args.cmd == "write":
        print(write_provenance(args.pred_dir, prov))
        return 0
    if args.cmd == "check":
        ok = provenance_matches(args.pred_dir, prov)
        print("MATCH" if ok else "MISMATCH")
        return 0 if ok else 1
    ap.print_help()
    return 2


def _selftest() -> int:
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d) / "checkpoint_epoch_0125.pth"
        ckpt.write_bytes(b"weights-A")
        cmap = Path(d) / "eval_cases_map.json"
        cmap.write_text('{"cases": {}}')
        pred_dir = Path(d) / "pred_0125"

        prov = build_provenance(str(ckpt), "nnUNetTrainer_OrbitalFOVCompletion",
                                "/x/nnUNetPlansFinetune.json", 0, str(cmap))
        assert prov["checkpoint_sha256"] == sha256_file(str(ckpt))
        write_provenance(str(pred_dir), prov)

        # identical run -> match (continue OK)
        assert provenance_matches(str(pred_dir), prov)
        # different checkpoint content -> mismatch (must re-predict)
        ckpt.write_bytes(b"weights-B")
        prov2 = build_provenance(str(ckpt), "nnUNetTrainer_OrbitalFOVCompletion",
                                 "/x/nnUNetPlansFinetune.json", 0, str(cmap))
        assert not provenance_matches(str(pred_dir), prov2)
        # no provenance file -> mismatch
        assert not provenance_matches(str(Path(d) / "nope"), prov)
    print("FOV-PROVENANCE SELF-TEST PASSED")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_selftest())
    raise SystemExit(main())
