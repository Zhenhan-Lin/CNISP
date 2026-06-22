"""
Pre-flight check for the v7 THICK end-to-end run.

Verifies (but does NOT regenerate) that every nnUNet output the v7 thick
training + inference needs is present under ``aligned_dir``:

  TRAIN obs   labels_dataset835_thick_train_step_XX/   (alpha_nn input, dual)
  TEST input  labels_dataset835_thick_step_XX/         (deployment latent-opt input)
  DENSE GT    labels/atlas_*.nii.gz                    (atlas Dice target)
              labels_dataset835/chk_*.nii.gz           (chk_* dense Dice target)
              metadata_dataset835/*.json               (chk_* native unmap)

Prints a per-bucket count summary and exits non-zero if any required bucket is
empty. These are produced by the nnUNet-side phases (nnunet-predict-sweep-train,
cnisp-prep-dataset835-gt, cnisp-prep-dataset835-sparse); we assume they ran.

Usage (from orbital_shape_prior_st1/):
    python -m diagnostics.preflight_v7_thick \
        -p configs/paths.yaml -c configs/train_v7_denoise.yaml
"""

import argparse
import sys
from pathlib import Path


def _count_step_dirs(aligned_dir: Path, prefix: str):
    """Return {dirname: n_niigz} for every ``{prefix}NN`` dir under aligned_dir."""
    out = {}
    for d in sorted(aligned_dir.glob(f"{prefix}*")):
        if d.is_dir():
            out[d.name] = len(list(d.glob("*.nii.gz")))
    return out


def main():
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--paths", required=True)
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--experiment", default="thick")
    args = ap.parse_args()

    with open(args.paths) as f:
        params = yaml.safe_load(f)
    with open(args.config) as f:
        params.update(yaml.safe_load(f))

    exp = args.experiment
    aligned_dir = Path(params["aligned_dir"])
    labels_dirname = params.get("labels_dirname", "labels")
    ds835_dirname = params.get("labels_dataset835_dirname", "labels_dataset835")
    meta835_dirname = params.get("metadata_dataset835_dirname", "metadata_dataset835")
    step_prefix = params.get("labels_dataset835_step_prefix", "labels_dataset835_step_")

    # exp-keyed prefixes (mirrors engine/test_label_sources.exp_step_prefix and
    # the train config's nnunet_patch_prefix).
    base = step_prefix[:-len("_step_")] if step_prefix.endswith("_step_") else step_prefix
    test_prefix = f"{base}_{exp}_step_"        # labels_dataset835_thick_step_
    train_prefix = f"{base}_{exp}_train_step_"  # labels_dataset835_thick_train_step_

    print(f"aligned_dir: {aligned_dir}")
    print(f"experiment : {exp}")
    if not aligned_dir.is_dir():
        print(f"\nERROR: aligned_dir does not exist: {aligned_dir}", file=sys.stderr)
        sys.exit(2)

    problems = []

    # 1. TRAIN obs (nnUNet thick sparse pred on the train split)
    train_dirs = _count_step_dirs(aligned_dir, train_prefix)
    print(f"\n[TRAIN obs] {train_prefix}XX/")
    if not train_dirs:
        problems.append(f"no TRAIN obs dirs matching {train_prefix}XX/")
    for name, n in train_dirs.items():
        flag = "" if n > 0 else "  <-- EMPTY"
        print(f"  {name}: {n} patches{flag}")
        if n == 0:
            problems.append(f"{name} is empty")

    # 2. TEST deployment input (nnUNet thick sparse pred per step)
    test_dirs = _count_step_dirs(aligned_dir, test_prefix)
    print(f"\n[TEST input] {test_prefix}XX/")
    if not test_dirs:
        problems.append(f"no TEST input dirs matching {test_prefix}XX/")
    for name, n in test_dirs.items():
        flag = "" if n > 0 else "  <-- EMPTY"
        print(f"  {name}: {n} patches{flag}")
        if n == 0:
            problems.append(f"{name} is empty")

    # 3. Dense Dice targets
    labels_dir = aligned_dir / labels_dirname
    ds835_dir = aligned_dir / ds835_dirname
    meta835_dir = aligned_dir / meta835_dirname
    n_atlas = len(list(labels_dir.glob("atlas_*.nii.gz"))) if labels_dir.is_dir() else 0
    n_chk = len(list(ds835_dir.glob("chk_*.nii.gz"))) if ds835_dir.is_dir() else 0
    n_meta = len(list(meta835_dir.glob("*.json"))) if meta835_dir.is_dir() else 0
    print("\n[DENSE targets]")
    print(f"  {labels_dir}/atlas_*.nii.gz       : {n_atlas}")
    print(f"  {ds835_dir}/chk_*.nii.gz          : {n_chk}")
    print(f"  {meta835_dir}/*.json              : {n_meta}")
    if n_atlas == 0 and n_chk == 0:
        problems.append("no dense Dice targets (atlas labels AND chk_* dataset835)")
    if n_chk > 0 and n_meta == 0:
        problems.append("chk_* dense targets present but metadata_dataset835 is empty")

    print("\n" + "=" * 60)
    if problems:
        print("PRE-FLIGHT FAILED. Missing/empty required nnUNet outputs:")
        for p in problems:
            print(f"  - {p}")
        print("\nGenerate them first (assumed already run), e.g.:")
        print("  bash run_pipeline.sh --config nnunet/configs_v7.yaml \\")
        print("       nnunet-predict-sweep-train cnisp-prep-dataset835-gt \\")
        print("       cnisp-prep-dataset835-sparse")
        sys.exit(2)
    print("PRE-FLIGHT OK: all thick train/test nnUNet outputs + dense targets present.")
    print("=" * 60)


if __name__ == "__main__":
    main()
    