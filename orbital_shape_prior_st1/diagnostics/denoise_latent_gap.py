"""
Problem 1 validation: train vs test-fit alpha_nn distribution gap.

Training optimises alpha_nn jointly as a per-case embedding (grad from the
nnUNet recon term). At test time alpha_nn is RE-FIT from scratch by
``optimize_latent`` (net frozen). If those two latents differ a lot, the Delta
correction learned on the training latents may not transfer to the test-fit
latents. This script quantifies that gap.

For N training cases it:
  1. reads the stored training ``latents_nn`` row, and
  2. re-fits alpha_nn from scratch (test-time optimize_latent, Delta OFF)
     against the SAME sparse nnUNet observation,
then reports mean/max ||train - testfit||, the latent norms, and the ratio
gap / ||train||.

DECISION GATE: if the mean relative gap exceeds --gap-threshold (default 0.20),
the script prints a STOP banner. Do NOT change the training scheme
automatically -- report the numbers first.

Usage (from orbital_shape_prior_st1/):
    python -m diagnostics.denoise_latent_gap \
        -p configs/paths.yaml -c configs/train_v7_denoise.yaml \
        --model-name orbital_ad_v7 --n-cases 16
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _full_grid_coords(label, spacing, offset, device):
    import torch
    idx = [torch.arange(label.shape[d]) for d in range(3)]
    grid = torch.stack(torch.meshgrid(idx, indexing="ij"), dim=-1).float()
    coords = (grid * spacing + offset).unsqueeze(0).to(device)
    return coords, label.unsqueeze(0).to(device)


def main():
    import yaml
    import torch
    from engine.dataset import create_data_loader, PhaseType
    from engine.train import create_model
    from engine.infer import load_model_checkpoint, optimize_latent

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--paths", required=True)
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--checkpoint", default="best", choices=["best", "latest"])
    ap.add_argument("--n-cases", type=int, default=16)
    ap.add_argument("--num-iters", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--gap-threshold", type=float, default=0.20)
    args = ap.parse_args()

    with open(args.paths) as f:
        params = yaml.safe_load(f)
    with open(args.config) as f:
        params.update(yaml.safe_load(f))
    if args.model_name:
        params["model_name"] = args.model_name
    params.setdefault("train_supervision", "dual")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = Path(params["model_basedir"]) / params["model_name"]
    model_state, _meta = load_model_checkpoint(model_dir, args.checkpoint, verbose=True)
    if "latents_nn" not in model_state or model_state["latents_nn"] is None:
        raise SystemExit(
            "Checkpoint has no 'latents_nn' -- this model was not trained with "
            "the dual-latent denoise framework (denoise.use_alpha_nn)."
        )
    latents_nn = model_state["latents_nn"].detach().to(device)

    net = create_model(params, torch.ones(3))
    net.load_state_dict(model_state["net"], strict=True)
    net = net.to(device).eval()

    dl = create_data_loader(params, PhaseType.TRAIN, verbose=True)
    ds = dl.dataset
    obs_src = getattr(ds, "bank_obs_source", None)
    if obs_src is None:
        raise SystemExit("Dataset has no bank_obs_source (need obs_sources:[nnunet]).")
    nnunet_items = [i for i in range(len(ds)) if obs_src[i] == "nnunet"]
    if not nnunet_items:
        raise SystemExit("No nnUNet-obs items in the training dataset.")

    gen = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(nnunet_items), generator=gen).tolist()
    chosen = [nnunet_items[i] for i in perm[: args.n_cases]]

    print(f"\nRe-fitting alpha_nn for {len(chosen)} training cases "
          f"(test-time optimize_latent, Delta OFF)...")
    latent_dim = params["latent_dim"]
    gaps, norms_train, norms_testfit = [], [], []
    for k, item in enumerate(chosen):
        caseid = ds.caseids[item]
        label = ds.labels_sparse[item]
        sp = ds.spacings_sparse[item]
        of = ds.offsets_sparse[item]
        coords, labels_batch = _full_grid_coords(label, sp, of, device)
        z_test = optimize_latent(
            net, labels_batch, coords, latent_dim=latent_dim, lr=args.lr,
            lat_reg_lambda=params.get("lat_reg_lambda", 1e-4),
            num_iters=args.num_iters, max_num_const_dsc=10, device=device,
            verbose=False, soft=True, label_smoothing=args.label_smoothing,
            delta=None,
        )[0]
        z_train = latents_nn[caseid]
        gap = torch.norm(z_train - z_test).item()
        gaps.append(gap)
        norms_train.append(torch.norm(z_train).item())
        norms_testfit.append(torch.norm(z_test).item())
        print(f"  [{k+1}/{len(chosen)}] {ds.casenames[item]}: "
              f"gap={gap:.3f}  |train|={norms_train[-1]:.3f}  "
              f"|testfit|={norms_testfit[-1]:.3f}")

    import statistics as st
    mean_gap = st.mean(gaps)
    max_gap = max(gaps)
    mean_norm_train = st.mean(norms_train)
    rel = mean_gap / max(mean_norm_train, 1e-8)
    print("\n==================== alpha_nn train/test gap ====================")
    print(f"  cases                : {len(chosen)}")
    print(f"  mean ||train-testfit||: {mean_gap:.4f}")
    print(f"  max  ||train-testfit||: {max_gap:.4f}")
    print(f"  mean ||train||        : {mean_norm_train:.4f}")
    print(f"  mean ||testfit||      : {st.mean(norms_testfit):.4f}")
    print(f"  relative gap          : {rel:.3f}  (threshold {args.gap_threshold})")
    print("=================================================================")
    if rel > args.gap_threshold:
        print("\n*** STOP: relative train/test alpha_nn gap exceeds threshold. ***")
        print("*** The Delta learned on training latents may not transfer to ***")
        print("*** test-fit latents. Report these numbers before changing the***")
        print("*** training scheme (e.g. fitting train alpha_nn test-time).   ***")
        sys.exit(3)
    print("\nGap within threshold: train/test alpha_nn distributions are close.")


if __name__ == "__main__":
    main()
