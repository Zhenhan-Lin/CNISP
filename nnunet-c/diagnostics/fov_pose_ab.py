#!/usr/bin/env python3
"""
FOV pose-aware CNISP fitting: A/B(/baseline) Dice comparison + visualization.

Compares the CNISP test-time fit on partial-FOV cases under three controlled
conditions -- run through the SAME optimizer + decode path, differing only in
``(valid_mask, optimize_tau)`` so the ablation is clean:

    baseline : mask = all-ones,  tau OFF   -- today's fit (no strategy)
    A_masked : mask = M_i (FOV),  tau OFF   -- isolates the FOV geometric mask
    B_full   : mask = M_i (FOV),  tau ON    -- full joint latent+translation fit

M_i is the acquired-FOV geometric mask (plan C1), projected from the truncation
sidecar's ``visible_box`` into the canonical patch grid via engine.fov_mask
(NOT the segmentation). The optimizer core is engine.pose_opt.

Two modes:
  --self-test : synthetic decoder + synthetic geometry, no model/data. Exercises
                M_i projection, the 3-condition fit, native-grid decode, region
                Dice, and the comparison PNG. Runs anywhere torch+numpy+mpl exist.
  --run       : real cases. Loads the CNISP checkpoint via the FOV corrector
                config, iterates the test casefile (default: the 3 pipeline-check
                cases), and writes every artifact under
                nnunet-c/data_fov_pereye_test/<case>_step<PP>/ :
                    obs.nii.gz gt.nii.gz fov_mask.nii.gz
                    pred_baseline.nii.gz pred_A_masked.nii.gz pred_B_full.nii.gz
                    comparison.png
                plus a top-level dice_ab.csv + summary.json.

Usage:
  python nnunet-c/diagnostics/fov_pose_ab.py --self-test
  python nnunet-c/diagnostics/fov_pose_ab.py --run \
      --config nnunet-c/configs/corrector_fov.yaml \
      -m orbital_ad_v6_5_gt -t configs/train_v6_5_gt.yaml -c configs/test_corrector.yaml \
      --checkpoint latest --experiment fov --test-label-source nnunet_pred \
      --steps 50,65,80 --test-casefile corrector_train_cases_fov.txt \
      --out-root nnunet-c/data_fov_pereye_test
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]            # /home/user/CNISP
CNISP_DIR = REPO / "orbital_shape_prior_st1"


# ── load the two decoder-agnostic cores by file path (torch/numpy only) ────────
def _load(mod_name, rel):
    spec = importlib.util.spec_from_file_location(mod_name, CNISP_DIR / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_pose = _load("pose_opt", "engine/pose_opt.py")
_fm = _load("fov_mask", "engine/fov_mask.py")
optimize_latent_translation = _pose.optimize_latent_translation
decode_with_tau = _pose.decode_with_tau
source_box_to_grid_mask = _fm.source_box_to_grid_mask
subpatch_affine = _fm.subpatch_affine


# ── metrics ────────────────────────────────────────────────────────────────
def _dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice of two boolean masks; both-empty == 1.0 (absent structure convention)."""
    a = a.astype(bool)
    b = b.astype(bool)
    s = a.sum() + b.sum()
    if s == 0:
        return 1.0
    return float(2.0 * (a & b).sum() / s)


def region_dices(pred: np.ndarray, gt: np.ndarray, fov_mask: np.ndarray,
                 num_classes: int) -> dict:
    """Per-class Dice over whole / visible / truncated regions (+ foreground means).

    ``fov_mask`` True == visible (acquired FOV). Region restriction zeroes both
    arrays outside the region, matching eval_corrector.py.
    """
    regions = {"whole": np.ones_like(fov_mask, bool),
               "visible": fov_mask.astype(bool),
               "truncated": ~fov_mask.astype(bool)}
    out = {}
    for rname, keep in regions.items():
        g = np.where(keep, gt, 0)
        p = np.where(keep, pred, 0)
        ds = [_dice(p == c, g == c) for c in range(1, num_classes)]
        for c, d in zip(range(1, num_classes), ds):
            out[f"dice_{rname}_c{c}"] = round(d, 5)
        out[f"dice_{rname}_mean"] = round(float(np.mean(ds)) if ds else 1.0, 5)
    return out


# ── one condition: fit z(+tau) then decode on the native grid ─────────────────
def fit_and_decode(net, coords_fit, labels_fit, valid_mask, coords_dec, dec_shape,
                   latent_dim, tau_max_mm, tau_sigma_mm, *, optimize_tau, steps,
                   lr_z, lr_tau, device):
    """Run one condition and return (pred_labels [dec_shape], z, tau, best_loss)."""
    import torch
    res = optimize_latent_translation(
        net, coords_fit, labels_fit, valid_mask, latent_dim,
        tau_max_mm, tau_sigma_mm,
        optimize_tau=optimize_tau, optimize_z=True,
        num_translation_steps=(steps["tau"] if optimize_tau else 0),
        num_joint_steps=steps["joint"],
        num_latent_refine_steps=steps["refine"],
        lr_z=lr_z, lr_tau=lr_tau,
    )
    probs = decode_with_tau(net, coords_dec, res["z_star"], res["tau_star"])
    pred = probs.argmax(dim=-1).reshape(dec_shape).cpu().numpy().astype(np.int16)
    tau = res["tau_star"].reshape(-1).tolist()
    return pred, res["z_star"], tau, res["best_loss"]


# ── visualization ────────────────────────────────────────────────────────────
def make_comparison_png(path, gt, obs, preds: dict, fov_mask, spacing=None):
    """Montage: rows = 3 orthogonal planes, cols = GT / obs / preds. All panels are
    sliced at ONE common index per axis (the same physical plane), chosen at the
    centroid of the union of every foreground so the plane actually cuts through the
    anatomy (not an empty geometric mid-slice). Aspect is voxel-spacing-aware so
    anisotropic patches are not squished. The FOV boundary is drawn on each panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("GT", gt), ("obs", obs)] + [(k, v) for k, v in preds.items()]
    shape = tuple(int(s) for s in np.asarray(gt).shape)
    sp = np.asarray(spacing, float) if spacing is not None else np.ones(3)

    # common slice index per axis = centroid of the union of all foreground voxels
    union = np.zeros(shape, bool)
    for _, v in panels:
        union |= (np.asarray(v) > 0)
    if union.any():
        idx = np.argwhere(union)
        center = [int(round(idx[:, a].mean())) for a in range(3)]
    else:
        center = [s // 2 for s in shape]

    ncol = len(panels)
    fig, axes = plt.subplots(3, ncol, figsize=(2.6 * ncol, 7.8))
    if ncol == 1:
        axes = axes[:, None]
    vmax = max(1, int(np.asarray(gt).max()))
    plane_names = ["axial (⊥ax0)", "coronal (⊥ax1)", "sagittal (⊥ax2)"]
    for r, ax_ax in enumerate(range(3)):
        rem = [a for a in range(3) if a != ax_ax]        # the two in-plane axes
        aspect = sp[rem[0]] / sp[rem[1]]                 # img is [rem1, rem0] after .T
        for c, (title, vol) in enumerate(panels):
            ax = axes[r, c]
            sl = [slice(None)] * 3
            sl[ax_ax] = center[ax_ax]
            img = np.asarray(vol)[tuple(sl)].T
            m = np.asarray(fov_mask)[tuple(sl)].T.astype(float)
            ax.imshow(img, origin="lower", cmap="tab10", vmin=0, vmax=vmax,
                      interpolation="nearest", aspect=aspect)
            if m.any() and not m.all():
                ax.contour(m, levels=[0.5], colors="w", linewidths=0.7)
            if r == 0:
                ax.set_title(title, fontsize=9)
            if c == 0:
                ax.set_ylabel(plane_names[r], fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(Path(path).parent.name + f"  slice@{tuple(center)}  "
                 "(white contour = FOV boundary; all panels = same physical plane)",
                 fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _save_nii(path, arr, affine):
    import nibabel as nib
    nib.save(nib.Nifti1Image(np.asarray(arr).astype(np.int16), affine), str(path))


# ── condition set (shared by --run and --self-test) ───────────────────────────
_STEP_SCHED = {"tau": 75, "joint": 500, "refine": 150}
_STEP_SCHED_FAST = {"tau": 40, "joint": 60, "refine": 20}   # self-test only


def _conditions():
    return [("baseline", False, False), ("A_masked", True, False), ("B_full", True, True)]
    #        name        use_mask optimize_tau


def run_one_case(net, latent_dim, num_classes, coords_fit, labels_fit, coords_dec,
                 dec_shape, fov_mask_bool, gt_dec, tau_max_mm, tau_sigma_mm, *,
                 steps, lr_z=1e-2, lr_tau=5e-3, device):
    """Run baseline/A/B for one case; return (preds{name->arr}, rows{name->dice dict})."""
    import torch
    ones = torch.ones(coords_fit.shape[0], device=device)
    mmask = torch.as_tensor(fov_mask_bool.reshape(-1).astype(np.float32), device=device)
    preds, taus, rows = {}, {}, {}
    for name, use_mask, opt_tau in _conditions():
        vmask = mmask if use_mask else ones
        pred, _z, tau, loss = fit_and_decode(
            net, coords_fit, labels_fit, vmask, coords_dec, dec_shape,
            latent_dim, tau_max_mm, tau_sigma_mm,
            optimize_tau=opt_tau, steps=steps, lr_z=lr_z, lr_tau=lr_tau, device=device)
        preds[name] = pred
        taus[name] = tau
        d = region_dices(pred, gt_dec, fov_mask_bool, num_classes)
        d["tau_mm"] = [round(t, 3) for t in tau]
        d["best_loss"] = round(float(loss), 5)
        rows[name] = d
    return preds, rows, taus


# ── --run : real cases ────────────────────────────────────────────────────────
def run_real(args) -> int:
    import torch
    import nibabel as nib
    import yaml
    for p in (str(CNISP_DIR), str(REPO)):
        if p not in sys.path:
            sys.path.insert(0, p)
    from engine.dataset import load_casenames, inner_crop_64mm
    from engine.train import create_model
    from engine.infer import (load_model_checkpoint, _load_labels_dense_per_case,
                              _build_label_obs_loader, _meta_path_for_case,
                              device as INFER_DEVICE)
    from engine.test_label_sources import build_run_layout, step_input_patch_path

    device = INFER_DEVICE

    def _load_yaml(p):
        with open(p) as f:
            return yaml.safe_load(f) or {}

    # CNISP-side params (paths/train/test yaml, resolved relative to CNISP_DIR)
    def _cnisp(p):
        p = Path(p)
        return p if p.is_absolute() else (CNISP_DIR / p)

    params = {}
    for key in (args.paths, args.train_config, args.config_cnisp):
        if key:
            params.update(_load_yaml(_cnisp(key)))
    params["model_name"] = args.model_name
    params["checkpoint"] = args.checkpoint
    params["test_label_source"] = args.test_label_source
    params["experiment"] = args.experiment
    params["run_tag"] = params.get("run_tag", "corrector_gt")
    if args.test_casefile:
        params["test_casefile"] = args.test_casefile
    if args.aligned_dir:
        params["aligned_dir"] = args.aligned_dir

    layout = build_run_layout(params)
    print(f"[fov-ab] aligned_dir = {params['aligned_dir']}  experiment={args.experiment}")

    # FOV sidecar (visible_box) -- resolved from the nnunet-c corrector config data_root
    trunc = None
    if args.trunc_manifest:
        trunc = json.load(open(args.trunc_manifest))
    elif args.config:
        for p in (str(REPO / "nnunet-c"), str(REPO)):
            if p not in sys.path:
                sys.path.insert(0, p)
        from lib.config import load_corrector_config
        cfg = load_corrector_config(str(REPO / args.config),
                                    caller_file=str(Path(__file__)))
        cd = cfg.get("corrector_data", {}) or {}
        dr = Path(cd.get("data_root", "nnunet-c/data"))
        dr = dr if dr.is_absolute() else (cfg["_resolved"]["repo_root"] / dr)
        tm = dr / "fov_truncation_manifest.json"
        if tm.is_file():
            trunc = json.load(open(tm))
            print(f"[fov-ab] sidecar -> {tm}")
        else:
            print(f"[fov-ab] WARNING: {tm} missing; M_i falls back to all-ones "
                  f"(baseline==A). Pass --trunc-manifest to fix.", file=sys.stderr)

    # model
    model_dir = Path(params["model_basedir"]) / params["model_name"]
    model_state, _ = load_model_checkpoint(model_dir, args.checkpoint, verbose=True)
    net = create_model(params, torch.ones(3)).to(device).eval()
    net.load_state_dict(model_state["net"], strict=True)
    latent_dim = int(params["latent_dim"])
    num_classes = int(net.num_classes)

    tau_max_mm = [float(args.tau_max)] * 3
    tau_sigma_mm = [float(args.tau_sigma)] * 3
    # Latent fit must be as strong as the deployed CNISP fit, or z barely leaves 0 and
    # the decode collapses toward the average shape (low Dice + "squished" look). Pull
    # lr + iteration budget from the same test config the real optimize_latent uses.
    lr_z = float(args.lr_z) if args.lr_z is not None else float(params.get("latent_lr", 1e-2))
    lr_tau = float(args.lr_tau)
    n_iters = int(args.iters) if args.iters is not None else int(params.get("latent_num_iters", 3000))
    steps = {"tau": min(150, max(50, n_iters // 20)),
             "joint": int(round(n_iters * 0.6)),
             "refine": n_iters - int(round(n_iters * 0.6))}
    print(f"[fov-ab] latent fit: lr_z={lr_z} lr_tau={lr_tau} iters={n_iters} "
          f"(tau_warmup={steps['tau']} joint={steps['joint']} refine={steps['refine']})")

    casefiles_dir = Path(params["casefiles_dir"])
    casenames_all = load_casenames(casefiles_dir / params["test_casefile"])
    labels_dense, spacings_dense, casenames = _load_labels_dense_per_case(layout, casenames_all)
    label_obs_loader = _build_label_obs_loader(layout)
    meta_for = _meta_path_for_case(layout)
    if label_obs_loader is None:
        print("[fov-ab] no label_obs override loader (need nnunet_pred source)", file=sys.stderr)
        return 2

    steps_list = [int(s) for s in args.steps.split(",") if s.strip()]
    out_root = _cnisp_repo_path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    csv_rows = []
    summary = {"config": args.config, "model": args.model_name, "cases": []}

    for ci, cn in enumerate(casenames):
        meta = json.load(open(meta_for(cn))) if Path(meta_for(cn)).is_file() else {}
        src_stem = Path(meta.get("original_nifti_path", cn)).name
        src_stem = src_stem.replace(".nii.gz", "").replace(".nii", "")
        for step in steps_list:
            ov = label_obs_loader(cn, step, 0)
            if ov is None:
                print(f"  {cn} step={step:02d}: SKIP (no obs patch)")
                continue
            label_obs, spacing_obs, offset_obs = ov
            spacing_dense = spacings_dense[ci]
            offset_dense = spacing_dense / 2.0
            # FOV: centre the 64 mm crop on the whole visible-eye centroid, not
            # the largest fragment (truncation can split the eye).
            inner = inner_crop_64mm(label_obs, spacing_obs, offset_obs,
                                    labels_dense[ci], spacing_dense, offset_dense,
                                    keep_all=True)
            sub_obs = inner["sub_sparse"]
            sub_gt = inner["sub_dense"]
            if tuple(sub_obs.shape) != tuple(sub_gt.shape):
                print(f"  {cn} step={step:02d}: sparse/dense sub shapes differ "
                      f"{sub_obs.shape} vs {sub_gt.shape}; FOV expects equal. skip",
                      file=sys.stderr)
                continue
            dec_shape = tuple(int(s) for s in sub_gt.shape)

            # coords for fit (obs grid) and decode (dense grid) -- equal for FOV
            coords_fit = _grid_coords(sub_obs.shape, spacing_obs,
                                      inner["sub_offset_sparse_local"], device)
            coords_dec = _grid_coords(sub_gt.shape, spacing_dense,
                                      inner["sub_offset_dense_local"], device)
            labels_fit = sub_obs.reshape(-1).long().to(device)
            gt_dec = sub_gt.cpu().numpy().astype(np.int16)

            # ── M_i : project visible_box into the sub-patch grid ──
            fov_mask = np.ones(dec_shape, bool)
            info = None
            if trunc is not None:
                for key in (src_stem, _source_of(cn), cn):
                    info = (trunc.get(str(key), {}) or {}).get(str(step))
                    if info:
                        break
                if info and "visible_box" in info:
                    obs_path = step_input_patch_path(layout, cn, step, 0)
                    A_disk = nib.load(str(obs_path)).affine
                    A_sub = subpatch_affine(A_disk, inner["sub_crop_lo_vox_dense"])
                    A_src = np.asarray(meta.get("original_affine"), dtype=np.float64)
                    fov_mask = source_box_to_grid_mask(dec_shape, A_sub, A_src,
                                                       info["visible_box"])
                    frac = float(fov_mask.mean())
                    print(f"  {cn} step={step:02d}: M_i visible frac={frac:.3f}")
                else:
                    print(f"  {cn} step={step:02d}: no sidecar entry "
                          f"(keys tried: {src_stem},{_source_of(cn)}); M_i=ones")

            preds, rows, taus = run_one_case(
                net, latent_dim, num_classes, coords_fit, labels_fit, coords_dec,
                dec_shape, fov_mask, gt_dec, tau_max_mm, tau_sigma_mm,
                steps=steps, lr_z=lr_z, lr_tau=lr_tau, device=device)

            # ── save artifacts under data_fov_pereye_test/<case>_step<PP>/ ──
            cdir = out_root / f"{cn}_step{step:02d}"
            cdir.mkdir(parents=True, exist_ok=True)
            aff = np.eye(4)
            aff[0, 0], aff[1, 1], aff[2, 2] = [float(x) for x in spacing_dense]
            _save_nii(cdir / "gt.nii.gz", gt_dec, aff)
            _save_nii(cdir / "obs.nii.gz", sub_obs.cpu().numpy(), aff)
            _save_nii(cdir / "fov_mask.nii.gz", fov_mask.astype(np.int16), aff)
            for name, pred in preds.items():
                _save_nii(cdir / f"pred_{name}.nii.gz", pred, aff)
            make_comparison_png(cdir / "comparison.png", gt_dec,
                                sub_obs.cpu().numpy(), preds, fov_mask,
                                spacing=[float(x) for x in spacing_dense])

            for name, d in rows.items():
                csv_rows.append({"case": cn, "step": step, "condition": name, **d})
            summary["cases"].append({"case": cn, "step": step,
                                     "visible_frac": round(float(fov_mask.mean()), 4),
                                     "dice_B_minus_baseline_whole": round(
                                         rows["B_full"]["dice_whole_mean"]
                                         - rows["baseline"]["dice_whole_mean"], 5),
                                     "dice_B_minus_baseline_truncated": round(
                                         rows["B_full"]["dice_truncated_mean"]
                                         - rows["baseline"]["dice_truncated_mean"], 5)})
            def _g(cond, reg):
                return rows.get(cond, {}).get(f"dice_{reg}_mean", float("nan"))
            print(f"  [{cn} step{step:02d}]  (baseline=unmasked  A=masked,tau off  "
                  f"B=masked+tau)")
            for reg in ("whole", "visible", "truncated"):
                print(f"      {reg:9s}  base={_g('baseline', reg):.3f}  "
                      f"A_masked={_g('A_masked', reg):.3f}  "
                      f"B_full={_g('B_full', reg):.3f}")
            print(f"      tau_B_mm={rows.get('B_full', {}).get('tau_mm')}  "
                  f"| mask-only gain (A-base, whole)="
                  f"{_g('A_masked','whole') - _g('baseline','whole'):+.3f}")

    if csv_rows:
        _write_csv(out_root / "dice_ab.csv", csv_rows)
        json.dump(summary, open(out_root / "summary.json", "w"), indent=2)
        print(f"[fov-ab] wrote {out_root/'dice_ab.csv'}  (+ summary.json, per-case PNGs)")
        return 0
    print("[fov-ab] no cases produced output", file=sys.stderr)
    return 1


def _grid_coords(shape, spacing, offset, device):
    import torch
    ids = [torch.arange(int(shape[d])) for d in range(3)]
    vox = torch.stack(torch.meshgrid(ids, indexing="ij"), dim=-1).float()
    coords = (vox * spacing + offset).reshape(-1, 3)
    return coords.to(device)


def _source_of(casename: str) -> str:
    """Strip a trailing _od/_os/_step token to recover the source key (best effort)."""
    s = casename
    for suf in ("_od", "_os", "_OD", "_OS"):
        if suf in s:
            s = s.split(suf)[0]
            break
    return s


def _cnisp_repo_path(p):
    p = Path(p)
    return p if p.is_absolute() else (REPO / p)


def _write_csv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (r.get(k) if not isinstance(r.get(k), list)
                            else json.dumps(r.get(k))) for k in keys})


# ── --self-test : synthetic decoder + synthetic geometry ──────────────────────
def self_test() -> int:
    import torch
    torch.manual_seed(0)

    class Ball(torch.nn.Module):
        num_classes = 2

        def __init__(self, c0, r):
            super().__init__()
            self.register_buffer("c0", torch.as_tensor(c0, dtype=torch.float32))
            self.r = float(r)
            self.p = torch.nn.Parameter(torch.zeros(1))

        def forward(self, latent, coords):
            # Enforce the real decoder's [B, *ST, 3] contract (expand z + cat), so a
            # flat [N, 3] would raise the same mismatch instead of broadcasting.
            spatial = coords.shape[1:-1]
            z = latent
            for _ in range(len(spatial)):
                z = z.unsqueeze(1)
            z = z.expand(*([-1] + list(spatial) + [-1]))
            _ = torch.cat([z, coords], dim=-1)
            fg = (self.r ** 2 - ((coords - self.c0) ** 2).sum(-1)) * 0.5 + 0.0 * latent.sum()
            return torch.stack([torch.zeros_like(fg), fg], dim=-1)

    # Real scenario: inner_crop centers the patch on the VISIBLE (drifted) centroid,
    # so the true anatomy (GT) sits OFFSET from the decoder's default center by the
    # drift g. The tau=0 baseline decodes the default (mis-positioned) shape; tau
    # should shift coords to recover GT. net default ball = c0 (= crop center);
    # decode(z, coords+tau) is a ball at (c0 - tau), so fitting the observation
    # (a truncated ball at A = c0 + g) drives tau -> c0 - A = -g, decoding GT.
    shape = (24, 24, 24)
    spacing = torch.tensor([1.0, 1.0, 1.0])
    offset = torch.tensor([0.0, 0.0, 0.0])
    coords = _grid_coords(shape, spacing, offset, torch.device("cpu"))
    c0, R = [11.5, 11.5, 11.5], 5.0            # decoder default center (crop center)
    net = Ball(c0, R)

    def ball_labels(center):
        c = torch.as_tensor(center, dtype=torch.float32)
        return (((coords - c) ** 2).sum(-1) < R ** 2).long()

    g = [3.0, -2.0, 0.0]                        # GT drift from the crop center
    A = [c0[i] + g[i] for i in range(3)]        # true anatomy center in the crop frame
    gt = ball_labels(A).reshape(shape).numpy().astype(np.int16)   # untruncated truth

    # visible_box: keep x < 17 -> the far edge of the ball at A (x up to 19.5) is cut
    visible_box = [[0, 17], [0, 24], [0, 24]]
    fov_mask = source_box_to_grid_mask(shape, np.eye(4), np.eye(4), visible_box)
    assert 0.0 < fov_mask.mean() < 1.0
    # observation = truncated GT (truncated region reads as background)
    obs_np = ball_labels(A).reshape(shape).numpy().copy()
    obs_np[~fov_mask] = 0
    obs_fit = torch.as_tensor(obs_np.reshape(-1)).long()

    preds, rows, taus = run_one_case(
        net, 4, 2, coords, obs_fit, coords, shape, fov_mask, gt,
        [8.0] * 3, [3.0] * 3, steps=_STEP_SCHED_FAST, device=torch.device("cpu"))

    for name in ("baseline", "A_masked", "B_full"):
        r = rows[name]
        print(f"  {name:9s}: whole={r['dice_whole_mean']:.3f} "
              f"visible={r['dice_visible_mean']:.3f} trunc={r['dice_truncated_mean']:.3f} "
              f"tau={r['tau_mm']}")

    tau_B = np.asarray(taus["B_full"])
    # tau_B should recover the drift: c0 - A = -g
    assert np.allclose(tau_B, -np.asarray(g), atol=1.2), f"tau_B={tau_B} expected ~{[-x for x in g]}"
    assert np.allclose(taus["baseline"], 0) and np.allclose(taus["A_masked"], 0)
    # the payoff: tau ON recovers GT far better than the tau=0 baseline, everywhere
    assert rows["B_full"]["dice_whole_mean"] > rows["baseline"]["dice_whole_mean"] + 0.05, \
        (rows["B_full"]["dice_whole_mean"], rows["baseline"]["dice_whole_mean"])
    assert rows["B_full"]["dice_truncated_mean"] >= rows["baseline"]["dice_truncated_mean"] - 1e-6
    print(f"  tau_B ~= -drift ({tau_B.round(2).tolist()} vs {[-x for x in g]})  "
          f"whole: {rows['baseline']['dice_whole_mean']:.3f} -> "
          f"{rows['B_full']['dice_whole_mean']:.3f}")

    # exercise the PNG + nii writers on a scratch dir
    import tempfile
    td = Path(tempfile.mkdtemp())
    make_comparison_png(td / "comparison.png", gt, obs_np, preds, fov_mask)
    assert (td / "comparison.png").stat().st_size > 0
    _save_nii(td / "pred_B.nii.gz", preds["B_full"], np.eye(4))
    print(f"  wrote comparison.png ({(td/'comparison.png').stat().st_size} B) + nii OK")
    print("\nALL FOV-POSE-AB SELF-TESTS PASSED")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="synthetic decoder + geometry; no model/data.")
    ap.add_argument("--run", action="store_true", help="real cases (needs a checkpoint).")
    # real-mode I/O
    ap.add_argument("--config", default="nnunet-c/configs/corrector_fov.yaml",
                    help="nnunet-c corrector config (resolves data_root -> FOV sidecar).")
    ap.add_argument("--trunc-manifest", default=None,
                    help="explicit fov_truncation_manifest.json (else derived from --config).")
    ap.add_argument("-m", "--model-name", default=None)
    ap.add_argument("-p", "--paths", default="configs/paths.yaml")
    ap.add_argument("-t", "--train-config", default=None)
    ap.add_argument("-c", "--config-cnisp", default=None,
                    help="CNISP test yaml (e.g. configs/test_corrector.yaml).")
    ap.add_argument("--checkpoint", default="latest")
    ap.add_argument("--experiment", default="fov")
    ap.add_argument("--test-label-source", default="nnunet_pred")
    ap.add_argument("--test-casefile", default=None)
    ap.add_argument("--aligned-dir", default=None)
    ap.add_argument("--steps", default="50,65,80")
    ap.add_argument("--tau-max", type=float, default=8.0,
                    help="per-axis tau bound (mm). Placeholder for pipeline check; replace "
                         "with 1.0-2.0x the 95th-pctile measured drift (plan §13).")
    ap.add_argument("--tau-sigma", type=float, default=3.0, help="per-axis tau penalty scale (mm).")
    ap.add_argument("--lr-z", type=float, default=None,
                    help="latent Adam lr (default: config latent_lr, ~1e-2).")
    ap.add_argument("--lr-tau", type=float, default=5e-3, help="translation Adam lr.")
    ap.add_argument("--iters", type=int, default=None,
                    help="total latent iterations (default: config latent_num_iters, ~3000).")
    ap.add_argument("--out-root", default="nnunet-c/data_fov_pereye_test")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if args.run:
        missing = [k for k in ("model_name", "train_config", "config_cnisp")
                   if getattr(args, k) is None]
        if missing:
            ap.error(f"--run needs {missing}")
        return run_real(args)
    ap.error("pass --self-test or --run")


if __name__ == "__main__":
    sys.exit(main())
