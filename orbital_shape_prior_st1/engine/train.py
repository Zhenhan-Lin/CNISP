"""
Training loop for orbital multi-class AutoDecoder.

Follows Amiranashvili et al. + Jansen et al.:
    - Per-shape latent codes optimized jointly with MLP weights
    - L2 regularization on latents with linear ramp-up (first 100 epochs)
    - CE + multi-class Dice loss
    - Periodic validation and checkpointing
"""

import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from models.multiclass_ad import MultiClassAutoDecoder
from models.denoise import LatentDenoiser
from models.losses import MultiClassShapeLoss, MultiClassDiceMetric
from engine.dataset import create_data_loader, EpochSubsetSampler, PhaseType
from engine.io_utils import RollingCheckpointWriter, Logger
from diagnostics.multiview_qc import compute_multiview_metrics


def create_model(params: dict, image_size: torch.Tensor) -> MultiClassAutoDecoder:
    return MultiClassAutoDecoder(
        latent_dim=params["latent_dim"],
        spatial_dim=3,
        image_size=image_size,
        num_classes=params["num_classes"],
        num_layers=params.get("op_num_layers", 8),
        layers_with_coords=params.get("op_coord_layers", [0, 4]),
    )

def assign_model_name(model_basedir: Path) -> str:
    date_str = datetime.now().strftime("%Y%m%d")
    # Find the next available index for orbital_ad_v
    idx = 1
    while True:
        candidate = f"orbital_ad_v{date_str}_{idx}"
        candidate_dir = model_basedir / candidate
        if not candidate_dir.exists():
            model_name = candidate
            break
        idx += 1
    return model_name

def train_one_epoch(
    dl: torch.utils.data.DataLoader,
    net: MultiClassAutoDecoder,
    latents: torch.nn.Parameter,
    optimizer: torch.optim.Optimizer,
    criterion: MultiClassShapeLoss,
    metric: MultiClassDiceMetric,
    lat_reg_lambda: float,
    device: torch.device,
    epoch: int,
    global_step: torch.Tensor,
    logger: Optional[SummaryWriter],
    log_this_epoch: bool,
):
    loss_running = 0.0
    n_losses = 0
    dice_running = 0.0
    n_examples = 0

    net.train()
    for batch in dl:
        labels = batch["labels"].to(device)
        coords = batch["coords"].to(device)
        latents_batch = latents[batch["caseids"]].to(device)

        optimizer.zero_grad()
        logits = net(latents_batch, coords)  # [B, *, C]
        loss = criterion(logits, labels)

        # Latent regularization with ramp-up
        lat_reg = torch.mean(torch.sum(torch.square(latents_batch), dim=1))
        if lat_reg_lambda > 0:
            ramp = min(1.0, epoch / 100.0)
            loss = loss + ramp * lat_reg_lambda * lat_reg

        loss.backward()
        optimizer.step()

        loss_running += loss.item()
        n_losses += 1

        with torch.no_grad():
            dice_info = metric(logits, labels)
            dice_running += dice_info["mean"] * labels.shape[0]
            n_examples += labels.shape[0]

        global_step += 1

    avg_loss = loss_running / max(n_losses, 1)
    avg_dice = dice_running / max(n_examples, 1)
    # lat_reg is only bound when the loop ran at least once; guard against an
    # empty loader (which train_model already rejects up front, but keep this
    # defensive so the metric never raises UnboundLocalError).
    lat_norm2 = lat_reg.item() if n_losses else 0.0
    ep = epoch + 1

    if log_this_epoch:
        if logger:
            logger.add_scalar("loss/train", avg_loss, global_step=ep)
            logger.add_scalar("dice/train", avg_dice, global_step=ep)
            logger.add_scalar("lat_norm2", lat_norm2, global_step=ep)

        print(f"[{ep}] loss={avg_loss:.4f}  dice={avg_dice:.3f}  "
              f"|z|²={lat_norm2:.2f}")

    return {"train_loss": avg_loss, "train_dice": avg_dice, "lat_norm2": lat_norm2}


def train_one_epoch_denoise(
    dl: torch.utils.data.DataLoader,
    net: MultiClassAutoDecoder,
    latents_gt: torch.nn.Parameter,
    latents_nn: torch.nn.Parameter,
    delta,                                   # LatentDenoiser or None
    optimizer: torch.optim.Optimizer,
    criterion: MultiClassShapeLoss,
    metric: MultiClassDiceMetric,
    lat_reg_lambda: float,
    lat_reg_lambda_nn: float,
    lambda_nn: float,
    lambda_denoise: float,
    eta: float,
    device: torch.device,
    epoch: int,
    global_step: torch.Tensor,
    logger,
    log_this_epoch: bool,
):
    """Dual-latent + Delta training epoch (denoise framework).

    Three terms (sg[.] = detach):
      L1 = DiceCE(F(alpha_GT, x_gt), onehot_GT)               -> F, alpha_GT
      L2 = DiceCE(F(alpha_nn, x_nn), onehot_nn)               -> F, alpha_nn
      L3 = DiceCE(F(sg[alpha_nn]+Delta(sg[alpha_nn]), x_gt),
                  onehot_GT) + eta*||Delta(sg[alpha_nn])||^2   -> F, Delta
      L  = L1 + lambda_nn*L2 + lambda_denoise*L3 + ramped L2 latent reg.

    The detach on alpha_nn in L3 routes its gradient exclusively to L2, so
    alpha_nn stays pinned to the nnUNet observation while Delta+F learn the
    correction. lat_reg_lambda(_nn) apply the (ramped) latent L2 separately so
    alpha_nn is not pulled to the origin (Problem 2b).
    """
    use_delta = delta is not None
    net.train()
    if use_delta:
        delta.train()

    loss_running = 0.0
    l1_run = l2_run = l3_run = 0.0
    n_losses = 0
    dice_running = 0.0
    n_examples = 0
    gap_run = 0.0
    dnorm_run = 0.0
    ramp = min(1.0, epoch / 100.0)

    for batch in dl:
        coords_gt = batch["coords_gt"].to(device)
        labels_gt = batch["labels_gt"].to(device)
        coords_nn = batch["coords_nn"].to(device)
        labels_nn = batch["labels_nn"].to(device)
        ids = batch["caseids"]

        z_gt = latents_gt[ids].to(device)
        z_nn = latents_nn[ids].to(device)

        optimizer.zero_grad()

        # ── Term 1: clean reconstruction (F + alpha_GT) ──
        logits_gt = net(z_gt, coords_gt)
        l1 = criterion(logits_gt, labels_gt)

        # ── Term 2: noisy reconstruction (F + alpha_nn) ──
        logits_nn = net(z_nn, coords_nn)
        l2 = criterion(logits_nn, labels_nn)

        loss = l1 + lambda_nn * l2

        # ── Term 3: denoise (F + Delta only; alpha_nn detached) ──
        if use_delta:
            z_nn_sg = z_nn.detach()
            resid = delta(z_nn_sg)
            z_hat = z_nn_sg + resid
            logits_dn = net(z_hat, coords_gt)
            l3_recon = criterion(logits_dn, labels_gt)
            l3_reg = eta * torch.mean(torch.sum(resid ** 2, dim=1))
            l3 = l3_recon + l3_reg
            loss = loss + lambda_denoise * l3
        else:
            l3 = torch.zeros((), device=device)

        # ── Ramped latent L2 (separate weights for the two tables) ──
        if lat_reg_lambda > 0:
            reg_gt = torch.mean(torch.sum(z_gt ** 2, dim=1))
            loss = loss + ramp * lat_reg_lambda * reg_gt
        if lat_reg_lambda_nn > 0:
            reg_nn = torch.mean(torch.sum(z_nn ** 2, dim=1))
            loss = loss + ramp * lat_reg_lambda_nn * reg_nn

        loss.backward()
        optimizer.step()

        loss_running += loss.item()
        l1_run += l1.item()
        l2_run += l2.item()
        l3_run += float(l3.item()) if use_delta else 0.0
        n_losses += 1

        with torch.no_grad():
            dice_info = metric(logits_gt, labels_gt)
            dice_running += dice_info["mean"] * labels_gt.shape[0]
            n_examples += labels_gt.shape[0]
            # Health metrics: alpha gap (collapse detector) + Delta magnitude.
            gap_run += torch.mean(torch.norm(z_nn - z_gt, dim=1)).item()
            if use_delta:
                dnorm_run += torch.mean(
                    torch.norm(delta(z_nn.detach()), dim=1)
                ).item()

        global_step += 1

    nb = max(n_losses, 1)
    avg_loss = loss_running / nb
    avg_dice = dice_running / max(n_examples, 1)
    metrics = {
        "train_loss": avg_loss,
        "train_dice": avg_dice,
        "lat_norm2": 0.0,  # kept for CSV schema compatibility
        "loss_recon_gt": l1_run / nb,
        "loss_recon_nn": l2_run / nb,
        "loss_denoise": l3_run / nb,
        "mean_alpha_gap": gap_run / nb,
        "mean_delta_norm": dnorm_run / nb,
    }

    ep = epoch + 1
    if log_this_epoch:
        if logger:
            logger.add_scalar("loss/train", avg_loss, global_step=ep)
            logger.add_scalar("dice/train", avg_dice, global_step=ep)
            logger.add_scalar("loss/recon_gt", metrics["loss_recon_gt"], global_step=ep)
            logger.add_scalar("loss/recon_nn", metrics["loss_recon_nn"], global_step=ep)
            logger.add_scalar("loss/denoise", metrics["loss_denoise"], global_step=ep)
            logger.add_scalar("health/mean_alpha_gap", metrics["mean_alpha_gap"], global_step=ep)
            logger.add_scalar("health/mean_delta_norm", metrics["mean_delta_norm"], global_step=ep)
        print(f"[{ep}] loss={avg_loss:.4f} dice={avg_dice:.3f} "
              f"L1={metrics['loss_recon_gt']:.4f} L2={metrics['loss_recon_nn']:.4f} "
              f"L3={metrics['loss_denoise']:.4f} "
              f"gap={metrics['mean_alpha_gap']:.3f} "
              f"|Δ|={metrics['mean_delta_norm']:.4f}")

    return metrics


def validate(
    dl: torch.utils.data.DataLoader,
    net: MultiClassAutoDecoder,
    latents: torch.nn.Parameter,
    criterion: MultiClassShapeLoss,
    metric: MultiClassDiceMetric,
    device: torch.device,
    epoch: int,
    logger: Optional[SummaryWriter],
):
    loss_running = 0.0
    dice_running = 0.0
    n_losses = 0
    n_examples = 0

    net.eval()
    with torch.no_grad():
        for batch in dl:
            labels = batch["labels"].to(device)
            coords = batch["coords"].to(device)
            latents_batch = latents[batch["caseids"]].to(device)
            logits = net(latents_batch, coords)

            loss = criterion(logits, labels)
            loss_running += loss.item()
            n_losses += 1

            dice_info = metric(logits, labels)
            dice_running += dice_info["mean"] * labels.shape[0]
            n_examples += labels.shape[0]

    avg_loss = loss_running / max(n_losses, 1)
    avg_dice = dice_running / max(n_examples, 1)

    if logger:
        logger.add_scalar("loss/val", avg_loss, global_step=epoch + 1)
        logger.add_scalar("dice/val", avg_dice, global_step=epoch + 1)
    print(f"[val] loss={avg_loss:.4f}  dice={avg_dice:.3f}")

    return {"val_loss": avg_loss, "val_dice": avg_dice}


def train_model(params: dict):
    """Main training entry point."""
    model_basedir = Path(params["model_basedir"])
    model_name = params.get("model_name")

    if model_name is None:
        model_name = assign_model_name(model_basedir)

    model_dir = model_basedir / model_name
    if model_dir.exists():
        print(f"WARNING: Model dir already exists: {model_dir}")
        print(f"  Existing checkpoints will be kept; new ones may overwrite.")
    model_dir.mkdir(parents=True, exist_ok=True)

    num_epochs = params["num_epochs"]
    log_every = params["log_epoch_count"]
    ckpt_every = params["checkpoint_epoch_count"]
    lat_reg_lambda = params["lat_reg_lambda"]
    lr = params["learning_rate"] * params["batch_size_train"]
    lr_lat = params["learning_rate_lat"]

    # Setup output directory
    sys.stdout = Logger(model_dir / "log.txt", "a")
    writer = SummaryWriter(log_dir=str(model_dir))
    ckpt_writer = RollingCheckpointWriter(
        model_dir, "checkpoint",
        params.get("max_num_checkpoints", 5), "pth"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    dl_train = create_data_loader(params, PhaseType.TRAIN)
    dl_val = create_data_loader(params, PhaseType.VAL)

    # Fail fast on an empty training set. The usual cause is a degradation
    # bank that produced zero items -- e.g. obs_sources=[nnunet] (v6-5) while
    # the nnUNet train-obs patches (labels_dataset835_{thin,thick}_train_step_XX/)
    # are missing under aligned_dir, so every (case, step) was skipped. Without
    # this guard the empty loader surfaces much later as a cryptic
    # UnboundLocalError in train_one_epoch.
    n_train_items = len(dl_train.dataset)
    if n_train_items == 0:
        obs_sources = (params.get("degradation_bank", {}) or {}).get(
            "obs_sources", ["gt"]
        )
        raise RuntimeError(
            "Training dataset is EMPTY (0 items in the degradation bank).\n"
            f"  obs_sources = {obs_sources}\n"
            "  If obs_sources is [nnunet] (v6-5), the nnUNet train-obs patches\n"
            "  must already exist under aligned_dir as\n"
            "    labels_dataset835_thin_train_step_XX/  and\n"
            "    labels_dataset835_thick_train_step_XX/\n"
            "  Generate them with the data-gen phase, e.g.:\n"
            "    bash run_pipeline.sh --config nnunet/configs.yaml "
            "nnunet-predict-sweep-train\n"
            "  (or copy that folder over from a server where it already exists).\n"
            "  See the '[bank] nnUNet-obs items added: 0 ...' line above for the\n"
            "  per-(case, step) skip reasons."
        )
    image_size = dl_train.dataset.image_size

    # Model
    net = create_model(params, image_size).to(device)
    print(net)
    print(f"image_size = {image_size.tolist()}")

    # Latents
    latent_dim = params["latent_dim"]
    n_train = len(dl_train.dataset)
    latents_train = torch.nn.Parameter(
        torch.normal(0.0, 1.0 / math.sqrt(latent_dim),
                     [n_train, latent_dim], device=device)
    )

    # ── Denoise framework (dual latent + Delta), all gated by config ──
    den = params.get("denoise") or {}
    denoise_enabled = bool(den.get("enabled", False))
    use_alpha_nn = denoise_enabled and bool(den.get("use_alpha_nn", True))
    use_delta = use_alpha_nn and bool(den.get("use_delta", True))
    lambda_nn = float(den.get("lambda_nn", 0.5))
    lambda_denoise = float(den.get("lambda_denoise", 1.0))
    eta = float(den.get("eta", 1.0e-2))
    # alpha_nn L2 must be weaker than alpha_GT's so term 2 (not L2) drives it.
    _l2_nn_cfg = den.get("lambda_l2_nn", None)
    lat_reg_lambda_nn = (
        float(_l2_nn_cfg) if _l2_nn_cfg is not None else 0.5 * lat_reg_lambda
    )

    latents_nn = None
    delta = None
    param_groups = [
        {"params": net.parameters(), "lr": lr},
        {"params": latents_train, "lr": lr_lat},
    ]
    if use_alpha_nn:
        if params.get("train_supervision") != "dual":
            raise ValueError(
                "denoise.use_alpha_nn=true requires train_supervision: dual "
                f"(got {params.get('train_supervision')!r})."
            )
        # Problem 2a: INDEPENDENT init draw (own generator), NOT a copy of
        # latents_train, so alpha_nn and alpha_GT start separated.
        gen_nn = torch.Generator(device="cpu").manual_seed(1234)
        latents_nn = torch.nn.Parameter(
            (torch.normal(0.0, 1.0 / math.sqrt(latent_dim),
                          [n_train, latent_dim], generator=gen_nn)).to(device)
        )
        param_groups.append({"params": latents_nn, "lr": lr_lat})
    if use_delta:
        delta = LatentDenoiser(
            latent_dim=latent_dim,
            hidden_dim=den.get("delta_hidden_dim") or None,
            num_hidden_layers=int(den.get("delta_num_hidden_layers", 2)),
        ).to(device)
        param_groups.append({"params": delta.parameters(), "lr": lr})

    optimizer = torch.optim.Adam(param_groups)

    if denoise_enabled:
        print(f"[denoise] enabled={denoise_enabled} use_alpha_nn={use_alpha_nn} "
              f"use_delta={use_delta} lambda_nn={lambda_nn} "
              f"lambda_denoise={lambda_denoise} eta={eta} "
              f"lat_reg_lambda_nn={lat_reg_lambda_nn}")

    def _model_state():
        """Checkpoint model_state; carries the dual latent + Delta when active.

        Back-compat: ``latents_nn`` / ``delta`` keys are simply absent for the
        original single-latent runs, so old loaders are unaffected.
        """
        ms = {"net": net.state_dict(), "latents_train": latents_train}
        if latents_nn is not None:
            ms["latents_nn"] = latents_nn
        if delta is not None:
            ms["delta"] = delta.state_dict()
        return ms

    criterion = MultiClassShapeLoss(
        ce_weight=params.get("loss_ce_weight", 1.0),
        dice_weight=params.get("loss_dice_weight", 1.0),
        dice_class_weights=params.get("dice_class_weights"),
    ).to(device)

    metric = MultiClassDiceMetric(
        num_classes=params["num_classes"]
    ).to(device)

    global_step = torch.tensor(0, dtype=torch.int64)

    # CSV logger
    csv_path = model_dir / "training_log.csv" if model_dir else None
    if csv_path:
        with open(csv_path, "w") as f:
            f.write("epoch,train_loss,train_dice,val_loss,val_dice,lat_norm2\n")

    best_val_dice = -1.0

    for epoch in range(num_epochs):
        log_this = (epoch % log_every == 0)

        # Strategy C (degradation bank): select random subset for this epoch
        if isinstance(dl_train.sampler, EpochSubsetSampler):
            dl_train.sampler.set_epoch(epoch)

        if use_alpha_nn:
            train_metrics = train_one_epoch_denoise(
                dl_train, net, latents_train, latents_nn, delta, optimizer,
                criterion, metric, lat_reg_lambda, lat_reg_lambda_nn,
                lambda_nn, lambda_denoise, eta, device, epoch, global_step,
                writer, log_this,
            )
        else:
            train_metrics = train_one_epoch(
                dl_train, net, latents_train, optimizer, criterion, metric,
                lat_reg_lambda, device, epoch, global_step,
                writer, log_this,
            )

        val_metrics = {}
        if log_this:
            val_metrics = validate(
                dl_val, net, latents_train, criterion, metric,
                device, epoch, writer,
            )

            # ── Multi-view diagnostics (Strategy B only, less frequent) ──
            if (dl_train.dataset.num_sparsify_offsets or 0) > 1 and epoch % (5 * log_every) == 0:
                mv_metrics = compute_multiview_metrics(
                    net, dl_train.dataset, latents_train, device,
                    params["num_classes"], max_scans=8,
                )
                md = mv_metrics.get("merged_dice_mean")
                ma = mv_metrics.get("multiview_acc_mean")
                if md is not None:
                    print(f"[diag] merged_dice={md:.3f}  multiview_acc={ma:.3f}")
                    writer.add_scalar("diag/merged_dice", mv_metrics["merged_dice_mean"], global_step=epoch + 1)
                    writer.add_scalar("diag/multiview_acc", mv_metrics["multiview_acc_mean"], global_step=epoch + 1)

        # Save best val checkpoint
        if log_this and val_metrics and model_dir:
            current_val_dice = val_metrics["val_dice"]
            if current_val_dice > best_val_dice:
                best_val_dice = current_val_dice
                best_path = model_dir / "best_checkpoint.pth"
                torch.save({
                    "model_state": _model_state(),
                    "optimizer_state": optimizer.state_dict(),
                    "num_steps_trained": int(global_step.item()),
                    "num_epochs_trained": epoch + 1,
                    "best_val_dice": best_val_dice,
                }, best_path)
                print(f"  ★ New best val dice: {best_val_dice:.4f} (epoch {epoch+1})")

        # Write CSV row on log epochs
        if log_this and csv_path:
            with open(csv_path, "a") as f:
                f.write(f"{epoch+1},"
                        f"{train_metrics['train_loss']:.6f},"
                        f"{train_metrics['train_dice']:.4f},"
                        f"{val_metrics.get('val_loss', ''):.6f},"
                        f"{val_metrics.get('val_dice', ''):.4f},"
                        f"{train_metrics['lat_norm2']:.4f}\n")

        if ckpt_writer and epoch % ckpt_every == 0:
            ckpt_writer.write_checkpoint(
                _model_state(),
                optimizer.state_dict(),
                int(global_step.item()), epoch + 1,
            )

    # Final checkpoint
    if ckpt_writer:
        ckpt_writer.write_checkpoint(
            _model_state(),
            optimizer.state_dict(),
            int(global_step.item()), num_epochs,
        )

    print("Training complete.")