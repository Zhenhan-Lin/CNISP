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

    optimizer = torch.optim.Adam([
        {"params": net.parameters(), "lr": lr},
        {"params": latents_train, "lr": lr_lat},
    ])

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
                    "model_state": {"net": net.state_dict(),
                                    "latents_train": latents_train},
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
                {"net": net.state_dict(), "latents_train": latents_train},
                optimizer.state_dict(),
                int(global_step.item()), epoch + 1,
            )

    # Final checkpoint
    if ckpt_writer:
        ckpt_writer.write_checkpoint(
            {"net": net.state_dict(), "latents_train": latents_train},
            optimizer.state_dict(),
            int(global_step.item()), num_epochs,
        )

    print("Training complete.")