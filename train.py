#!/usr/bin/env python3
"""
Training script for HAC (Robust Cross-Modal Remote Sensing Image Retrieval via Maximal Correlation Augmentation).
Follows Wang et al., IEEE TGRS 2024:
- Pretrained AlexNet backbones for optical and SAR modalities.
- Parallel Euclidean and HGRMC projection heads (512-D).
- Label classifier P(·) predicting 10 semantic classes.
- Optimizer: Adam with weight decay 1e-4.
- Learning rates: 1e-4 for backbone, 5e-4 for heads, linear decay from epoch 25 to 50.
- Loss weights: alpha=0.001, beta=0.001.
- Saves best checkpoint, final checkpoint, training history CSV, and loss curves.
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import sys
import json
import time
import argparse
import logging
import contextlib
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from utils import set_seed, get_device, plot_loss_curves
from model import HACModel
from losses import HACLoss
from dataset import EuroSATPairDataset, sample_in_batch_pairs

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("train")


def train_epoch(
    model: HACModel,
    loader: DataLoader,
    criterion: HACLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int = 10,
    use_amp: bool = True
) -> Dict[str, float]:
    """
    Run one full training epoch.
    """
    model.train()
    total_loss_sum = 0.0
    j_p_sum = 0.0
    j_euc_sum = 0.0
    j_i_sum = 0.0
    j_r_sum = 0.0
    j_hgr_sum = 0.0
    j_c_sum = 0.0
    j_cr_sum = 0.0
    num_batches = 0

    device_type = "cuda" if device.type == "cuda" else ("mps" if device.type == "mps" else "cpu")
    enable_autocast = use_amp and (device.type in ("cuda", "mps"))

    for batch in loader:
        opt_imgs = batch["optical"].to(device)
        sar_imgs = batch["sar"].to(device)
        labels = batch["label_id"].to(device)

        # In-batch positive and negative sampling (Section IV-C)
        v_pos, v_neg, s_pos, s_neg = sample_in_batch_pairs(labels)

        # Targets as one-hot for MSE prediction loss (Eq. 10)
        targets_one_hot = F.one_hot(labels, num_classes=num_classes).float()

        optimizer.zero_grad()

        # Forward pass through optical and SAR branches with optional mixed precision
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=enable_autocast):
            out = model(opt_imgs, sar_imgs)
            f_e = out["f_e"].float()
            g_e = out["g_e"].float()
            f_h = out["f_h"].float()
            g_h = out["g_h"].float()

            pred_f_e = out["pred_f_e"].float()
            pred_f_h = out["pred_f_h"].float()
            pred_g_e = out["pred_g_e"].float()
            pred_g_h = out["pred_g_h"].float()

            # Compute all losses (Eqs. 3-11)
            loss_dict = criterion(
                f_e=f_e,
                g_e=g_e,
                f_e_pos=f_e[v_pos],
                f_e_neg=f_e[v_neg],
                g_e_pos=g_e[s_pos],
                g_e_neg=g_e[s_neg],
                f_h=f_h,
                g_h=g_h,
                f_h_pos=f_h[v_pos],
                f_h_neg=f_h[v_neg],
                g_h_pos=g_h[s_pos],
                g_h_neg=g_h[s_neg],
                pred_f_e=pred_f_e,
                pred_f_h=pred_f_h,
                pred_g_e=pred_g_e,
                pred_g_h=pred_g_h,
                targets_one_hot=targets_one_hot
            )

            total_loss = loss_dict["total_loss"]

        total_loss.backward()
        optimizer.step()

        total_loss_sum += total_loss.item()
        j_p_sum += loss_dict["j_p"].item()
        j_euc_sum += loss_dict["j_euc"].item()
        j_i_sum += loss_dict["j_i"].item()
        j_r_sum += loss_dict["j_r"].item()
        j_hgr_sum += loss_dict["j_hgr"].item()
        j_c_sum += loss_dict["j_c"].item()
        j_cr_sum += loss_dict["j_cr"].item()
        num_batches += 1

    return {
        "total_loss": total_loss_sum / max(num_batches, 1),
        "j_p": j_p_sum / max(num_batches, 1),
        "j_euc": j_euc_sum / max(num_batches, 1),
        "j_i": j_i_sum / max(num_batches, 1),
        "j_r": j_r_sum / max(num_batches, 1),
        "j_hgr": j_hgr_sum / max(num_batches, 1),
        "j_c": j_c_sum / max(num_batches, 1),
        "j_cr": j_cr_sum / max(num_batches, 1),
    }


def validate_epoch(
    model: HACModel,
    loader: DataLoader,
    criterion: HACLoss,
    device: torch.device,
    num_classes: int = 10,
    use_amp: bool = True
) -> Dict[str, float]:
    """
    Run evaluation on the validation split.
    """
    model.eval()
    total_loss_sum = 0.0
    j_p_sum = 0.0
    j_euc_sum = 0.0
    j_hgr_sum = 0.0
    num_batches = 0

    device_type = "cuda" if device.type == "cuda" else ("mps" if device.type == "mps" else "cpu")
    enable_autocast = use_amp and (device.type in ("cuda", "mps"))

    with torch.no_grad():
        for batch in loader:
            opt_imgs = batch["optical"].to(device)
            sar_imgs = batch["sar"].to(device)
            labels = batch["label_id"].to(device)

            v_pos, v_neg, s_pos, s_neg = sample_in_batch_pairs(labels)
            targets_one_hot = F.one_hot(labels, num_classes=num_classes).float()

            with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=enable_autocast):
                out = model(opt_imgs, sar_imgs)
                f_e = out["f_e"].float()
                g_e = out["g_e"].float()
                f_h = out["f_h"].float()
                g_h = out["g_h"].float()

                pred_f_e = out["pred_f_e"].float()
                pred_f_h = out["pred_f_h"].float()
                pred_g_e = out["pred_g_e"].float()
                pred_g_h = out["pred_g_h"].float()

                loss_dict = criterion(
                    f_e=f_e,
                    g_e=g_e,
                    f_e_pos=f_e[v_pos],
                    f_e_neg=f_e[v_neg],
                    g_e_pos=g_e[s_pos],
                    g_e_neg=g_e[s_neg],
                    f_h=f_h,
                    g_h=g_h,
                    f_h_pos=f_h[v_pos],
                    f_h_neg=f_h[v_neg],
                    g_h_pos=g_h[s_pos],
                    g_h_neg=g_h[s_neg],
                    pred_f_e=pred_f_e,
                    pred_f_h=pred_f_h,
                    pred_g_e=pred_g_e,
                    pred_g_h=pred_g_h,
                    targets_one_hot=targets_one_hot
                )

            total_loss_sum += loss_dict["total_loss"].item()
            j_p_sum += loss_dict["j_p"].item()
            j_euc_sum += loss_dict["j_euc"].item()
            j_hgr_sum += loss_dict["j_hgr"].item()
            num_batches += 1

    return {
        "val_total_loss": total_loss_sum / max(num_batches, 1),
        "val_j_p": j_p_sum / max(num_batches, 1),
        "val_j_euc": j_euc_sum / max(num_batches, 1),
        "val_j_hgr": j_hgr_sum / max(num_batches, 1)
    }


def main():
    parser = argparse.ArgumentParser(description="Train HAC Cross-Modal Retrieval Model")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing data manifests")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size (paper uses 512 on A100; default 128 for Apple Silicon / workstations)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (paper uses 50; default 15 for fast convergence)")
    parser.add_argument("--lr_backbone", type=float, default=0.0001, help="Initial LR for backbone (default: 1e-4)")
    parser.add_argument("--lr_heads", type=float, default=0.0005, help="Initial LR for projection heads (default: 5e-4)")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay for Adam (default: 1e-4)")
    parser.add_argument("--decay_start_epoch", type=int, default=8, help="Epoch to start linear LR decay (default: 8, or 25 for 50 epochs)")
    parser.add_argument("--alpha", type=float, default=0.001, help="Weight for Euclidean loss (default: 0.001)")
    parser.add_argument("--beta", type=float, default=0.001, help="Weight for HGRMC loss (default: 0.001)")
    parser.add_argument("--margin", type=float, default=0.3, help="Margin parameter m for ranking loss (default: 0.3)")
    parser.add_argument("--feature_dim", type=int, default=512, help="Dimensionality of projected features (default: 512)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker processes (default: 0)")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('mps', 'cuda', 'cpu', or None for auto)")
    parser.add_argument("--no_amp", action="store_true", help="Disable automatic mixed precision")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Optional sample limit for quick smoke test")

    args = parser.parse_args()

    # 1. Reproducibility
    set_seed(args.seed)
    device = get_device(args.device)
    logger.info(f"Using compute device: {device}")
    use_amp = not args.no_amp
    logger.info(f"Automatic mixed precision (AMP): {use_amp}")

    # Output directories
    out_dir = Path(args.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    plots_dir = out_dir / "plots"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load class mapping metadata
    data_dir = Path(args.data_dir)
    mapping_path = data_dir / "class_mapping.json"
    if not mapping_path.exists():
        logger.error(f"Class mapping not found at {mapping_path}. Run prepare_dataset.py first.")
        sys.exit(1)

    with open(mapping_path) as f:
        meta = json.load(f)
    num_classes = len(meta["classes"])
    class_mapping = meta["class_to_id"]

    # 3. Datasets and DataLoaders
    train_csv = data_dir / "train.csv"
    val_csv = data_dir / "val.csv"

    train_dataset = EuroSATPairDataset(train_csv)
    val_dataset = EuroSATPairDataset(val_csv)

    if args.max_train_samples and args.max_train_samples < len(train_dataset):
        logger.info(f"Subsetting training dataset to {args.max_train_samples} samples for quick test.")
        train_dataset = Subset(train_dataset, list(range(args.max_train_samples)))
        val_dataset = Subset(val_dataset, list(range(min(len(val_dataset), args.max_train_samples // 4))))

    logger.info(f"Training pairs: {len(train_dataset)}, Validation pairs: {len(val_dataset)}")

    eff_batch_size = min(args.batch_size, len(train_dataset))
    train_loader = DataLoader(
        train_dataset,
        batch_size=eff_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eff_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False
    )

    # 4. Initialize model
    model = HACModel(
        num_classes=num_classes,
        feature_dim=args.feature_dim,
        pretrained=True
    ).to(device)

    # 5. Loss and Optimizer
    criterion = HACLoss(
        alpha=args.alpha,
        beta=args.beta,
        margin=args.margin
    ).to(device)

    param_groups = model.get_parameter_groups(
        lr_backbone=args.lr_backbone,
        lr_heads=args.lr_heads,
        weight_decay=args.weight_decay
    )
    optimizer = torch.optim.Adam(param_groups)

    # 6. Linear LR Scheduler starting at decay_start_epoch (Section VI-B)
    decay_start = min(args.decay_start_epoch, max(1, args.epochs // 2))

    def lr_lambda(epoch: int) -> float:
        if epoch < decay_start:
            return 1.0
        remaining_epochs = args.epochs - decay_start
        if remaining_epochs <= 0:
            return 1.0
        progress = (epoch - decay_start) / float(remaining_epochs)
        return max(0.0, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # 7. Training loop
    history = []
    best_val_loss = float("inf")
    start_time = time.time()

    logger.info("=" * 60)
    logger.info(f"Starting HAC training for {args.epochs} epochs")
    logger.info(f"  Batch size: {eff_batch_size}")
    logger.info(f"  Alpha: {args.alpha}, Beta: {args.beta}, Margin: {args.margin}")
    logger.info(f"  Initial LRs: Backbone={args.lr_backbone}, Heads={args.lr_heads}")
    logger.info(f"  Linear LR decay starting at epoch {decay_start} -> 0 at epoch {args.epochs}")
    logger.info("=" * 60)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_classes=num_classes,
            use_amp=use_amp
        )

        # Validate
        val_metrics = validate_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            use_amp=use_amp
        )

        # Clear MPS cache between epochs to prevent memory fragmentation
        if device.type == "mps":
            torch.mps.empty_cache()

        # LR step
        scheduler.step()
        curr_lr_backbone = optimizer.param_groups[0]["lr"]
        curr_lr_heads = optimizer.param_groups[1]["lr"]
        epoch_dur = time.time() - epoch_start

        record = {
            "epoch": epoch,
            "train_total_loss": train_metrics["total_loss"],
            "train_j_p": train_metrics["j_p"],
            "train_j_euc": train_metrics["j_euc"],
            "train_j_i": train_metrics["j_i"],
            "train_j_r": train_metrics["j_r"],
            "train_j_hgr": train_metrics["j_hgr"],
            "train_j_c": train_metrics["j_c"],
            "train_j_cr": train_metrics["j_cr"],
            "val_total_loss": val_metrics["val_total_loss"],
            "val_j_p": val_metrics["val_j_p"],
            "val_j_euc": val_metrics["val_j_euc"],
            "val_j_hgr": val_metrics["val_j_hgr"],
            "lr_backbone": curr_lr_backbone,
            "lr_heads": curr_lr_heads,
            "epoch_time_sec": epoch_dur
        }
        history.append(record)

        logger.info(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Train: {train_metrics['total_loss']:.4f} (J_P={train_metrics['j_p']:.4f}, "
            f"J_EUC={train_metrics['j_euc']:.4f}, J_HGR={train_metrics['j_hgr']:.4f}) | "
            f"Val: {val_metrics['val_total_loss']:.4f} | "
            f"Time: {epoch_dur:.1f}s"
        )

        # Save checkpoint metadata
        checkpoint_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "best_val_loss": min(best_val_loss, val_metrics["val_total_loss"]),
            "config": vars(args),
            "class_mapping": class_mapping,
            "seed": args.seed,
            "num_classes": num_classes,
            "feature_dim": args.feature_dim
        }

        # Track best model checkpoint
        if val_metrics["val_total_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_total_loss"]
            torch.save(checkpoint_dict, ckpt_dir / "best_model.pth")
            logger.info(f"  -> Best model saved with val loss: {best_val_loss:.4f}")

    # Save final model checkpoint
    torch.save(checkpoint_dict, ckpt_dir / "final_model.pth")
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time / 60.0:.2f} minutes.")

    # Save history and generate loss plots
    history_df = pd.DataFrame(history)
    history_df.to_csv(out_dir / "training_history.csv", index=False)
    plot_loss_curves(history_df, plots_dir)
    logger.info(f"Saved training history to {out_dir / 'training_history.csv'}")
    logger.info(f"Saved loss plots to {plots_dir}/")


if __name__ == "__main__":
    main()
