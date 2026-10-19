#!/usr/bin/env python3
"""
Utility functions for HAC:
- Device selection (MPS, CUDA, CPU)
- Reproducible random seeds
- SAR and Optical image loading and preprocessing
- Plotting loss curves
"""

import os
import random
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union
import numpy as np
import pandas as pd
from PIL import Image
import tifffile
import torch
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Union_Path = Union[str, Path]


def set_seed(seed: int = 42):
    """
    Set random seed across Python, NumPy, and PyTorch for strict reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device(preferred: Optional[str] = None) -> torch.device:
    """
    Select compute device following priority: preferred -> MPS -> CUDA -> CPU.
    """
    if preferred:
        pref = preferred.lower()
        if pref == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if pref == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if pref == "cpu":
            return torch.device("cpu")

    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_sar_image(path: Union_Path) -> Image.Image:
    """
    Load 2-channel Sentinel-1 SAR TIFF image (VV, VH in dB) and convert to
    a 3-channel composite image compatible with ImageNet-pretrained AlexNet:
      - Channel 0: Normalized VV backscatter (clipped to [-35, 5] dB -> [0, 1])
      - Channel 1: Normalized VH backscatter (clipped to [-45, 0] dB -> [0, 1])
      - Channel 2: Normalized cross-polarization ratio VV - VH (clipped to [0, 25] dB -> [0, 1])
    """
    arr = tifffile.imread(str(path))
    if arr.ndim == 2:
        # Single channel fallback: repeat to 3 channels
        c0 = np.clip((arr + 35.0) / 40.0, 0.0, 1.0)
        rgb_arr = np.stack([c0, c0, c0], axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] >= 2:
        vv = arr[:, :, 0]
        vh = arr[:, :, 1]
        ratio = vv - vh
        c0 = np.clip((vv + 35.0) / 40.0, 0.0, 1.0)
        c1 = np.clip((vh + 45.0) / 45.0, 0.0, 1.0)
        c2 = np.clip(ratio / 25.0, 0.0, 1.0)
        rgb_arr = np.stack([c0, c1, c2], axis=-1)
    else:
        raise ValueError(f"Unexpected SAR array shape: {arr.shape} in {path}")

    uint8_img = (rgb_arr * 255.0).astype(np.uint8)
    return Image.fromarray(uint8_img, mode="RGB")



def load_rgb_image(path: Union_Path) -> Image.Image:
    """
    Load optical RGB image as PIL Image in RGB mode.
    """
    img = Image.open(str(path))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def get_image_transforms(img_size: int = 224, is_train: bool = True) -> T.Compose:
    """
    Standard image transformations for AlexNet (224x224, ImageNet normalization).
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if is_train:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])


def plot_loss_curves(history_df: pd.DataFrame, output_dir: Union_Path):
    """
    Generate and save training and validation loss plots:
      1. train_loss.png (Training loss vs epoch)
      2. val_loss.png (Validation loss vs epoch)
      3. loss_components.png (Individual loss components: J_P, J_EUC, J_HGR)
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    epochs = history_df["epoch"].values

    # 1. Total Train Loss
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, history_df["train_total_loss"], "b-o", label="Train Total Loss", linewidth=2)
    if "val_total_loss" in history_df.columns:
        ax.plot(epochs, history_df["val_total_loss"], "r--s", label="Val Total Loss", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training & Validation Loss vs Epoch", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path / "train_loss.png", dpi=150)
    plt.close(fig)

    # 2. Validation Loss specifically (required by assignment)
    if "val_total_loss" in history_df.columns:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(epochs, history_df["val_total_loss"], "r-s", label="Validation Loss", linewidth=2)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Validation Loss", fontsize=12)
        ax.set_title("Validation Loss vs Epoch", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.6)
        fig.tight_layout()
        fig.savefig(out_path / "val_loss.png", dpi=150)
        plt.close(fig)

    # 3. Individual loss components
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Prediction loss J_P
    ax1.plot(epochs, history_df["train_j_p"], "m-o", label="Train J_P (Prediction MSE)", linewidth=2)
    if "val_j_p" in history_df.columns:
        ax1.plot(epochs, history_df["val_j_p"], "m--", label="Val J_P", linewidth=1.5)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("J_P Loss", fontsize=12)
    ax1.set_title("Semantic Prediction Loss (J_P)", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Metric losses: J_EUC and J_HGR
    ax2.plot(epochs, history_df["train_j_euc"], "g-o", label="Train J_EUC (Euclidean)", linewidth=2)
    ax2.plot(epochs, history_df["train_j_hgr"], "c-s", label="Train J_HGR (Soft-HGR)", linewidth=2)
    if "val_j_euc" in history_df.columns:
        ax2.plot(epochs, history_df["val_j_euc"], "g--", label="Val J_EUC", linewidth=1.5)
    if "val_j_hgr" in history_df.columns:
        ax2.plot(epochs, history_df["val_j_hgr"], "c--", label="Val J_HGR", linewidth=1.5)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Loss Value", fontsize=12)
    ax2.set_title("Metric Losses: J_EUC and J_HGR", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_path / "loss_components.png", dpi=150)
    plt.close(fig)
