#!/usr/bin/env python3
"""
Dataset and DataLoader definitions for Cross-Modal Remote Sensing (EuroSAT RGB and SAR).
- EuroSATPairDataset: Loads paired optical and SAR images for training and validation.
- SingleModalityDataset: Loads images from a single modality for feature extraction in retrieval evaluation.
- sample_in_batch_pairs: Selects positive and negative sample indices within the batch according to Section IV-C.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from utils import load_rgb_image, load_sar_image, get_image_transforms


class EuroSATPairDataset(Dataset):
    """
    Dataset yielding paired optical (RGB) and SAR images with shared label.
    Each item corresponds to a verified pair_id with identical geographic scene.
    """
    def __init__(
        self,
        manifest_csv: Union[str, Path],
        transform=None,
        root_dir: Optional[Union[str, Path]] = None
    ):
        super().__init__()
        self.root_dir = Path(root_dir) if root_dir else Path.cwd()
        df = pd.read_csv(manifest_csv)

        # Separate and align RGB and SAR by pair_id
        rgb_df = df[df["modality"] == "rgb"].set_index("pair_id")
        sar_df = df[df["modality"] == "sar"].set_index("pair_id")

        common_stems = sorted(list(set(rgb_df.index).intersection(set(sar_df.index))))
        self.pairs = []
        for stem in common_stems:
            rgb_row = rgb_df.loc[stem]
            sar_row = sar_df.loc[stem]
            label_id = int(rgb_row["label_id"])
            label = rgb_row["label"]
            self.pairs.append({
                "pair_id": stem,
                "rgb_path": str(self.root_dir / rgb_row["path"]),
                "sar_path": str(self.root_dir / sar_row["path"]),
                "label_id": label_id,
                "label": label
            })

        self.transform = transform if transform is not None else get_image_transforms(is_train=True)
        self.labels = np.array([p["label_id"] for p in self.pairs])

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int, str]]:
        item = self.pairs[idx]
        rgb_img = load_rgb_image(item["rgb_path"])
        sar_img = load_sar_image(item["sar_path"])

        rgb_tensor = self.transform(rgb_img)
        sar_tensor = self.transform(sar_img)

        return {
            "optical": rgb_tensor,
            "sar": sar_tensor,
            "label_id": item["label_id"],
            "pair_id": item["pair_id"]
        }


class SingleModalityDataset(Dataset):
    """
    Dataset for single modality image extraction (queries or gallery).
    """
    def __init__(
        self,
        manifest_csv: Union[str, Path],
        modality: str,
        transform=None,
        root_dir: Optional[Union[str, Path]] = None
    ):
        super().__init__()
        self.root_dir = Path(root_dir) if root_dir else Path.cwd()
        self.modality = modality.lower()
        df = pd.read_csv(manifest_csv)
        self.data = df[df["modality"] == self.modality].reset_index(drop=True)

        self.transform = transform if transform is not None else get_image_transforms(is_train=False)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int, str]]:
        row = self.data.iloc[idx]
        img_path = str(self.root_dir / row["path"])

        if self.modality == "rgb":
            img = load_rgb_image(img_path)
        elif self.modality == "sar":
            img = load_sar_image(img_path)
        else:
            raise ValueError(f"Unsupported modality: {self.modality}")

        tensor = self.transform(img)
        return {
            "image": tensor,
            "label_id": int(row["label_id"]),
            "pair_id": str(row["pair_id"]),
            "path": img_path
        }


def sample_in_batch_pairs(
    labels: torch.Tensor,
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    For each anchor in the batch, select in-batch positive and negative sample indices
    following Section IV-C:
      - s_pos_idx: Positive SAR index for optical anchor i (same class)
      - s_neg_idx: Negative SAR index for optical anchor i (different class)
      - v_pos_idx: Positive optical index for SAR anchor i (same class)
      - v_neg_idx: Negative optical index for SAR anchor i (different class)

    Args:
        labels: 1D Tensor of shape (N,) containing class label IDs
    Returns:
        (v_pos_idx, v_neg_idx, s_pos_idx, s_neg_idx)
    """
    N = labels.size(0)
    device = labels.device

    v_pos_idx = torch.empty(N, dtype=torch.long, device=device)
    v_neg_idx = torch.empty(N, dtype=torch.long, device=device)
    s_pos_idx = torch.empty(N, dtype=torch.long, device=device)
    s_neg_idx = torch.empty(N, dtype=torch.long, device=device)

    # Boolean masks
    labels_row = labels.unsqueeze(1)  # (N, 1)
    labels_col = labels.unsqueeze(0)  # (1, N)
    same_class = (labels_row == labels_col)  # (N, N)
    diff_class = ~same_class                # (N, N)

    for i in range(N):
        # Positive candidates: same class excluding self if possible
        pos_candidates = torch.where(same_class[i])[0]
        pos_other = pos_candidates[pos_candidates != i]
        if len(pos_other) > 0:
            rand_pos_s = pos_other[torch.randint(0, len(pos_other), (1,)).item()]
            rand_pos_v = pos_other[torch.randint(0, len(pos_other), (1,)).item()]
        else:
            # Fallback to self if no other sample of same class exists in batch
            rand_pos_s = i
            rand_pos_v = i

        # Negative candidates: different class
        neg_candidates = torch.where(diff_class[i])[0]
        if len(neg_candidates) > 0:
            rand_neg_s = neg_candidates[torch.randint(0, len(neg_candidates), (1,)).item()]
            rand_neg_v = neg_candidates[torch.randint(0, len(neg_candidates), (1,)).item()]
        else:
            # Fallback if entire batch is of a single class (degenerate case)
            rand_neg_s = i
            rand_neg_v = i

        s_pos_idx[i] = rand_pos_s
        s_neg_idx[i] = rand_neg_s
        v_pos_idx[i] = rand_pos_v
        v_neg_idx[i] = rand_neg_v

    return v_pos_idx, v_neg_idx, s_pos_idx, s_neg_idx
