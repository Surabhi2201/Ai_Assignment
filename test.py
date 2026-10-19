#!/usr/bin/env python3
"""
Testing and Evaluation script for HAC Cross-Modal Retrieval.
Evaluates both retrieval directions on the held-out test dataset:
- Optical-to-SAR (RGB -> SAR)
- SAR-to-Optical (SAR -> RGB)
Computes full-gallery Mean Average Precision (mAP), NDCG, and saves results.
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils import set_seed, get_device
from model import HACModel
from dataset import SingleModalityDataset
from retrieval import perform_cross_modal_retrieval
from metrics import compute_map, compute_ndcg

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("test")


def extract_all_features(
    model: HACModel,
    loader: DataLoader,
    modality: str,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, list]:
    """
    Extract Euclidean and HGRMC features for all images in the dataset.
    Returns:
        (feats_euc, feats_hgr, labels_np, pair_ids)
    """
    model.eval()
    all_euc = []
    all_hgr = []
    all_labels = []
    all_pair_ids = []

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            labels = batch["label_id"].cpu().numpy()
            pair_ids = batch["pair_id"]

            feat_e, feat_h = model.extract_features(imgs, modality=modality)
            all_euc.append(feat_e.cpu())
            all_hgr.append(feat_h.cpu())
            all_labels.append(labels)
            all_pair_ids.extend(pair_ids)

    feats_euc = torch.cat(all_euc, dim=0)
    feats_hgr = torch.cat(all_hgr, dim=0)
    labels_np = np.concatenate(all_labels, axis=0)

    return feats_euc, feats_hgr, labels_np, all_pair_ids


def evaluate_retrieval(
    query_euc: torch.Tensor,
    query_hgr: torch.Tensor,
    query_labels: np.ndarray,
    gallery_euc: torch.Tensor,
    gallery_hgr: torch.Tensor,
    gallery_labels: np.ndarray,
    fusion_mode: str = "ds",
    b1: float = 0.95,
    b2: float = 0.95,
    top_k: Optional[int] = None
) -> Dict[str, float]:
    """
    Perform cross-modal retrieval and compute mAP and NDCG.
    """
    ranked_indices, scores = perform_cross_modal_retrieval(
        query_euc=query_euc,
        query_hgr=query_hgr,
        gallery_euc=gallery_euc,
        gallery_hgr=gallery_hgr,
        fusion_mode=fusion_mode,
        b1=b1,
        b2=b2
    )

    map_score, per_query_ap = compute_map(
        ranked_gallery_indices=ranked_indices,
        query_labels=query_labels,
        gallery_labels=gallery_labels,
        top_k=top_k
    )

    ndcg_score = compute_ndcg(
        ranked_gallery_indices=ranked_indices,
        query_labels=query_labels,
        gallery_labels=gallery_labels,
        top_k=top_k
    )

    return {
        "mAP": map_score,
        "NDCG": ndcg_score,
        "per_query_ap": per_query_ap,
        "ranked_indices": ranked_indices.cpu().numpy()
    }


def main():
    parser = argparse.ArgumentParser(description="Test HAC Cross-Modal Retrieval Model")
    parser.add_argument("--test_csv", type=str, default="data/test.csv", help="Path to test manifest CSV")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_model.pth", help="Model checkpoint path")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for feature extraction (default: 128)")
    parser.add_argument("--fusion_mode", type=str, default="ds", choices=["ds", "sum", "euclidean_only", "hgrmc_only"],
                        help="Retrieval score fusion method (default: 'ds')")
    parser.add_argument("--b1", type=float, default=0.95, help="DS confidence parameter for Euclidean score (default: 0.95)")
    parser.add_argument("--b2", type=float, default=0.95, help="DS confidence parameter for HGRMC score (default: 0.95)")
    parser.add_argument("--top_k", type=int, default=None, help="Top-K retrieval limit (default: None for full gallery)")
    parser.add_argument("--output_csv", type=str, default="outputs/test_results.csv", help="Where to save test results")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('mps', 'cuda', 'cpu')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    logger.info(f"Using compute device: {device}")

    # Check test CSV
    test_path = Path(args.test_csv)
    if not test_path.exists():
        logger.error(f"Test CSV not found at {test_path}. Run prepare_dataset.py first.")
        sys.exit(1)

    # Check checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        # Fallback to final_model.pth if best_model.pth not yet found
        fallback = ckpt_path.parent / "final_model.pth"
        if fallback.exists():
            ckpt_path = fallback
        else:
            logger.error(f"Checkpoint not found at {ckpt_path}. Train the model first.")
            sys.exit(1)

    logger.info(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    num_classes = checkpoint.get("num_classes", 10)
    feature_dim = checkpoint.get("feature_dim", 512)

    # Instantiate model without downloading weights (loads from checkpoint)
    model = HACModel(
        num_classes=num_classes,
        feature_dim=feature_dim,
        pretrained=False
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info(f"Model restored successfully (Epoch {checkpoint.get('epoch', 'N/A')}).")

    # Create single-modality datasets for feature extraction
    rgb_test_dataset = SingleModalityDataset(test_path, modality="rgb")
    sar_test_dataset = SingleModalityDataset(test_path, modality="sar")

    logger.info(f"Extracting features for {len(rgb_test_dataset)} RGB images...")
    rgb_loader = DataLoader(rgb_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    rgb_euc, rgb_hgr, rgb_labels, rgb_stems = extract_all_features(model, rgb_loader, modality="rgb", device=device)

    logger.info(f"Extracting features for {len(sar_test_dataset)} SAR images...")
    sar_loader = DataLoader(sar_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    sar_euc, sar_hgr, sar_labels, sar_stems = extract_all_features(model, sar_loader, modality="sar", device=device)

    logger.info("Performing bidirectional retrieval evaluation...")

    # 1. Optical-to-SAR (Query: RGB, Gallery: SAR)
    res_opt2sar = evaluate_retrieval(
        query_euc=rgb_euc,
        query_hgr=rgb_hgr,
        query_labels=rgb_labels,
        gallery_euc=sar_euc,
        gallery_hgr=sar_hgr,
        gallery_labels=sar_labels,
        fusion_mode=args.fusion_mode,
        b1=args.b1,
        b2=args.b2,
        top_k=args.top_k
    )

    # 2. SAR-to-Optical (Query: SAR, Gallery: RGB)
    res_sar2opt = evaluate_retrieval(
        query_euc=sar_euc,
        query_hgr=sar_hgr,
        query_labels=sar_labels,
        gallery_euc=rgb_euc,
        gallery_hgr=rgb_hgr,
        gallery_labels=rgb_labels,
        fusion_mode=args.fusion_mode,
        b1=args.b1,
        b2=args.b2,
        top_k=args.top_k
    )

    opt2sar_map = res_opt2sar["mAP"]
    sar2opt_map = res_sar2opt["mAP"]
    mean_map = (opt2sar_map + sar2opt_map) / 2.0

    opt2sar_ndcg = res_opt2sar["NDCG"]
    sar2opt_ndcg = res_sar2opt["NDCG"]
    mean_ndcg = (opt2sar_ndcg + sar2opt_ndcg) / 2.0

    # Print results in required format
    print("\n" + "=" * 50)
    print("        CROSS-MODAL RETRIEVAL TEST RESULTS        ")
    print("=" * 50)
    print(f"RGB-to-SAR mAP: {opt2sar_map:.4f}")
    print(f"SAR-to-RGB mAP: {sar2opt_map:.4f}")
    print(f"Mean bidirectional mAP: {mean_map:.4f}")
    print(f"RGB-to-SAR NDCG: {opt2sar_ndcg:.4f}")
    print(f"SAR-to-RGB NDCG: {sar2opt_ndcg:.4f}")
    print(f"Mean bidirectional NDCG: {mean_ndcg:.4f}")
    print(f"Number of RGB queries: {len(rgb_labels)}")
    print(f"Number of SAR queries: {len(sar_labels)}")
    print(f"Number of gallery images: {len(sar_labels)}")
    print(f"Checkpoint used: {ckpt_path}")
    print(f"Fusion mode: {args.fusion_mode} (b1={args.b1}, b2={args.b2})")
    print("=" * 50 + "\n")

    # Save detailed summary to CSV
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame([{
        "task": "RGB-to-SAR",
        "mAP": opt2sar_map,
        "NDCG": opt2sar_ndcg,
        "num_queries": len(rgb_labels),
        "num_gallery": len(sar_labels),
        "fusion_mode": args.fusion_mode,
        "checkpoint": str(ckpt_path)
    }, {
        "task": "SAR-to-RGB",
        "mAP": sar2opt_map,
        "NDCG": sar2opt_ndcg,
        "num_queries": len(sar_labels),
        "num_gallery": len(rgb_labels),
        "fusion_mode": args.fusion_mode,
        "checkpoint": str(ckpt_path)
    }, {
        "task": "Bidirectional Mean",
        "mAP": mean_map,
        "NDCG": mean_ndcg,
        "num_queries": len(rgb_labels) + len(sar_labels),
        "num_gallery": len(rgb_labels) + len(sar_labels),
        "fusion_mode": args.fusion_mode,
        "checkpoint": str(ckpt_path)
    }])
    summary_df.to_csv(out_csv, index=False)
    logger.info(f"Saved test results to {out_csv}")


if __name__ == "__main__":
    main()
