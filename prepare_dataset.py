#!/usr/bin/env python3
"""
Dataset preparation script for Cross-Modal Remote Sensing Image Retrieval (HAC).
Scans EuroSAT_RGB and EuroSAT-SAR datasets, validates integrity, creates stratified
train (70%), validation (15%), and test (15%) splits with reproducible seed (42),
and saves manifests and class mappings.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("prepare_dataset")

EXPECTED_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake"
]

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def scan_modality(data_dir: Path, modality_name: str) -> List[Dict]:
    """
    Recursively scan a directory for image files, verifying classes and extensions.
    """
    records = []
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    for class_name in EXPECTED_CLASSES:
        class_path = data_dir / class_name
        if not class_path.is_dir():
            logger.warning(f"Class folder {class_name} not found in {data_dir}")
            continue

        for entry in class_path.iterdir():
            if entry.name.startswith("."):
                continue
            ext = entry.suffix.lower()
            if ext in VALID_EXTENSIONS:
                pair_id = entry.stem
                records.append({
                    "rel_path": str(entry.as_posix()),
                    "label": class_name,
                    "pair_id": pair_id,
                    "modality": modality_name
                })
    return records


def prepare_dataset(
    rgb_dir: str = "EuroSAT_RGB",
    sar_dir: str = "EuroSAT-SAR",
    output_dir: str = "data",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
):
    logger.info("Starting dataset scan and validation...")
    rgb_path = Path(rgb_dir)
    sar_path = Path(sar_dir)

    # Scan both modalities
    rgb_records = scan_modality(rgb_path, "rgb")
    sar_records = scan_modality(sar_path, "sar")

    logger.info(f"Scanned {len(rgb_records)} RGB images and {len(sar_records)} SAR images.")

    # Build class mapping
    class_to_id = {cls_name: i for i, cls_name in enumerate(sorted(EXPECTED_CLASSES))}
    id_to_class = {i: cls_name for cls_name, i in class_to_id.items()}

    # Pair alignment check
    rgb_df = pd.DataFrame(rgb_records)
    sar_df = pd.DataFrame(sar_records)

    rgb_stems = set(rgb_df["pair_id"])
    sar_stems = set(sar_df["pair_id"])
    common_stems = rgb_stems.intersection(sar_stems)

    logger.info(f"Unique RGB pair_ids: {len(rgb_stems)}, Unique SAR pair_ids: {len(sar_stems)}")
    logger.info(f"Exact stem matches between RGB and SAR: {len(common_stems)}")

    if len(common_stems) == len(rgb_stems) == len(sar_stems):
        logger.info("Verified 1-to-1 correspondence between EuroSAT_RGB and EuroSAT-SAR patches.")
    else:
        logger.warning("Discrepancies found between RGB and SAR file stems. Matching subset will be used.")

    # Create stratified splits based on pairs so paired optical & SAR images always belong to the same split
    # This prevents any data leakage across splits.
    rng = np.random.RandomState(seed)

    # Group by class to ensure exact class stratification
    pair_splits = {}
    for cls_name in EXPECTED_CLASSES:
        cls_stems = sorted(list(rgb_df[rgb_df["label"] == cls_name]["pair_id"].unique()))
        cls_stems = [s for s in cls_stems if s in common_stems]
        rng.shuffle(cls_stems)

        n_total = len(cls_stems)
        n_train = int(round(n_total * train_ratio))
        n_val = int(round(n_total * val_ratio))
        # Ensure test gets the remainder to sum to n_total exactly
        n_test = n_total - n_train - n_val

        train_stems = cls_stems[:n_train]
        val_stems = cls_stems[n_train:n_train + n_val]
        test_stems = cls_stems[n_train + n_val:]

        for s in train_stems:
            pair_splits[s] = "train"
        for s in val_stems:
            pair_splits[s] = "val"
        for s in test_stems:
            pair_splits[s] = "test"

    # Assign splits and label_ids
    def build_split_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df[df["pair_id"].isin(common_stems)].copy()
        df["split"] = df["pair_id"].map(pair_splits)
        df["label_id"] = df["label"].map(class_to_id)
        df = df.rename(columns={"rel_path": "path"})
        return df[["path", "label", "label_id", "modality", "split", "pair_id"]]

    all_rgb = build_split_df(rgb_df)
    all_sar = build_split_df(sar_df)
    all_data = pd.concat([all_rgb, all_sar], ignore_index=True)

    # Create output directory
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = all_data[all_data["split"] == "train"].sort_values(by=["label_id", "modality", "pair_id"]).reset_index(drop=True)
    val_df = all_data[all_data["split"] == "val"].sort_values(by=["label_id", "modality", "pair_id"]).reset_index(drop=True)
    test_df = all_data[all_data["split"] == "test"].sort_values(by=["label_id", "modality", "pair_id"]).reset_index(drop=True)

    # Save CSVs
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    # Save class mapping and metadata
    metadata = {
        "class_to_id": class_to_id,
        "id_to_class": id_to_class,
        "classes": sorted(EXPECTED_CLASSES),
        "random_seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "counts": {
            "train": {
                "total": len(train_df),
                "rgb": len(train_df[train_df["modality"] == "rgb"]),
                "sar": len(train_df[train_df["modality"] == "sar"]),
            },
            "val": {
                "total": len(val_df),
                "rgb": len(val_df[val_df["modality"] == "rgb"]),
                "sar": len(val_df[val_df["modality"] == "sar"]),
            },
            "test": {
                "total": len(test_df),
                "rgb": len(test_df[test_df["modality"] == "rgb"]),
                "sar": len(test_df[test_df["modality"] == "sar"]),
            }
        }
    }

    with open(out_dir / "class_mapping.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Dataset preparation complete:")
    logger.info(f"  Train samples: {len(train_df)} (RGB: {metadata['counts']['train']['rgb']}, SAR: {metadata['counts']['train']['sar']})")
    logger.info(f"  Val samples:   {len(val_df)} (RGB: {metadata['counts']['val']['rgb']}, SAR: {metadata['counts']['val']['sar']})")
    logger.info(f"  Test samples:  {len(test_df)} (RGB: {metadata['counts']['test']['rgb']}, SAR: {metadata['counts']['test']['sar']})")

    # Verification: check per-class distribution across splits
    logger.info("Per-class split breakdown:")
    for cls_name in sorted(EXPECTED_CLASSES):
        c_train = len(train_df[(train_df["label"] == cls_name) & (train_df["modality"] == "rgb")])
        c_val = len(val_df[(val_df["label"] == cls_name) & (val_df["modality"] == "rgb")])
        c_test = len(test_df[(test_df["label"] == cls_name) & (test_df["modality"] == "rgb")])
        logger.info(f"  {cls_name:<22}: Train={c_train:<5} Val={c_val:<5} Test={c_test:<5}")

    logger.info(f"Files saved in {out_dir}/: train.csv, val.csv, test.csv, class_mapping.json")


def main():
    parser = argparse.ArgumentParser(description="Prepare EuroSAT RGB/SAR dataset splits")
    parser.add_argument("--rgb_dir", type=str, default="EuroSAT_RGB", help="Path to EuroSAT RGB directory")
    parser.add_argument("--sar_dir", type=str, default="EuroSAT-SAR", help="Path to EuroSAT SAR directory")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory for manifests")
    parser.add_argument("--train_ratio", type=float, default=0.70, help="Train ratio (default: 0.70)")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation ratio (default: 0.15)")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test ratio (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")

    args = parser.parse_args()
    prepare_dataset(
        rgb_dir=args.rgb_dir,
        sar_dir=args.sar_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
