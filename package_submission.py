#!/usr/bin/env python3
"""
Packaging script for final assignment deliverable: submission.zip.
Assembles the complete project including:
- All source codes, tests, README, requirements, and academic report.pdf.
- Saved trained model checkpoints and training/validation loss plots.
- The COMPLETE dataset (all 27,000 EuroSAT_RGB and 27,000 EuroSAT-SAR images).
- Data manifests (train.csv, val.csv, test.csv, class_mapping.json).
Excludes .venv, __pycache__, .DS_Store, and temporary files.
"""

import os
import sys
import zipfile
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("package_submission")

REQUIRED_SOURCE_FILES = [
    "prepare_dataset.py",
    "dataset.py",
    "model.py",
    "losses.py",
    "retrieval.py",
    "metrics.py",
    "utils.py",
    "train.py",
    "test.py",
    "README.md",
    "requirements.txt",
    "report.pdf",
    "tests/test_hac.py"
]

REQUIRED_DATA_FILES = [
    "data/train.csv",
    "data/val.csv",
    "data/test.csv",
    "data/class_mapping.json"
]

REQUIRED_OUTPUT_FILES = [
    "outputs/checkpoints/best_model.pth",
    "outputs/checkpoints/final_model.pth",
    "outputs/plots/train_loss.png",
    "outputs/plots/val_loss.png",
    "outputs/plots/loss_components.png",
    "outputs/training_history.csv",
    "outputs/test_results.csv"
]

DATASET_DIRECTORIES = [
    "EuroSAT_RGB",
    "EuroSAT-SAR"
]


def scan_dataset_files(root: Path):
    """
    Recursively scans EuroSAT_RGB and EuroSAT-SAR for all valid images,
    ignoring .DS_Store and temporary files.
    """
    valid_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    image_paths = []
    for dir_name in DATASET_DIRECTORIES:
        dataset_dir = root / dir_name
        if not dataset_dir.exists():
            logger.warning(f"Dataset directory not found: {dataset_dir}")
            continue
        for class_dir in sorted(dataset_dir.iterdir()):
            if class_dir.is_dir() and not class_dir.name.startswith("."):
                for img_path in sorted(class_dir.iterdir()):
                    if img_path.is_file() and img_path.suffix.lower() in valid_extensions:
                        image_paths.append(img_path.relative_to(root))
    return image_paths


def create_submission_zip(zip_filename: str = "submission.zip"):
    root = Path.cwd()
    zip_path = root / zip_filename

    logger.info(f"Preparing complete deliverable: {zip_path}")

    # Check required deliverables
    missing = []
    for f in REQUIRED_SOURCE_FILES + REQUIRED_DATA_FILES + REQUIRED_OUTPUT_FILES:
        p = root / f
        if not p.exists():
            missing.append(f)

    if missing:
        logger.warning(f"The following deliverables are missing: {missing}")

    # Scan full dataset
    logger.info("Scanning complete EuroSAT_RGB and EuroSAT-SAR datasets...")
    dataset_files = scan_dataset_files(root)
    logger.info(f"Identified {len(dataset_files)} total images to pack into archive.")

    total_files = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # 1. Source files and tests
        for f in REQUIRED_SOURCE_FILES:
            p = root / f
            if p.exists():
                zf.write(p, arcname=f)
                total_files += 1

        # 2. Manifests
        for f in REQUIRED_DATA_FILES:
            p = root / f
            if p.exists():
                zf.write(p, arcname=f)
                total_files += 1

        # 3. Outputs & Checkpoints
        for f in REQUIRED_OUTPUT_FILES:
            p = root / f
            if p.exists():
                zf.write(p, arcname=f)
                total_files += 1

        # 4. Complete datasets
        logger.info("Packaging complete EuroSAT_RGB and EuroSAT-SAR dataset images...")
        log_interval = max(len(dataset_files) // 10, 1)
        for idx, img_rel in enumerate(dataset_files, 1):
            p = root / img_rel
            zf.write(p, arcname=str(img_rel))
            total_files += 1
            if idx % log_interval == 0 or idx == len(dataset_files):
                pct = (idx / len(dataset_files)) * 100
                logger.info(f"  Progress: {idx}/{len(dataset_files)} images ({pct:.1f}%)")

    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    logger.info("=" * 60)
    logger.info("Complete submission ZIP created successfully!")
    logger.info(f"  Destination:        {zip_path}")
    logger.info(f"  Total files packed: {total_files}")
    logger.info(f"  Archive size:       {zip_size_mb:.2f} MB")
    logger.info("=" * 60)


if __name__ == "__main__":
    create_submission_zip()
