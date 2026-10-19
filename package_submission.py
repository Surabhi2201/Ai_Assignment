#!/usr/bin/env python3
"""
Packaging script for final assignment deliverable: submission.zip.
Assembles all code, README, requirements, test data, saved checkpoints,
plots, training history, test results, unit tests, and PDF report.
Excludes .venv, __pycache__, .DS_Store, and non-test dataset images.
"""

import os
import sys
import zipfile
import logging
from pathlib import Path
import pandas as pd

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
    "generate_report.py",
    "package_submission.py",
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
    "outputs/plots/train_loss.png",
    "outputs/plots/val_loss.png",
    "outputs/plots/loss_components.png",
    "outputs/training_history.csv",
    "outputs/test_results.csv"
]


def create_submission_zip(
    zip_filename: str = "submission.zip",
    include_test_images: bool = True
):
    root = Path.cwd()
    zip_path = root / zip_filename

    logger.info(f"Preparing deliverable: {zip_path}")

    # Check required files
    missing = []
    for f in REQUIRED_SOURCE_FILES + REQUIRED_DATA_FILES + REQUIRED_OUTPUT_FILES:
        p = root / f
        if not p.exists():
            missing.append(f)

    if missing:
        logger.warning(f"The following required deliverables are currently missing: {missing}")
        logger.warning("Continuing to package available files...")

    # Load test images list
    test_csv = root / "data" / "test.csv"
    test_image_paths = []
    if test_csv.exists() and include_test_images:
        test_df = pd.read_csv(test_csv)
        test_image_paths = [p for p in test_df["path"] if (root / p).exists()]
        logger.info(f"Identified {len(test_image_paths)} held-out test images to include in package.")

    # Create ZIP
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

        # 4. Held-out test images
        if include_test_images:
            logger.info("Packaging test images into zip...")
            for img_rel in test_image_paths:
                p = root / img_rel
                zf.write(p, arcname=img_rel)
                total_files += 1

    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    logger.info("=" * 60)
    logger.info(f"Submission zip created successfully!")
    logger.info(f"  Destination: {zip_path}")
    logger.info(f"  Total files packed: {total_files}")
    logger.info(f"  Archive size: {zip_size_mb:.2f} MB")
    logger.info("=" * 60)


if __name__ == "__main__":
    create_submission_zip()
