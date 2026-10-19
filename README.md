# Reproducing HAC: Robust Cross-Modal Remote Sensing Image Retrieval via Maximal Correlation Augmentation

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.14](https://img.shields.io/badge/PyTorch-2.14-orange.svg)](https://pytorch.org/)
[![Device](https://img.shields.io/badge/Device-Apple%20Silicon%20MPS%20%7C%20CUDA%20%7C%20CPU-success.svg)]()
[![License](https://img.shields.io/badge/License-Academic%20Evaluation-lightgrey.svg)]()

This repository provides a complete, reproducible Python implementation of the research paper:
> **Zhuoyue Wang, Xueqian Wang, Gang Li, and Chengxi Li**, *"Robust Cross-Modal Remote Sensing Image Retrieval via Maximal Correlation Augmentation"*, **IEEE Transactions on Geoscience and Remote Sensing (TGRS)**, vol. 62, Art no. 4705517, 2024. [DOI: 10.1109/TGRS.2024.3406606](https://doi.org/10.1109/TGRS.2024.3406606)

---

## 1. Quick Start Commands (Single-Line Commands)

As specified in the assignment evaluation guidelines, all scripts can be executed directly from the terminal from the extracted project root directory.

### E1. Testing with Included Trained Model and Test Data:
```bash
python test.py
```
*Automatically loads `data/test.csv`, restores `outputs/checkpoints/best_model.pth`, performs Dempster–Shafer cross-modal retrieval for Optical-to-SAR and SAR-to-Optical, and prints the bidirectional test mAP.*

### E2. Training the Model from Scratch:
```bash
python train.py --seed 42
```
*Trains the HAC network with modality-specific AlexNet backbones, saves the best model checkpoint to `outputs/checkpoints/best_model.pth`, final checkpoint to `outputs/checkpoints/final_model.pth`, and generates all loss plots.*

### Optional: Dataset Preparation & Stratified Split Generation:
```bash
python prepare_dataset.py --seed 42
```
*Scans `EuroSAT_RGB` and `EuroSAT-SAR`, validates image integrity, creates reproducible 70/15/15 stratified splits, and saves `data/train.csv`, `data/val.csv`, `data/test.csv`, and `data/class_mapping.json`.*

### Optional: Run Automated Test Suite:
```bash
python -m pytest -q
```
*Executes all 15 unit tests covering loss mathematics, gradients, DS evidence fusion, feature extraction, and end-to-end optimization.*

---

## 2. Project Architecture & Methodology

Cross-modal remote sensing retrieval involves querying with an image from one sensor modality (e.g., optical RGB) to find semantically relevant scenes in a gallery of another sensor modality (e.g., synthetic aperture radar, SAR) and vice versa.

### HAC Framework Overview

Existing approaches primarily minimize/maximize Euclidean distances in a common latent space. However, Euclidean metrics lack scale normalization and decorrelation, making them fragile in the presence of remote sensing image noise (e.g., speckle in SAR and atmospheric haze in optical).

HAC addresses this through two key algorithmic innovations:
1. **Training Phase (Equations 3–11):**
   - **Dual Modality-Specific Feature Extraction:** Optical and SAR images are processed through separate AlexNet backbones pretrained on ImageNet to extract 4096-dimensional high-level features from `fc3`.
   - **Parallel Projection Heads:** Each modality projects the 4096-D representation into two parallel 512-D spaces:
     - **Euclidean Features ($f^E, g^E$):** Constrained by similarity loss $J_I$ and margin ranking loss $J_R$ ($m=0.3$).
     - **HGRMC Features ($f^H, g^H$):** Constrained by soft-HGR maximal correlation loss $J_C$ and correlation ranking loss $J_{CR}$. The soft-HGR objective maximizes cross-modal statistical dependency while enforcing batch covariance decorrelation.
   - **Supervised Semantic Prediction ($J_P$):** A shared linear classifier $P(\cdot)$ maps 512-D features to one-hot class vectors using MSE loss.
   - **Combined Total Loss:** $J = J_P + \alpha J_{\mathrm{EUC}} + \beta J_{\mathrm{HGR}}$ with $\alpha=0.001, \beta=0.001$.

2. **Retrieval Phase (Equations 12–15):**
   - **Euclidean Score ($s_{\mathrm{EUC}}$):** Cosine similarity between query and gallery Euclidean features, min-max normalized to $[0, 1]$.
   - **HGRMC Score ($s_{\mathrm{HGR}}$):** Inner product correlation between query and gallery HGRMC features, min-max normalized to $[0, 1]$.
   - **Dempster–Shafer (DS) Evidence Theory Fusion:**
     - Masses for reliable ($A_1$), unreliable ($A_2$), and uncertainty ($A_3$) hypotheses with confidence $b_1=0.95, b_2=0.95$.
     - Conflict coefficient $Z = m_1(A_1)m_2(A_2) + m_1(A_2)m_2(A_1)$.
     - Combined mass $m_c$ and plausibility scoring: $\mathrm{PLB}(A_1) = m_c(A_1) + m_c(A_3)$.
     - Candidates ranked in descending order of $\mathrm{PLB}(A_1)$.

---

## 3. Dataset Substitution & Pairing Verification

### Dataset Substitution Note
* **Original Paper:** Evaluated on CBRSIR_VS (26,901 optical-SAR pairs; optical from Google Earth at 1-m resolution, SAR from Sentinel-1 at 10-m resolution).
* **This Assignment:** Evaluated on **EuroSAT** (Sentinel-2 Optical RGB and Sentinel-1 Dual-Polarization SAR from HuggingFace `wangyi111/EuroSAT-SAR`).
* Numerical retrieval scores reflect the properties, resolution, and class structures of EuroSAT and are not expected to be identical to CBRSIR_VS numerical values.

### Pairing Structure
Each sample in EuroSAT-SAR has an exact corresponding sample in EuroSAT_RGB with matching filename stems (e.g., `AnnualCrop_1.jpg` $\leftrightarrow$ `AnnualCrop_1.tif`). The datasets contain 10 balanced land-use categories:
`AnnualCrop`, `Forest`, `HerbaceousVegetation`, `Highway`, `Industrial`, `Pasture`, `PermanentCrop`, `Residential`, `River`, `SeaLake`.

### SAR Dual-Polarization Preprocessing
EuroSAT-SAR images are 2-channel 32-bit floating-point TIFF files containing Sentinel-1 VV and VH backscatter coefficients in decibels (dB). To feed them into standard 3-channel AlexNet backbones, we synthesize an authoritative 3-channel false-color composite:
- **Channel 0:** Normalized VV backscatter (clipped to $[-35, 5]$ dB $\rightarrow [0, 1]$)
- **Channel 1:** Normalized VH backscatter (clipped to $[-45, 0]$ dB $\rightarrow [0, 1]$)
- **Channel 2:** Cross-polarization ratio $\mathrm{VV} - \mathrm{VH}$ (clipped to $[0, 25]$ dB $\rightarrow [0, 1]$)
- Standard ImageNet normalization: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.

---

## 4. Directory Structure

```text
AI_Assign_Surabhi/
├── prepare_dataset.py      # Dataset scanning, verification, and stratified splitting
├── dataset.py              # PyTorch Dataset and in-batch sampling loaders
├── model.py                # HAC architecture: modality AlexNets, projection heads, classifier
├── losses.py               # Equations (3)-(11): Euclidean, Soft-HGR, Prediction, Total loss
├── retrieval.py            # Equations (12)-(15): Euclidean & HGRMC scoring, DS evidence fusion
├── metrics.py              # Full-gallery mAP (Eq. 16), NDCG (Eqs. 17-18), PR curves
├── utils.py                # Device selection, seeds, SAR/RGB loading, loss plot generation
├── train.py                # Training entry point (saves checkpoints, history, and plots)
├── test.py                 # Independent testing entry point (loads checkpoint, computes mAP)
├── generate_report.py      # Generates comprehensive research report (report.pdf)
├── package_submission.py   # Assembles submission.zip
│
├── data/
│   ├── train.csv           # 70% stratified training split (18,900 pairs)
│   ├── val.csv             # 15% stratified validation split (4,050 pairs)
│   ├── test.csv            # 15% stratified test split (4,050 pairs)
│   └── class_mapping.json  # Class to ID mapping and split metadata
│
├── outputs/
│   ├── checkpoints/
│   │   ├── best_model.pth  # Best validation loss checkpoint
│   │   └── final_model.pth # Final epoch checkpoint
│   ├── plots/
│   │   ├── train_loss.png  # Training loss vs epoch
│   │   ├── val_loss.png    # Validation loss vs epoch
│   │   └── loss_components.png # Individual J_P, J_EUC, J_HGR components
│   ├── training_history.csv # Epoch-by-epoch metrics log
│   └── test_results.csv    # Evaluated mAP and NDCG test results
│
├── tests/
│   └── test_hac.py         # 15 automated unit tests runnable via pytest
│
├── requirements.txt        # Python library dependencies
├── README.md               # Complete project documentation
├── report.pdf              # Concise academic research report
└── submission.zip          # Final submission archive
```

---

## 5. Environment & Dependencies

The implementation requires Python 3.12+ and PyTorch 2.0+. Install dependencies with:
```bash
pip install -r requirements.txt
```

### Hardware Support
Automatic hardware detection prioritizes:
1. **Apple Silicon MPS** (`mps` backend with automatic mixed precision `autocast`).
2. **NVIDIA CUDA** (`cuda` backend with GPU acceleration).
3. **CPU** fallback.

---

## 6. Evaluation Metric: Mean Average Precision (mAP)

For query $q$ across gallery candidates $k \in \{1, \dots, G\}$:
$$\mathrm{AP}(q) = \frac{1}{R_q} \sum_{k=1}^G P_q(k) \, r_q(k)$$
where $R_q$ is the total number of relevant gallery images, $P_q(k)$ is precision at rank $k$, and $r_q(k) \in \{0, 1\}$ indicates relevance (matching class label).
The Mean Average Precision is:
$$\mathrm{mAP} = \frac{1}{Q} \sum_{q=1}^Q \mathrm{AP}(q)$$

Bidirectional evaluation measures:
- **Optical-to-SAR (RGB $\rightarrow$ SAR):** Query = RGB, Gallery = SAR
- **SAR-to-Optical (SAR $\rightarrow$ RGB):** Query = SAR, Gallery = RGB
- **Mean Bidirectional mAP:** Average of both directions.

---

## 7. Submission Checklist & Reproducibility Verification

- [x] All source code in clean, separate `.py` files (no Jupyter notebooks).
- [x] Train/validation/test manifests generated with fixed seed 42.
- [x] Held-out test dataset packaged for offline evaluation.
- [x] Saved checkpoint (`best_model.pth`) contains all weights and metadata.
- [x] Training and validation loss plots generated.
- [x] Bidirectional mAP reported for RGB $\rightarrow$ SAR and SAR $\rightarrow$ RGB.
- [x] Single-line commands tested and verified from terminal.
- [x] 15 automated unit tests passing (`python -m pytest -q`).
- [x] Short report compiled to `report.pdf`.
- [x] Complete self-contained `submission.zip` created.
# Ai_Assignment
# Ai_Assignment
