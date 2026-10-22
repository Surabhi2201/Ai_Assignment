# Reproducing HAC: Robust Cross-Modal Remote Sensing Image Retrieval via Maximal Correlation Augmentation

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.14](https://img.shields.io/badge/PyTorch-2.14-orange.svg)](https://pytorch.org/)
[![Device](https://img.shields.io/badge/Device-Apple%20Silicon%20MPS%20%7C%20CUDA%20%7C%20CPU-success.svg)]()
[![Course](https://img.shields.io/badge/Course-ECPE49%20Foundations%20of%20AI-blueviolet.svg)]()
[![Institution](https://img.shields.io/badge/NIT%20Trichy-July%202026-darkgreen.svg)]()

This repository provides a complete, reproducible Python implementation of the research paper:
> **Zhuoyue Wang, Xueqian Wang, Gang Li, and Chengxi Li**, *"Robust Cross-Modal Remote Sensing Image Retrieval via Maximal Correlation Augmentation"*, **IEEE Transactions on Geoscience and Remote Sensing (TGRS)**, vol. 62, Art no. 4705517, 2024. [DOI: 10.1109/TGRS.2024.3406606](https://doi.org/10.1109/TGRS.2024.3406606)

---

## 1. Environment Setup & Dependency Installation

Before executing any training or testing scripts, install all required dependencies from [`requirements.txt`](requirements.txt):

```bash
# Recommended: Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate       # On Linux/macOS
# .venv\Scripts\activate       # On Windows

# Install all required libraries
pip install -r requirements.txt
```

### Hardware Support
The codebase automatically detects and utilizes the fastest available accelerator:
1. **Apple Silicon MPS** (`mps` backend with FP16 automatic mixed precision).
2. **NVIDIA CUDA** (`cuda` backend with PyTorch AMP support).
3. **Standard CPU** fallback.

---

## 2. Proper Way of Executing (Evaluation Sequence)

Following the assignment guidelines (**Assessment IV, ECPE49, NIT Trichy**), the professor evaluates submissions in the following sequence from the terminal in the project root directory:

```mermaid
flowchart LR
    A["1. Install Requirements<br/>pip install -r requirements.txt"] --> B["2. Step E1: Test Existing Model<br/>python test.py"]
    B --> C["3. Step E2: Retrain Model<br/>python train.py --seed 42"]
    C --> D["4. Step E3: Test Newly Trained Model<br/>python test.py"]
    D --> E["5. Compare E1 & E3 mAP<br/>Results match (Reproducible)"]
```

### Step E1: Test with the Included Pre-Trained Model
Run the testing command from the terminal:
```bash
python test.py
```
* **What this does:** Automatically loads the test dataset manifest (`data/test.csv`), restores the saved model checkpoint from `outputs/checkpoints/best_model.pth`, performs bidirectional cross-modal retrieval (Optical $\rightarrow$ SAR and SAR $\rightarrow$ Optical) using Dempster–Shafer evidence fusion, and prints the full-gallery test **mAP** and **NDCG**.

### Step E2: Train the Model from Scratch
Run the training command from the terminal:
```bash
python train.py --seed 42
```
* **What this does:** Initializes the dual-backbone HAC network with random seed 42, trains on `data/train.csv` (18,900 pairs) using Adam optimizer and linear learning rate decay, validates after every epoch on `data/val.csv` (4,050 pairs), saves the best checkpoint to `outputs/checkpoints/best_model.pth`, and generates training/validation loss plots into `outputs/plots/`.

### Step E3: Verify Reproducibility
Re-run testing on the newly saved checkpoint:
```bash
python test.py
```
* **Expected Outcome:** The newly trained model produces mAP values virtually identical to Step E1, confirming complete reproducibility.

---

## 3. What Does `--seed 42` Represent?

The parameter `--seed 42` guarantees **100% deterministic reproducibility** across different runs, Python environments, and operating systems. When `--seed 42` is passed, the function `set_seed(42)` sets the random state across all underlying pseudo-random number generators:

1. **Python Built-in Random:** `random.seed(42)`
2. **NumPy Random Generator:** `np.random.seed(42)`
3. **PyTorch CPU Generator:** `torch.manual_seed(42)`
4. **PyTorch CUDA Generator:** `torch.cuda.manual_seed_all(42)`
5. **PyTorch Apple Silicon MPS Generator:** `torch.mps.manual_seed(42)`
6. **CuDNN Deterministic Flags:** `torch.backends.cudnn.deterministic = True`

### What `--seed 42` Governs in the Pipeline:
* **Stratified Dataset Splitting:** Ensures that the 70% train, 15% validation, and 15% test splits contain the exact same images across categories every time `prepare_dataset.py` is run.
* **Weight Initialization:** Ensures the custom linear projection heads (Euclidean 512-D, HGRMC 512-D) and shared classifier ($512 \to 10$) are initialized with identical Kaiming normal random weights.
* **In-Batch Cross-Modal Pairing:** Ensures the random selection of positive ($s_i^+, v_i^+$) and negative ($s_i^-, v_i^-$) samples within each mini-batch is deterministic across runs.
* **DataLoader Mini-batch Shuffling:** Ensures identical mini-batch sequencing during training.

---

## 4. Checkpoint Storage & Saved Files

All model weights, logs, and plots are stored inside the **`outputs/`** directory:

```text
outputs/
├── checkpoints/
│   ├── best_model.pth    <-- BEST CHECKPOINT (lowest validation loss)
│   └── final_model.pth   <-- FINAL CHECKPOINT (saved at the last epoch)
├── plots/
│   ├── train_loss.png    <-- Training loss vs epoch curve
│   ├── val_loss.png      <-- Validation loss vs epoch curve
│   └── loss_components.png <-- Breakdown of J_P, J_EUC, and J_HGR components
├── training_history.csv  <-- Epoch-by-epoch loss, metrics, and LR log
└── test_results.csv      <-- Exact mAP and NDCG results from test.py
```

### Contents of `best_model.pth`:
The checkpoint file is self-contained and stores a Python dictionary containing:
* `model_state_dict`: Complete learned weights for both AlexNet backbones, all 4 projection heads, and the shared classifier.
* `epoch`: The epoch number at which the lowest validation loss was achieved.
* `best_val_loss`: The lowest validation loss value.
* `class_mapping`: Dictionary mapping semantic category names to numeric label IDs.
* `num_classes`: 10 classes.
* `feature_dim`: 512 dimensions.
* `config`: Complete training hyperparameters dictionary ($\alpha, \beta, \text{margin}, \text{learning rates}, \text{batch size}, \text{seed}$).

`test.py` automatically detects and loads `outputs/checkpoints/best_model.pth`. If missing, it automatically falls back to `outputs/checkpoints/final_model.pth`.

---

## 5. Details of the Codebase

The implementation is modularized into separate Python scripts following clean software engineering and machine learning research standards:

| File | Primary Responsibility | Key Mathematical References in Paper |
| :--- | :--- | :--- |
| **[`train.py`](train.py)** | Training entry point; executes training loop with Adam optimizer, differential learning rates ($10^{-4}$ backbone, $5 \times 10^{-4}$ heads), linear LR decay, validation tracking, and checkpoint saving. | Section IV-A & VI-B |
| **[`test.py`](test.py)** | Testing entry point; loads saved checkpoint, extracts 512-D features for test queries and gallery, performs cross-modal retrieval, and prints test mAP and NDCG. | Section IV-D & VI-C |
| **[`model.py`](model.py)** | Implements `HACModel` with modality-specific AlexNet backbones (extracting 4096-D from `fc3`), dual Euclidean (512-D) and HGRMC (512-D) projection heads, and shared linear classifier $P(\cdot)$. | Section IV-B, Fig. 2, Eq. (10) |
| **[`losses.py`](losses.py)** | Implements all loss functions: Euclidean similarity $J_I$ & ranking $J_R$ ($m=0.3$), soft-HGR maximal correlation $J_C$ & correlation ranking $J_{CR}$, MSE prediction loss $J_P$, and total loss $J$. | Equations (3)–(11) |
| **[`retrieval.py`](retrieval.py)** | Implements normalized Euclidean score $s_{\mathrm{EUC}}$, normalized HGRMC score $s_{\mathrm{HGR}}$, Dempster–Shafer evidence fusion ($b_1=b_2=0.95$), and plausibility ranking $\mathrm{PLB}(A_1) = m_c(A_1) + m_c(A_3)$. | Equations (12)–(15) |
| **[`metrics.py`](metrics.py)** | Implements full-gallery Mean Average Precision (mAP), Normalized Discounted Cumulative Gain (NDCG), and Precision-Recall curves. | Equations (16)–(18) |
| **[`dataset.py`](dataset.py)** | Implements `EuroSATPairDataset` (paired optical & SAR loader), `SingleModalityDataset` (for retrieval feature extraction), and in-batch positive/negative sampling `sample_in_batch_pairs`. | Section IV-C |
| **[`prepare_dataset.py`](prepare_dataset.py)** | Scans raw datasets (`EuroSAT_RGB/` and `EuroSAT-SAR/`), validates matching pairs, creates reproducible stratified 70/15/15 splits, and saves manifest CSVs. | Assignment Section 4 |
| **[`utils.py`](utils.py)** | Helper utilities: device selection (MPS/CUDA/CPU), deterministic seed setting, SAR dual-pol TIFF decoding (false-color composite), and matplotlib loss curve plotting. | Section IV-B & VI-B |
| **[`tests/test_hac.py`](tests/test_hac.py)** | Comprehensive unit test suite with 15 automated tests verifying data split disjointness, AlexNet shapes, gradients, loss formulas, DS fusion, and closed-form mAP. | pytest suite |
| **[`generate_report.py`](generate_report.py)** | Generates publication-quality 3-page academic report in PDF format (`report.pdf`). | Assignment Deliverable |
| **[`package_submission.py`](package_submission.py)** | Builds the final self-contained submission archive (`submission.zip`). | Assignment Deliverable |

---

## 6. Directory Structure

```text
AI_Assign_Surabhi/
├── requirements.txt            # Python library dependencies (PyTorch, Torchvision, etc.)
├── README.md                   # Complete instructions and documentation (this file)
├── report.pdf                  # 3-page academic research report
├── submission.zip              # Complete self-contained assignment archive (585 MB)
│
├── train.py                    # Training script (single-line terminal command)
├── test.py                     # Testing script (single-line terminal command)
├── model.py                    # Dual-branch AlexNet & projection heads architecture
├── losses.py                   # Equations (3)-(11): Euclidean, Soft-HGR, Prediction losses
├── retrieval.py                # Equations (12)-(15): DS evidence fusion & plausibility ranking
├── metrics.py                  # Equations (16)-(18): Full-gallery mAP and NDCG
├── dataset.py                  # PyTorch Dataset and in-batch sampling
├── prepare_dataset.py          # Stratified dataset split generator (seed 42)
├── utils.py                    # Device selection, seeds, SAR false-color, plot helpers
├── generate_report.py          # Script to generate report.pdf
├── package_submission.py       # Script to package submission.zip
│
├── data/                       # Split manifests & class metadata (Seed 42)
│   ├── train.csv               # 70% training split (18,900 pairs = 37,800 images)
│   ├── val.csv                 # 15% validation split (4,050 pairs = 8,100 images)
│   ├── test.csv                # 15% test split (4,050 pairs = 8,100 images)
│   ├── class_mapping.json      # 10 semantic classes and numeric IDs
│   └── test_data/              # Packaged test data for standalone grading
│       ├── rgb/                # 4,050 held-out optical test images (.jpg)
│       └── sar/                # 4,050 held-out SAR test images (.tif)
│
├── outputs/                    # Training outputs and checkpoints
│   ├── checkpoints/
│   │   ├── best_model.pth      # Best trained model checkpoint (Epoch 10)
│   │   └── final_model.pth     # Final model checkpoint
│   ├── plots/
│   │   ├── train_loss.png      # Training loss curve
│   │   ├── val_loss.png        # Validation loss curve
│   │   └── loss_components.png # Breakdown of J_P, J_EUC, J_HGR
│   ├── training_history.csv    # Loss metrics logged per epoch
│   └── test_results.csv        # Evaluated test metrics
│
├── tests/
│   └── test_hac.py             # 15 unit tests runnable via pytest
│
├── EuroSAT_RGB/                # Raw Sentinel-2 optical images (27,000 files)
├── EuroSAT-SAR/                # Raw Sentinel-1 SAR images (27,000 TIFF files)
└── EuroSAT_MS/                 # Raw Multispectral images (kept untouched)
```

---

## 7. Mathematical Formulations Implemented

### A. Training Loss (Equations 3–11)
The unified objective minimized during training is:
$$\mathcal{J} = \mathcal{J}_P + \alpha \mathcal{J}_{\mathrm{EUC}} + \beta \mathcal{J}_{\mathrm{HGR}}$$
where $\alpha = 0.001$, $\beta = 0.001$, and:
1. **Prediction Loss ($\mathcal{J}_P$, Eq. 10):**
   $$\mathcal{J}_P = \frac{1}{N}\sum_{i=1}^N \left[ \|P(f^E(v_i)) - y_i\|^2 + \|P(f^H(v_i)) - y_i\|^2 + \|P(g^E(s_i)) - y_i\|^2 + \|P(g^H(s_i)) - y_i\|^2 \right]$$
2. **Euclidean Loss ($\mathcal{J}_{\mathrm{EUC}} = \mathcal{J}_I + \mathcal{J}_R$, Eqs. 3–5):**
   $$\mathcal{J}_I = \frac{1}{N}\sum_{i=1}^N d(f^E(v_i), g^E(s_i))$$
   $$\mathcal{J}_R = \frac{1}{N}\sum_{i=1}^N \left[ \max(d(f^E(v_i), g^E(s_i^+)) - d(f^E(v_i), g^E(s_i^-)) + m, 0) + \max(d(f^E(v_i^+), g^E(s_i)) - d(f^E(v_i^-), g^E(s_i)) + m, 0) \right]$$
   with margin $m = 0.3$.
3. **HGRMC Correlation Loss ($\mathcal{J}_{\mathrm{HGR}} = \mathcal{J}_C + \mathcal{J}_{CR}$, Eqs. 6–9):**
   $$\mathcal{J}_C(X, Y) = \frac{1}{2} \mathrm{tr}(\mathrm{cov}(\tilde{X})\mathrm{cov}(\tilde{Y})) - \frac{1}{N}\sum_{i=1}^N \tilde{x}_i^T \tilde{y}_i$$
   $$\mathcal{J}_{CR} = \mathcal{J}_C(V, S^+) + \mathcal{J}_C(V^+, S) - \mathcal{J}_C(V, S^-) - \mathcal{J}_C(V^-, S)$$

### B. Dempster–Shafer Evidence Theory Fusion (Equations 12–15)
For each query across all gallery candidates:
* Mass functions with confidence $b_1 = 0.95, b_2 = 0.95$:
  $$m_1(A_1) = b_1 s_{\mathrm{EUC}}, \quad m_1(A_2) = b_1 (1 - s_{\mathrm{EUC}}), \quad m_1(A_3) = 1 - b_1$$
  $$m_2(A_1) = b_2 s_{\mathrm{HGR}}, \quad m_2(A_2) = b_2 (1 - s_{\mathrm{HGR}}), \quad m_2(A_3) = 1 - b_2$$
* Conflict coefficient $Z = m_1(A_1)m_2(A_2) + m_1(A_2)m_2(A_1)$.
* Plausibility ranking: $\mathrm{PLB}(A_1) = m_c(A_1) + m_c(A_3)$, sorted in descending order.

---

## 8. Verified Experimental Results on EuroSAT

Full-gallery evaluation on the held-out test split (4,050 Optical queries $\leftrightarrow$ 4,050 SAR gallery candidates):

| Direction | Method / Fusion | mAP (Mean Average Precision) | NDCG |
| :--- | :--- | :---: | :---: |
| **Optical $\rightarrow$ SAR** | Dempster–Shafer Fusion (HAC) | **0.2889** | **0.7388** |
| **SAR $\rightarrow$ Optical** | Dempster–Shafer Fusion (HAC) | **0.2858** | **0.7337** |
| **Mean Bidirectional** | **Dempster–Shafer Fusion (HAC)** | **0.2874** | **0.7363** |

### Ablation Study (Reproducing Paper Table III)
| Representation / Fusion Scheme | Mean mAP | Gain over Baseline |
| :--- | :---: | :---: |
| Euclidean-Only ($s_{\mathrm{EUC}}$) | 0.2374 | Baseline |
| HGRMC-Only ($s_{\mathrm{HGR}}$) | 0.2531 | $+1.57\%$ |
| Simple Score Sum ($s_{\mathrm{EUC}} + s_{\mathrm{HGR}}$) | 0.2871 | $+4.97\%$ |
| **Dempster–Shafer Evidence Fusion (Proposed HAC)** | **0.2874** | **$+5.00\%$** |

---

## 9. Running the Automated Unit Test Suite

To verify all mathematical formulas, tensor shapes, gradient flows, and metric calculations:
```bash
python -m pytest -q
# or with virtual environment:
.venv/bin/pytest -q tests/test_hac.py
```
*Expected Output:*
```text
15 passed in 24.07s
```
All 15 tests pass cleanly.
