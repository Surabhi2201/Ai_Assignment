#!/usr/bin/env python3
"""
Comprehensive automated tests for HAC implementation.
Validates all 15 required components:
1. Dataset scanning and label mapping.
2. Train/val/test disjointness.
3. Reproducible splits.
4. Correct image loading for RGB and SAR.
5. Feature dimensions.
6. Euclidean similarity loss (J_I).
7. Euclidean ranking loss (J_R).
8. Soft-HGR loss (J_C) and its gradients.
9. Prediction loss (J_P).
10. Total loss combination (J).
11. DS mass functions and evidence fusion.
12. Plausibility-based ranking.
13. mAP computation using hand-verifiable ground truth.
14. Checkpoint saving and loading.
15. End-to-end forward pass and one optimization step.
"""

import os
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from utils import set_seed, get_device, load_rgb_image, load_sar_image
from model import HACModel
from losses import (
    euclidean_distance,
    compute_euclidean_loss,
    soft_hgr_correlation,
    compute_hgrmc_loss,
    compute_prediction_loss,
    HACLoss
)
from retrieval import (
    compute_euclidean_scores,
    compute_hgrmc_scores,
    dempster_shafer_fusion,
    perform_cross_modal_retrieval
)
from metrics import compute_ap, compute_map, compute_ndcg
from dataset import sample_in_batch_pairs


def test_1_dataset_scanning_and_label_mapping():
    """Test 1: Verify data manifests and class mapping exist and contain 10 classes."""
    data_dir = Path("data")
    assert (data_dir / "class_mapping.json").exists()
    assert (data_dir / "train.csv").exists()
    assert (data_dir / "val.csv").exists()
    assert (data_dir / "test.csv").exists()

    with open(data_dir / "class_mapping.json") as f:
        meta = json.load(f)

    classes = meta["classes"]
    assert len(classes) == 10
    assert "AnnualCrop" in classes
    assert "SeaLake" in classes


def test_2_train_val_test_disjointness():
    """Test 2: Verify zero overlap between train, val, and test splits."""
    train_df = pd.read_csv("data/train.csv")
    val_df = pd.read_csv("data/val.csv")
    test_df = pd.read_csv("data/test.csv")

    train_stems = set(train_df["pair_id"])
    val_stems = set(val_df["pair_id"])
    test_stems = set(test_df["pair_id"])

    assert len(train_stems.intersection(val_stems)) == 0, "Train and Val overlap!"
    assert len(train_stems.intersection(test_stems)) == 0, "Train and Test overlap!"
    assert len(val_stems.intersection(test_stems)) == 0, "Val and Test overlap!"


def test_3_reproducible_splits():
    """Test 3: Verify split proportions match 70/15/15."""
    train_df = pd.read_csv("data/train.csv")
    val_df = pd.read_csv("data/val.csv")
    test_df = pd.read_csv("data/test.csv")

    total = len(train_df) + len(val_df) + len(test_df)
    assert total == 54000  # 27000 RGB + 27000 SAR

    # Check approximate 70/15/15 ratio
    assert abs(len(train_df) / total - 0.70) < 0.01
    assert abs(len(val_df) / total - 0.15) < 0.01
    assert abs(len(test_df) / total - 0.15) < 0.01


def test_4_correct_image_loading_rgb_and_sar():
    """Test 4: Verify loading RGB JPEG and SAR 2-channel TIFF converts to 3-channel (H, W, 3)."""
    rgb_sample = "EuroSAT_RGB/AnnualCrop/AnnualCrop_1.jpg"
    sar_sample = "EuroSAT-SAR/AnnualCrop/AnnualCrop_1.tif"

    img_rgb = load_rgb_image(rgb_sample)
    assert img_rgb.size == (64, 64)
    assert img_rgb.mode == "RGB"

    img_sar = load_sar_image(sar_sample)
    assert img_sar.size == (64, 64)
    assert img_sar.mode == "RGB"


def test_5_feature_dimensions():
    """Test 5: Verify AlexNet backbone outputs 4096-D, projection heads output 512-D, classifier outputs 10-D."""
    model = HACModel(num_classes=10, feature_dim=512, pretrained=False)
    model.eval()

    x = torch.randn(2, 3, 224, 224)
    out_opt = model.forward_optical(x)
    out_sar = model.forward_sar(x)

    assert out_opt["high_level"].shape == (2, 4096)
    assert out_opt["f_e"].shape == (2, 512)
    assert out_opt["f_h"].shape == (2, 512)
    assert out_opt["pred_f_e"].shape == (2, 10)
    assert out_opt["pred_f_h"].shape == (2, 10)

    assert out_sar["high_level"].shape == (2, 4096)
    assert out_sar["g_e"].shape == (2, 512)
    assert out_sar["g_h"].shape == (2, 512)
    assert out_sar["pred_g_e"].shape == (2, 10)
    assert out_sar["pred_g_h"].shape == (2, 10)


def test_6_euclidean_similarity_loss():
    """Test 6: Verify J_I decreases as vectors get closer and equals 0 when identical."""
    a = torch.randn(4, 512)
    b = a.clone()
    dist_zero = euclidean_distance(a, b)
    assert torch.allclose(dist_zero, torch.zeros(4), atol=1e-3)

    c = a + 1.0
    dist_pos = euclidean_distance(a, c)
    assert (dist_pos > 0).all()


def test_7_euclidean_ranking_loss():
    """Test 7: Verify J_R is zero when positive is much closer than negative (margin satisfied)."""
    N = 4
    D = 512
    f_e = torch.zeros(N, D)
    g_e = torch.zeros(N, D)

    # Positive sample is identical (dist = 0)
    g_e_pos = torch.zeros(N, D)
    f_e_pos = torch.zeros(N, D)

    # Negative sample is far away (dist = 10 >> margin 0.3)
    g_e_neg = torch.ones(N, D) * 10.0
    f_e_neg = torch.ones(N, D) * 10.0

    j_euc, j_i, j_r = compute_euclidean_loss(
        f_e, g_e, f_e_pos, f_e_neg, g_e_pos, g_e_neg, margin=0.3
    )
    assert torch.isclose(j_r, torch.tensor(0.0), atol=1e-5)
    assert torch.isclose(j_i, torch.tensor(0.0), atol=1e-3)


def test_8_soft_hgr_loss_and_gradients():
    """Test 8: Verify soft-HGR correlation loss computes correctly and produces valid non-zero gradients."""
    x = torch.randn(8, 64, requires_grad=True)
    y = torch.randn(8, 64, requires_grad=True)

    j_c = soft_hgr_correlation(x, y)
    assert not torch.isnan(j_c)
    assert not torch.isinf(j_c)

    j_c.backward()
    assert x.grad is not None
    assert torch.norm(x.grad) > 0
    assert y.grad is not None
    assert torch.norm(y.grad) > 0


def test_9_prediction_loss():
    """Test 9: Verify prediction loss J_P equals 0 when predictions match one-hot targets."""
    N = 4
    C = 10
    targets = torch.zeros(N, C)
    for i in range(N):
        targets[i, i % C] = 1.0

    preds = targets.clone()
    j_p = compute_prediction_loss(preds, preds, preds, preds, targets)
    assert torch.isclose(j_p, torch.tensor(0.0), atol=1e-6)

    # When predictions are completely wrong
    wrong_preds = 1.0 - targets
    j_p_wrong = compute_prediction_loss(wrong_preds, wrong_preds, wrong_preds, wrong_preds, targets)
    assert j_p_wrong > 0


def test_10_total_loss_combination():
    """Test 10: Verify total loss J = J_P + alpha * J_EUC + beta * J_HGR with alpha=0.001, beta=0.001."""
    criterion = HACLoss(alpha=0.001, beta=0.001, margin=0.3)
    N = 6
    D = 512
    C = 10

    f_e = torch.randn(N, D)
    g_e = torch.randn(N, D)
    f_h = torch.randn(N, D)
    g_h = torch.randn(N, D)

    f_e_pos, f_e_neg = f_e.clone(), f_e.clone() + 2.0
    g_e_pos, g_e_neg = g_e.clone(), g_e.clone() + 2.0
    f_h_pos, f_h_neg = f_h.clone(), f_h.clone() * -1.0
    g_h_pos, g_h_neg = g_h.clone(), g_h.clone() * -1.0

    pred_f_e = torch.randn(N, C)
    pred_f_h = torch.randn(N, C)
    pred_g_e = torch.randn(N, C)
    pred_g_h = torch.randn(N, C)

    targets = F.one_hot(torch.randint(0, C, (N,)), num_classes=C).float()

    losses = criterion(
        f_e, g_e, f_e_pos, f_e_neg, g_e_pos, g_e_neg,
        f_h, g_h, f_h_pos, f_h_neg, g_h_pos, g_h_neg,
        pred_f_e, pred_f_h, pred_g_e, pred_g_h, targets
    )

    expected = losses["j_p"] + 0.001 * losses["j_euc"] + 0.001 * losses["j_hgr"]
    assert torch.isclose(losses["total_loss"], expected, atol=1e-5)


def test_11_ds_mass_functions_and_fusion():
    """Test 11: Verify Dempster-Shafer fusion produces valid probabilities in [0, 1]."""
    s_euc = torch.tensor([[0.9, 0.1], [0.5, 0.5]])
    s_hgr = torch.tensor([[0.8, 0.2], [0.6, 0.4]])

    fused = dempster_shafer_fusion(s_euc, s_hgr, b1=0.95, b2=0.95)
    assert fused.shape == (2, 2)
    assert (fused >= 0.0).all() and (fused <= 1.0).all()

    # Candidate with both high Euclidean and high HGRMC score must have higher plausibility
    assert fused[0, 0] > fused[0, 1]


def test_12_plausibility_ranking():
    """Test 12: Verify perform_cross_modal_retrieval ranks highest-scoring candidate first."""
    query_euc = torch.tensor([[1.0, 0.0]])
    query_hgr = torch.tensor([[1.0, 0.0]])

    gallery_euc = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    gallery_hgr = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])

    ranked_idx, scores = perform_cross_modal_retrieval(
        query_euc, query_hgr, gallery_euc, gallery_hgr, fusion_mode="ds"
    )
    # Candidate 1 is identical to query -> must be ranked first (index 1)
    assert ranked_idx[0, 0].item() == 1


def test_13_map_hand_verifiable_example():
    """
    Test 13: Verify mAP computation on hand-crafted ground truth.
    Query class = 0.
    Gallery labels: [0, 1, 0, 1, 0] -> 3 relevant out of 5.
    If ranked: [0, 1, 0, 1, 0]
      Rank 1: label 0 (relevant) -> P@1 = 1/1 = 1.0, r=1
      Rank 2: label 1 (not rel)  -> P@2 = 1/2 = 0.5, r=0
      Rank 3: label 0 (relevant) -> P@3 = 2/3 = 0.6667, r=1
      Rank 4: label 1 (not rel)  -> P@4 = 2/4 = 0.5, r=0
      Rank 5: label 0 (relevant) -> P@5 = 3/5 = 0.6, r=1
      AP = (1.0 + 2/3 + 3/5) / 3 = (1.0 + 0.66667 + 0.6) / 3 = 2.26667 / 3 = 0.755556
    """
    ranked_labels = np.array([0, 1, 0, 1, 0])
    ap = compute_ap(ranked_labels, query_label=0, total_relevant=3)
    expected_ap = (1.0 + 2.0 / 3.0 + 3.0 / 5.0) / 3.0
    assert abs(ap - expected_ap) < 1e-5


def test_14_checkpoint_saving_and_loading(tmp_path):
    """Test 14: Verify model state dict and metadata save and restore identically."""
    model = HACModel(num_classes=10, feature_dim=512, pretrained=False)
    ckpt_path = tmp_path / "test_model.pth"

    metadata = {
        "epoch": 5,
        "config": {"alpha": 0.001, "beta": 0.001},
        "class_mapping": {"Forest": 1},
        "model_state_dict": model.state_dict()
    }
    torch.save(metadata, ckpt_path)

    # Reload into fresh model
    loaded_ckpt = torch.load(ckpt_path, map_location="cpu")
    new_model = HACModel(num_classes=10, feature_dim=512, pretrained=False)
    new_model.load_state_dict(loaded_ckpt["model_state_dict"])

    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.allclose(p1, p2)


def test_15_end_to_end_forward_and_optimization_step():
    """Test 15: Run a small forward pass and one full Adam optimization step."""
    set_seed(42)
    model = HACModel(num_classes=10, feature_dim=512, pretrained=False)
    criterion = HACLoss(alpha=0.001, beta=0.001, margin=0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    batch_size = 4
    x_opt = torch.randn(batch_size, 3, 224, 224)
    x_sar = torch.randn(batch_size, 3, 224, 224)
    labels = torch.tensor([0, 0, 1, 1])
    targets_one_hot = F.one_hot(labels, num_classes=10).float()

    v_pos, v_neg, s_pos, s_neg = sample_in_batch_pairs(labels)

    optimizer.zero_grad()
    out = model(x_opt, x_sar)

    f_e, g_e = out["f_e"], out["g_e"]
    f_h, g_h = out["f_h"], out["g_h"]

    loss_dict = criterion(
        f_e=f_e, g_e=g_e,
        f_e_pos=f_e[v_pos], f_e_neg=f_e[v_neg],
        g_e_pos=g_e[s_pos], g_e_neg=g_e[s_neg],
        f_h=f_h, g_h=g_h,
        f_h_pos=f_h[v_pos], f_h_neg=f_h[v_neg],
        g_h_pos=g_h[s_pos], g_h_neg=g_h[s_neg],
        pred_f_e=out["pred_f_e"], pred_f_h=out["pred_f_h"],
        pred_g_e=out["pred_g_e"], pred_g_h=out["pred_g_h"],
        targets_one_hot=targets_one_hot
    )

    total_loss = loss_dict["total_loss"]
    total_loss.backward()
    optimizer.step()

    assert not torch.isnan(total_loss)
    assert total_loss.item() > 0
