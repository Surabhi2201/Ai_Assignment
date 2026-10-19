#!/usr/bin/env python3
"""
Loss functions for HAC (Robust Cross-Modal Remote Sensing Image Retrieval via Maximal Correlation Augmentation).
Implements Equations (3) through (11) from Wang et al., IEEE TGRS 2024:
- Euclidean similarity loss (J_I) and ranking loss (J_R) -> J_EUC (Eqs. 3-5)
- Soft-HGR correlation loss (J_C) and correlation ranking loss (J_CR) -> J_HGR (Eqs. 6-9)
- Semantic prediction MSE loss -> J_P (Eq. 10)
- Total loss: J = J_P + alpha * J_EUC + beta * J_HGR (Eq. 11)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


def euclidean_distance(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Numerically stable Euclidean distance between paired vectors x and y.
    Args:
        x: Tensor of shape (N, D)
        y: Tensor of shape (N, D)
        eps: Small constant to avoid zero gradients at d=0
    Returns:
        Tensor of shape (N,) containing ||x_i - y_i||_2
    """
    diff_sq = torch.sum((x - y) ** 2, dim=-1)
    return torch.sqrt(diff_sq + eps)


def compute_euclidean_loss(
    f_e: torch.Tensor,
    g_e: torch.Tensor,
    f_e_pos: torch.Tensor,
    f_e_neg: torch.Tensor,
    g_e_pos: torch.Tensor,
    g_e_neg: torch.Tensor,
    margin: float = 0.3,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Euclidean similarity loss J_I and ranking loss J_R (Eqs. 3-5).
    Args:
        f_e: Optical anchor Euclidean features (N, D)
        g_e: SAR anchor Euclidean features (N, D)
        f_e_pos: Positive optical features for g_e (N, D)
        f_e_neg: Negative optical features for g_e (N, D)
        g_e_pos: Positive SAR features for f_e (N, D)
        g_e_neg: Negative SAR features for f_e (N, D)
        margin: Margin parameter m (default: 0.3)
    Returns:
        (J_EUC, J_I, J_R)
    """
    N = f_e.size(0)

    # Similarity loss: J_I = (1/N) * sum d(f^E(v_i), g^E(s_i))
    d_anchor = euclidean_distance(f_e, g_e, eps=eps)
    j_i = torch.mean(d_anchor)

    # Ranking loss (two-directional triplet margin ranking):
    # max(d(f^E(v_i), g^E(s_i^+)) - d(f^E(v_i), g^E(s_i^-)) + m, 0)
    d_v_pos_s = euclidean_distance(f_e, g_e_pos, eps=eps)
    d_v_neg_s = euclidean_distance(f_e, g_e_neg, eps=eps)
    loss_v_anchor = F.relu(d_v_pos_s - d_v_neg_s + margin)

    # max(d(f^E(v_i^+), g^E(s_i)) - d(f^E(v_i^-), g^E(s_i)) + m, 0)
    d_pos_v_s = euclidean_distance(f_e_pos, g_e, eps=eps)
    d_neg_v_s = euclidean_distance(f_e_neg, g_e, eps=eps)
    loss_s_anchor = F.relu(d_pos_v_s - d_neg_v_s + margin)

    j_r = torch.mean(loss_v_anchor + loss_s_anchor)
    j_euc = j_i + j_r
    return j_euc, j_i, j_r


def soft_hgr_correlation(
    x: torch.Tensor,
    y: torch.Tensor,
    cov_unbiased: bool = False
) -> torch.Tensor:
    """
    Compute soft-HGR correlation loss J_C between two feature representations (Eq. 7).
    Minimizing J_C maximizes the cross-modal correlation while enforcing decorrelation.
    J_C(X, Y) = 0.5 * tr(cov(X_tilde) * cov(Y_tilde)) - (1/N) * sum(x_tilde_i^T * y_tilde_i)

    Args:
        x: Tensor of shape (N, D)
        y: Tensor of shape (N, D)
        cov_unbiased: Whether to use N-1 (unbiased) or N in covariance
    Returns:
        Scalar tensor J_C
    """
    N, D = x.size()
    if N <= 1:
        # Avoid division by zero on degenerate single-sample batch
        return torch.tensor(0.0, device=x.device, requires_grad=True)

    # Center representations (Eq. 8)
    x_mean = torch.mean(x, dim=0, keepdim=True)
    y_mean = torch.mean(y, dim=0, keepdim=True)
    x_tilde = x - x_mean
    y_tilde = y - y_mean

    # Inner product term: (1/N) * sum(x_tilde_i^T * y_tilde_i)
    # This is consistent with correlation maximization
    inner_prod = torch.mean(torch.sum(x_tilde * y_tilde, dim=-1))

    # Decorrelation regularization term: 0.5 * tr(cov(x_tilde) * cov(y_tilde))
    # cov(x) = (1/N) * x_tilde^T * x_tilde
    # tr(cov(x) * cov(y)) = (1/N^2) * tr(x_tilde^T * x_tilde * y_tilde^T * y_tilde)
    #                     = (1/N^2) * ||x_tilde * y_tilde^T||_F^2
    denom = (N - 1) if cov_unbiased else N
    cov_x = torch.matmul(x_tilde.t(), x_tilde) / denom  # (D, D)
    cov_y = torch.matmul(y_tilde.t(), y_tilde) / denom  # (D, D)
    tr_cov = 0.5 * torch.sum(cov_x * cov_y)  # Frobenius inner product = tr(cov_x * cov_y)

    j_c = tr_cov - inner_prod
    return j_c


def compute_hgrmc_loss(
    f_h: torch.Tensor,
    g_h: torch.Tensor,
    f_h_pos: torch.Tensor,
    f_h_neg: torch.Tensor,
    g_h_pos: torch.Tensor,
    g_h_neg: torch.Tensor,
    cov_unbiased: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute HGRMC correlation loss J_C and correlation ranking loss J_CR (Eqs. 6-9).
    J_HGR = J_C + J_CR
    J_CR = J_C(V, S^+) + J_C(V^+, S) - J_C(V, S^-) - J_C(V^-, S)
    """
    # Primary correlation loss on anchor pairs
    j_c = soft_hgr_correlation(f_h, g_h, cov_unbiased=cov_unbiased)

    # Correlation ranking loss (Eq. 9)
    # Positive pairs: maximize correlation (minimize J_C)
    j_c_v_spos = soft_hgr_correlation(f_h, g_h_pos, cov_unbiased=cov_unbiased)
    j_c_vpos_s = soft_hgr_correlation(f_h_pos, g_h, cov_unbiased=cov_unbiased)

    # Negative pairs: minimize correlation (maximize J_C, so subtract in loss)
    j_c_v_sneg = soft_hgr_correlation(f_h, g_h_neg, cov_unbiased=cov_unbiased)
    j_c_vneg_s = soft_hgr_correlation(f_h_neg, g_h, cov_unbiased=cov_unbiased)

    j_cr = (j_c_v_spos + j_c_vpos_s) - (j_c_v_sneg + j_c_vneg_s)
    j_hgr = j_c + j_cr

    return j_hgr, j_c, j_cr


def compute_prediction_loss(
    pred_f_e: torch.Tensor,
    pred_f_h: torch.Tensor,
    pred_g_e: torch.Tensor,
    pred_g_h: torch.Tensor,
    targets_one_hot: torch.Tensor
) -> torch.Tensor:
    """
    Compute semantic prediction MSE loss J_P over Euclidean and HGRMC features
    from both optical and SAR modalities (Eq. 10).
    J_P = (1/N) * sum [ ||P(f^E(v_i)) - y_i||^2 + ||P(f^H(v_i)) - y_i||^2
                        + ||P(g^E(s_i)) - y_i||^2 + ||P(g^H(s_i)) - y_i||^2 ]
    """
    loss_fe = torch.sum((pred_f_e - targets_one_hot) ** 2, dim=-1)
    loss_fh = torch.sum((pred_f_h - targets_one_hot) ** 2, dim=-1)
    loss_ge = torch.sum((pred_g_e - targets_one_hot) ** 2, dim=-1)
    loss_gh = torch.sum((pred_g_h - targets_one_hot) ** 2, dim=-1)

    j_p = torch.mean(loss_fe + loss_fh + loss_ge + loss_gh)
    return j_p


class HACLoss(nn.Module):
    """
    Unified HAC Loss module implementing Eq. (11):
    J = J_P + alpha * J_EUC + beta * J_HGR
    """
    def __init__(
        self,
        alpha: float = 0.001,
        beta: float = 0.001,
        margin: float = 0.3,
        cov_unbiased: bool = False
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.margin = margin
        self.cov_unbiased = cov_unbiased

    def forward(
        self,
        f_e: torch.Tensor,
        g_e: torch.Tensor,
        f_e_pos: torch.Tensor,
        f_e_neg: torch.Tensor,
        g_e_pos: torch.Tensor,
        g_e_neg: torch.Tensor,
        f_h: torch.Tensor,
        g_h: torch.Tensor,
        f_h_pos: torch.Tensor,
        f_h_neg: torch.Tensor,
        g_h_pos: torch.Tensor,
        g_h_neg: torch.Tensor,
        pred_f_e: torch.Tensor,
        pred_f_h: torch.Tensor,
        pred_g_e: torch.Tensor,
        pred_g_h: torch.Tensor,
        targets_one_hot: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # Euclidean loss
        j_euc, j_i, j_r = compute_euclidean_loss(
            f_e, g_e, f_e_pos, f_e_neg, g_e_pos, g_e_neg, margin=self.margin
        )

        # HGRMC loss
        j_hgr, j_c, j_cr = compute_hgrmc_loss(
            f_h, g_h, f_h_pos, f_h_neg, g_h_pos, g_h_neg, cov_unbiased=self.cov_unbiased
        )

        # Prediction loss
        j_p = compute_prediction_loss(
            pred_f_e, pred_f_h, pred_g_e, pred_g_h, targets_one_hot
        )

        # Total combined loss
        total_loss = j_p + self.alpha * j_euc + self.beta * j_hgr

        return {
            "total_loss": total_loss,
            "j_p": j_p,
            "j_euc": j_euc,
            "j_i": j_i,
            "j_r": j_r,
            "j_hgr": j_hgr,
            "j_c": j_c,
            "j_cr": j_cr
        }
