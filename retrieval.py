#!/usr/bin/env python3
"""
Retrieval module for HAC (Robust Cross-Modal Remote Sensing Image Retrieval via Maximal Correlation Augmentation).
Implements Section IV-D and Equations (12)-(15):
- Cosine similarity for Euclidean feature space + min-max normalization -> s_EUC
- HGRMC correlation score + min-max normalization -> s_HGR
- Dempster-Shafer (DS) Evidence Theory fusion with confidence parameters b1, b2
- Plausibility calculation PLB(A1) = m_c(A1) + m_c(A3)
- Gallery ranking and query retrieval
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Union, Dict


def compute_euclidean_scores(
    query_feats: torch.Tensor,
    gallery_feats: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute normalized Euclidean retrieval scores s_EUC (Section IV-D).
    Cosine similarity between query and gallery, then min-max normalized to [0, 1].

    Args:
        query_feats: (Q, D) Euclidean features
        gallery_feats: (G, D) Euclidean features
    Returns:
        s_euc: (Q, G) normalized scores in [0, 1]
    """
    # L2 normalize features to compute cosine similarity
    q_norm = F.normalize(query_feats, p=2, dim=-1)
    g_norm = F.normalize(gallery_feats, p=2, dim=-1)

    # Cosine similarities: (Q, G)
    cos_sim = torch.matmul(q_norm, g_norm.t())

    # Min-max normalization per query across gallery candidates
    sim_min = cos_sim.min(dim=-1, keepdim=True)[0]
    sim_max = cos_sim.max(dim=-1, keepdim=True)[0]
    diff = sim_max - sim_min

    # Safe normalization: handle case where max == min
    s_euc = torch.where(
        diff > eps,
        (cos_sim - sim_min) / (diff + eps),
        torch.full_like(cos_sim, 0.5)
    )
    return torch.clamp(s_euc, 0.0, 1.0)


def compute_hgrmc_scores(
    query_feats: torch.Tensor,
    gallery_feats: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute normalized HGRMC correlation scores s_HGR (Section IV-D).
    Inner product of learned HGRMC features, then min-max normalized to [0, 1].

    Args:
        query_feats: (Q, D) HGRMC features
        gallery_feats: (G, D) HGRMC features
    Returns:
        s_hgr: (Q, G) normalized scores in [0, 1]
    """
    # Correlation inner product: (Q, G)
    corr = torch.matmul(query_feats, gallery_feats.t())

    # Min-max normalization per query across gallery candidates
    corr_min = corr.min(dim=-1, keepdim=True)[0]
    corr_max = corr.max(dim=-1, keepdim=True)[0]
    diff = corr_max - corr_min

    s_hgr = torch.where(
        diff > eps,
        (corr - corr_min) / (diff + eps),
        torch.full_like(corr, 0.5)
    )
    return torch.clamp(s_hgr, 0.0, 1.0)


def dempster_shafer_fusion(
    s_euc: torch.Tensor,
    s_hgr: torch.Tensor,
    b1: float = 0.95,
    b2: float = 0.95,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Fuse Euclidean and HGRMC scores using Dempster-Shafer (DS) Evidence Theory (Eqs. 12-15).

    Hypotheses:
      A1: The score is reliable
      A2: The score is not reliable
      A3: The score is reliable or not (uncertainty)
      A4: Empty set (mass = 0)

    Mass functions:
      m1(A1) = b1 * s_EUC,  m1(A2) = b1 * (1 - s_EUC),  m1(A3) = 1 - b1
      m2(A1) = b2 * s_HGR,  m2(A2) = b2 * (1 - s_HGR),  m2(A3) = 1 - b2

    Conflict coefficient:
      Z = m1(A1)*m2(A2) + m1(A2)*m2(A1)

    Combined mass:
      m_c(A1) = (m1(A1)m2(A1) + m1(A1)m2(A3) + m1(A3)m2(A1)) / (1 - Z)
      m_c(A3) = (m1(A3)m2(A3)) / (1 - Z)

    Plausibility:
      PLB(A1) = m_c(A1) + m_c(A3)

    Args:
        s_euc: (Q, G) normalized Euclidean scores in [0, 1]
        s_hgr: (Q, G) normalized HGRMC scores in [0, 1]
        b1: Confidence parameter for Euclidean score (default: 0.95)
        b2: Confidence parameter for HGRMC score (default: 0.95)
        eps: Small constant for numerical stability
    Returns:
        plb_a1: (Q, G) plausibility scores
    """
    # Mass functions for Euclidean criterion m1
    m1_a1 = b1 * s_euc
    m1_a2 = b1 * (1.0 - s_euc)
    m1_a3 = 1.0 - b1

    # Mass functions for HGRMC criterion m2
    m2_a1 = b2 * s_hgr
    m2_a2 = b2 * (1.0 - s_hgr)
    m2_a3 = 1.0 - b2

    # Conflict coefficient Z (Eq. 14)
    Z = m1_a1 * m2_a2 + m1_a2 * m2_a1

    # Safe denominator: 1 - Z
    denom = torch.clamp(1.0 - Z, min=eps)

    # Combined masses (Eq. 14)
    num_a1 = m1_a1 * m2_a1 + m1_a1 * m2_a3 + m1_a3 * m2_a1
    num_a3 = m1_a3 * m2_a3

    mc_a1 = num_a1 / denom
    mc_a3 = num_a3 / denom

    # Plausibility function PLB(A1) = mc(A1) + mc(A3) (Eq. 15)
    plb_a1 = mc_a1 + mc_a3
    return plb_a1


def perform_cross_modal_retrieval(
    query_euc: torch.Tensor,
    query_hgr: torch.Tensor,
    gallery_euc: torch.Tensor,
    gallery_hgr: torch.Tensor,
    fusion_mode: str = "ds",
    b1: float = 0.95,
    b2: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform cross-modal retrieval given query and gallery features.

    Args:
        query_euc: (Q, D) query Euclidean features
        query_hgr: (Q, D) query HGRMC features
        gallery_euc: (G, D) gallery Euclidean features
        gallery_hgr: (G, D) gallery HGRMC features
        fusion_mode: One of 'ds' (Dempster-Shafer, default), 'sum', 'euclidean_only', 'hgrmc_only'
        b1: DS confidence parameter for Euclidean score (0.95)
        b2: DS confidence parameter for HGRMC score (0.95)
    Returns:
        ranked_indices: (Q, G) gallery indices sorted in descending order of retrieval score
        scores: (Q, G) sorted retrieval scores
    """
    s_euc = compute_euclidean_scores(query_euc, gallery_euc)
    s_hgr = compute_hgrmc_scores(query_hgr, gallery_hgr)

    if fusion_mode == "ds":
        final_scores = dempster_shafer_fusion(s_euc, s_hgr, b1=b1, b2=b2)
    elif fusion_mode == "sum":
        final_scores = s_euc + s_hgr
    elif fusion_mode == "euclidean_only":
        final_scores = s_euc
    elif fusion_mode == "hgrmc_only":
        final_scores = s_hgr
    else:
        raise ValueError(f"Unknown fusion mode: {fusion_mode}")

    # Sort descending
    ranked_scores, ranked_indices = torch.sort(final_scores, dim=-1, descending=True)
    return ranked_indices, ranked_scores
