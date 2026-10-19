#!/usr/bin/env python3
"""
Evaluation metrics for Cross-Modal Remote Sensing Image Retrieval:
- Mean Average Precision (mAP) according to Equation (16)
- Normalized Discounted Cumulative Gain (NDCG) according to Equations (17)-(18)
- Precision-Recall curve computation
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union


def compute_ap(
    retrieved_labels: np.ndarray,
    query_label: int,
    total_relevant: Optional[int] = None
) -> float:
    """
    Compute Average Precision (AP) for a single query (Eq. 16).
    AP = (1 / R_q) * sum_{k=1}^G P(k) * r(k)
    where:
      G is the total number of gallery candidates evaluated,
      R_q is the total number of relevant candidates in the gallery,
      P(k) is the precision at rank k,
      r(k) is binary relevance indicator (1 if retrieved_label == query_label, else 0).

    Args:
        retrieved_labels: 1D array of class labels for ranked gallery candidates
        query_label: True class label of the query
        total_relevant: Total number of relevant samples in gallery (if None, len of matching labels)
    Returns:
        AP score in [0, 1]
    """
    relevance = (retrieved_labels == query_label).astype(np.float64)
    n_rel_in_retrieved = np.sum(relevance)

    if total_relevant is None:
        total_relevant = n_rel_in_retrieved

    if total_relevant == 0 or n_rel_in_retrieved == 0:
        return 0.0

    # Cumulative sum of relevant items up to each rank k
    cumsum_rel = np.cumsum(relevance)
    ranks = np.arange(1, len(relevance) + 1, dtype=np.float64)
    precision_at_k = cumsum_rel / ranks

    # AP is sum(P(k) * r(k)) / R_q
    ap = np.sum(precision_at_k * relevance) / float(total_relevant)
    return float(ap)


def compute_map(
    ranked_gallery_indices: Union[torch.Tensor, np.ndarray],
    query_labels: Union[torch.Tensor, np.ndarray],
    gallery_labels: Union[torch.Tensor, np.ndarray],
    top_k: Optional[int] = None
) -> Tuple[float, np.ndarray]:
    """
    Compute Mean Average Precision (mAP) across all queries.

    Args:
        ranked_gallery_indices: (Q, G) indices of gallery items sorted by descending score
        query_labels: (Q,) ground truth labels of queries
        gallery_labels: (G,) ground truth labels of gallery candidates
        top_k: If set, evaluates AP over top-K retrieved candidates. Default None (full gallery).
    Returns:
        (mAP, per_query_ap)
    """
    if isinstance(ranked_gallery_indices, torch.Tensor):
        ranked_gallery_indices = ranked_gallery_indices.cpu().numpy()
    if isinstance(query_labels, torch.Tensor):
        query_labels = query_labels.cpu().numpy()
    if isinstance(gallery_labels, torch.Tensor):
        gallery_labels = gallery_labels.cpu().numpy()

    Q, G = ranked_gallery_indices.shape
    k = G if top_k is None else min(top_k, G)

    # Precompute total relevant count for each class in gallery
    unique_classes, counts = np.unique(gallery_labels, return_counts=True)
    class_rel_counts = dict(zip(unique_classes, counts))

    ap_list = []
    for q_idx in range(Q):
        q_label = query_labels[q_idx]
        top_indices = ranked_gallery_indices[q_idx, :k]
        top_labels = gallery_labels[top_indices]
        r_q = class_rel_counts.get(q_label, 0)

        ap = compute_ap(top_labels, q_label, total_relevant=r_q)
        ap_list.append(ap)

    ap_array = np.array(ap_list)
    map_score = float(np.mean(ap_array))
    return map_score, ap_array


def compute_ndcg(
    ranked_gallery_indices: Union[torch.Tensor, np.ndarray],
    query_labels: Union[torch.Tensor, np.ndarray],
    gallery_labels: Union[torch.Tensor, np.ndarray],
    top_k: Optional[int] = None
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG) across queries (Eqs. 17-18).
    DCG_k = sum_{i=1}^k r(i) / log2(i + 1)
    NDCG_k = DCG_k / IDCG_k
    """
    if isinstance(ranked_gallery_indices, torch.Tensor):
        ranked_gallery_indices = ranked_gallery_indices.cpu().numpy()
    if isinstance(query_labels, torch.Tensor):
        query_labels = query_labels.cpu().numpy()
    if isinstance(gallery_labels, torch.Tensor):
        gallery_labels = gallery_labels.cpu().numpy()

    Q, G = ranked_gallery_indices.shape
    k = G if top_k is None else min(top_k, G)

    unique_classes, counts = np.unique(gallery_labels, return_counts=True)
    class_rel_counts = dict(zip(unique_classes, counts))

    # Discounts: log2(i + 1) for i = 1..k -> log2(2..k+1)
    ranks = np.arange(1, k + 1, dtype=np.float64)
    discounts = 1.0 / np.log2(ranks + 1.0)

    ndcg_list = []
    for q_idx in range(Q):
        q_label = query_labels[q_idx]
        top_indices = ranked_gallery_indices[q_idx, :k]
        top_labels = gallery_labels[top_indices]
        rel = (top_labels == q_label).astype(np.float64)

        dcg = np.sum(rel * discounts)

        # Ideal DCG: all R_q relevant items placed at the front
        total_rel = class_rel_counts.get(q_label, 0)
        n_ideal = min(k, total_rel)
        if n_ideal == 0:
            continue
        idcg = np.sum(discounts[:n_ideal])

        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_list.append(ndcg)

    return float(np.mean(ndcg_list)) if ndcg_list else 0.0


def compute_precision_recall_curve(
    ranked_gallery_indices: Union[torch.Tensor, np.ndarray],
    query_labels: Union[torch.Tensor, np.ndarray],
    gallery_labels: Union[torch.Tensor, np.ndarray],
    num_points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute interpolated Precision-Recall curve over queries.
    """
    if isinstance(ranked_gallery_indices, torch.Tensor):
        ranked_gallery_indices = ranked_gallery_indices.cpu().numpy()
    if isinstance(query_labels, torch.Tensor):
        query_labels = query_labels.cpu().numpy()
    if isinstance(gallery_labels, torch.Tensor):
        gallery_labels = gallery_labels.cpu().numpy()

    Q, G = ranked_gallery_indices.shape
    recall_levels = np.linspace(0.0, 1.0, num_points)
    precision_levels = np.zeros(num_points)

    valid_queries = 0
    for q_idx in range(Q):
        q_label = query_labels[q_idx]
        top_indices = ranked_gallery_indices[q_idx]
        top_labels = gallery_labels[top_indices]
        rel = (top_labels == q_label).astype(np.float64)
        total_rel = np.sum(gallery_labels == q_label)
        if total_rel == 0:
            continue

        cumsum_rel = np.cumsum(rel)
        ranks = np.arange(1, G + 1, dtype=np.float64)
        prec = cumsum_rel / ranks
        rec = cumsum_rel / float(total_rel)

        # Interpolated precision
        interp_p = np.zeros(num_points)
        for i, r_thresh in enumerate(recall_levels):
            idx = np.where(rec >= r_thresh)[0]
            if len(idx) > 0:
                interp_p[i] = np.max(prec[idx])
            else:
                interp_p[i] = 0.0

        precision_levels += interp_p
        valid_queries += 1

    if valid_queries > 0:
        precision_levels /= valid_queries

    return recall_levels, precision_levels
