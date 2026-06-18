from typing import Dict, List

import numpy as np
from loguru import logger


def compute_recall_at_k(
    query_labels: np.ndarray,
    retrieved_labels: np.ndarray,
    k_values: List[int] = None,
) -> Dict[str, float]:
    if k_values is None:
        k_values = [1, 5, 10]

    num_queries = len(query_labels)
    if num_queries == 0:
        logger.warning("Empty query set — returning zero recall")
        return {f"recall_at_{k}": 0.0 for k in k_values}

    results = {}

    for k in k_values:
        effective_k = min(k, retrieved_labels.shape[1])

        top_k_retrieved = retrieved_labels[:, :effective_k]

        matches = np.any(
            top_k_retrieved == query_labels[:, np.newaxis],
            axis=1,
        )

        recall = float(np.mean(matches))
        results[f"recall_at_{k}"] = round(recall, 4)

        logger.debug("Recall@{}: {:.4f} ({}/{} queries)", k, recall, matches.sum(), num_queries)

    return results


def compute_pairwise_distances(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
) -> np.ndarray:
    query_sq = np.sum(query_embeddings ** 2, axis=1, keepdims=True)

    gallery_sq = np.sum(gallery_embeddings ** 2, axis=1, keepdims=True).T

    cross_term = -2.0 * query_embeddings @ gallery_embeddings.T

    distances = query_sq + gallery_sq + cross_term

    distances = np.maximum(distances, 0.0)
    return np.sqrt(distances)


def evaluate_retrieval(
    query_embeddings: np.ndarray,
    query_labels: np.ndarray,
    gallery_embeddings: np.ndarray,
    gallery_labels: np.ndarray,
    k_values: List[int] = None,
) -> Dict[str, float]:
    if k_values is None:
        k_values = [1, 5, 10]

    logger.info(
        "Evaluating retrieval | queries={} | gallery={} | k_values={}",
        len(query_embeddings), len(gallery_embeddings), k_values,
    )

    if len(gallery_embeddings) == 0:
        logger.warning("Empty gallery set — returning zero recall")
        return {f"recall_at_{k}": 0.0 for k in k_values}

    distances = compute_pairwise_distances(query_embeddings, gallery_embeddings)

    max_k = max(k_values)
    sorted_indices = np.argsort(distances, axis=1)[:, :max_k]

    retrieved_labels = gallery_labels[sorted_indices]

    return compute_recall_at_k(query_labels, retrieved_labels, k_values)
