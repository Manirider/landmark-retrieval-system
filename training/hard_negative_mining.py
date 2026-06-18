import torch
from loguru import logger


def compute_pairwise_distance_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    dot_product = torch.mm(embeddings, embeddings.t())

    square_norm = torch.diag(dot_product)

    distances = square_norm.unsqueeze(0) + square_norm.unsqueeze(1) - 2.0 * dot_product

    distances = torch.clamp(distances, min=0.0)

    distances = torch.sqrt(distances + 1e-8)

    return distances


def mine_hard_triplets(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.3,
) -> tuple:
    batch_size = embeddings.size(0)
    device = embeddings.device

    distance_matrix = compute_pairwise_distance_matrix(embeddings)

    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)

    identity_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
    positive_mask = labels_equal & ~identity_mask

    negative_mask = ~labels_equal

    positive_distances = distance_matrix.clone()
    positive_distances[~positive_mask] = -1.0
    hardest_positive_dist, hardest_positive_idx = positive_distances.max(dim=1)

    negative_distances = distance_matrix.clone()
    negative_distances[~negative_mask] = float("inf")
    hardest_negative_dist, hardest_negative_idx = negative_distances.min(dim=1)

    has_positive = positive_mask.any(dim=1)
    has_negative = negative_mask.any(dim=1)
    valid_anchors = has_positive & has_negative

    if valid_anchors.sum() == 0:
        logger.warning("No valid triplets found in batch — check label distribution")
        return (
            torch.empty(0, embeddings.size(1), device=device),
            torch.empty(0, embeddings.size(1), device=device),
            torch.empty(0, embeddings.size(1), device=device),
        )

    valid_indices = valid_anchors.nonzero(as_tuple=True)[0]
    anchors = embeddings[valid_indices]
    hardest_positives = embeddings[hardest_positive_idx[valid_indices]]
    hardest_negatives = embeddings[hardest_negative_idx[valid_indices]]

    num_active = (
        hardest_positive_dist[valid_indices] - hardest_negative_dist[valid_indices] + margin > 0
    ).sum().item()

    logger.debug(
        "Hard mining | valid_triplets={}/{} | active_triplets={} | "
        "avg_pos_dist={:.4f} | avg_neg_dist={:.4f}",
        len(valid_indices), batch_size, num_active,
        hardest_positive_dist[valid_indices].mean().item(),
        hardest_negative_dist[valid_indices].mean().item(),
    )

    return anchors, hardest_positives, hardest_negatives
