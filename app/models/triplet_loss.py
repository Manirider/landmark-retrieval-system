import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0.3) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        distance_positive = self._euclidean_distance(anchor, positive)
        distance_negative = self._euclidean_distance(anchor, negative)

        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()

    @staticmethod
    def _euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.sum((x - y) ** 2, dim=1) + 1e-8)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(margin={self.margin})"
