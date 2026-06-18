from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MobileNetEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim

        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)

        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        backbone_out_features = 576

        self.embedding_head = nn.Sequential(
            nn.Linear(backbone_out_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.embedding_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)

        features = self.avgpool(features)
        features = torch.flatten(features, 1)

        embeddings = self.embedding_head(features)

        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    def get_embedding_dim(self) -> int:
        return self.embedding_dim


def create_embedding_model(
    embedding_dim: int = 128,
    pretrained: bool = True,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> MobileNetEmbedding:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MobileNetEmbedding(
        embedding_dim=embedding_dim,
        pretrained=pretrained,
    )

    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

    model = model.to(device)
    return model
