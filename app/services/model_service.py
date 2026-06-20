from pathlib import Path
from typing import Optional

import torch
import numpy as np
from loguru import logger

from app.models.mobilenet_embedding import MobileNetEmbedding


class ModelService:
    def __init__(
        self,
        model_path: Path,
        embedding_dim: int = 128,
        backbone: str = "efficientnet_b3",
        device: Optional[torch.device] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.embedding_dim = embedding_dim
        self.backbone = backbone
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Optional[MobileNetEmbedding] = None

    def load(self) -> None:
        self._model = MobileNetEmbedding(
            embedding_dim=self.embedding_dim,
            pretrained=True,
            backbone=self.backbone,
        )

        if self.model_path.exists():
            state_dict = torch.load(
                str(self.model_path),
                map_location=self.device,
                weights_only=True,
            )
            self._model.load_state_dict(state_dict)
            logger.info("Loaded trained model from {}", self.model_path)
        else:
            logger.warning(
                "Model checkpoint not found at {} — using pretrained weights",
                self.model_path,
            )

        self._model = self._model.to(self.device)
        self._model.eval()

    @property
    def model(self) -> MobileNetEmbedding:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._model

    @torch.no_grad()
    def extract_embedding(self, image_tensor: torch.Tensor) -> np.ndarray:
        image_tensor = image_tensor.to(self.device)
        embeddings = self.model(image_tensor)
        return embeddings.cpu().numpy()

    @torch.no_grad()
    def extract_embeddings_batch(
        self,
        image_tensors: torch.Tensor,
        batch_size: int = 32,
    ) -> np.ndarray:
        all_embeddings = []
        num_images = len(image_tensors)

        for start_idx in range(0, num_images, batch_size):
            end_idx = min(start_idx + batch_size, num_images)
            batch = image_tensors[start_idx:end_idx].to(self.device)
            embeddings = self.model(batch)
            all_embeddings.append(embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
