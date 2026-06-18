from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from loguru import logger


class FAISSService:
    def __init__(
        self,
        index_path: Path,
        id_mapping: np.ndarray,
    ) -> None:
        self.index_path = Path(index_path)
        self.id_mapping = id_mapping
        self._index: Optional[faiss.Index] = None

    def load(self) -> None:
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")

        try:
            self._index = faiss.read_index(str(self.index_path))
            logger.info(
                "FAISS index loaded | vectors={} | dimension={}",
                self._index.ntotal,
                self._index.d,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load FAISS index: {e}") from e

    @property
    def index(self) -> faiss.Index:
        if self._index is None:
            raise RuntimeError("FAISS index not loaded. Call load() first.")
        return self._index

    @property
    def index_size(self) -> int:
        if self._index is None:
            return 0
        return self._index.ntotal

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding = query_embedding.astype(np.float32)

        distances, indices = self.index.search(query_embedding, top_k)
        return distances, indices

    def search_with_ids(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict[str, object]]:
        distances, indices = self.search(query_embedding, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue

            landmark_id = str(self.id_mapping[idx])
            score = float(1.0 / (1.0 + dist))

            results.append({
                "landmark_id": landmark_id,
                "distance": float(dist),
                "score": round(score, 4),
            })

        return results

    @property
    def is_loaded(self) -> bool:
        return self._index is not None
