import time
from typing import Dict, List

import numpy as np
from loguru import logger

from app.services.model_service import ModelService
from app.services.faiss_service import FAISSService
from app.utils.image_utils import load_image, preprocess_image, get_inference_transforms


class RetrievalService:
    def __init__(
        self,
        model_service: ModelService,
        faiss_service: FAISSService,
        landmark_map: Dict[str, str],
        top_k: int = 5,
    ) -> None:
        self.model_service = model_service
        self.faiss_service = faiss_service
        self.landmark_map = landmark_map
        self.top_k = top_k
        self._transform = get_inference_transforms()

    def retrieve(
        self,
        image_bytes: bytes,
        top_k: int = None,
    ) -> List[Dict[str, object]]:
        k = top_k or self.top_k
        start_time = time.perf_counter()

        image = load_image(image_bytes)
        image_tensor = preprocess_image(image, self._transform)

        embedding = self.model_service.extract_embedding(image_tensor)

        search_k = min(k * 5, self.faiss_service.index_size)
        raw_results = self.faiss_service.search_with_ids(embedding, top_k=search_k)

        aggregated = self._aggregate_results(raw_results)

        sorted_results = sorted(aggregated, key=lambda x: x["score"], reverse=True)[:k]

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info("Retrieval completed in {:.1f} ms | results={}", elapsed_ms, len(sorted_results))

        return sorted_results

    def _aggregate_results(
        self,
        raw_results: List[Dict[str, object]],
    ) -> List[Dict[str, object]]:
        landmark_scores: Dict[str, List[float]] = {}
        for result in raw_results:
            lid = result["landmark_id"]
            score = result["score"]
            if lid not in landmark_scores:
                landmark_scores[lid] = []
            landmark_scores[lid].append(score)

        aggregated = []
        for lid, scores in landmark_scores.items():
            max_score = max(scores)
            mean_score = float(np.mean(scores))
            combined_score = 0.7 * max_score + 0.3 * mean_score

            name = self.landmark_map.get(str(lid), f"Unknown Landmark ({lid})")
            aggregated.append({
                "landmark_name": name,
                "score": round(combined_score, 4),
            })

        return aggregated
