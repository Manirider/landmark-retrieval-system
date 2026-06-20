from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from loguru import logger

from app.core.config import get_settings
from app.core.logger import setup_logger
from app.services.model_service import ModelService
from app.services.faiss_service import FAISSService
from app.utils.mapping_utils import load_landmark_map, load_id_mapping


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    setup_logger()
    settings = get_settings()

    logger.info("Starting Landmark Retrieval System")
    logger.info("Configuration loaded | embedding_dim={} | top_k={}", settings.embedding_dim, settings.top_k)

    model_path = settings.resolve_path(settings.model_path)
    model_service = ModelService(
        model_path=model_path,
        embedding_dim=settings.embedding_dim,
        backbone=settings.model_backbone,
    )
    model_service.load()
    app.state.model_service = model_service
    logger.info("Model loaded | path={}", model_path)

    index_path = settings.resolve_path(settings.index_path)
    id_mapping_path = settings.resolve_path(settings.id_mapping_path)

    id_mapping = load_id_mapping(id_mapping_path)
    faiss_service = FAISSService(
        index_path=index_path,
        id_mapping=id_mapping,
    )
    faiss_service.load()
    app.state.faiss_service = faiss_service
    logger.info("FAISS index loaded | path={} | vectors={}", index_path, faiss_service.index_size)

    landmark_map_path = settings.resolve_path(settings.landmark_map_path)
    landmark_map = load_landmark_map(landmark_map_path)
    app.state.landmark_map = landmark_map
    logger.info("Landmark map loaded | path={} | landmarks={}", landmark_map_path, len(landmark_map))

    logger.info("All resources loaded — system ready to serve requests")

    yield

    logger.info("Shutting down Landmark Retrieval System")

    del app.state.model_service
    del app.state.faiss_service
    del app.state.landmark_map
    logger.info("Cleanup complete")
