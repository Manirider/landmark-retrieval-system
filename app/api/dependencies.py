from fastapi import Depends, Request

from app.core.config import Settings, get_settings
from app.services.retrieval_service import RetrievalService
from app.services.model_service import ModelService
from app.services.faiss_service import FAISSService


def get_model_service(request: Request) -> ModelService:
    return request.app.state.model_service


def get_faiss_service(request: Request) -> FAISSService:
    return request.app.state.faiss_service


def get_landmark_map(request: Request) -> dict:
    return request.app.state.landmark_map


def get_retrieval_service(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> RetrievalService:
    return RetrievalService(
        model_service=request.app.state.model_service,
        faiss_service=request.app.state.faiss_service,
        landmark_map=request.app.state.landmark_map,
        top_k=settings.top_k,
    )
