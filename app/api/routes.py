from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, Request
from loguru import logger

from app.api.schemas import HealthResponse, RetrievalResult, ErrorResponse
from app.api.dependencies import get_retrieval_service
from app.services.retrieval_service import RetrievalService

from typing import List

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={
        503: {"model": ErrorResponse, "description": "Service uninitialized or unhealthy"},
    },
    summary="Health Check",
    description="Returns service health status. Use for liveness and readiness probes.",
    tags=["System"],
)
async def health_check(request: Request) -> HealthResponse:
    app = request.app
    model_loaded = getattr(app.state, "model_service", None) is not None and app.state.model_service.is_loaded
    faiss_loaded = getattr(app.state, "faiss_service", None) is not None and app.state.faiss_service.is_loaded
    map_loaded = getattr(app.state, "landmark_map", None) is not None

    if not (model_loaded and faiss_loaded and map_loaded):
        raise HTTPException(
            status_code=503,
            detail="Service uninitialized: model, index, or mappings not loaded"
        )
    return HealthResponse(status="healthy")


@router.post(
    "/retrieve",
    response_model=List[RetrievalResult],
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image"},
        422: {"model": ErrorResponse, "description": "Missing image field"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Retrieve Landmarks",
    description=(
        "Upload an image to identify landmarks. Returns a ranked list of "
        "the top-K most likely landmarks with confidence scores."
    ),
    tags=["Retrieval"],
)
async def retrieve_landmarks(
    image: UploadFile = File(..., description="Image file (JPEG, PNG)"),
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
) -> List[RetrievalResult]:
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content type: {image.content_type}. Expected an image file.",
        )

    content_length = image.headers.get("content-length")
    if content_length and int(content_length) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File size exceeds the 10MB limit.")

    try:
        image_bytes = await image.read()
    except Exception as e:
        logger.error("Failed to read uploaded file: {}", e)
        raise HTTPException(status_code=400, detail="Failed to read uploaded file") from e

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File size exceeds the 10MB limit.")

    try:
        results = retrieval_service.retrieve(image_bytes)
    except ValueError as e:
        logger.warning("Invalid image: {}", e)
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}") from e
    except Exception as e:
        logger.error("Retrieval failed: {}", e)
        raise HTTPException(status_code=500, detail="Internal retrieval error") from e

    return [
        RetrievalResult(landmark_name=r["landmark_name"], score=r["score"])
        for r in results
    ]
