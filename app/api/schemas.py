from typing import List

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(
        ...,
        description="Service health status",
        json_schema_extra={"example": "healthy"},
    )


class RetrievalResult(BaseModel):
    landmark_name: str = Field(
        ...,
        description="Human-readable name of the predicted landmark",
        json_schema_extra={"example": "Eiffel Tower"},
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1",
        json_schema_extra={"example": 0.96},
    )


class RetrievalResponse(BaseModel):
    results: List[RetrievalResult] = Field(
        ...,
        description="Ranked list of landmark predictions",
    )


class ErrorResponse(BaseModel):
    detail: str = Field(
        ...,
        description="Human-readable error description",
        json_schema_extra={"example": "Invalid image format"},
    )
