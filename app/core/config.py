from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    api_host: str = Field(default="0.0.0.0", description="API bind host")
    api_port: int = Field(default=8000, description="API bind port")

    model_path: str = Field(
        default="artifacts/model.pth",
        description="Path to the trained model checkpoint",
    )
    index_path: str = Field(
        default="artifacts/landmarks.index",
        description="Path to the FAISS index file",
    )
    id_mapping_path: str = Field(
        default="artifacts/id_mapping.npy",
        description="Path to the index-to-landmark ID mapping",
    )
    landmark_map_path: str = Field(
        default="data/landmark_map.json",
        description="Path to the landmark ID → name mapping JSON",
    )

    top_k: int = Field(default=5, ge=1, le=100, description="Number of results to return")
    embedding_dim: int = Field(default=128, ge=16, le=2048, description="Embedding vector dimension")

    learning_rate: float = Field(default=0.001, gt=0, description="Optimizer learning rate")
    batch_size: int = Field(default=32, ge=1, description="Training batch size")
    num_epochs: int = Field(default=20, ge=1, description="Number of training epochs")
    margin: float = Field(default=0.3, gt=0, description="Triplet loss margin")

    num_classes: int = Field(default=50, ge=2, description="Number of landmark classes")
    min_images_per_class: int = Field(default=10, ge=2, description="Minimum images per class")
    train_split: float = Field(default=0.8, gt=0, lt=1, description="Train/test split ratio")

    log_level: str = Field(default="INFO", description="Logging level")

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent.parent

    def resolve_path(self, relative_path: str) -> Path:
        path = Path(relative_path)
        if path.is_absolute():
            return path
        return self.project_root / path

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
