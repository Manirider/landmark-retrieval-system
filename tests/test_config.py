import os
import pytest
from pathlib import Path
from unittest.mock import patch


class TestSettings:
    def test_default_settings(self):
        from app.core.config import Settings
        settings = Settings()

        assert settings.api_host == "0.0.0.0"
        assert settings.api_port == 8000
        assert settings.top_k == 5
        assert settings.embedding_dim == 128
        assert settings.margin == 0.3
        assert settings.log_level == "INFO"

    def test_model_path_default(self):
        from app.core.config import Settings
        settings = Settings()
        assert settings.model_path == "artifacts/model.pth"

    def test_index_path_default(self):
        from app.core.config import Settings
        settings = Settings()
        assert settings.index_path == "artifacts/landmarks.index"

    def test_landmark_map_path_default(self):
        from app.core.config import Settings
        settings = Settings()
        assert settings.landmark_map_path == "data/landmark_map.json"

    def test_environment_variable_override(self):
        with patch.dict(os.environ, {"TOP_K": "10", "EMBEDDING_DIM": "256"}):
            from app.core.config import Settings
            settings = Settings()
            assert settings.top_k == 10
            assert settings.embedding_dim == 256

    def test_resolve_path_relative(self):
        from app.core.config import Settings
        settings = Settings()
        resolved = settings.resolve_path("artifacts/model.pth")
        assert resolved.is_absolute()
        assert str(resolved).endswith("model.pth")

    def test_resolve_path_absolute(self):
        from app.core.config import Settings
        settings = Settings()
        if os.name == "nt":
            abs_path = "C:\\absolute\\path\\model.pth"
        else:
            abs_path = "/absolute/path/model.pth"
        resolved = settings.resolve_path(abs_path)
        assert str(resolved) == abs_path

    def test_project_root_exists(self):
        from app.core.config import Settings
        settings = Settings()

        assert isinstance(settings.project_root, Path)

    def test_batch_size_positive(self):
        from app.core.config import Settings
        settings = Settings()
        assert settings.batch_size > 0

    def test_learning_rate_positive(self):
        from app.core.config import Settings
        settings = Settings()
        assert settings.learning_rate > 0

    def test_train_split_valid_range(self):
        from app.core.config import Settings
        settings = Settings()
        assert 0 < settings.train_split < 1

    def test_get_settings_returns_instance(self):
        from app.core.config import get_settings

        get_settings.cache_clear()
        settings = get_settings()
        assert isinstance(settings, object)
        assert hasattr(settings, "api_port")
        assert hasattr(settings, "embedding_dim")
        get_settings.cache_clear()

    def test_get_settings_caching(self):
        from app.core.config import get_settings
        get_settings.cache_clear()
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2
        get_settings.cache_clear()
