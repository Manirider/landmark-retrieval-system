import io
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np


@pytest.fixture
def mock_app():
    with patch("app.core.startup.lifespan") as mock_lifespan:
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def noop_lifespan(app):
            app.state.model_service = MagicMock()
            app.state.faiss_service = MagicMock()
            app.state.landmark_map = {
                "10000": "Eiffel Tower",
                "10001": "Statue of Liberty",
            }
            yield

        mock_lifespan.side_effect = noop_lifespan

        from app.main import create_app
        app = create_app()

        model_service = MagicMock()
        model_service.extract_embedding.return_value = np.random.randn(1, 128).astype(np.float32)
        model_service.is_loaded = True
        app.state.model_service = model_service

        faiss_service = MagicMock()
        faiss_service.index_size = 100
        faiss_service.is_loaded = True
        faiss_service.search_with_ids.return_value = [
            {"landmark_id": "10000", "distance": 0.1, "score": 0.91},
            {"landmark_id": "10001", "distance": 0.5, "score": 0.67},
            {"landmark_id": "10000", "distance": 0.2, "score": 0.83},
        ]
        app.state.faiss_service = faiss_service

        app.state.landmark_map = {
            "10000": "Eiffel Tower",
            "10001": "Statue of Liberty",
        }

        yield app


@pytest.fixture
def client(mock_app):
    return TestClient(mock_app)


def create_test_image(size=(224, 224), color=(255, 0, 0), fmt="JPEG") -> bytes:
    image = Image.new("RGB", size, color=color)
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    buffer.seek(0)
    return buffer.read()


class TestRetrieveEndpoint:
    def test_retrieve_valid_image(self, client):
        image_bytes = create_test_image()
        response = client.post(
            "/retrieve",
            files={"image": ("test.jpg", image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200

    def test_retrieve_returns_list(self, client):
        image_bytes = create_test_image()
        response = client.post(
            "/retrieve",
            files={"image": ("test.jpg", image_bytes, "image/jpeg")},
        )
        data = response.json()
        assert isinstance(data, list)

    def test_retrieve_result_schema(self, client):
        image_bytes = create_test_image()
        response = client.post(
            "/retrieve",
            files={"image": ("test.jpg", image_bytes, "image/jpeg")},
        )
        data = response.json()
        if len(data) > 0:
            result = data[0]
            assert "landmark_name" in result
            assert "score" in result
            assert isinstance(result["landmark_name"], str)
            assert isinstance(result["score"], (int, float))

    def test_retrieve_scores_in_range(self, client):
        image_bytes = create_test_image()
        response = client.post(
            "/retrieve",
            files={"image": ("test.jpg", image_bytes, "image/jpeg")},
        )
        data = response.json()
        for result in data:
            assert 0 <= result["score"] <= 1.0

    def test_retrieve_png_image(self, client):
        image_bytes = create_test_image(fmt="PNG")
        response = client.post(
            "/retrieve",
            files={"image": ("test.png", image_bytes, "image/png")},
        )
        assert response.status_code == 200

    def test_retrieve_missing_image_field(self, client):
        response = client.post("/retrieve")
        assert response.status_code == 422

    def test_retrieve_empty_file(self, client):
        response = client.post(
            "/retrieve",
            files={"image": ("empty.jpg", b"", "image/jpeg")},
        )
        assert response.status_code == 400

    def test_retrieve_invalid_content_type(self, client):
        response = client.post(
            "/retrieve",
            files={"image": ("test.txt", b"not an image", "text/plain")},
        )
        assert response.status_code == 400

    def test_retrieve_content_type_json(self, client):
        image_bytes = create_test_image()
        response = client.post(
            "/retrieve",
            files={"image": ("test.jpg", image_bytes, "image/jpeg")},
        )
        assert "application/json" in response.headers["content-type"]

    def test_retrieve_get_not_allowed(self, client):
        response = client.get("/retrieve")
        assert response.status_code == 405
