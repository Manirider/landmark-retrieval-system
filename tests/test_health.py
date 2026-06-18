import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def mock_app_state():
    with patch("app.core.startup.lifespan") as mock_lifespan:
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def noop_lifespan(app):
            app.state.model_service = MagicMock()
            app.state.faiss_service = MagicMock()
            app.state.landmark_map = {"10000": "Test Landmark"}
            yield

        mock_lifespan.side_effect = noop_lifespan

        from app.main import create_app
        app = create_app()
        app.state.model_service = MagicMock()
        app.state.faiss_service = MagicMock()
        app.state.landmark_map = {"10000": "Test Landmark"}
        yield app


@pytest.fixture
def client(mock_app_state):
    return TestClient(mock_app_state)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_healthy_status(self, client):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_response_schema(self, client):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert isinstance(data["status"], str)

    def test_health_content_type(self, client):
        response = client.get("/health")
        assert "application/json" in response.headers["content-type"]

    def test_health_method_not_allowed(self, client):
        response = client.post("/health")
        assert response.status_code == 405
