"""Tests for SMTR API server."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from smtr.runtime.api_server import app, set_llm


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_llm():
    """Create mock LLM."""
    llm = MagicMock()
    llm._model_name = "test-model"
    llm._generate = MagicMock(return_value='{"plan": ["a", "b"], "explanation": "test"}')
    return llm


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_returns_200(self, client):
        """Test health endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_format(self, client):
        """Test health response format."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "model_loaded" in data
        assert data["status"] == "healthy"


class TestChatCompletionsEndpoint:
    """Test chat completions endpoint."""

    def test_chat_completions_requires_model(self, client):
        """Test that model field is required."""
        response = client.post("/v1/chat/completions", json={"messages": []})
        assert response.status_code == 422

    def test_chat_completions_with_mock_llm(self, client, mock_llm):
        """Test chat completions with mock LLM."""
        set_llm(mock_llm)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]

    def test_chat_completion_response_format(self, client, mock_llm):
        """Test chat completion response format."""
        set_llm(mock_llm)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

        data = response.json()
        assert "id" in data
        assert "object" in data
        assert data["object"] == "chat.completion"
        assert "created" in data
        assert "model" in data


class TestDemoEndpoint:
    """Test demo endpoint."""

    def test_demo_endpoint_with_mock(self, client, mock_llm):
        """Test demo endpoint with mock LLM."""
        set_llm(mock_llm)

        with patch("smtr.runtime.graph.run_demo") as mock_run_demo:
            mock_run_demo.return_value = {
                "task": "test task",
                "team_success": True,
                "team_reward": 1.0,
                "router_trace": [],
            }

            response = client.post("/smtr/demo", json={"seed": 7, "use_real_llm": True})

            assert response.status_code == 200
            data = response.json()
            assert "task" in data
            assert "team_success" in data


class TestPipelineEndpoint:
    """Test pipeline endpoint."""

    def test_pipeline_endpoint_with_mock(self, client, mock_llm):
        """Test pipeline endpoint with mock LLM."""
        set_llm(mock_llm)

        with patch("smtr.runtime.graph.run_episode") as mock_run_episode:
            mock_run_episode.return_value = {
                "task": "test task",
                "team_success": True,
                "team_reward": 1.0,
                "team_summary": "success",
                "router_trace": [],
                "agent_outputs": {"planner": {}, "executor": {}, "critic": {}},
            }

            response = client.post(
                "/smtr/run-pipeline",
                json={
                    "task": "test task",
                    "seed": 7,
                    "top_k": 4,
                    "use_real_llm": True,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert "task" in data
            assert "team_success" in data
            assert "agent_outputs_keys" in data


class TestLLMInjection:
    """Test LLM injection functionality."""

    def test_set_llm(self, client, mock_llm):
        """Test setting LLM instance."""
        set_llm(mock_llm)

        response = client.get("/health")
        data = response.json()

        assert data["model_loaded"] is True
        assert data["model_name"] == "test-model"
