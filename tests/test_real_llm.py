"""Tests for RealLLM adapter."""

from unittest.mock import MagicMock, patch

from smtr.runtime.real_llm import RealLLM


class TestRealLLMInterface:
    """Test RealLLM interface compatibility."""

    def test_plan_returns_dict(self):
        """Test that plan() returns a dict with required keys."""
        with patch.object(RealLLM, "__init__", lambda self, **kwargs: None):
            llm = RealLLM()
            llm._model_name = "test-model"
            llm._client = None
            llm._model = None
            llm._tokenizer = None
            llm._generate = MagicMock(return_value='{"plan": ["a", "b"], "explanation": "test"}')

            result = llm.plan("task", {"valid_sequence": ["a", "b"]}, [])

            assert isinstance(result, dict)
            assert "plan" in result
            assert "explanation" in result
            assert "local_trace" in result
            assert result["local_trace"]["source"] == "real_llm"

    def test_summarize_execution_returns_string(self):
        """Test that summarize_execution() returns a string."""
        with patch.object(RealLLM, "__init__", lambda self, **kwargs: None):
            llm = RealLLM()
            llm._model_name = "test-model"
            llm._client = None
            llm._model = None
            llm._tokenizer = None
            llm._generate = MagicMock(return_value="Summary of execution")

            result = llm.summarize_execution([{"ok": True}, {"ok": False}])

            assert isinstance(result, str)

    def test_plan_fallback_on_parse_error(self):
        """Test plan() fallback when JSON parsing fails."""
        with patch.object(RealLLM, "__init__", lambda self, **kwargs: None):
            llm = RealLLM()
            llm._model_name = "test-model"
            llm._client = None
            llm._model = None
            llm._tokenizer = None
            llm._generate = MagicMock(return_value="invalid json")

            result = llm.plan("task", {"valid_sequence": ["a", "b"]}, [])

            assert "plan" in result
            # When parsing fails, plan is empty list from _parse_json_response
            assert isinstance(result["plan"], list)

    def test_parse_json_with_code_block(self):
        """Test parsing JSON from markdown code block."""
        with patch.object(RealLLM, "__init__", lambda self, **kwargs: None):
            llm = RealLLM()
            raw = '```json\n{"plan": ["x"], "explanation": "y"}\n```'
            result = llm._parse_json_response(raw)
            assert result["plan"] == ["x"]

    def test_parse_json_extracts_object(self):
        """Test extracting JSON object from mixed text."""
        with patch.object(RealLLM, "__init__", lambda self, **kwargs: None):
            llm = RealLLM()
            raw = 'Here is the response: {"plan": ["a"], "explanation": "b"} and more text'
            result = llm._parse_json_response(raw)
            assert result["plan"] == ["a"]


class TestRealLLMGeneration:
    """Test RealLLM generation methods."""

    def test_generate_locally_mocked(self):
        """Test local generation with mocked model."""
        with patch.object(RealLLM, "__init__", lambda self, **kwargs: None):
            llm = RealLLM()
            llm._client = None
            llm._model_name = "test"
            llm._max_new_tokens = 100
            llm._temperature = 0.1

            mock_tokenizer = MagicMock()
            mock_tokenizer.return_value = {"input_ids": MagicMock(shape=(1, 5))}
            mock_tokenizer.eos_token_id = 0
            mock_tokenizer.decode = MagicMock(return_value="generated text")

            mock_model = MagicMock()
            mock_model.device = "cpu"
            mock_model.generate = MagicMock(return_value=[[1, 2, 3, 4, 5, 6, 7]])

            llm._tokenizer = mock_tokenizer
            llm._model = mock_model

            with patch("smtr.runtime.real_llm.torch", create=True):
                result = llm._generate_locally("prompt")

            assert result == "generated text"

    def test_generate_via_api_mocked(self):
        """Test API generation with mocked client."""
        with patch.object(RealLLM, "__init__", lambda self, **kwargs: None):
            llm = RealLLM()
            llm._model_name = "test-model"
            llm._max_new_tokens = 100
            llm._temperature = 0.1

            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "api response"}}]
            }
            mock_client.post.return_value = mock_response

            llm._client = mock_client

            result = llm._generate_via_api("prompt")

            assert result == "api response"
            mock_client.post.assert_called_once()


class TestRealLLMPrompts:
    """Test prompt building methods."""

    def test_build_plan_prompt(self):
        """Test plan prompt construction."""
        with patch.object(RealLLM, "__init__", lambda self, **kwargs: None):
            llm = RealLLM()
            observation = {"valid_sequence": ["a", "b"], "target_artifact": "target"}
            payloads = [{"goal_summary": "goal", "steps": ["step1"]}]

            prompt = llm._build_plan_prompt("task", observation, payloads)

            assert "task" in prompt.lower()
            assert "valid_sequence" in prompt
            assert "goal" in prompt

    def test_build_summarize_prompt(self):
        """Test summarize prompt construction."""
        with patch.object(RealLLM, "__init__", lambda self, **kwargs: None):
            llm = RealLLM()
            results = [{"action": "a", "ok": True}, {"action": "b", "ok": False}]

            prompt = llm._build_summarize_prompt(results)

            assert "summarize" in prompt.lower()
            assert "action" in prompt
