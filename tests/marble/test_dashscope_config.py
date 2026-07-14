import json
from pathlib import Path

from smtr.marble.database_smoke import _write_marble_config
from smtr.marble.runtime_preflight import DEFAULT_DASHSCOPE_BASE_URL


def test_dashscope_model_is_written_as_litellm_openai_compatible_model(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("MARBLE_LLM_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("DASHSCOPE_MODEL", raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-dashscope-secret-value")

    config_path = tmp_path / "marble_config.yaml"
    _write_marble_config(
        task={"environment": {}, "task": {}, "agents": []},
        config_path=config_path,
        raw_result_path=tmp_path / "result.jsonl",
        generation_seed=0,
    )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["llm"] == "openai/qwen3.7-max"
    assert DEFAULT_DASHSCOPE_BASE_URL
