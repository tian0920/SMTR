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


def test_marble_output_path_is_written_absolute(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = Path("workspace/marble_config.yaml")
    raw_result_path = Path("workspace/result.jsonl")

    _write_marble_config(
        task={"environment": {}, "task": {}, "agents": []},
        config_path=config_path,
        raw_result_path=raw_result_path,
        generation_seed=0,
    )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_path = Path(config["output"]["file_path"])
    assert output_path.is_absolute()
    assert output_path == raw_result_path.resolve()
    assert not raw_result_path.exists()


def test_marble_config_does_not_embed_api_key_names_or_values(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-dashscope-secret-value")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-secret-value")
    config_path = tmp_path / "workspace" / "marble_config.yaml"

    _write_marble_config(
        task={"environment": {}, "task": {}, "agents": []},
        config_path=config_path,
        raw_result_path=tmp_path / "workspace" / "result.jsonl",
        generation_seed=0,
    )

    text = config_path.read_text(encoding="utf-8")
    assert "DASHSCOPE_API_KEY" not in text
    assert "OPENAI_API_KEY" not in text
    assert "Authorization" not in text
    assert "Bearer" not in text
    assert "sk-dashscope-secret-value" not in text
    assert "sk-openai-secret-value" not in text
