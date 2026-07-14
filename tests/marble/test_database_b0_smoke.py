from pathlib import Path

from smtr.marble.database_smoke import run_database_b0_smoke


def test_b0_smoke_does_not_start_engine_when_preflight_fails(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_API_KEY", raising=False)
    monkeypatch.delenv("MARBLE_LLM_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    summary = run_database_b0_smoke(
        marble_root=Path("/home/ecs-user/MARBLE"),
        task_id="1",
        generation_seed=0,
        output_dir=Path("artifacts/marble/outputs/database_b0_smoke_test"),
    )

    assert summary["real_engine_executed"] is False
    assert summary["native_evaluator_executed"] is False
    assert summary["b0_memory_absent"] is True
