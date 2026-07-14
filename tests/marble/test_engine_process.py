import os
import stat
import time
from pathlib import Path

from smtr.marble.engine_process import _engine_environment, _text_digest, run_marble_engine_process
from smtr.marble.runtime_preflight import DEFAULT_DASHSCOPE_BASE_URL


def test_engine_process_records_exit_logs_and_digests(monkeypatch, tmp_path: Path) -> None:
    marble_root = tmp_path / "MARBLE"
    raw_result = tmp_path / "result.jsonl"
    _write_fake_marble_python(
        marble_root=marble_root,
        body="echo 'hello token=secret Bearer abc123 sk-abc123456789'\nexit 3\n",
    )
    _write_fake_sudo(monkeypatch, tmp_path, exit_code=0)
    (marble_root / "pyproject.toml").write_text('version = "0.0.test"\n', encoding="utf-8")

    result = run_marble_engine_process(
        marble_root=marble_root,
        config_path=tmp_path / "config.yaml",
        raw_result_path=raw_result,
        output_dir=tmp_path / "logs",
        timeout_seconds=5,
    )

    assert result.command[0].endswith(".venv/bin/python")
    assert result.exit_code == 3
    assert result.stdout_digest
    assert result.stderr_digest
    assert result.stdout_log_path is not None
    stdout = Path(result.stdout_log_path).read_text(encoding="utf-8")
    assert "secret" not in stdout
    assert "Bearer abc123" not in stdout
    assert "sk-abc123456789" not in stdout
    assert result.stdout_digest == _text_digest(stdout)
    assert result.real_engine_executed is False


def test_engine_timeout_marks_not_executed(monkeypatch, tmp_path: Path) -> None:
    marble_root = tmp_path / "MARBLE"
    _write_fake_marble_python(marble_root=marble_root, body="sleep 5\n")
    _write_fake_sudo(monkeypatch, tmp_path, exit_code=0)

    result = run_marble_engine_process(
        marble_root=marble_root,
        config_path=tmp_path / "config.yaml",
        raw_result_path=tmp_path / "missing.jsonl",
        output_dir=tmp_path / "logs",
        timeout_seconds=1,
    )

    assert result.timed_out is True
    assert result.real_engine_executed is False


def test_old_raw_result_is_deleted_before_execution(monkeypatch, tmp_path: Path) -> None:
    marble_root = tmp_path / "MARBLE"
    raw_result = tmp_path / "result.jsonl"
    raw_result.write_text('{"old": true}\n', encoding="utf-8")
    _write_fake_marble_python(marble_root=marble_root, body="exit 0\n")
    _write_fake_sudo(monkeypatch, tmp_path, exit_code=0)

    result = run_marble_engine_process(
        marble_root=marble_root,
        config_path=tmp_path / "config.yaml",
        raw_result_path=raw_result,
        output_dir=tmp_path / "logs",
        timeout_seconds=5,
    )

    assert not raw_result.exists()
    assert result.raw_result_exists is False
    assert result.real_engine_executed is False


def test_exit_zero_empty_result_is_not_real_execution(monkeypatch, tmp_path: Path) -> None:
    marble_root = tmp_path / "MARBLE"
    raw_result = tmp_path / "result.jsonl"
    _write_fake_marble_python(
        marble_root=marble_root,
        body=f": > {raw_result}\nexit 0\n",
    )
    _write_fake_sudo(monkeypatch, tmp_path, exit_code=0)

    result = run_marble_engine_process(
        marble_root=marble_root,
        config_path=tmp_path / "config.yaml",
        raw_result_path=raw_result,
        output_dir=tmp_path / "logs",
        timeout_seconds=5,
    )

    assert result.exit_code == 0
    assert result.raw_result_exists is True
    assert result.raw_result_nonempty is False
    assert result.real_engine_executed is False


def test_exit_zero_unparseable_result_is_not_real_execution(
    monkeypatch,
    tmp_path: Path,
) -> None:
    marble_root = tmp_path / "MARBLE"
    raw_result = tmp_path / "result.jsonl"
    _write_fake_marble_python(
        marble_root=marble_root,
        body=f"echo 'not json' > {raw_result}\nexit 0\n",
    )
    _write_fake_sudo(monkeypatch, tmp_path, exit_code=0)

    result = run_marble_engine_process(
        marble_root=marble_root,
        config_path=tmp_path / "config.yaml",
        raw_result_path=raw_result,
        output_dir=tmp_path / "logs",
        timeout_seconds=5,
    )

    assert result.raw_result_parseable is False
    assert result.real_engine_executed is False


def test_stale_result_is_not_real_execution(monkeypatch, tmp_path: Path) -> None:
    marble_root = tmp_path / "MARBLE"
    raw_result = tmp_path / "result.jsonl"
    _write_fake_marble_python(
        marble_root=marble_root,
        body=(
            f"echo '{{\"ok\": true}}' > {raw_result}\n"
            f"touch -t 200001010000 {raw_result}\n"
            "exit 0\n"
        ),
    )
    _write_fake_sudo(monkeypatch, tmp_path, exit_code=0)

    result = run_marble_engine_process(
        marble_root=marble_root,
        config_path=tmp_path / "config.yaml",
        raw_result_path=raw_result,
        output_dir=tmp_path / "logs",
        timeout_seconds=5,
    )

    assert result.raw_result_parseable is True
    assert result.raw_result_fresh is False
    assert result.real_engine_executed is False


def test_cleanup_failure_is_recorded(monkeypatch, tmp_path: Path) -> None:
    marble_root = tmp_path / "MARBLE"
    raw_result = tmp_path / "result.jsonl"
    _write_fake_marble_python(
        marble_root=marble_root,
        body=f"echo '{{\"ok\": true}}' > {raw_result}\nexit 0\n",
    )
    _write_fake_sudo(monkeypatch, tmp_path, exit_code=9)

    result = run_marble_engine_process(
        marble_root=marble_root,
        config_path=tmp_path / "config.yaml",
        raw_result_path=raw_result,
        output_dir=tmp_path / "logs",
        timeout_seconds=5,
    )

    assert result.real_engine_executed is True
    assert result.cleanup_succeeded is False
    assert result.cleanup_exit_code == 9


def test_normal_mock_subprocess_can_be_valid(monkeypatch, tmp_path: Path) -> None:
    marble_root = tmp_path / "MARBLE"
    raw_result = tmp_path / "result.jsonl"
    _write_fake_marble_python(
        marble_root=marble_root,
        body=f"echo '{{\"ok\": true}}' > {raw_result}\nexit 0\n",
    )
    _write_fake_sudo(monkeypatch, tmp_path, exit_code=0)

    result = run_marble_engine_process(
        marble_root=marble_root,
        config_path=tmp_path / "config.yaml",
        raw_result_path=raw_result,
        output_dir=tmp_path / "logs",
        timeout_seconds=1,
    )

    assert result.real_engine_executed is True
    assert result.raw_result_exists is True
    assert result.raw_result_nonempty is True
    assert result.raw_result_fresh is True
    assert result.raw_result_parseable is True
    assert result.cleanup_succeeded is True


def test_dashscope_key_is_mapped_to_openai_compatible_env(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-dashscope-secret-value")

    env = _engine_environment(tmp_path / "MARBLE")

    assert env["OPENAI_API_KEY"] == "sk-dashscope-secret-value"
    assert env["OPENAI_BASE_URL"] == DEFAULT_DASHSCOPE_BASE_URL
    assert env["OPENAI_API_BASE"] == DEFAULT_DASHSCOPE_BASE_URL


def test_dashscope_endpoint_prefers_dashscope_key_when_openai_key_also_exists(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-secret-value")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-dashscope-secret-value")

    env = _engine_environment(tmp_path / "MARBLE")

    assert env["OPENAI_API_KEY"] == "sk-dashscope-secret-value"
    assert env["OPENAI_BASE_URL"] == DEFAULT_DASHSCOPE_BASE_URL
    assert env["OPENAI_API_BASE"] == DEFAULT_DASHSCOPE_BASE_URL


def test_runtime_shim_is_added_for_litellm_openai_compatible_calls(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-dashscope-secret-value")
    shim_dir = tmp_path / "shim"

    env = _engine_environment(tmp_path / "MARBLE", shim_dir=shim_dir)

    assert env["PYTHONPATH"].split(":")[0] == str(shim_dir)
    shim = shim_dir / "sitecustomize.py"
    assert shim.exists()
    text = shim.read_text(encoding="utf-8")
    assert "enable_thinking" in text
    assert "sk-dashscope-secret-value" not in text


def _write_fake_marble_python(*, marble_root: Path, body: str) -> None:
    (marble_root / ".venv/bin").mkdir(parents=True)
    (marble_root / "marble").mkdir(parents=True)
    python = marble_root / ".venv/bin/python"
    python.write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8")
    python.chmod(python.stat().st_mode | stat.S_IXUSR)
    (marble_root / "marble/main.py").write_text("# fake\n", encoding="utf-8")
    compose = marble_root / "marble/environments/db_env_docker/docker-compose.yml"
    compose.parent.mkdir(parents=True)
    compose.write_text("services: {}\n", encoding="utf-8")


def _write_fake_sudo(monkeypatch, tmp_path: Path, *, exit_code: int) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sudo = bin_dir / "sudo"
    sudo.write_text(
        "#!/usr/bin/env bash\n"
        "echo cleanup stdout\n"
        "echo cleanup stderr >&2\n"
        f"exit {exit_code}\n",
        encoding="utf-8",
    )
    sudo.chmod(sudo.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    time.sleep(0.01)
