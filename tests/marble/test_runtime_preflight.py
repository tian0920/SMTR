import json
import stat
from pathlib import Path

from smtr.marble.runtime_preflight import run_database_runtime_preflight, write_runtime_preflight


def test_preflight_reports_missing_litellm(tmp_path: Path) -> None:
    marble_root = tmp_path / "MARBLE"
    python = marble_root / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text(
        "#!/usr/bin/env sh\n"
        "if [ \"$1\" = \"--version\" ]; then echo 'Python 3.12.3'; exit 0; fi\n"
        "case \"$2\" in *litellm*)\n"
        "  echo \"ModuleNotFoundError: No module named litellm\" >&2\n"
        "  exit 1\n"
        ";; esac\n"
        "echo '{\"version\": \"0.test\"}'\n",
        encoding="utf-8",
    )
    python.chmod(python.stat().st_mode | stat.S_IXUSR)
    compose = marble_root / "marble/environments/db_env_docker/docker-compose.yml"
    compose.parent.mkdir(parents=True)
    compose.write_text("services: {}\n", encoding="utf-8")

    result = run_database_runtime_preflight(marble_root=marble_root)

    check = next(item for item in result.checks if item.name == "litellm_import")
    assert check.passed is False
    assert "requirements-marble" in check.detail


def test_preflight_artifact_does_not_leak_api_key(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret-value")
    output = Path("artifacts/marble/manifests/runtime_preflight_test.json")

    write_runtime_preflight(marble_root=Path("/home/ecs-user/MARBLE"), output_path=output)
    payload = output.read_text(encoding="utf-8")

    assert "sk-test-secret-value" not in payload
    data = json.loads(payload)
    assert "checks" in data
