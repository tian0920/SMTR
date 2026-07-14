"""Runtime preflight checks for the real MARBLE database engine."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from smtr.marble.artifacts import assert_marble_artifact_path

DEFAULT_DASHSCOPE_BASE_URL = (
    "https://llm-jhxtd03gjg0gd2o2.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
)
DEFAULT_DASHSCOPE_MODEL = "qwen3.7-max"


@dataclass(frozen=True)
class RuntimeCheck:
    name: str
    passed: bool
    detail: str
    blocking: bool


@dataclass(frozen=True)
class MarbleRuntimePreflightResult:
    checks: tuple[RuntimeCheck, ...]
    ready: bool


def run_database_runtime_preflight(*, marble_root: Path) -> MarbleRuntimePreflightResult:
    checks: list[RuntimeCheck] = []
    python_executable = _marble_python(marble_root)
    checks.append(_path_check("marble_root_exists", marble_root, "Set --marble-root correctly."))
    checks.append(_python_version_check(python_executable))
    checks.append(
        _path_check(
            "required_compose_file_exists",
            marble_root / "marble/environments/db_env_docker/docker-compose.yml",
            "Check out the full MARBLE repository with DB docker assets.",
        )
    )
    checks.append(
        _path_check(
            "workspaces_writable",
            Path("artifacts/marble/workspaces"),
            "Create artifacts/marble/workspaces or fix permissions.",
            must_exist=False,
            write=True,
        )
    )

    checks.extend(
        [
            _import_check(
                python_executable,
                marble_root,
                "marble",
                "Install MARBLE in the selected MARBLE Python environment.",
            ),
            _import_check(
                python_executable,
                marble_root,
                "litellm",
                "Install requirements-marble.txt; expected litellm ^1.52.1.",
            ),
            _import_check(
                python_executable,
                marble_root,
                "marble.engine.engine",
                "Install MARBLE dependencies and verify marble.engine.engine imports.",
            ),
            _import_check(
                python_executable,
                marble_root,
                "marble.environments.db_env",
                "Install MARBLE database dependencies such as psycopg2-binary.",
            ),
            _import_check(
                python_executable,
                marble_root,
                "marble.evaluator.evaluator",
                (
                    "Install MARBLE evaluator dependencies and run from a "
                    "complete MARBLE checkout."
                ),
            ),
        ]
    )

    checks.extend(
        [
            _command_check("docker_executable", ("docker", "--version"), "Install Docker."),
            _command_check(
                "docker_daemon_reachable",
                ("docker", "info"),
                "Start Docker and ensure the current user can access the daemon.",
            ),
            _command_check(
                "docker_compose_available",
                ("docker", "compose", "version"),
                "Install Docker Compose v2.",
            ),
            _port_check(5432),
            _command_check(
                "sudo_non_interactive_available",
                ("sudo", "-n", "true"),
                "Configure passwordless sudo for MARBLE's fixed sudo docker compose calls.",
            ),
            _llm_provider_check(),
            _model_check(),
            _base_url_check(),
        ]
    )
    ready = all(check.passed for check in checks if check.blocking)
    return MarbleRuntimePreflightResult(checks=tuple(checks), ready=ready)


def write_runtime_preflight(
    *, marble_root: Path, output_path: Path
) -> MarbleRuntimePreflightResult:
    assert_marble_artifact_path(output_path)
    result = run_database_runtime_preflight(marble_root=marble_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "ready": result.ready,
                "blocking_failures": [
                    check.name for check in result.checks if check.blocking and not check.passed
                ],
                "selected_python": str(_marble_python(marble_root)),
                "selected_python_version": _python_version_detail(
                    _marble_python(marble_root)
                ),
                "marble_imports": {
                    check.name: check.passed
                    for check in result.checks
                    if check.name.endswith("_import")
                },
                "docker_available": _check_passed(result, "docker_executable"),
                "compose_available": _check_passed(result, "docker_compose_available"),
                "llm_key_configured": _check_passed(
                    result, "required_api_key_presence"
                ),
                "llm_model_configured": _check_passed(result, "configured_model_name"),
                "llm_base_url_configured": _check_passed(
                    result, "configured_base_url"
                ),
                "checks": [asdict(check) for check in result.checks],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def _check_passed(result: MarbleRuntimePreflightResult, name: str) -> bool:
    return any(check.name == name and check.passed for check in result.checks)


def _path_check(
    name: str,
    path: Path,
    fix: str,
    *,
    must_exist: bool = True,
    write: bool = False,
) -> RuntimeCheck:
    if must_exist and not path.exists():
        return RuntimeCheck(name, False, f"{path} not found. {fix}", True)
    if write:
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".smtr_write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            return RuntimeCheck(name, False, f"{path} is not writable: {exc}. {fix}", True)
    return RuntimeCheck(name, True, str(path), True)


def _import_check(python: Path, marble_root: Path, module: str, fix: str) -> RuntimeCheck:
    code = (
        "import importlib, importlib.metadata, json\n"
        f"m=importlib.import_module({module!r})\n"
        f"top={module.split('.', maxsplit=1)[0]!r}\n"
        "v=getattr(m, '__version__', None)\n"
        "\ntry:\n    v=v or importlib.metadata.version(top)\nexcept Exception:\n    pass\n"
        "print(json.dumps({'version': v}))\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(marble_root)
    completed = subprocess.run(
        (str(python), "-c", code),
        cwd=marble_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        message = detail[-1] if detail else f"exit_code={completed.returncode}"
        return RuntimeCheck(
            module.replace(".", "_") + "_import",
            False,
            f"{message}. {fix}",
            True,
        )
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        version = payload.get("version")
    except (IndexError, json.JSONDecodeError):
        version = None
    detail = f"imported {module}"
    if version:
        detail += f" version={version}"
    return RuntimeCheck(module.replace(".", "_") + "_import", True, detail, True)


def _marble_python(marble_root: Path) -> Path:
    candidate = marble_root / ".venv/bin/python"
    return candidate if candidate.exists() else Path(sys.executable)


def _python_version_check(python: Path) -> RuntimeCheck:
    detail = _python_version_detail(python)
    if detail.startswith("Could not execute"):
        return RuntimeCheck("marble_python_version", False, detail, True)
    parts = detail.split()
    version = parts[1] if len(parts) > 1 else "0"
    major_minor = tuple(int(item) for item in version.split(".")[:2])
    passed = (3, 9) <= major_minor < (3, 13)
    return RuntimeCheck(
        "marble_python_version",
        passed,
        f"{detail} at {python}. MARBLE requires Python >=3.9,<3.13.",
        True,
    )


def _python_version_detail(python: Path) -> str:
    completed = subprocess.run(
        (str(python), "--version"),
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    detail = (completed.stdout or completed.stderr).strip()
    if completed.returncode != 0:
        return f"Could not execute {python}. Use MARBLE .venv or Python >=3.9,<3.13."
    return detail


def _command_check(name: str, command: tuple[str, ...], fix: str) -> RuntimeCheck:
    executable = shutil.which(command[0])
    if executable is None:
        return RuntimeCheck(name, False, f"{command[0]} not found. {fix}", True)
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception as exc:
        return RuntimeCheck(name, False, f"{type(exc).__name__}: {exc}. {fix}", True)
    output = (completed.stdout or completed.stderr).strip().splitlines()
    detail = output[0] if output else f"exit_code={completed.returncode}"
    return RuntimeCheck(
        name,
        completed.returncode == 0,
        f"{detail}. {fix if completed.returncode else ''}".strip(),
        True,
    )


def _port_check(port: int) -> RuntimeCheck:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        result = sock.connect_ex(("127.0.0.1", port))
    if result == 0:
        return RuntimeCheck(
            "postgresql_port_availability",
            False,
            (
                f"127.0.0.1:{port} is already in use; stop the conflicting "
                "service before MARBLE starts its fixed DB."
            ),
            True,
        )
    return RuntimeCheck("postgresql_port_availability", True, f"127.0.0.1:{port} available", True)


def _llm_provider_check() -> RuntimeCheck:
    key_names = (
        "OPENAI_API_KEY",
        "DASHSCOPE_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "AZURE_API_KEY",
    )
    present = [name for name in key_names if os.environ.get(name)]
    if present:
        return RuntimeCheck("required_api_key_presence", True, f"present={','.join(present)}", True)
    return RuntimeCheck(
        "required_api_key_presence",
        False,
        (
            "No supported LLM API key is present; set OPENAI_API_KEY, "
            "DASHSCOPE_API_KEY, or another MARBLE-supported provider key."
        ),
        True,
    )


def _model_check() -> RuntimeCheck:
    model = (
        os.environ.get("MARBLE_LLM_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or os.environ.get("DASHSCOPE_MODEL")
    )
    if model:
        return RuntimeCheck("configured_model_name", True, f"model={model}", True)
    if _dashscope_compatible_runtime_configured():
        return RuntimeCheck(
            "configured_model_name",
            True,
            f"model={DEFAULT_DASHSCOPE_MODEL} (default DashScope-compatible smoke model)",
            True,
        )
    return RuntimeCheck(
        "configured_model_name",
        False,
        (
            "Set MARBLE_LLM_MODEL, OPENAI_MODEL, or DASHSCOPE_MODEL to the model "
            "MARBLE should use."
        ),
        True,
    )


def _base_url_check() -> RuntimeCheck:
    base_url = (
        os.environ.get("MARBLE_LLM_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
        or os.environ.get("DASHSCOPE_BASE_URL")
    )
    if base_url:
        return RuntimeCheck("configured_base_url", True, f"base_url={base_url}", False)
    if os.environ.get("DASHSCOPE_API_KEY"):
        return RuntimeCheck(
            "configured_base_url",
            True,
            f"base_url={DEFAULT_DASHSCOPE_BASE_URL} (default DashScope-compatible URL)",
            False,
        )
    return RuntimeCheck(
        "configured_base_url",
        False,
        "No OpenAI-compatible base URL configured; this is only required for custom endpoints.",
        False,
    )


def _dashscope_compatible_runtime_configured() -> bool:
    return bool(
        os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("DASHSCOPE_BASE_URL")
        or os.environ.get("MARBLE_LLM_BASE_URL")
    )
