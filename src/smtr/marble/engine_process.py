"""Subprocess boundary for invoking the real MARBLE engine."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from smtr.counterfactual.decision_points import canonical_digest


@dataclass(frozen=True)
class MarbleEngineProcessResult:
    command: tuple[str, ...]
    working_directory: str
    exit_code: int
    timed_out: bool
    stdout_digest: str
    stderr_digest: str
    raw_result_path: str | None
    real_engine_executed: bool
    engine_version: str | None
    environment_digest: str
    started_at: str
    ended_at: str

    def to_json(self) -> dict[str, object]:
        return asdict(self)


def run_marble_engine_process(
    *,
    marble_root: Path,
    config_path: Path,
    raw_result_path: Path | None,
    timeout_seconds: int = 900,
) -> MarbleEngineProcessResult:
    env = _engine_environment(marble_root)
    python = _marble_python(marble_root)
    command = (
        str(python),
        str(marble_root / "marble/main.py"),
        "--config_path",
        str(config_path),
    )
    started = _now()
    timed_out = False
    try:
        completed = subprocess.run(
            command,
            cwd=marble_root,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        exit_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        exit_code = -9
        stdout = (
            (exc.stdout or "").decode()
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        stderr = (
            (exc.stderr or "").decode()
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        _cleanup_database(marble_root)
    ended = _now()
    return MarbleEngineProcessResult(
        command=command,
        working_directory=str(marble_root),
        exit_code=exit_code,
        timed_out=timed_out,
        stdout_digest=_text_digest(stdout),
        stderr_digest=_text_digest(stderr),
        raw_result_path=(
            str(raw_result_path) if raw_result_path and raw_result_path.exists() else None
        ),
        real_engine_executed=(
            not timed_out
            and exit_code == 0
            and bool(raw_result_path and raw_result_path.exists())
        ),
        engine_version=_engine_version(marble_root),
        environment_digest=canonical_digest(_sanitized_environment(env)),
        started_at=started,
        ended_at=ended,
    )


def write_engine_process_result(path: Path, result: MarbleEngineProcessResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _engine_environment(marble_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(marble_root) if not pythonpath else f"{marble_root}:{pythonpath}"
    return env


def _marble_python(marble_root: Path) -> Path:
    candidate = marble_root / ".venv/bin/python"
    return candidate if candidate.exists() else Path(os.sys.executable)


def _sanitized_environment(env: dict[str, str]) -> dict[str, str]:
    sanitized: dict[str, str] = {}
    for key, value in sorted(env.items()):
        upper = key.upper()
        if any(token in upper for token in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
            sanitized[key] = "<redacted-present>" if value else "<empty>"
        elif key in {"PATH", "PYTHONPATH", "MARBLE_LLM_MODEL", "OPENAI_MODEL"}:
            sanitized[key] = value
    return sanitized


def _text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _engine_version(marble_root: Path) -> str | None:
    pyproject = marble_root / "pyproject.toml"
    if not pyproject.exists():
        return None
    for line in pyproject.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("version"):
            return line.split("=", maxsplit=1)[1].strip().strip('"')
    return None


def _cleanup_database(marble_root: Path) -> None:
    compose_dir = marble_root / "marble/environments/db_env_docker"
    if compose_dir.exists():
        subprocess.run(
            ("sudo", "docker", "compose", "down", "-v"),
            cwd=compose_dir,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
