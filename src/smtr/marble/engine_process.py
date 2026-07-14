"""Subprocess boundary for invoking the real MARBLE engine."""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from smtr.counterfactual.decision_points import canonical_digest
from smtr.marble.runtime_preflight import DEFAULT_DASHSCOPE_BASE_URL

DEFAULT_ENGINE_TIMEOUT_SECONDS = 900
DEFAULT_TERMINATION_GRACE_SECONDS = 5.0
LAST_OBSERVED_STAGE_PARSER_VERSION = "marble_engine_log_markers_v1"


@dataclass(frozen=True)
class MarbleEngineProcessResult:
    command: tuple[str, ...]
    working_directory: str
    selected_python: str
    config_path: str
    engine_timeout_seconds: int
    engine_timeout_source: str
    engine_duration_seconds: float
    exit_code: int
    timed_out: bool
    engine_termination_requested: bool
    engine_termination_signal: str | None
    engine_termination_grace_period_seconds: float
    engine_kill_escalated: bool
    last_observed_stage: str
    last_observed_stage_parser_version: str
    stdout_digest: str
    stderr_digest: str
    stdout_log_path: str | None
    stderr_log_path: str | None
    raw_result_path: str | None
    raw_result_exists: bool
    raw_result_nonempty: bool
    raw_result_fresh: bool
    raw_result_parseable: bool
    raw_result_identity_verified: bool
    raw_result_identity_failure_reason: str | None
    real_engine_executed: bool
    engine_version: str | None
    environment_digest: str
    started_at: str
    ended_at: str
    cleanup_exit_code: int | None
    cleanup_succeeded: bool
    cleanup_failure_reason: str | None
    cleanup_stdout_log_path: str | None
    cleanup_stderr_log_path: str | None
    cleanup_stdout_digest: str | None
    cleanup_stderr_digest: str | None

    def to_json(self) -> dict[str, object]:
        return asdict(self)


def run_marble_engine_process(
    *,
    marble_root: Path,
    config_path: Path,
    raw_result_path: Path | None,
    output_dir: Path | None = None,
    run_identity: dict[str, str] | None = None,
    timeout_seconds: int = DEFAULT_ENGINE_TIMEOUT_SECONDS,
    timeout_source: str = "default",
    termination_grace_period_seconds: float = DEFAULT_TERMINATION_GRACE_SECONDS,
) -> MarbleEngineProcessResult:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if termination_grace_period_seconds < 0:
        raise ValueError("termination_grace_period_seconds must be non-negative")
    log_dir = output_dir or (raw_result_path.parent if raw_result_path else config_path.parent)
    log_dir.mkdir(parents=True, exist_ok=True)
    env = _engine_environment(marble_root, shim_dir=log_dir / "runtime_shim")
    python = _marble_python(marble_root)
    if raw_result_path and raw_result_path.exists():
        raw_result_path.unlink()
    command = (
        str(python),
        str(marble_root / "marble/main.py"),
        "--config_path",
        str(config_path.resolve()),
    )
    started_at_timestamp = time.time()
    started = _now()
    timed_out = False
    termination_requested = False
    termination_signal: str | None = None
    kill_escalated = False
    process = subprocess.Popen(
        command,
        cwd=marble_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        termination_requested = True
        termination_signal = "SIGTERM"
        _terminate_process_group(process, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=termination_grace_period_seconds)
        except subprocess.TimeoutExpired as exc:
            kill_escalated = True
            termination_signal = "SIGKILL"
            _terminate_process_group(process, signal.SIGKILL)
            stdout, stderr = process.communicate()
            stdout = _combine_timeout_output(exc.stdout, stdout)
            stderr = _combine_timeout_output(exc.stderr, stderr)
    exit_code = process.returncode if process.returncode is not None else -9
    ended_at_timestamp = time.time()
    stdout_log = _write_log(log_dir / "stdout.log", stdout)
    stderr_log = _write_log(log_dir / "stderr.log", stderr)
    last_stage = _last_observed_stage(stdout=stdout, stderr=stderr)
    cleanup = _cleanup_database(marble_root, log_dir=log_dir)
    ended = _now()
    raw_validation = _validate_raw_result(
        raw_result_path=raw_result_path,
        started_at_timestamp=started_at_timestamp,
        run_identity=run_identity or {},
    )
    real_engine_executed = (
        not timed_out
        and exit_code == 0
        and raw_validation["raw_result_exists"]
        and raw_validation["raw_result_nonempty"]
        and raw_validation["raw_result_fresh"]
        and raw_validation["raw_result_parseable"]
    )
    return MarbleEngineProcessResult(
        command=command,
        working_directory=str(marble_root),
        selected_python=str(python),
        config_path=str(config_path.resolve()),
        engine_timeout_seconds=timeout_seconds,
        engine_timeout_source=timeout_source,
        engine_duration_seconds=round(ended_at_timestamp - started_at_timestamp, 3),
        exit_code=exit_code,
        timed_out=timed_out,
        engine_termination_requested=termination_requested,
        engine_termination_signal=termination_signal,
        engine_termination_grace_period_seconds=termination_grace_period_seconds,
        engine_kill_escalated=kill_escalated,
        last_observed_stage=last_stage,
        last_observed_stage_parser_version=LAST_OBSERVED_STAGE_PARSER_VERSION,
        stdout_digest=stdout_log["digest"],
        stderr_digest=stderr_log["digest"],
        stdout_log_path=stdout_log["path"],
        stderr_log_path=stderr_log["path"],
        raw_result_path=str(raw_result_path) if raw_result_path else None,
        raw_result_exists=raw_validation["raw_result_exists"],
        raw_result_nonempty=raw_validation["raw_result_nonempty"],
        raw_result_fresh=raw_validation["raw_result_fresh"],
        raw_result_parseable=raw_validation["raw_result_parseable"],
        raw_result_identity_verified=raw_validation["raw_result_identity_verified"],
        raw_result_identity_failure_reason=raw_validation[
            "raw_result_identity_failure_reason"
        ],
        real_engine_executed=real_engine_executed,
        engine_version=_engine_version(marble_root),
        environment_digest=canonical_digest(_sanitized_environment(env)),
        started_at=started,
        ended_at=ended,
        cleanup_exit_code=cleanup["exit_code"],
        cleanup_succeeded=cleanup["succeeded"],
        cleanup_failure_reason=cleanup["failure_reason"],
        cleanup_stdout_log_path=cleanup["stdout_log_path"],
        cleanup_stderr_log_path=cleanup["stderr_log_path"],
        cleanup_stdout_digest=cleanup["stdout_digest"],
        cleanup_stderr_digest=cleanup["stderr_digest"],
    )


def write_engine_process_result(path: Path, result: MarbleEngineProcessResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _engine_environment(marble_root: Path, *, shim_dir: Path | None = None) -> dict[str, str]:
    env = dict(os.environ)
    pythonpath = env.get("PYTHONPATH")
    path_entries = []
    if shim_dir is not None:
        _write_litellm_runtime_shim(shim_dir)
        path_entries.append(str(shim_dir))
    path_entries.append(str(marble_root))
    if pythonpath:
        path_entries.append(pythonpath)
    env["PYTHONPATH"] = ":".join(path_entries)
    base_url = (
        env.get("MARBLE_LLM_BASE_URL")
        or env.get("OPENAI_BASE_URL")
        or env.get("OPENAI_API_BASE")
        or env.get("DASHSCOPE_BASE_URL")
    )
    if not base_url and env.get("DASHSCOPE_API_KEY"):
        base_url = DEFAULT_DASHSCOPE_BASE_URL
    if env.get("DASHSCOPE_API_KEY") and (
        not env.get("OPENAI_API_KEY") or _is_dashscope_compatible_base_url(base_url)
    ):
        env["OPENAI_API_KEY"] = env["DASHSCOPE_API_KEY"]
    if base_url:
        env["OPENAI_BASE_URL"] = base_url
        env["OPENAI_API_BASE"] = base_url
        env["SMTR_OPENAI_COMPAT_BASE_URL"] = base_url
    if env.get("DASHSCOPE_API_KEY") and "SMTR_LLM_ENABLE_THINKING" not in env:
        env["SMTR_LLM_ENABLE_THINKING"] = "true"
    return env


def _is_dashscope_compatible_base_url(base_url: str | None) -> bool:
    if not base_url:
        return False
    host = urlparse(base_url).hostname or ""
    return host.endswith(".aliyuncs.com") or host == "dashscope.aliyuncs.com"


def _write_litellm_runtime_shim(shim_dir: Path) -> None:
    shim_dir.mkdir(parents=True, exist_ok=True)
    (shim_dir / "sitecustomize.py").write_text(
        """
from __future__ import annotations

import os

try:
    import litellm
except Exception:
    litellm = None

if litellm is not None and not getattr(litellm, "_smtr_openai_compat_patch", False):
    _smtr_original_completion = litellm.completion

    def _smtr_completion(*args, **kwargs):
        base_url = os.environ.get("SMTR_OPENAI_COMPAT_BASE_URL")
        api_key = os.environ.get("OPENAI_API_KEY")
        if base_url and not kwargs.get("base_url"):
            kwargs["base_url"] = base_url
        if api_key and not kwargs.get("api_key"):
            kwargs["api_key"] = api_key
        if os.environ.get("SMTR_LLM_ENABLE_THINKING", "").lower() in {"1", "true", "yes"}:
            extra_body = dict(kwargs.get("extra_body") or {})
            extra_body.setdefault("enable_thinking", True)
            kwargs["extra_body"] = extra_body
        return _smtr_original_completion(*args, **kwargs)

    litellm.completion = _smtr_completion
    litellm._smtr_openai_compat_patch = True
""".lstrip(),
        encoding="utf-8",
    )


def _marble_python(marble_root: Path) -> Path:
    candidate = marble_root / ".venv/bin/python"
    return candidate if candidate.exists() else Path(os.sys.executable)


def _sanitized_environment(env: dict[str, str]) -> dict[str, str]:
    sanitized: dict[str, str] = {}
    for key, value in sorted(env.items()):
        upper = key.upper()
        if any(token in upper for token in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
            sanitized[key] = "<redacted-present>" if value else "<empty>"
        elif key in {
            "PATH",
            "PYTHONPATH",
            "MARBLE_LLM_MODEL",
            "OPENAI_MODEL",
            "DASHSCOPE_MODEL",
            "MARBLE_LLM_BASE_URL",
            "OPENAI_BASE_URL",
            "OPENAI_API_BASE",
            "DASHSCOPE_BASE_URL",
            "SMTR_OPENAI_COMPAT_BASE_URL",
            "SMTR_LLM_ENABLE_THINKING",
        }:
            sanitized[key] = value
    return sanitized


def _text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _redact(text: str) -> str:
    redacted = text
    patterns = [
        (
            (
                r"(?i)(OPENAI_API_KEY|DASHSCOPE_API_KEY|ANTHROPIC_API_KEY|"
                r"AZURE_OPENAI_API_KEY)\s*=\s*\S+"
            ),
            r"\1=<redacted>",
        ),
        (
            r"(?i)(api_key|token|password)(['\"]?\s*[:=]\s*['\"]?)[^'\"\s,}]+",
            r"\1\2<redacted>",
        ),
        (r"Bearer\s+[A-Za-z0-9._\-]+", "Bearer <redacted>"),
        (r"sk-[A-Za-z0-9_\-]{8,}", "sk-<redacted>"),
    ]
    for pattern, replacement in patterns:
        redacted = re.sub(pattern, replacement, redacted)
    for key, value in os.environ.items():
        upper = key.upper()
        if value and any(token in upper for token in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
            redacted = redacted.replace(value, "<redacted>")
    return redacted


def _write_log(path: Path, text: str) -> dict[str, str]:
    redacted = _redact(text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(redacted, encoding="utf-8")
    return {"path": str(path), "digest": _text_digest(redacted)}


def _combine_timeout_output(partial: str | bytes | None, final: str | None) -> str:
    left = partial.decode() if isinstance(partial, bytes) else (partial or "")
    right = final or ""
    return right if right.startswith(left) else left + right


def _terminate_process_group(process: subprocess.Popen[str], sig: signal.Signals) -> None:
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        return
    except OSError:
        if sig == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()


def _last_observed_stage(*, stdout: str, stderr: str) -> str:
    text = f"{stdout}\n{stderr}".lower()
    markers = (
        ("raw_output_writing_started", ("jsonl", "output")),
        ("final_answer_generated", ("final answer",)),
        ("tool_sql_execution_started", ("sql", "query", "tool")),
        ("first_model_response_received", ("model response", "completion response")),
        ("first_model_request_started", ("model request", "completion request")),
        ("agents_initialized", ("agent initialized", "initializing agent", "agent")),
        ("docker_database_started", ("starting docker containers",)),
        ("engine_process_started", ("starting engine with configuration",)),
    )
    for stage, tokens in markers:
        if all(token in text for token in tokens):
            return stage
    return "unknown"


def _validate_raw_result(
    *,
    raw_result_path: Path | None,
    started_at_timestamp: float,
    run_identity: dict[str, str],
) -> dict[str, Any]:
    if raw_result_path is None:
        return _raw_validation(
            exists=False,
            nonempty=False,
            fresh=False,
            parseable=False,
            identity_verified=False,
            identity_failure_reason="raw_result_path_not_configured",
        )
    exists = raw_result_path.exists()
    nonempty = exists and raw_result_path.stat().st_size > 0
    fresh = exists and raw_result_path.stat().st_mtime >= started_at_timestamp
    parseable = False
    identity_verified = False
    identity_failure_reason: str | None = None
    records: list[dict[str, Any]] = []
    if nonempty:
        try:
            for line in raw_result_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError("JSONL item is not an object")
                records.append(item)
            parseable = bool(records)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            identity_failure_reason = f"raw_result_parse_failed: {type(exc).__name__}"
    if parseable:
        if run_identity:
            identity_verified = any(
                all(str(record.get(key)) == str(value) for key, value in run_identity.items())
                for record in records
            )
            if not identity_verified:
                identity_failure_reason = "raw_result_identity_mismatch"
        else:
            identity_failure_reason = "raw_result_identity_not_configured"
    return _raw_validation(
        exists=exists,
        nonempty=nonempty,
        fresh=fresh,
        parseable=parseable,
        identity_verified=identity_verified,
        identity_failure_reason=identity_failure_reason,
    )


def _raw_validation(
    *,
    exists: bool,
    nonempty: bool,
    fresh: bool,
    parseable: bool,
    identity_verified: bool,
    identity_failure_reason: str | None,
) -> dict[str, Any]:
    return {
        "raw_result_exists": exists,
        "raw_result_nonempty": nonempty,
        "raw_result_fresh": fresh,
        "raw_result_parseable": parseable,
        "raw_result_identity_verified": identity_verified,
        "raw_result_identity_failure_reason": identity_failure_reason,
    }


def _engine_version(marble_root: Path) -> str | None:
    pyproject = marble_root / "pyproject.toml"
    if not pyproject.exists():
        return None
    for line in pyproject.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("version"):
            return line.split("=", maxsplit=1)[1].strip().strip('"')
    return None


def _cleanup_database(marble_root: Path, *, log_dir: Path) -> dict[str, Any]:
    compose_dir = marble_root / "marble/environments/db_env_docker"
    stdout = ""
    stderr = ""
    exit_code: int | None = None
    failure_reason: str | None = None
    if not compose_dir.exists():
        failure_reason = f"compose_dir_not_found: {compose_dir}"
    else:
        try:
            completed = subprocess.run(
                ("sudo", "docker", "compose", "down", "-v"),
                cwd=compose_dir,
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
            exit_code = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr
            if completed.returncode != 0:
                failure_reason = f"cleanup_exit_code={completed.returncode}"
        except Exception as exc:
            exit_code = -1
            failure_reason = f"cleanup_failed: {type(exc).__name__}: {exc}"
    stdout_log = _write_log(log_dir / "cleanup_stdout.log", stdout)
    stderr_log = _write_log(log_dir / "cleanup_stderr.log", stderr)
    return {
        "exit_code": exit_code,
        "succeeded": failure_reason is None,
        "failure_reason": failure_reason,
        "stdout_log_path": stdout_log["path"],
        "stderr_log_path": stderr_log["path"],
        "stdout_digest": stdout_log["digest"],
        "stderr_digest": stderr_log["digest"],
    }


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
