"""Subprocess boundary for invoking the real MARBLE engine."""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from smtr.counterfactual.decision_points import canonical_digest
from smtr.marble.environment.docker_slot_pool import DockerSlot
from smtr.marble.runtime_preflight import DEFAULT_DASHSCOPE_BASE_URL

DEFAULT_ENGINE_TIMEOUT_SECONDS = 900
DEFAULT_TERMINATION_GRACE_SECONDS = 5.0

_LITELLM_SHIM_TEMPLATE = """
from __future__ import annotations

import os

try:
    import litellm
except Exception:
    litellm = None

if litellm is not None and not getattr(litellm, "_smtr_compat_patch", False):
    _smtr_original_completion = litellm.completion

    def _smtr_completion(*args, **kwargs):
        # --- URL / key routing ---
        base_url = os.environ.get("SMTR_OPENAI_COMPAT_BASE_URL")
        api_key = os.environ.get("OPENAI_API_KEY")
        if base_url and not kwargs.get("base_url"):
            kwargs["base_url"] = base_url
        if api_key and not kwargs.get("api_key"):
            kwargs["api_key"] = api_key
        _thinking_env = os.environ.get("SMTR_LLM_ENABLE_THINKING", "")
        if _thinking_env:
            extra_body = dict(kwargs.get("extra_body") or {})
            _thinking_on = _thinking_env.lower() in {"1", "true", "yes"}
            extra_body.setdefault("enable_thinking", _thinking_on)
            kwargs["extra_body"] = extra_body
        # --- Runtime visibility audit ---
        _audit_path = os.environ.get("SMTR_VISIBILITY_AUDIT_PATH")
        if _audit_path:
            try:
                _smtr_audit_completion(_audit_path, args, kwargs)
            except Exception as _exc:
                try:
                    _err_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audit_error.log')
                    with open(_err_path, 'a') as _ef:
                        import traceback as _tb
                        _ef.write(type(_exc).__name__ + ': ' + str(_exc) + chr(10))
                        _tb.print_exc(file=_ef)
                except Exception:
                    pass
        return _smtr_original_completion(*args, **kwargs)

    def _smtr_audit_completion(audit_path, args, kwargs):
        import json as _json
        import hashlib as _hashlib
        import re as _re
        import time as _time
        import fcntl as _fcntl
        messages = kwargs.get("messages") or (args[1] if len(args) > 1 else [])
        if not messages:
            return
        full_text = ""
        for m in messages:
            content = m.get("content", "") if isinstance(m, dict) else str(m)
            full_text += content + "\\n"
        agent_match = _re.search(r"You are (agent\\w+):", full_text)
        agent_id = agent_match.group(1) if agent_match else "unknown"
        mem_pattern = _re.compile(
            r"\\[SMTR_PROCEDURAL_MEMORY:id=([^:]+):intervention=([^\\]]+)\\]"
        )
        visible_ids = []
        intervention_ids = set()
        for m in messages:
            content = m.get("content", "") if isinstance(m, dict) else str(m)
            for match in mem_pattern.finditer(content):
                mid = match.group(1)
                iid = match.group(2)
                if mid not in visible_ids:
                    visible_ids.append(mid)
                intervention_ids.add(iid)
        meta = {}
        meta_path = os.environ.get("SMTR_RUN_METADATA_PATH")
        if meta_path:
            try:
                with open(meta_path, "r") as f:
                    meta = _json.loads(f.read())
            except Exception:
                pass
        receiver_ids = set()
        ri = os.environ.get("SMTR_RECEIVER_AGENT_IDS", "")
        if ri:
            receiver_ids = set(x.strip() for x in ri.split(",") if x.strip())
        mem_digest = _hashlib.sha256(
            ",".join(visible_ids).encode("utf-8")
        ).hexdigest()
        msgs_digest = _hashlib.sha256(full_text.encode("utf-8")).hexdigest()
        record = {
            "schema_version": "1.0",
            "run_id": meta.get("run_id", os.environ.get("SMTR_RUN_ID", "unknown")),
            "task_id": meta.get("task_id", os.environ.get("SMTR_TASK_ID", "unknown")),
            "scenario": meta.get("scenario", os.environ.get("SMTR_SCENARIO", "unknown")),
            "method": meta.get("method", os.environ.get("SMTR_METHOD", "unknown")),
            "branch": meta.get("branch", os.environ.get("SMTR_BRANCH", "unknown")),
            "agent_id": agent_id,
            "agent_role": None,
            "receiver_agent": agent_id in receiver_ids,
            "turn_id": 0,
            "visible_memory_ids": visible_ids,
            "ordered_memory_digest": mem_digest,
            "memory_payload_digest": os.environ.get("SMTR_MEMORY_PAYLOAD_DIGEST", ""),
            "system_prompt_digest": "",
            "messages_digest": msgs_digest,
            "intervention_id": (
                list(intervention_ids)[0] if intervention_ids
                else os.environ.get("SMTR_INTERVENTION_ID")
            ),
            "invocation_index": 0,
            "timestamp_utc": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        }
        line = _json.dumps(record, sort_keys=True) + "\\n"
        p = audit_path
        with open(p, "a", encoding="utf-8") as fh:
            try:
                _fcntl.flock(fh.fileno(), _fcntl.LOCK_EX)
                fh.write(line)
                fh.flush()
            finally:
                _fcntl.flock(fh.fileno(), _fcntl.LOCK_UN)

    litellm.completion = _smtr_completion
    litellm._smtr_compat_patch = True
""".lstrip()


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
    memory_injection: dict[str, Any] | None = None,
    run_metadata: dict[str, str] | None = None,
    docker_slot: DockerSlot | None = None,
    api_key: str | None = None,
) -> MarbleEngineProcessResult:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if termination_grace_period_seconds < 0:
        raise ValueError("termination_grace_period_seconds must be non-negative")
    log_dir = output_dir or (raw_result_path.parent if raw_result_path else config_path.parent)
    log_dir.mkdir(parents=True, exist_ok=True)
    env = _engine_environment(
        marble_root,
        shim_dir=log_dir / "runtime_shim",
        memory_injection=memory_injection,
        visibility_audit_path=log_dir / "memory_visibility_audit.jsonl",
        run_metadata=run_metadata,
        docker_slot=docker_slot,
        api_key=api_key,
    )
    python = _marble_python(marble_root)
    if raw_result_path and raw_result_path.exists():
        raw_result_path.unlink()
    # Write a thin launcher that forces sitecustomize loading via exec(compile(...)).
    # Belt-and-suspenders: PYTHONPATH is already absolute, but the launcher
    # guarantees loading regardless of sys.path quirks.
    _shim_dir = log_dir / "runtime_shim"
    _launcher = _shim_dir / "_smtr_launcher.py"
    _main_py = str((marble_root / "marble/main.py").resolve())
    _sc_abs = str((_shim_dir / "sitecustomize.py").resolve())
    _launcher.write_text(
        "import sys, os, runpy, traceback\n"
        "_shim = " + repr(_sc_abs) + "\n"
        "sys.path.insert(0, os.path.dirname(_shim))\n"
        "try:\n"
        "    with open(_shim) as _f:\n"
        "        exec(compile(_f.read(), _shim, 'exec'), "
        "{'__file__': _shim, '__name__': 'sitecustomize'})\n"
        "except Exception as _e:\n"
        "    traceback.print_exc()\n"
        "sys.argv[0] = " + repr(_main_py) + "\n"
        "runpy.run_path(sys.argv[0], run_name='__main__')\n",
        encoding="utf-8",
    )
    command = (
        str(python),
        str(_launcher.resolve()),
        "--config_path",
        str(config_path.resolve()),
    )
    started_at_timestamp = time.time()
    started = _now()
    timed_out = False
    termination_requested = False
    termination_signal: str | None = None
    kill_escalated = False
    # MARBLE uses relative paths (e.g. evaluator/evaluator_prompts.json)
    # so CWD must be marble_root/marble (the package directory)
    engine_cwd = marble_root / "marble"
    process = subprocess.Popen(
        command,
        cwd=engine_cwd if engine_cwd.exists() else marble_root,
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
    cleanup = _cleanup_database(marble_root, log_dir=log_dir, docker_slot=docker_slot)
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


def _engine_environment(
    marble_root: Path,
    *,
    shim_dir: Path | None = None,
    memory_injection: dict[str, Any] | None = None,
    visibility_audit_path: Path | None = None,
    run_metadata: dict[str, str] | None = None,
    docker_slot: DockerSlot | None = None,
    api_key: str | None = None,
) -> dict[str, str]:
    env = dict(os.environ)
    pythonpath = env.get("PYTHONPATH")
    path_entries = []
    if shim_dir is not None:
        _write_runtime_shim(
            shim_dir,
            memory_injection=memory_injection,
            visibility_audit_path=visibility_audit_path,
            run_metadata=run_metadata,
        )
        # Use absolute path — subprocess CWD differs from caller CWD
        path_entries.append(str(shim_dir.resolve()))
    path_entries.append(str(marble_root))
    # Ensure MARBLE venv bin is on PATH so `python` resolves correctly
    venv_bin = marble_root / ".venv" / "bin"
    if venv_bin.exists():
        existing_path = env.get("PATH", "")
        env["PATH"] = f"{venv_bin}:{existing_path}" if existing_path else str(venv_bin)
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
    # --- Parallel slot overrides ---
    if docker_slot is not None:
        for key, value in docker_slot.engine_env.items():
            env[key] = value
    if api_key:
        env["DASHSCOPE_API_KEY"] = api_key
        env["OPENAI_API_KEY"] = api_key
    return env


def _is_dashscope_compatible_base_url(base_url: str | None) -> bool:
    if not base_url:
        return False
    host = urlparse(base_url).hostname or ""
    return host.endswith(".aliyuncs.com") or host == "dashscope.aliyuncs.com"


def _write_runtime_shim(
    shim_dir: Path,
    *,
    memory_injection: dict[str, Any] | None = None,
    visibility_audit_path: Path | None = None,
    run_metadata: dict[str, str] | None = None,
) -> None:
    """Write a combined sitecustomize.py with unified litellm patch + optional memory injection."""
    shim_dir.mkdir(parents=True, exist_ok=True)
    parts: list[str] = [_LITELLM_SHIM_TEMPLATE]
    if visibility_audit_path:
        audit_path_str = str(visibility_audit_path.resolve())
        # Write run metadata file for the shim to read
        meta_path_str = ""
        if run_metadata:
            meta_path = shim_dir / "run_metadata.json"
            meta_path.write_text(
                json.dumps(run_metadata, sort_keys=True), encoding="utf-8"
            )
            meta_path_str = str(meta_path.resolve())
        # Write a small env-setup snippet that always runs (even for B0)
        parts.append(_build_audit_env_snippet(
            audit_path=audit_path_str,
            metadata_path=meta_path_str,
            run_metadata=run_metadata or {},
            receiver_agent_ids=memory_injection.get("receiver_agent_ids", []) if memory_injection else [],
            intervention_id=memory_injection.get("intervention_id", "") if memory_injection else "",
            memory_digest=_compute_memory_digest(memory_injection),
        ))
        if memory_injection:
            payload_json = json.dumps(memory_injection, sort_keys=True)
            parts.append(
                _build_memory_injection_code(
                    payload_json=payload_json,
                    audit_path=audit_path_str,
                    metadata_path=meta_path_str,
                    receiver_agent_ids=memory_injection.get("receiver_agent_ids", []),
                    intervention_id=memory_injection.get("intervention_id", ""),
                )
            )
    (shim_dir / "sitecustomize.py").write_text(
        "\n".join(parts), encoding="utf-8"
    )


def _write_litellm_runtime_shim(shim_dir: Path) -> None:
    """Legacy wrapper — writes only the litellm shim."""
    _write_runtime_shim(shim_dir)


def _compute_memory_digest(memory_injection: dict[str, Any] | None) -> str:
    """SHA-256 of the canonical JSON representation of the injection payload."""
    if not memory_injection:
        return ""
    return hashlib.sha256(
        json.dumps(memory_injection, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _build_audit_env_snippet(
    *,
    audit_path: str,
    metadata_path: str = "",
    run_metadata: dict[str, str] | None = None,
    receiver_agent_ids: list[str] | None = None,
    intervention_id: str = "",
    memory_digest: str = "",
) -> str:
    """Generate a small Python snippet that sets env vars for the audit.

    This runs for ALL methods (B0, AllShare, SMTR, share, withhold) so that
    the unified litellm.completion patch always writes visibility records.
    """
    safe_audit = audit_path.replace("\\", "\\\\")
    safe_meta = metadata_path.replace("\\", "\\\\")
    safe_receivers = ",".join(receiver_agent_ids or [])
    meta = run_metadata or {}
    return f"""

# --- SMTR Visibility Audit Env Setup ---
import os as _os
_os.environ["SMTR_VISIBILITY_AUDIT_PATH"] = '{safe_audit}'
if '{safe_meta}':
    _os.environ["SMTR_RUN_METADATA_PATH"] = '{safe_meta}'
_os.environ["SMTR_RUN_ID"] = '{meta.get("run_id", "unknown")}'
_os.environ["SMTR_TASK_ID"] = '{meta.get("task_id", "unknown")}'
_os.environ["SMTR_SCENARIO"] = '{meta.get("scenario", "unknown")}'
_os.environ["SMTR_METHOD"] = '{meta.get("method", "unknown")}'
_os.environ["SMTR_BRANCH"] = '{meta.get("branch", "unknown")}'
_os.environ["SMTR_RECEIVER_AGENT_IDS"] = '{safe_receivers}'
_os.environ["SMTR_MEMORY_PAYLOAD_DIGEST"] = '{memory_digest}'
_os.environ["SMTR_INTERVENTION_ID"] = '{intervention_id}'
"""


def _build_memory_injection_code(
    *,
    payload_json: str,
    audit_path: str,
    metadata_path: str = "",
    receiver_agent_ids: list[str] | None = None,
    intervention_id: str = "",
) -> str:
    """Generate Python code for memory injection with structured markers.

    The generated code monkey-patches Engine.start() to inject memories
    wrapped in [SMTR_PROCEDURAL_MEMORY] markers.  The unified litellm
    completion patch (already in the shim) detects these markers in the
    final messages and writes RuntimeVisibilityRecords.
    """
    safe_payload = payload_json.replace("\\", "\\\\").replace("'", "\\'") if payload_json else ""
    return f"""

# --- SMTR Memory Injection ---
def _smtr_inject_memories():
    import json as _json
    payload_str = '{safe_payload}'
    if not payload_str:
        return
    try:
        payload = _json.loads(payload_str)
    except Exception:
        return
    receiver_ids = set(payload.get("receiver_agent_ids", []))
    memory_payloads = payload.get("memory_payloads", [])
    memory_ids = payload.get("memory_ids", [])
    intervention_id = payload.get("intervention_id", "unknown")
    if not receiver_ids or not memory_payloads:
        return
    try:
        from marble.engine.engine import Engine
    except ImportError:
        return
    _original_start = Engine.start
    def _patched_start(self):
        for agent in self.agents:
            if agent.agent_id in receiver_ids:
                for idx, mem_payload in enumerate(memory_payloads):
                    mem_id = memory_ids[idx] if idx < len(memory_ids) else f"mem_{{idx}}"
                    wrapped = (
                        f"[SMTR_PROCEDURAL_MEMORY:id={{mem_id}}:intervention={{intervention_id}}]"
                        f"\\n{{mem_payload}}"
                        f"\\n[/SMTR_PROCEDURAL_MEMORY]"
                    )
                    agent.memory.update("smtr_procedural", wrapped)
        return _original_start(self)
    Engine.start = _patched_start

_smtr_inject_memories()
"""


def _marble_python(marble_root: Path) -> Path:
    candidate = marble_root / ".venv/bin/python"
    return candidate if candidate.exists() else Path(sys.executable)


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


def _cleanup_database(
    marble_root: Path,
    *,
    log_dir: Path,
    docker_slot: DockerSlot | None = None,
) -> dict[str, Any]:
    compose_dir = marble_root / "marble/environments/db_env_docker"
    stdout = ""
    stderr = ""
    exit_code: int | None = None
    failure_reason: str | None = None
    if not compose_dir.exists():
        failure_reason = f"compose_dir_not_found: {compose_dir}"
    else:
        # Use slot-specific compose project when available
        if docker_slot is not None:
            cmd = (
                "sudo", "docker", "compose",
                "-p", docker_slot.compose_project,
                "-f", str(compose_dir / "docker-compose.yml"),
                "down", "-v",
            )
        else:
            cmd = ("sudo", "docker", "compose", "down", "-v")
        try:
            completed = subprocess.run(
                cmd,
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
