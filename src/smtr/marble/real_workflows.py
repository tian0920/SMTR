"""Executable real-data workflows; failures are retained but never promoted."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from smtr.counterfactual.decision_points import canonical_digest
from smtr.marble.database_smoke import run_database_b0_smoke
from smtr.marble.engine_process import DEFAULT_ENGINE_TIMEOUT_SECONDS
from smtr.marble.real_data import RealDatabaseTrajectory, SplitName, file_sha256


def collect_database_trajectories(
    *,
    marble_root: Path,
    dataset_manifest_path: Path,
    split_manifest_path: Path,
    split: str,
    output_dir: Path,
    generation_seeds: list[int],
    task_ids: list[str] | None = None,
    task_count: int | None = None,
    resume: bool = False,
    engine_timeout_seconds: int = DEFAULT_ENGINE_TIMEOUT_SECONDS,
    engine_timeout_source: str = "default",
) -> dict[str, Any]:
    if split != "train":
        raise ValueError("real source trajectory collection is train-only")
    dataset = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    splits = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    if file_sha256(Path(dataset["dataset_absolute_path"])) != dataset["dataset_sha256"]:
        raise ValueError("MARBLE dataset hash no longer matches frozen manifest")
    allowed = {str(record["task_id"]) for record in splits["records"] if record["split"] == split}
    selected = sorted(allowed & set(task_ids or allowed), key=_task_sort_key)
    if task_count is not None:
        selected = selected[:task_count]
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "trajectory_index.jsonl"
    index: list[dict[str, Any]] = []
    for task_id in selected:
        for seed in generation_seeds:
            trajectory_id = canonical_digest(
                {"task_id": task_id, "split": split, "generation_seed": seed}
            )[:24]
            run_dir = output_dir / "trajectories" / trajectory_id
            record_path = run_dir / "trajectory.json"
            if resume and record_path.exists():
                index.append(json.loads(record_path.read_text(encoding="utf-8")))
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            summary: dict[str, Any] | None = None
            try:
                summary = run_database_b0_smoke(
                    marble_root=marble_root,
                    task_id=task_id,
                    generation_seed=seed,
                    output_dir=run_dir,
                    engine_timeout_seconds=engine_timeout_seconds,
                    engine_timeout_source=engine_timeout_source,
                )
                record = _normalize_smoke(
                    summary=summary,
                    trajectory_id=trajectory_id,
                    task_id=task_id,
                    split=split,
                    generation_seed=seed,
                    dataset=dataset,
                    dataset_manifest_path=dataset_manifest_path,
                    split_manifest_path=split_manifest_path,
                    run_dir=run_dir,
                )
                payload = record.model_dump(mode="json")
                payload["valid"] = True
            except Exception as exc:
                payload = _invalid_trajectory_payload(
                    trajectory_id=trajectory_id,
                    task_id=task_id,
                    split=split,
                    generation_seed=seed,
                    summary=summary,
                    invalid_reason=str(exc),
                )
            record_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            index.append(payload)
    index_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in index),
        encoding="utf-8",
    )
    return {
        "attempted": len(index),
        "valid": sum(bool(item.get("valid")) for item in index),
        "invalid": sum(not bool(item.get("valid")) for item in index),
        "task_ids": selected,
        "index": str(index_path),
    }


def _normalize_smoke(
    *,
    summary: dict[str, Any],
    trajectory_id: str,
    task_id: str,
    split: str,
    generation_seed: int,
    dataset: dict[str, Any],
    dataset_manifest_path: Path,
    split_manifest_path: Path,
    run_dir: Path,
) -> RealDatabaseTrajectory:
    raw_path_value = summary.get("raw_result_path")
    if not raw_path_value:
        raise ValueError("raw result path not recorded")
    raw_path = Path(raw_path_value)
    if not summary.get("raw_result_exists"):
        raise ValueError("raw result file missing")
    if not summary.get("raw_result_nonempty"):
        raise ValueError("raw result file empty")
    if not summary.get("raw_result_fresh"):
        raise ValueError("raw result file stale")
    if not summary.get("raw_result_parseable"):
        raise ValueError("raw result file unparseable")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    structured = _structured_trace(raw)
    return RealDatabaseTrajectory(
        trajectory_id=trajectory_id,
        task_id=task_id,
        split=cast(SplitName, split),
        generation_seed=generation_seed,
        model_id=str(summary.get("model_id") or "unknown"),
        dataset_manifest_sha256=file_sha256(dataset_manifest_path),
        split_manifest_sha256=file_sha256(split_manifest_path),
        marble_commit=dataset["marble_commit"],
        smtr_commit=dataset["smtr_commit"],
        initial_database_fingerprint=summary.get("initial_logical_fingerprint") or {},
        initial_logical_database_digest=str(
            summary.get("initial_logical_database_digest")
            or summary.get("initial_state_digest")
            or ""
        ),
        agent_identities=structured["agents"],
        agent_messages=structured["messages"],
        agent_actions=structured["actions"],
        tool_calls=structured["tool_calls"],
        sql_statements=structured["sql"],
        observations=structured["observations"],
        errors=structured["errors"],
        final_answer=structured["final_answer"],
        raw_result_path=str(raw_path),
        raw_result_sha256=file_sha256(raw_path),
        native_evaluator_executed=bool(summary.get("native_evaluator_executed")),
        native_evaluator_output=summary.get("outcome") or {},
        score=float(summary.get("outcome", {}).get("score") or 0.0),
        task_success=bool(summary.get("outcome", {}).get("success")),
        real_engine_executed=bool(summary.get("real_engine_executed")),
        cleanup_succeeded=bool(summary.get("cleanup_succeeded")),
        environment_valid=bool(summary.get("environment_valid")),
        raw_result_exists=bool(summary.get("raw_result_exists")),
        raw_result_nonempty=bool(summary.get("raw_result_nonempty")),
        raw_result_fresh=bool(summary.get("raw_result_fresh")),
        raw_result_parseable=bool(summary.get("raw_result_parseable")),
        started_at=str(summary.get("started_at") or "unknown"),
        completed_at=str(summary.get("completed_at") or "unknown"),
        stdout_log_path=str(summary.get("stdout_log_path") or run_dir / "stdout.log"),
        stderr_log_path=str(summary.get("stderr_log_path") or run_dir / "stderr.log"),
        workspace_path=str(run_dir / "workspace"),
    )


def _structured_trace(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("MARBLE raw result is not an object")
    messages = raw.get("messages") or raw.get("agent_messages") or raw.get("history") or []
    actions = raw.get("actions") or raw.get("agent_actions") or []
    tool_calls = raw.get("tool_calls") or []
    if not actions and not tool_calls:
        raise ValueError("MARBLE raw result lacks structured action/tool trace")
    sql = raw.get("sql_statements") or [
        str(call.get("arguments", {}).get("sql") or call.get("sql"))
        for call in tool_calls
        if isinstance(call, dict) and (call.get("sql") or call.get("arguments", {}).get("sql"))
    ]
    return {
        "agents": raw.get("agents") or [],
        "messages": messages,
        "actions": actions,
        "tool_calls": tool_calls,
        "sql": sql,
        "observations": raw.get("observations") or [],
        "errors": raw.get("errors") or [],
        "final_answer": str(raw.get("final_answer") or raw.get("answer") or ""),
    }


def _invalid_trajectory_payload(
    *,
    trajectory_id: str,
    task_id: str,
    split: str,
    generation_seed: int,
    summary: dict[str, Any] | None,
    invalid_reason: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "trajectory_id": trajectory_id,
        "task_id": task_id,
        "split": split,
        "generation_seed": generation_seed,
        "valid": False,
        "invalid_reason": invalid_reason,
        "failure_layer": _failure_layer(summary=summary, invalid_reason=invalid_reason),
    }
    if summary is None:
        return payload
    for key in (
        "runtime_preflight_ready",
        "preflight_blocking_failures",
        "engine_exit_code",
        "engine_timed_out",
        "engine_timeout_seconds",
        "engine_timeout_source",
        "engine_duration_seconds",
        "engine_termination_requested",
        "engine_termination_signal",
        "engine_termination_grace_period_seconds",
        "engine_kill_escalated",
        "last_observed_stage",
        "last_observed_stage_parser_version",
        "engine_working_directory",
        "engine_config_path",
        "selected_python",
        "real_engine_executed",
        "raw_result_path",
        "raw_result_exists",
        "raw_result_nonempty",
        "raw_result_fresh",
        "raw_result_parseable",
        "raw_result_identity_verified",
        "native_evaluator_executed",
        "cleanup_exit_code",
        "cleanup_succeeded",
        "cleanup_failure_reason",
        "environment_valid",
        "stdout_log_path",
        "stderr_log_path",
    ):
        if key in summary:
            payload[key] = summary[key]
    payload["structured_trace_present"] = False
    return payload


def _failure_layer(*, summary: dict[str, Any] | None, invalid_reason: str) -> str:
    reason = invalid_reason.lower()
    if summary is None:
        return "unknown"
    if summary.get("runtime_preflight_ready") is False:
        return "preflight_failure"
    if summary.get("engine_timed_out") is True:
        return "engine_execution_timeout"
    if _nonzero_exit(summary.get("engine_exit_code")):
        return "engine_execution_failure"
    if (
        "raw result" in reason
        or not summary.get("raw_result_path")
        or not summary.get("raw_result_exists")
        or not summary.get("raw_result_nonempty")
        or not summary.get("raw_result_fresh")
        or not summary.get("raw_result_parseable")
    ):
        return "raw_result_failure"
    if "structured" in reason or "trace" in reason:
        return "structured_trace_failure"
    if summary.get("native_evaluator_executed") is False:
        return "native_evaluator_failure"
    if summary.get("cleanup_succeeded") is False:
        return "cleanup_failure"
    if "provenance" in reason:
        return "provenance_failure"
    return "unknown"


def _nonzero_exit(value: Any) -> bool:
    if value is None:
        return False
    try:
        return int(value) != 0
    except (TypeError, ValueError):
        return True


def _task_sort_key(task_id: str) -> tuple[int, str]:
    return (0, f"{int(task_id):09d}") if task_id.isdigit() else (1, task_id)
