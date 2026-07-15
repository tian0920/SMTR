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
                payload = json.loads(record_path.read_text(encoding="utf-8"))
                index.append(_index_record(payload, record_path))
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
                )
                record = _normalize_smoke(
                    summary=summary,
                    trajectory_id=trajectory_id,
                    task_id=task_id,
                    split=split,
                    generation_seed=seed,
                    dataset=dataset,
                )
                payload = record.model_dump(mode="json")
            except Exception as exc:
                payload = _invalid_trajectory_payload(
                    trajectory_id=trajectory_id,
                    task_id=task_id,
                    split=split,
                    generation_seed=seed,
                    summary=summary,
                    failure_reason=_failure_reason(summary=summary, error=str(exc)),
                )
            record_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            index.append(_index_record(payload, record_path))
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
    if not summary.get("real_engine_executed"):
        raise ValueError("real engine did not produce a valid raw result")
    if not summary.get("native_evaluator_executed"):
        raise ValueError("native evaluator did not run")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    structured = _structured_trace(raw)
    return RealDatabaseTrajectory(
        trajectory_id=trajectory_id,
        task_id=task_id,
        split=cast(SplitName, split),
        generation_seed=generation_seed,
        model_id=str(summary.get("model_id") or "unknown"),
        source_dataset_version=str(dataset.get("dataset_sha256") or dataset.get("version") or ""),
        messages=structured["messages"],
        actions=structured["actions"],
        tool_calls=structured["tool_calls"],
        sql_statements=structured["sql"],
        observations=structured["observations"],
        errors=structured["errors"],
        final_answer=structured["final_answer"],
        score=float(summary.get("outcome", {}).get("score") or 0.0),
        task_success=bool(summary.get("outcome", {}).get("success")),
        valid=True,
        failure_reason=None,
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
    failure_reason: str,
) -> dict[str, Any]:
    return {
        "schema_version": "database_trajectory_v1",
        "trajectory_id": trajectory_id,
        "task_id": task_id,
        "split": split,
        "generation_seed": generation_seed,
        "model_id": str((summary or {}).get("model_id") or "unknown"),
        "source_dataset_version": None,
        "messages": [],
        "actions": [],
        "tool_calls": [],
        "sql_statements": [],
        "observations": [],
        "errors": [],
        "final_answer": "",
        "score": None,
        "task_success": None,
        "valid": False,
        "failure_reason": failure_reason,
    }


def _failure_reason(*, summary: dict[str, Any] | None, error: str) -> str:
    reason = error.lower()
    if summary is None:
        return "unknown"
    if summary.get("runtime_preflight_ready") is False:
        return "preflight_failed"
    if summary.get("engine_timed_out") is True:
        return "engine_timeout"
    if _nonzero_exit(summary.get("engine_exit_code")):
        return "engine_failed"
    if (
        "raw result" in reason
        or not summary.get("raw_result_path")
        or not summary.get("raw_result_exists")
        or not summary.get("raw_result_nonempty")
        or not summary.get("raw_result_fresh")
        or not summary.get("raw_result_parseable")
    ):
        return "raw_result_invalid"
    if "structured" in reason or "trace" in reason:
        return "trace_missing"
    if summary.get("native_evaluator_executed") is False:
        return "evaluator_failed"
    return "unknown"


def _index_record(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    return {
        "trajectory_id": payload["trajectory_id"],
        "task_id": payload["task_id"],
        "split": payload["split"],
        "generation_seed": payload["generation_seed"],
        "valid": bool(payload.get("valid")),
        "failure_reason": payload.get("failure_reason"),
        "path": str(path),
    }


def _nonzero_exit(value: Any) -> bool:
    if value is None:
        return False
    try:
        return int(value) != 0
    except (TypeError, ValueError):
        return True


def _task_sort_key(task_id: str) -> tuple[int, str]:
    return (0, f"{int(task_id):09d}") if task_id.isdigit() else (1, task_id)
