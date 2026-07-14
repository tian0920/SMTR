"""Quality audit for the real MARBLE database data pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def audit_real_database_mvp(
    *,
    dataset_manifest_path: Path,
    split_manifest_path: Path,
    trajectory_index_path: Path | None = None,
    memory_pool_path: Path | None = None,
    candidate_manifest_path: Path | None = None,
    paired_records_path: Path | None = None,
) -> dict[str, Any]:
    dataset = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    splits = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    trajectories = _trajectory_records(trajectory_index_path)
    memories = _jsonl(memory_pool_path)
    candidates = (
        json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
        if candidate_manifest_path and candidate_manifest_path.exists()
        else {"edges": []}
    )
    pairs = _jsonl(paired_records_path)
    edges = candidates.get("edges", [])
    split_counts = {
        split: sum(record["split"] == split for record in splits["records"])
        for split in ("train", "validation", "test")
    }
    valid_trajectories = [item for item in trajectories if item.get("valid")]
    invalid_trajectories = [item for item in trajectories if not item.get("valid")]
    valid_pairs = [item for item in pairs if item.get("paired_record_valid")]
    primary_layers = [_primary_failure_layer(item) for item in invalid_trajectories]
    report = {
        "dataset_task_count": dataset["total_tasks"],
        **{f"{name}_count": count for name, count in split_counts.items()},
        "trajectory_attempted": len(trajectories),
        "trajectory_valid": len(valid_trajectories),
        "trajectory_invalid": len(invalid_trajectories),
        "memory_generated": len(memories),
        "memory_accepted": len(memories),
        "memory_rejected": 0,
        "memory_duplicate_count": sum(bool(item.get("duplicate_cluster_id")) for item in memories),
        "recipient_task_count": len({edge["recipient_task_id"] for edge in edges}),
        "candidate_edge_count": len(edges),
        "self_pair_count": sum(
            edge["source_task_id"] == edge["recipient_task_id"] for edge in edges
        ),
        "same_group_pair_count": sum(
            edge["source_group_id"] == edge["recipient_group_id"] for edge in edges
        ),
        "cross_split_leakage_count": sum(edge.get("source_split") != "train" for edge in edges),
        "paired_attempted": len(pairs),
        "paired_valid": len(valid_pairs),
        "paired_invalid": len(pairs) - len(valid_pairs),
        "initial_state_mismatch_count": sum(
            not item.get("initial_state_match", False) for item in valid_pairs
        ),
        "memory_intervention_failure_count": sum(
            not item.get("memory_intervention_verified", False) for item in valid_pairs
        ),
        "preflight_failure_count": primary_layers.count("preflight_failure"),
        "engine_failure_count": sum(
            item.get("real_engine_executed") is False
            or _primary_failure_layer(item)
            in {"engine_execution_timeout", "engine_execution_failure"}
            for item in invalid_trajectories
        ),
        "engine_execution_timeout_count": primary_layers.count("engine_execution_timeout"),
        "engine_execution_failure_count": primary_layers.count("engine_execution_failure"),
        "raw_result_failure_count": sum(
            _raw_result_failed(item) for item in invalid_trajectories
        ),
        "raw_result_primary_failure_count": primary_layers.count("raw_result_failure"),
        "structured_trace_failure_count": primary_layers.count("structured_trace_failure"),
        "evaluator_failure_count": primary_layers.count("native_evaluator_failure"),
        "evaluator_skipped_count": sum(
            item.get("native_evaluator_executed") is False for item in invalid_trajectories
        ),
        "evaluator_skipped_due_to_upstream_failure_count": sum(
            item.get("native_evaluator_executed") is False
            and _primary_failure_layer(item)
            in {
                "preflight_failure",
                "engine_execution_timeout",
                "engine_execution_failure",
                "raw_result_failure",
                "structured_trace_failure",
            }
            for item in invalid_trajectories
        ),
        "cleanup_failure_count": sum(
            item.get("cleanup_succeeded") is False for item in invalid_trajectories
        ),
        "cleanup_primary_failure_count": primary_layers.count("cleanup_failure"),
        "provenance_failure_count": primary_layers.count("provenance_failure"),
        "unknown_invalid_reason_count": primary_layers.count("unknown"),
        "classified_primary_failures": sum(layer != "unknown" for layer in primary_layers),
        "unclassified_invalid_count": primary_layers.count("unknown"),
        "primary_failure_layer_counts": {
            layer: primary_layers.count(layer)
            for layer in (
                "preflight_failure",
                "engine_execution_timeout",
                "engine_execution_failure",
                "raw_result_failure",
                "structured_trace_failure",
                "native_evaluator_failure",
                "cleanup_failure",
                "provenance_failure",
                "unknown",
            )
        },
        "missing_provenance_count": 0,
    }
    report.update(
        {
            "READY_FOR_MARBLE_DATASET_SNAPSHOT": (
                dataset.get("dataset_sha256") is not None
                and report["train_count"] + report["validation_count"] + report["test_count"]
                == report["dataset_task_count"]
            ),
            "READY_FOR_MARBLE_TRAJECTORY_COLLECTION": len(valid_trajectories) >= 1,
            "READY_FOR_MARBLE_REAL_MEMORY_POOL": len(memories) >= 1,
            "READY_FOR_MARBLE_REAL_PAIRED_DATA": len(valid_pairs) >= 1,
            "READY_FOR_FORMAL_MARBLE_EXPERIMENT": False,
        }
    )
    return report


def _jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _trajectory_records(path: Path | None) -> list[dict[str, Any]]:
    rows = _jsonl(path)
    if path is None:
        return rows
    base = path.parent / "trajectories"
    records: list[dict[str, Any]] = []
    for row in rows:
        trajectory_id = row.get("trajectory_id")
        if not trajectory_id:
            records.append(row)
            continue
        run_dir = base / str(trajectory_id)
        trajectory = _json_file(run_dir / "trajectory.json")
        b0 = _json_file(run_dir / "b0_smoke.json")
        engine = _json_file(run_dir / "engine_process.json")
        merged = {**row, **trajectory}
        _merge_if_absent(
            merged,
            b0,
            (
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
            ),
        )
        engine_map = {
            "exit_code": "engine_exit_code",
            "timed_out": "engine_timed_out",
            "engine_timeout_seconds": "engine_timeout_seconds",
            "engine_timeout_source": "engine_timeout_source",
            "engine_duration_seconds": "engine_duration_seconds",
            "engine_termination_requested": "engine_termination_requested",
            "engine_termination_signal": "engine_termination_signal",
            "engine_termination_grace_period_seconds": (
                "engine_termination_grace_period_seconds"
            ),
            "engine_kill_escalated": "engine_kill_escalated",
            "last_observed_stage": "last_observed_stage",
            "last_observed_stage_parser_version": "last_observed_stage_parser_version",
            "working_directory": "engine_working_directory",
            "config_path": "engine_config_path",
            "selected_python": "selected_python",
        }
        for source, target in engine_map.items():
            if target not in merged and source in engine:
                merged[target] = engine[source]
        _merge_if_absent(
            merged,
            engine,
            (
                "real_engine_executed",
                "raw_result_path",
                "raw_result_exists",
                "raw_result_nonempty",
                "raw_result_fresh",
                "raw_result_parseable",
                "raw_result_identity_verified",
                "cleanup_exit_code",
                "cleanup_succeeded",
                "cleanup_failure_reason",
            ),
        )
        records.append(merged)
    return records


def _json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _merge_if_absent(target: dict[str, Any], source: dict[str, Any], keys: tuple[str, ...]) -> None:
    for key in keys:
        if key not in target and key in source:
            target[key] = source[key]


def _raw_result_failed(item: dict[str, Any]) -> bool:
    reason = str(item.get("invalid_reason", "")).lower()
    return (
        "raw result" in reason
        or item.get("raw_result_path") in (None, "")
        or item.get("raw_result_exists") is False
        or item.get("raw_result_nonempty") is False
        or item.get("raw_result_fresh") is False
        or item.get("raw_result_parseable") is False
    )


def _primary_failure_layer(item: dict[str, Any]) -> str:
    explicit = item.get("failure_layer")
    if explicit:
        return str(explicit)
    reason = str(item.get("invalid_reason", "")).lower()
    if item.get("runtime_preflight_ready") is False:
        return "preflight_failure"
    if item.get("engine_timed_out") is True:
        return "engine_execution_timeout"
    if _nonzero_exit(item.get("engine_exit_code")):
        return "engine_execution_failure"
    if _raw_result_failed(item):
        return "raw_result_failure"
    if "structured" in reason or "trace" in reason:
        return "structured_trace_failure"
    if item.get("native_evaluator_executed") is False or "evaluator" in reason:
        return "native_evaluator_failure"
    if item.get("cleanup_succeeded") is False or "cleanup" in reason:
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
