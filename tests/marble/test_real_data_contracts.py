from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from smtr.marble.dataset import build_marble_dataset_manifest
from smtr.marble.real_audit import audit_real_database_mvp
from smtr.marble.real_data import (
    CandidateEdge,
    ProceduralRoutingCard,
    ProcedurePayload,
    RealDatabaseTrajectory,
    RealPairedRecord,
    RealProceduralMemory,
    build_cross_task_candidates,
    extract_procedural_memories,
)
from smtr.marble.real_workflows import _invalid_trajectory_payload, _normalize_smoke
from smtr.marble.splits import create_split_manifest, validate_split_manifest

MARBLE_ROOT = Path("/home/ecs-user/MARBLE")


def _trajectory(**overrides) -> RealDatabaseTrajectory:
    values = {
        "trajectory_id": "traj-1",
        "task_id": "1",
        "split": "train",
        "generation_seed": 0,
        "model_id": "real-model",
        "dataset_manifest_sha256": "d",
        "split_manifest_sha256": "s",
        "marble_commit": "m",
        "smtr_commit": "c",
        "initial_database_fingerprint": {"schema": "x"},
        "initial_logical_database_digest": "logical",
        "agent_identities": [{"agent_id": "agent1"}],
        "agent_messages": [{"role": "assistant", "content": "inspect"}],
        "agent_actions": [{"name": "query_db"}],
        "tool_calls": [{"name": "query_db"}],
        "sql_statements": ["SELECT 1"],
        "observations": [{"rows": 1}],
        "errors": [],
        "final_answer": "diagnosis",
        "raw_result_path": "/tmp/raw.json",
        "raw_result_sha256": "raw",
        "native_evaluator_executed": True,
        "native_evaluator_output": {"score": 1},
        "score": 1.0,
        "task_success": True,
        "real_engine_executed": True,
        "cleanup_succeeded": True,
        "environment_valid": True,
        "raw_result_exists": True,
        "raw_result_nonempty": True,
        "raw_result_fresh": True,
        "raw_result_parseable": True,
        "started_at": "2026-01-01T00:00:00Z",
        "completed_at": "2026-01-01T00:01:00Z",
        "stdout_log_path": "/tmp/stdout.log",
        "stderr_log_path": "/tmp/stderr.log",
        "workspace_path": "/tmp/workspace",
    }
    values.update(overrides)
    return RealDatabaseTrajectory.model_validate(values)


def _memory(task_id: str = "1", group_id: str = "g1") -> RealProceduralMemory:
    return RealProceduralMemory(
        memory_id=f"m-{task_id}",
        source_task_id=task_id,
        source_trajectory_id=f"t-{task_id}",
        source_split="train",
        source_group_id=group_id,
        routing_card=ProceduralRoutingCard(
            goal_summary="database performance diagnosis",
            task_tags=["database"],
            precondition_summary="monitoring available",
            expected_effect="better diagnosis",
            known_risks=["query cost"],
        ),
        procedure_payload=ProcedurePayload(
            applicable_context="database incident",
            ordered_steps=["inspect monitoring evidence"],
            failure_signals=["timeout"],
            recovery_actions=["narrow query"],
        ),
        extractor_type="rule",
        extractor_version="1",
        extraction_model="none",
        extraction_seed=0,
        extraction_prompt_hash="p",
        dataset_manifest_sha256="d",
        split_manifest_sha256="s",
        source_trajectory_sha256="t",
        created_at="now",
    )


def test_real_database_manifest_is_deterministic_and_complete(tmp_path: Path) -> None:
    left = build_marble_dataset_manifest(marble_root=MARBLE_ROOT, scenarios={"database"})
    right = build_marble_dataset_manifest(marble_root=MARBLE_ROOT, scenarios={"database"})
    assert left == right
    assert left.total_tasks == 100
    assert len(left.ordered_task_ids) == len(set(left.ordered_task_ids)) == 100
    assert left.dataset_sha256


def test_database_hash_changes_and_missing_required_field_is_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "multiagentbench/database/database_main.jsonl"
    path.parent.mkdir(parents=True)
    base = {
        "scenario": "database",
        "task_id": 1,
        "task": {"content": "diagnose"},
        "environment": {"init_sql": "CREATE TABLE sample(id INT);"},
    }
    path.write_text(json.dumps(base) + "\n", encoding="utf-8")
    first = build_marble_dataset_manifest(
        marble_root=tmp_path, scenarios={"database"}
    )
    path.write_text(json.dumps({**base, "task_id": 2}) + "\n", encoding="utf-8")
    second = build_marble_dataset_manifest(
        marble_root=tmp_path, scenarios={"database"}
    )
    assert first.dataset_sha256 != second.dataset_sha256
    path.write_text(
        json.dumps({"scenario": "database", "task_id": 3, "task": {"content": "x"}})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="init_sql"):
        build_marble_dataset_manifest(marble_root=tmp_path, scenarios={"database"})


def test_database_split_is_disjoint_grouped_and_complete(tmp_path: Path) -> None:
    dataset = build_marble_dataset_manifest(marble_root=MARBLE_ROOT, scenarios={"database"})
    path = tmp_path / "dataset.json"
    path.write_text(dataset.model_dump_json(), encoding="utf-8")
    split = create_split_manifest(dataset_manifest_path=path, seed=7)
    validate_split_manifest(split, expected_task_ids=set(dataset.ordered_task_ids))
    assert split.split_counts == {"train": 70, "validation": 10, "test": 20}


@pytest.mark.parametrize(
    "field",
    ["real_engine_executed", "native_evaluator_executed", "cleanup_succeeded"],
)
def test_invalid_real_trajectory_is_rejected(field: str) -> None:
    with pytest.raises(ValidationError):
        _trajectory(**{field: False})


def test_memory_extraction_rejects_non_train_and_separates_card_payload() -> None:
    with pytest.raises(ValueError, match="train"):
        extract_procedural_memories(
            [_trajectory(split="validation")], group_by_task={"1": "g1"}, created_at="now"
        )
    memories = extract_procedural_memories(
        [_trajectory()], group_by_task={"1": "g1"}, created_at="now"
    )
    dumped = memories[0].routing_card.model_dump()
    assert "ordered_steps" not in dumped
    assert "final_answer" not in json.dumps(memories[0].model_dump())


def test_candidates_reject_self_and_same_group_and_are_deterministic() -> None:
    with pytest.raises(ValidationError, match="self-pair"):
        CandidateEdge(
            recipient_task_id="1",
            recipient_split="validation",
            recipient_group_id="g2",
            memory_id="m",
            source_task_id="1",
            source_split="train",
            source_group_id="g1",
            retrieval_score=1,
        )
    memories = [_memory("1", "g1"), _memory("2", "g2")]
    kwargs = {
        "memories": memories,
        "recipients": [{"task_id": "3", "group_id": "g3", "instruction": "database diagnosis"}],
        "dataset_manifest_sha256": "d",
        "split_manifest_sha256": "s",
        "created_at": "fixed",
    }
    assert build_cross_task_candidates(**kwargs) == build_cross_task_candidates(**kwargs)


def test_pair_label_and_branch_evidence_are_enforced() -> None:
    memory = _memory()
    common = dict(
        pair_id="p",
        recipient_task_id="2",
        recipient_split="validation",
        memory_id=memory.memory_id,
        source_task_id="1",
        source_trajectory_id="t1",
        source_split="train",
        generation_seed=0,
        branch_order="share_then_withhold",
        withhold_trajectory_id="w",
        share_trajectory_id="s",
        y_withhold=0,
        y_share=1,
        tau=1,
        withhold_score=0.0,
        share_score=1.0,
        withhold_task_success=False,
        share_task_success=True,
        initial_state_match=True,
        initial_logical_digest_match=True,
        memory_intervention_verified=True,
        withhold_real_engine_executed=True,
        share_real_engine_executed=True,
        withhold_native_evaluator_executed=True,
        share_native_evaluator_executed=True,
        withhold_cleanup_succeeded=True,
        share_cleanup_succeeded=True,
        routing_card_snapshot=memory.routing_card,
        memory_payload_sha256="payload",
        candidate_manifest_sha256="candidate",
        dataset_manifest_sha256="dataset",
        split_manifest_sha256="split",
        paired_record_valid=True,
    )
    assert RealPairedRecord(**common).tau == 1
    with pytest.raises(ValidationError):
        RealPairedRecord(**{**common, "initial_state_match": False})


def test_collector_normalizes_raw_result_path_from_b0_summary(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        json.dumps(
            {
                "agents": [{"agent_id": "agent1"}],
                "actions": [{"name": "query_db"}],
                "tool_calls": [{"name": "query_db", "sql": "SELECT 1"}],
                "final_answer": "diagnosis",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps({"marble_commit": "m", "smtr_commit": "s"}),
        encoding="utf-8",
    )
    split_path = tmp_path / "split.json"
    split_path.write_text("{}", encoding="utf-8")
    record = _normalize_smoke(
        summary={
            "raw_result_path": str(raw),
            "raw_result_exists": True,
            "raw_result_nonempty": True,
            "raw_result_fresh": True,
            "raw_result_parseable": True,
            "native_evaluator_executed": True,
            "outcome": {"score": 1.0, "success": True},
            "real_engine_executed": True,
            "cleanup_succeeded": True,
            "environment_valid": True,
            "initial_logical_fingerprint": {"schema": "x"},
            "initial_state_digest": "initial",
        },
        trajectory_id="traj",
        task_id="1",
        split="train",
        generation_seed=0,
        dataset={"marble_commit": "m", "smtr_commit": "s"},
        dataset_manifest_path=dataset_path,
        split_manifest_path=split_path,
        run_dir=tmp_path,
    )

    assert record.raw_result_path == str(raw)
    assert record.real_engine_executed is True
    assert record.native_evaluator_executed is True


def test_audit_deep_reads_shallow_index_and_classifies_invalid_trajectories(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps({"total_tasks": 3, "dataset_sha256": "dataset"}),
        encoding="utf-8",
    )
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps(
            {
                "records": [
                    {"task_id": "1", "split": "train"},
                    {"task_id": "2", "split": "validation"},
                    {"task_id": "3", "split": "test"},
                ]
            }
        ),
        encoding="utf-8",
    )
    index = tmp_path / "trajectory_index.jsonl"
    rows = [
        {
            "trajectory_id": f"traj-{task_id}",
            "task_id": str(task_id),
            "split": "train",
            "generation_seed": 0,
            "valid": False,
            "invalid_reason": "raw result file missing",
        }
        for task_id in (1, 2, 3)
    ]
    index.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    for row in rows:
        run_dir = tmp_path / "trajectories" / row["trajectory_id"]
        run_dir.mkdir(parents=True)
        (run_dir / "trajectory.json").write_text(json.dumps(row), encoding="utf-8")
        (run_dir / "b0_smoke.json").write_text(
            json.dumps(
                {
                    "real_engine_executed": False,
                    "native_evaluator_executed": False,
                    "raw_result_path": str(run_dir / "workspace/marble_output.jsonl"),
                    "raw_result_exists": False,
                    "raw_result_fresh": False,
                    "raw_result_parseable": False,
                    "cleanup_succeeded": True,
                    "environment_valid": False,
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "engine_process.json").write_text(
            json.dumps(
                {
                    "exit_code": -9,
                    "timed_out": True,
                    "engine_timeout_seconds": 3,
                    "engine_timeout_source": "cli",
                    "engine_duration_seconds": 3.01,
                    "engine_termination_requested": True,
                    "engine_termination_signal": "SIGKILL",
                    "engine_kill_escalated": True,
                    "last_observed_stage": "docker_database_started",
                    "real_engine_executed": False,
                    "raw_result_path": str(run_dir / "workspace/marble_output.jsonl"),
                    "raw_result_exists": False,
                    "raw_result_fresh": False,
                    "raw_result_parseable": False,
                    "cleanup_succeeded": True,
                }
            ),
            encoding="utf-8",
        )

    report = audit_real_database_mvp(
        dataset_manifest_path=dataset_path,
        split_manifest_path=split_path,
        trajectory_index_path=index,
    )

    assert report["trajectory_invalid"] == 3
    assert report["classified_primary_failures"] == 3
    assert report["unclassified_invalid_count"] == 0
    assert report["engine_execution_timeout_count"] == 3
    assert report["engine_execution_failure_count"] == 0
    assert report["engine_failure_count"] == 3
    assert report["raw_result_failure_count"] == 3
    assert report["evaluator_skipped_due_to_upstream_failure_count"] == 3


def test_timeout_invalid_payload_preserves_timeout_as_primary_failure() -> None:
    payload = _invalid_trajectory_payload(
        trajectory_id="traj",
        task_id="19",
        split="train",
        generation_seed=0,
        invalid_reason="raw result file missing",
        summary={
            "engine_timed_out": True,
            "engine_exit_code": -9,
            "engine_timeout_seconds": 3,
            "engine_timeout_source": "cli",
            "engine_duration_seconds": 3.0,
            "engine_termination_requested": True,
            "engine_termination_signal": "SIGKILL",
            "engine_kill_escalated": True,
            "real_engine_executed": False,
            "raw_result_path": "/tmp/raw.jsonl",
            "raw_result_exists": False,
            "raw_result_fresh": False,
            "raw_result_parseable": False,
            "native_evaluator_executed": False,
            "cleanup_succeeded": True,
            "last_observed_stage": "docker_database_started",
        },
    )

    assert payload["failure_layer"] == "engine_execution_timeout"
    assert payload["raw_result_exists"] is False
    assert payload["engine_timeout_seconds"] == 3
    assert payload["last_observed_stage"] == "docker_database_started"
