from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from smtr.marble.dataset import build_marble_dataset_manifest
from smtr.marble.real_data import (
    CandidateSet,
    ProceduralRoutingCard,
    ProcedurePayload,
    RealDatabaseTrajectory,
    RealPairedRecord,
    RealProceduralMemory,
    build_cross_task_candidates,
    extract_procedural_memories,
)
from smtr.marble.real_workflows import _index_record, _invalid_trajectory_payload, _normalize_smoke
from smtr.marble.splits import create_split_manifest, validate_split_manifest

MARBLE_ROOT = Path("/home/ecs-user/MARBLE")


def _trajectory(**overrides) -> RealDatabaseTrajectory:
    values = {
        "trajectory_id": "traj-1",
        "task_id": "1",
        "split": "train",
        "generation_seed": 0,
        "model_id": "real-model",
        "source_dataset_version": "dataset-v1",
        "messages": [{"role": "assistant", "content": "inspect"}],
        "actions": [{"name": "query_db"}],
        "tool_calls": [{"name": "query_db"}],
        "sql_statements": ["SELECT 1"],
        "observations": [{"rows": 1}],
        "errors": [],
        "final_answer": "diagnosis",
        "score": 1.0,
        "task_success": True,
        "valid": True,
        "failure_reason": None,
    }
    values.update(overrides)
    return RealDatabaseTrajectory.model_validate(values)


def _memory(task_id: str = "1") -> RealProceduralMemory:
    return RealProceduralMemory(
        memory_id=f"m-{task_id}",
        source_task_id=task_id,
        source_trajectory_id=f"t-{task_id}",
        routing_card=ProceduralRoutingCard(
            goal_summary="database performance diagnosis",
            task_tags=["database"],
            precondition_summary="monitoring available",
            expected_effect="better diagnosis",
            known_risks=["query cost"],
        ),
        procedure_payload=ProcedurePayload(
            preconditions=["database incident"],
            steps=["inspect monitoring evidence"],
            failure_signals=["timeout"],
            recovery_actions=["narrow query"],
        ),
    )


def test_real_database_manifest_is_deterministic_and_complete() -> None:
    left = build_marble_dataset_manifest(marble_root=MARBLE_ROOT, scenarios={"database"})
    right = build_marble_dataset_manifest(marble_root=MARBLE_ROOT, scenarios={"database"})
    assert left == right
    assert left.total_tasks == 100
    assert len(left.ordered_task_ids) == len(set(left.ordered_task_ids)) == 100
    assert left.dataset_sha256


def test_database_split_is_disjoint_grouped_and_complete(tmp_path: Path) -> None:
    dataset = build_marble_dataset_manifest(marble_root=MARBLE_ROOT, scenarios={"database"})
    path = tmp_path / "dataset.json"
    path.write_text(dataset.model_dump_json(), encoding="utf-8")
    split = create_split_manifest(dataset_manifest_path=path, seed=7)
    validate_split_manifest(split, expected_task_ids=set(dataset.ordered_task_ids))
    assert split.split_counts == {"train": 70, "validation": 10, "test": 20}


def test_valid_trajectory_requires_trace_and_native_outcome() -> None:
    with pytest.raises(ValidationError):
        _trajectory(actions=[], tool_calls=[], sql_statements=[])
    with pytest.raises(ValidationError):
        _trajectory(score=None)
    invalid = _trajectory(
        valid=False,
        failure_reason="engine_timeout",
        score=None,
        task_success=None,
    )
    assert invalid.failure_reason == "engine_timeout"


def test_memory_extraction_reads_valid_train_only_and_separates_card_payload() -> None:
    with pytest.raises(ValueError, match="train"):
        extract_procedural_memories(
            [_trajectory(split="validation")], group_by_task={"1": "g1"}, created_at="now"
        )
    memories = extract_procedural_memories(
        [_trajectory()], group_by_task={"1": "g1"}, created_at="now"
    )
    assert "steps" not in memories[0].routing_card.model_dump()
    dumped = json.dumps(memories[0].model_dump(mode="json")).lower()
    assert "final_answer" not in dumped
    assert "y_withhold" not in dumped


def test_candidate_manifest_is_grouped_and_does_not_store_pair_labels() -> None:
    with pytest.raises(ValidationError, match="length"):
        CandidateSet(recipient_task_id="2", candidate_memory_ids=["m1"], retrieval_scores=[])
    manifest = build_cross_task_candidates(
        memories=[_memory("1"), _memory("2")],
        recipients=[{"task_id": "3", "group_id": "g3", "instruction": "database diagnosis"}],
        group_by_task={"1": "g1", "2": "g2", "3": "g3"},
        top_k=2,
    )
    assert manifest.candidates[0].candidate_memory_ids == ["m-1", "m-2"]
    assert "Y_share" not in manifest.model_dump_json()


def test_candidate_builder_rejects_self_and_same_group_without_storing_group() -> None:
    manifest = build_cross_task_candidates(
        memories=[_memory("1"), _memory("2")],
        recipients=[{"task_id": "1", "group_id": "g2", "instruction": "database diagnosis"}],
        group_by_task={"1": "g1", "2": "g2"},
        top_k=2,
    )
    assert manifest.excluded_self_count == 1
    assert manifest.excluded_group_count == 1
    assert manifest.candidates[0].candidate_memory_ids == []


def test_pair_schema_enforces_initial_state_intervention_and_tau() -> None:
    common = dict(
        pair_id="p",
        recipient_task_id="2",
        memory_id="m-1",
        source_task_id="1",
        generation_seed=0,
        Y_withhold=0,
        Y_share=1,
        tau=1,
        withhold_score=0.0,
        share_score=1.0,
        initial_state_match=True,
        memory_intervention_verified=True,
        valid=True,
        failure_reason=None,
    )
    assert RealPairedRecord(**common).tau == 1
    with pytest.raises(ValidationError, match="tau"):
        RealPairedRecord(**{**common, "tau": 0})
    with pytest.raises(ValidationError, match="state match"):
        RealPairedRecord(**{**common, "initial_state_match": False})


def test_collector_normalizes_to_minimal_trajectory_schema(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        json.dumps(
            {
                "actions": [{"name": "query_db"}],
                "tool_calls": [{"name": "query_db", "sql": "SELECT 1"}],
                "final_answer": "diagnosis",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    record = _normalize_smoke(
        summary={
            "raw_result_path": str(raw),
            "raw_result_exists": True,
            "raw_result_nonempty": True,
            "raw_result_fresh": True,
            "raw_result_parseable": True,
            "real_engine_executed": True,
            "native_evaluator_executed": True,
            "outcome": {"score": 1.0, "success": True},
            "model_id": "model",
        },
        trajectory_id="traj",
        task_id="1",
        split="train",
        generation_seed=0,
        dataset={"dataset_sha256": "dataset"},
    )

    payload = record.model_dump(mode="json")
    assert payload["valid"] is True
    assert payload["actions"] == [{"name": "query_db"}]
    assert "raw_result_path" not in payload
    assert "engine_timeout_source" not in payload


def test_invalid_trajectory_and_index_are_minimal(tmp_path: Path) -> None:
    payload = _invalid_trajectory_payload(
        trajectory_id="traj",
        task_id="19",
        split="train",
        generation_seed=0,
        summary={"engine_timed_out": True, "engine_exit_code": -9},
        failure_reason="engine_timeout",
    )
    index = _index_record(payload, tmp_path / "trajectories/traj/trajectory.json")

    assert payload["valid"] is False
    assert payload["failure_reason"] == "engine_timeout"
    assert set(index) == {
        "trajectory_id",
        "task_id",
        "split",
        "generation_seed",
        "valid",
        "failure_reason",
        "path",
    }
