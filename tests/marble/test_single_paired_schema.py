"""Single paired schema version (清单 Writer-Agnostic §20).

All formal paired records use the unified ``marble_candidate_pair_v4``
schema. No v2, v3, or legacy variant exists.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from smtr.marble.real_pairs import (
    PAIRED_SCHEMA_VERSION,
    paired_result_to_record,
)


def _edge() -> dict:
    return {
        "edge_id": "e1",
        "task_id": "t1",
        "receiver_agent_id": "r1",
        "receiver_role": "executor",
        "receiver_capabilities": ["sql"],
        "candidate_memory_id": "m1",
        "candidate_rank": 1,
        "candidate_score": 0.9,
        "memory_source_agent_id": "w1",
        "memory_source_task_id": "train_t1",
        "memory_source_trajectory_id": "traj_1",
        "memory_source_split": "train",
    }


def _pair_result() -> MagicMock:
    pr = MagicMock()
    pr.task_id = "t1"
    pr.scenario = "database"
    pr.candidate_memory_id = "m1"
    for branch in (pr.share, pr.withhold):
        branch.outcome.success = True
        branch.outcome.environment_valid = True
        branch.outcome.native_evaluator_executed = True
        branch.real_engine_executed = True
        branch.runtime_visibility_verified = True
        branch.outcome.evaluator_type = "native"
        branch.outcome.status = "success"
        branch.outcome.native_evaluator_status = "passed"
        branch.initial_digest = "d1"
        branch.result_digest = "rd1"
    return pr


def test_schema_version_is_v4():
    assert PAIRED_SCHEMA_VERSION == "marble_candidate_pair_v4"


def test_serialized_record_uses_v4():
    record = paired_result_to_record(
        pair_result=_pair_result(),
        edge=_edge(),
        seed=0,
    )
    assert record["schema_version"] == "marble_candidate_pair_v4"


def test_no_legacy_schema_variants():
    forbidden = {"marble_candidate_pair_v2", "marble_candidate_pair_v3"}
    record = paired_result_to_record(
        pair_result=_pair_result(),
        edge=_edge(),
        seed=0,
    )
    assert record["schema_version"] not in forbidden
