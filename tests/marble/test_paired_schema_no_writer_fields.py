"""Paired schema must not contain any writer_* fields (清单 Writer-Agnostic §19).

Asserts that the serialized formal paired record does not carry any
legacy writer identity fields — not even as empty strings.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from smtr.marble.real_pairs import paired_result_to_record


def _edge() -> dict:
    return {
        "edge_id": "e1",
        "task_id": "t1",
        "receiver_agent_id": "r1",
        "receiver_role": "executor",
        "receiver_capabilities": ["sql"],
        "receiver_tool_names": ["psql"],
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
    pr.paired_label = "positive_transfer"
    pr.paired_record_valid = True
    pr.invalid_reason = None
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


FORBIDDEN_WRITER_FIELDS = {
    "writer_agent_id",
    "writer_role",
    "writer_capabilities",
    "writer_tool_names",
    "writer_model_name",
    "writer_receiver_match_type",
    "writer_receiver_score",
}


def test_paired_record_contains_no_writer_fields():
    record = paired_result_to_record(
        pair_result=_pair_result(),
        edge=_edge(),
        seed=0,
    )
    for field in FORBIDDEN_WRITER_FIELDS:
        assert field not in record, (
            f"forbidden writer field '{field}' found in paired record"
        )
