"""Core-valid pair filtering (清单 Test 10 / P0-16 / P0-17).

A paired record is core-valid only when share/withhold hold constant the
target task, receiver, generation seed, initial environment state and
agent/tool configuration, the only treatment difference is the candidate
memory exposure, both branches carry MARBLE native team outcomes, and the
runtime visibility contract holds (target receiver sees the memory only in
the share branch; non-target agents never see the payload).

Any single violation must invalidate the pair, and invalid pairs must never
enter the critic training set.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from smtr.marble.core_validity import (
    core_validity_exclusion_reasons,
    filter_core_paired_records,
    is_core_valid_pair,
)
from smtr.router.transfer_features import load_paired_records_with_metadata


def _branch(team_success: bool = True) -> dict:
    return {
        "team_success": team_success,
        "local_success": None,
        "environment_valid": True,
        "native_evaluator_executed": True,
        "real_engine_executed": True,
        "runtime_visibility_verified": True,
        "cleanup_succeeded": True,
    }


def _valid_record() -> dict:
    """A fully core-valid paired record covering every enforced field."""
    return {
        "task_id": "t1",
        "receiver_agent_id": "r1",
        "candidate_memory_id": "m1",
        "generation_seed": 3,
        "edge_id": "t1|r1|m1",
        "share": _branch(team_success=True),
        "withhold": _branch(team_success=False),
        "share_generation_seed": 3,
        "withhold_generation_seed": 3,
        "share_receiver_agent_id": "r1",
        "withhold_receiver_agent_id": "r1",
        "non_target_payload_leakage": False,
        "valid": True,
        "invalid_reason": None,
        "digests": {
            "share_task_digest": "d-task",
            "withhold_task_digest": "d-task",
            "share_agent_config_digest": "d-agent",
            "withhold_agent_config_digest": "d-agent",
            "share_tool_config_digest": "d-tool",
            "withhold_tool_config_digest": "d-tool",
            "share_initial_digest": "d-init",
            "withhold_initial_digest": "d-init",
            "share_initial_logical_digest": "d-logic",
            "withhold_initial_logical_digest": "d-logic",
        },
    }


def _mutate(mutator) -> dict:
    rec = copy.deepcopy(_valid_record())
    mutator(rec)
    return rec


# Each of the eight failure conditions must invalidate the pair on its own.
FAILURE_CASES = [
    (
        "initial_state_mismatch",
        lambda r: r["digests"].update({"withhold_initial_digest": "d-other"}),
        "mismatched_initial_state_digest",
    ),
    (
        "receiver_mismatch",
        lambda r: r.update({"withhold_receiver_agent_id": "r2"}),
        "mismatched_receiver",
    ),
    (
        "seed_mismatch",
        lambda r: r.update({"share_generation_seed": 99}),
        "mismatched_generation_seed",
    ),
    (
        "tool_config_mismatch",
        lambda r: r["digests"].update({"share_tool_config_digest": "d-other"}),
        "mismatched_tool_config_digest",
    ),
    (
        "native_outcome_missing",
        lambda r: r["share"].update({"native_evaluator_executed": False}),
        "share_native_outcome_missing",
    ),
    (
        "share_visibility_failure",
        lambda r: r["share"].update({"runtime_visibility_verified": False}),
        "share_visibility_failure",
    ),
    (
        "withhold_visibility_failure",
        lambda r: r["withhold"].update({"runtime_visibility_verified": False}),
        "withhold_visibility_failure",
    ),
    (
        "non_target_payload_leakage",
        lambda r: r.update({"non_target_payload_leakage": True}),
        "non_target_payload_leakage",
    ),
]


class TestCoreValidPair:
    def test_valid_record_passes(self):
        assert is_core_valid_pair(_valid_record())
        assert core_validity_exclusion_reasons(_valid_record()) == []

    @pytest.mark.parametrize(
        ("name", "mutator", "expected_reason"),
        FAILURE_CASES,
        ids=[case[0] for case in FAILURE_CASES],
    )
    def test_each_failure_condition_invalidates_pair(
        self, name, mutator, expected_reason
    ):
        rec = _mutate(mutator)
        assert not is_core_valid_pair(rec), name
        assert expected_reason in core_validity_exclusion_reasons(rec)

    def test_invalid_pairs_are_excluded_with_rate_and_reasons(self):
        invalid_visibility = _mutate(
            lambda r: r["withhold"].update({"runtime_visibility_verified": False})
        )
        invalid_leakage = _mutate(lambda r: r.update({"non_target_payload_leakage": True}))
        result = filter_core_paired_records(
            [_valid_record(), invalid_visibility, invalid_leakage]
        )
        assert result["valid_paired_records"] == 1
        assert result["excluded_paired_records"] == 2
        assert result["invalid_pair_rate"] == pytest.approx(2 / 3)
        assert result["exclusion_reasons"]["withhold_visibility_failure"] == 1
        assert result["exclusion_reasons"]["non_target_payload_leakage"] == 1
        # Invalid pairs are excluded, never silently relabelled as failures.
        assert all(is_core_valid_pair(rec) for rec in result["valid_records"])

    def test_invalid_pairs_never_enter_training_set(self, tmp_path: Path):
        valid = _valid_record()
        invalid = _mutate(
            lambda r: r["share"].update({"runtime_visibility_verified": False})
        )
        records_path = tmp_path / "paired_records.jsonl"
        records_path.write_text(
            "\n".join(json.dumps(rec) for rec in (valid, invalid)) + "\n",
            encoding="utf-8",
        )
        pool_path = tmp_path / "memory_pool.json"
        pool_path.write_text(
            json.dumps({
                "memory_id": "m1",
                "payload": {"procedure": "step"},
                "routing_card": {
                    "goal_summary": "goal",
                    "task_tags": [],
                    "required_tools": [],
                    "required_capabilities": [],
                    "execution_role_tags": [],
                    "environment_constraints": [],
                    "precondition_tags": [],
                    "procedure_type": "diagnostic",
                    "procedure_length_bucket": "short",
                    "read_write_scope": "read",
                    "evidence_count": 1,
                },
            })
            + "\n",
            encoding="utf-8",
        )
        loaded = load_paired_records_with_metadata(records_path, pool_path)
        assert len(loaded) == 1
        _, _, kept_record = loaded[0]
        assert kept_record["edge_id"] == "t1|r1|m1"
        assert is_core_valid_pair(kept_record)
