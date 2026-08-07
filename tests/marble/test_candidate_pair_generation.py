"""Tests for candidate-level pair generation wiring.

The generator now uses the shared-control protocol (清单 Shared-Control
第1/5章): one ``run_no_memory_control`` per (task, receiver, seed) group
and one ``run_candidate_share`` per treatment edge, paired by
``assemble_shared_control_pair``. The legacy ``run_pair`` path must no
longer be used.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from tests.marble._shared_control_harness import (
    AGENT_CONFIG_DIGEST,
    CONTROL_RAW_RESULT_DIGEST,
    INITIAL_DIGEST,
    TASK_DIGEST,
    TOOL_CONFIG_DIGEST,
    run_generate,
)


def test_each_candidate_seed_calls_run_candidate_share_once(tmp_path):
    """Each candidate/seed runs one share; each group/seed runs one control."""
    out = run_generate(
        tmp_path,
        entries=[{
            "task_id": "t1",
            "receiver_agent_id": "r1",
            "memory_ids": ["m1", "m2"],
        }],
        seeds=[0, 1, 2],
    )
    runner = out["runner"]
    # 2 candidates x 3 seeds = 6 share executions
    assert len(runner.share_calls) == 6
    # 1 control group x 3 seeds = 3 shared controls
    assert len(runner.control_calls) == 3
    assert out["result"]["attempted"] == 6


def test_never_calls_legacy_run_pair(tmp_path):
    """The legacy run_pair / run_single_branch APIs must never be used."""
    out = run_generate(
        tmp_path,
        entries=[{
            "task_id": "t1",
            "receiver_agent_id": "r1",
            "memory_ids": ["m1", "m2"],
        }],
        seeds=[0, 1, 2],
    )
    runner = out["runner"]
    # FakeSharedControlRunner.run_pair raises AssertionError when called,
    # so a successful run already proves it was never used.
    assert len(runner.control_calls) + len(runner.share_calls) > 0
    assert not hasattr(runner, "run_single_branch")


def test_record_digests_come_from_paired_branch_result(tmp_path):
    """Record digests must come from the branch results, not fabricated."""
    out = run_generate(
        tmp_path,
        entries=[{
            "task_id": "t1",
            "receiver_agent_id": "r1",
            "memory_ids": ["m1", "m2"],
        }],
        seeds=[0, 1, 2],
        limit_pairs=1,
    )
    records = out["records"]
    # 1 edge x 3 seeds = 3 replicate records
    assert len(records) == 3
    replicate_ids = {rec["replicate_id"] for rec in records}
    assert len(replicate_ids) == 3

    rec = records[0]
    assert rec["schema_version"] == "marble_candidate_pair_v4"
    assert rec["control_group_id"]
    assert rec["control_reused"] is True

    # Share digests must match the mocked share audit values.
    assert rec["digests"]["share_initial_digest"] == INITIAL_DIGEST
    assert rec["digests"]["share_task_digest"] == TASK_DIGEST
    assert rec["digests"]["share_tool_config_digest"] == TOOL_CONFIG_DIGEST
    assert rec["digests"]["share_agent_config_digest"] == AGENT_CONFIG_DIGEST

    # Withhold digests come from the shared control audit, and the
    # control-specific digests reference the same execution.
    assert rec["digests"]["withhold_initial_digest"] == INITIAL_DIGEST
    assert rec["digests"]["control_initial_digest"] == INITIAL_DIGEST
    assert rec["digests"]["control_task_digest"] == TASK_DIGEST
    assert rec["digests"]["control_tool_config_digest"] == TOOL_CONFIG_DIGEST
    assert (
        rec["digests"]["control_raw_result_digest"] == CONTROL_RAW_RESULT_DIGEST
    )
    assert rec["control_raw_result_digest"] == CONTROL_RAW_RESULT_DIGEST


def test_invalid_pair_not_used_by_training_loader(tmp_path):
    """Invalid pairs must not be loaded for critic training."""
    from smtr.marble.real_pairs import paired_result_to_record

    mock_outcome_share = MagicMock()
    mock_outcome_share.success = True
    mock_outcome_share.environment_valid = True
    mock_outcome_share.native_evaluator_executed = True

    mock_outcome_withhold = MagicMock()
    mock_outcome_withhold.success = False
    mock_outcome_withhold.environment_valid = True
    mock_outcome_withhold.native_evaluator_executed = True

    mock_share = MagicMock()
    mock_share.outcome = mock_outcome_share
    mock_share.real_engine_executed = True
    mock_share.runtime_visibility_verified = True
    mock_share.cleanup_succeeded = True
    mock_share.initial_digest = "digest_abc"
    mock_share.initial_logical_fingerprint = {"combined_digest": "logical_abc"}
    mock_share.agent_config_digest = "agent_digest"
    mock_share.task_digest = "task_digest"
    mock_share.tool_config_digest = "tool_digest"

    mock_withhold = MagicMock()
    mock_withhold.outcome = mock_outcome_withhold
    mock_withhold.real_engine_executed = True
    mock_withhold.runtime_visibility_verified = True
    mock_withhold.cleanup_succeeded = True
    mock_withhold.initial_digest = "digest_abc"
    mock_withhold.initial_logical_fingerprint = {"combined_digest": "logical_abc"}
    mock_withhold.agent_config_digest = "agent_digest"
    mock_withhold.task_digest = "task_digest"
    mock_withhold.tool_config_digest = "tool_digest"
    mock_withhold.raw_result_digest = "raw_digest"

    mock_result = MagicMock()
    mock_result.scenario = "database"
    mock_result.task_id = "t1"
    mock_result.candidate_memory_id = "m1"
    mock_result.share = mock_share
    mock_result.withhold = mock_withhold
    mock_result.paired_label = "positive_transfer"
    mock_result.paired_record_valid = False
    mock_result.invalid_reason = "engine_timeout"
    mock_result.branch_execution_order = "share_then_withhold"

    edge = {
        "receiver_agent_id": "r1", "receiver_role": "executor",
        "receiver_capabilities": [],
        "candidate_rank": 1, "candidate_score": 0.8,
    }

    record = paired_result_to_record(pair_result=mock_result, edge=edge, seed=0)
    assert record["valid"] is False
    assert record["invalid_reason"] == "engine_timeout"
    # Label is always derived from the canonical nested outcomes, even for
    # invalid records (which the training loader excludes via `valid`).
    assert record["label"] == "positive_transfer"
