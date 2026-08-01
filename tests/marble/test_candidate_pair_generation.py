"""Tests for candidate-level pair generation wiring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


def _write_split_manifest(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "records": [
            {"task_id": "t1", "split": "validation"},
            {"task_id": "t2", "split": "train"},
        ]
    }), encoding="utf-8")


def _write_dataset_manifest(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "tasks": [
            {"task_id": "t1", "scenario": "database", "agents": []},
            {"task_id": "t2", "scenario": "database", "agents": []},
        ]
    }), encoding="utf-8")


def _write_candidate_manifest(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "candidates": [{
            "task_id": "t1",
            "receiver_agent_id": "r1",
            "receiver_role": "executor",
            "receiver_capabilities": ["sql"],
            "task_instruction": "test task",
            "environment_signature": [],
            "candidate_records": [
                {"memory_id": "m1", "writer_agent_id": "w1", "writer_role": "planner",
                 "writer_capabilities": ["plan"], "rank": 1, "score": 0.8},
                {"memory_id": "m2", "writer_agent_id": "w2", "writer_role": "executor",
                 "writer_capabilities": ["sql"], "rank": 2, "score": 0.5},
            ],
        }],
    }), encoding="utf-8")


def _write_memory_pool(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    memories = [
        {"memory_id": "m1", "payload": {"procedure": "Step 1. Do X."},
         "routing_card": {"writer": {"agent_id": "w1", "role": "planner"}}},
        {"memory_id": "m2", "payload": {"procedure": "Step 1. Do Y."},
         "routing_card": {"writer": {"agent_id": "w2", "role": "executor"}}},
    ]
    path.write_text("".join(json.dumps(m) + "\n" for m in memories), encoding="utf-8")


def _make_mock_pair_result(task_id: str, memory_id: str):
    """Create a mock PairedBranchResult."""
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

    result = MagicMock()
    result.scenario = "database"
    result.task_id = task_id
    result.candidate_memory_id = memory_id
    result.share = mock_share
    result.withhold = mock_withhold
    result.paired_label = "positive_transfer"
    result.paired_record_valid = True
    result.invalid_reason = None
    result.branch_execution_order = "share_then_withhold"
    return result


def test_each_candidate_seed_calls_run_pair_once(tmp_path):
    """Each candidate/seed combination must call run_pair exactly once."""
    split_manifest = tmp_path / "splits.json"
    dataset_manifest = tmp_path / "dataset.json"
    candidate_manifest = tmp_path / "candidates.json"
    memory_pool = tmp_path / "memories.jsonl"
    output_dir = tmp_path / "output"

    _write_split_manifest(split_manifest)
    _write_dataset_manifest(dataset_manifest)
    _write_candidate_manifest(candidate_manifest)
    _write_memory_pool(memory_pool)

    mock_runner = MagicMock()
    mock_runner.run_pair.side_effect = lambda **kwargs: _make_mock_pair_result(
        task_id=kwargs["task"]["task_id"],
        memory_id=kwargs["candidate_memory"]["memory_id"],
    )

    with patch("smtr.marble.branch_runner.MarblePairedBranchRunner", return_value=mock_runner), \
         patch("smtr.marble.paired_context.build_pair_execution_context") as mock_ctx:
        mock_context = MagicMock()
        mock_context.task = {"task_id": "t1", "scenario": "database"}
        mock_context.initial_state_bundle = MagicMock()
        mock_context.agent_config = {}
        mock_ctx.return_value = mock_context

        from smtr.marble.real_pairs import generate_candidate_level_pairs

        result = generate_candidate_level_pairs(
            marble_root=tmp_path / "marble",
            dataset_manifest_path=dataset_manifest,
            split_manifest_path=split_manifest,
            split="validation",
            candidate_manifest_path=candidate_manifest,
            memory_pool_path=memory_pool,
            generation_seeds=[0, 1],
            output_dir=output_dir,
        )

    # 2 candidates x 2 seeds = 4 calls
    assert mock_runner.run_pair.call_count == 4
    assert result["attempted"] == 4


def test_never_calls_run_single_branch(tmp_path):
    """run_single_branch must never be called."""
    split_manifest = tmp_path / "splits.json"
    dataset_manifest = tmp_path / "dataset.json"
    candidate_manifest = tmp_path / "candidates.json"
    memory_pool = tmp_path / "memories.jsonl"
    output_dir = tmp_path / "output"

    _write_split_manifest(split_manifest)
    _write_dataset_manifest(dataset_manifest)
    _write_candidate_manifest(candidate_manifest)
    _write_memory_pool(memory_pool)

    mock_runner = MagicMock()
    mock_runner.run_pair.side_effect = lambda **kwargs: _make_mock_pair_result(
        task_id=kwargs["task"]["task_id"],
        memory_id=kwargs["candidate_memory"]["memory_id"],
    )

    with patch("smtr.marble.branch_runner.MarblePairedBranchRunner", return_value=mock_runner), \
         patch("smtr.marble.paired_context.build_pair_execution_context") as mock_ctx:
        mock_context = MagicMock()
        mock_context.task = {"task_id": "t1", "scenario": "database"}
        mock_context.initial_state_bundle = MagicMock()
        mock_context.agent_config = {}
        mock_ctx.return_value = mock_context

        from smtr.marble.real_pairs import generate_candidate_level_pairs

        generate_candidate_level_pairs(
            marble_root=tmp_path / "marble",
            dataset_manifest_path=dataset_manifest,
            split_manifest_path=split_manifest,
            split="validation",
            candidate_manifest_path=candidate_manifest,
            memory_pool_path=memory_pool,
            generation_seeds=[0],
            output_dir=output_dir,
        )

    assert not mock_runner.run_single_branch.called


def test_record_digests_come_from_paired_branch_result(tmp_path):
    """Record digests must come from PairedBranchResult, not fabricated."""
    split_manifest = tmp_path / "splits.json"
    dataset_manifest = tmp_path / "dataset.json"
    candidate_manifest = tmp_path / "candidates.json"
    memory_pool = tmp_path / "memories.jsonl"
    output_dir = tmp_path / "output"

    _write_split_manifest(split_manifest)
    _write_dataset_manifest(dataset_manifest)
    _write_candidate_manifest(candidate_manifest)
    _write_memory_pool(memory_pool)

    mock_runner = MagicMock()
    mock_runner.run_pair.side_effect = lambda **kwargs: _make_mock_pair_result(
        task_id=kwargs["task"]["task_id"],
        memory_id=kwargs["candidate_memory"]["memory_id"],
    )

    with patch("smtr.marble.branch_runner.MarblePairedBranchRunner", return_value=mock_runner), \
         patch("smtr.marble.paired_context.build_pair_execution_context") as mock_ctx:
        mock_context = MagicMock()
        mock_context.task = {"task_id": "t1", "scenario": "database"}
        mock_context.initial_state_bundle = MagicMock()
        mock_context.agent_config = {}
        mock_ctx.return_value = mock_context

        from smtr.marble.real_pairs import generate_candidate_level_pairs

        generate_candidate_level_pairs(
            marble_root=tmp_path / "marble",
            dataset_manifest_path=dataset_manifest,
            split_manifest_path=split_manifest,
            split="validation",
            candidate_manifest_path=candidate_manifest,
            memory_pool_path=memory_pool,
            generation_seeds=[0],
            limit_pairs=1,
            output_dir=output_dir,
        )

    # Read output records
    records_path = output_dir / "paired_records.jsonl"
    records = [json.loads(l) for l in records_path.read_text().splitlines() if l.strip()]
    assert len(records) == 1
    rec = records[0]
    # Digests must match the mock values
    assert rec["digests"]["share_initial_digest"] == "digest_abc"
    assert rec["digests"]["withhold_initial_digest"] == "digest_abc"
    assert rec["digests"]["share_task_digest"] == "task_digest"
    assert rec["digests"]["share_tool_config_digest"] == "tool_digest"


def test_invalid_pair_not_used_by_training_loader(tmp_path):
    """Invalid pairs must not be loaded for critic training."""
    from smtr.marble.real_pairs import paired_result_to_record

    # Create an invalid pair result
    mock_result = _make_mock_pair_result("t1", "m1")
    mock_result.paired_record_valid = False
    mock_result.invalid_reason = "engine_timeout"
    mock_result.paired_label = None

    edge = {
        "receiver_agent_id": "r1", "receiver_role": "executor",
        "receiver_capabilities": [], "writer_agent_id": "w1",
        "writer_role": "planner", "writer_capabilities": [],
        "candidate_rank": 1, "candidate_score": 0.8,
    }

    record = paired_result_to_record(pair_result=mock_result, edge=edge, seed=0)
    assert record["valid"] is False
    assert record["invalid_reason"] == "engine_timeout"
    assert record["label"] is None
