"""Context propagation tests: training loader must preserve receiver context,
and train/inference feature tokens must match for the same paired record."""

from __future__ import annotations

import json
from pathlib import Path

from smtr.core.types import CandidateExposureInput
from smtr.marble.paired_evaluation import build_receiver_state_from_entry
from smtr.router.transfer_features import (
    HashingTransferFeatureEncoder,
    build_routing_card_from_pool_entry,
    load_paired_records_for_training,
)


def _memory_pool_entry() -> dict:
    return {
        "memory_id": "dbproc-traj00000001",
        "payload": {
            "memory_id": "dbproc-traj00000001",
            "procedure": "1. Execute a SELECT query via sql_tool",
            "preconditions": ["Requires tools: sql_tool"],
            "postconditions": ["A supported database diagnosis is identified."],
            "provenance": {
                "source_agent_id": "writer01",
                "source_agent_role": "executor",
                "source_task_id": "101",
                "source_trajectory_id": "traj00000001",
                "source_split": "train",
                "source_scenario": "database",
            },
            "version": "v3",
        },
        "routing_card": {
            "memory_id": "dbproc-traj00000001",
            "goal_summary": "Diagnose database issue using 3-step evidence method.",
            "task_tags": ["database", "select"],
            "required_tools": ["sql_tool"],
            "required_capabilities": ["sql"],
            "execution_role_tags": ["executor"],
            "environment_constraints": ["read-only SQL"],
            "precondition_tags": ["sql_tool available"],
            "procedure_type": "diagnostic",
            "procedure_length_bucket": "short",
            "read_write_scope": "read",
            "evidence_count": 1,
        },
    }


def _candidate_entry() -> dict:
    return {
        "task_id": "205",
        "receiver_agent_id": "receiver07",
        "receiver_role": "critic",
        "receiver_capabilities": ["review"],
        "receiver_tool_names": ["review_tool"],
        "receiver_model_name": "qwen-plus",
        "task_instruction": "Verify the database latency diagnosis with evidence.",
        "environment_signature": ["mysql", "read-only"],
    }


def _paired_record(entry: dict) -> dict:
    """A paired record persisted by real_pairs.paired_result_to_record."""
    return {
        "record_type": "marble_candidate_level_pair",
        "schema_version": "v2",
        "scenario": "database",
        "task_id": entry["task_id"],
        "generation_seed": 0,
        "receiver_agent_id": entry["receiver_agent_id"],
        "receiver_role": entry["receiver_role"],
        "receiver_capabilities": entry["receiver_capabilities"],
        "receiver_tool_names": entry["receiver_tool_names"],
        "receiver_model_name": entry["receiver_model_name"],
        "candidate_memory_id": "dbproc-traj00000001",
        "writer_agent_id": "writer01",
        "writer_role": "executor",
        "writer_capabilities": ["sql"],
        "writer_tool_names": ["sql_tool", "monitor_tool"],
        "writer_model_name": "qwen-max",
        "selected_prefix_memory_ids": [],
        "candidate_rank": 1,
        "candidate_score": 0.5,
        "task_instruction": entry["task_instruction"],
        "environment_signature": entry["environment_signature"],
        "subtask": None,
        "local_context_summary": "receiver local context",
        "team_context_summary": "team context",
        "share": {"team_success": True},
        "withhold": {"team_success": False},
        "label": "positive_transfer",
        "valid": True,
        "invalid_reason": None,
        "branch_execution_order": "share_then_withhold",
    }


def _write_fixtures(tmp_path: Path) -> tuple[Path, Path, dict]:
    entry = _candidate_entry()
    record = _paired_record(entry)
    pool_entry = _memory_pool_entry()

    records_path = tmp_path / "paired_records.jsonl"
    records_path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    pool_path = tmp_path / "memory_pool.jsonl"
    pool_path.write_text(json.dumps(pool_entry, sort_keys=True) + "\n", encoding="utf-8")
    return records_path, pool_path, entry


def test_training_loader_preserves_receiver_context(tmp_path: Path) -> None:
    records_path, pool_path, entry = _write_fixtures(tmp_path)

    items = load_paired_records_for_training(records_path, pool_path)
    assert len(items) == 1
    training_item, label = items[0]
    assert label == "positive_transfer"

    rs = training_item.receiver_state
    assert rs.task_instruction == entry["task_instruction"]
    assert rs.environment_signature == tuple(entry["environment_signature"])
    assert rs.local_context_summary == "receiver local context"
    assert rs.team_context_summary == "team context"
    assert rs.receiver.agent_id == entry["receiver_agent_id"]
    assert rs.receiver.role == "critic"
    assert rs.receiver.capabilities == ("review",)
    assert rs.receiver.tool_names == ("review_tool",)
    assert rs.receiver.model_name == "qwen-plus"

    card = training_item.candidate_card
    assert card.goal_summary == (
        "Diagnose database issue using 3-step evidence method."
    )
    assert card.task_tags == ("database", "select")
    assert card.required_tools == ("sql_tool",)
    assert card.required_capabilities == ("sql",)
    assert card.environment_constraints == ("read-only SQL",)
    # Writer-agnostic: the card carries no writer profile at all.
    assert not hasattr(card, "writer")

    # SMTR-v1: S = empty, never reconstructed from records.
    assert training_item.selected_prefix_cards == ()


def test_train_and_inference_feature_tokens_match(tmp_path: Path) -> None:
    records_path, pool_path, entry = _write_fixtures(tmp_path)

    # Training path: loader from persisted paired record.
    training_input, _ = load_paired_records_for_training(records_path, pool_path)[0]

    # Inference path: evaluation builder from candidate entry + memory pool.
    pool_entry = json.loads(pool_path.read_text(encoding="utf-8").splitlines()[0])
    evaluation_input = CandidateExposureInput(
        receiver_state=build_receiver_state_from_entry(entry),
        candidate_card=build_routing_card_from_pool_entry(pool_entry),
        selected_prefix_cards=(),
    )

    encoder = HashingTransferFeatureEncoder(feature_block="full")
    assert encoder.tokens(training_input) == encoder.tokens(evaluation_input)

    # Also for the ablation block that removes receiver-compatibility
    # interaction features.
    encoder_no_ci = HashingTransferFeatureEncoder(
        feature_block="no_compatibility_interaction"
    )
    assert encoder_no_ci.tokens(training_input) == encoder_no_ci.tokens(evaluation_input)
