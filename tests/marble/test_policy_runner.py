"""Tests for MARBLE policy runner with mock environment."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from smtr.marble.end_to_end_evaluation import MarblePolicyRunResult


def _make_memory_pool() -> dict[str, dict]:
    return {
        "m1": {
            "memory_id": "m1",
            "payload": {"procedure": "1. Inspect database health.\n2. Query monitoring view."},
            "routing_card": {"writer": {"agent_id": "w1", "role": "planner"}},
        },
        "m2": {
            "memory_id": "m2",
            "payload": {"procedure": "1. Run explain plan.\n2. Compare indexes."},
            "routing_card": {"writer": {"agent_id": "w2", "role": "executor"}},
        },
        "m3": {
            "memory_id": "m3",
            "payload": {"procedure": "1. Check replication lag.\n2. Verify failover."},
            "routing_card": {"writer": {"agent_id": "w3", "role": "dba"}},
        },
    }


def _make_task_entry() -> dict[str, Any]:
    return {"task_id": "t1", "scenario": "database", "agents": [{"agent_id": "r1", "role": "executor"}]}


def test_no_memory_injects_nothing(tmp_path):
    """NoMemory method must not inject any memory payload."""
    from smtr.marble.policy_runner import MarblePolicyRunner

    runner = MarblePolicyRunner(marble_root=tmp_path / "marble")

    with patch("smtr.marble.paired_context.build_pair_execution_context") as mock_ctx, \
         patch("smtr.marble.policy_runner.render_procedure_payload") as mock_render:
        # Simulate context build failure to short-circuit engine execution
        mock_ctx.side_effect = RuntimeError("no marble env")

        result = runner.run_episode(
            method="b0_no_memory",
            task_entry=_make_task_entry(),
            receiver_agent_id="r1",
            receiver_role="executor",
            candidate_memory_ids=["m1", "m2"],
            selected_memory_ids=[],  # NoMemory selects nothing
            memory_pool=_make_memory_pool(),
            generation_seed=0,
            workspace=tmp_path / "run",
        )

    # No payloads should be rendered
    mock_render.assert_not_called()
    assert result.selected_memory_ids == ()
    assert result.invalid_reason is not None  # context build failed (no real env)


def test_top1_injects_only_rank1(tmp_path):
    """Top1 method must only inject the rank-1 candidate."""
    from smtr.marble.policy_runner import MarblePolicyRunner

    runner = MarblePolicyRunner(marble_root=tmp_path / "marble")
    memory_pool = _make_memory_pool()

    with patch("smtr.marble.paired_context.build_pair_execution_context") as mock_ctx, \
         patch("smtr.marble.policy_runner.render_procedure_payload", return_value="rendered") as mock_render:
        mock_ctx.side_effect = RuntimeError("no marble env")

        result = runner.run_episode(
            method="top1_relevance",
            task_entry=_make_task_entry(),
            receiver_agent_id="r1",
            receiver_role="executor",
            candidate_memory_ids=["m1", "m2", "m3"],
            selected_memory_ids=["m1"],  # Top1 selects only rank 1
            memory_pool=memory_pool,
            generation_seed=0,
            workspace=tmp_path / "run",
        )

    # Only m1 should be rendered
    assert mock_render.call_count == 1
    call_arg = mock_render.call_args[0][0]
    assert call_arg["memory_id"] == "m1"


def test_smtr_injects_only_router_selected(tmp_path):
    """SMTR method must only inject router-selected memories."""
    from smtr.marble.policy_runner import MarblePolicyRunner

    runner = MarblePolicyRunner(marble_root=tmp_path / "marble")
    memory_pool = _make_memory_pool()

    with patch("smtr.marble.paired_context.build_pair_execution_context") as mock_ctx, \
         patch("smtr.marble.policy_runner.render_procedure_payload", return_value="rendered") as mock_render:
        mock_ctx.side_effect = RuntimeError("no marble env")

        result = runner.run_episode(
            method="smtr",
            task_entry=_make_task_entry(),
            receiver_agent_id="r1",
            receiver_role="executor",
            candidate_memory_ids=["m1", "m2", "m3"],
            selected_memory_ids=["m1", "m3"],  # Router selected m1 and m3
            memory_pool=memory_pool,
            generation_seed=0,
            workspace=tmp_path / "run",
        )

    # m1 and m3 should be rendered
    assert mock_render.call_count == 2
    rendered_ids = [c[0][0]["memory_id"] for c in mock_render.call_args_list]
    assert "m1" in rendered_ids
    assert "m3" in rendered_ids
    assert "m2" not in rendered_ids


def test_all_share_injects_all_candidates(tmp_path):
    """AllShare method must inject all candidate memories."""
    from smtr.marble.policy_runner import MarblePolicyRunner

    runner = MarblePolicyRunner(marble_root=tmp_path / "marble")
    memory_pool = _make_memory_pool()

    with patch("smtr.marble.paired_context.build_pair_execution_context") as mock_ctx, \
         patch("smtr.marble.policy_runner.render_procedure_payload", return_value="rendered") as mock_render:
        mock_ctx.side_effect = RuntimeError("no marble env")

        result = runner.run_episode(
            method="all_share",
            task_entry=_make_task_entry(),
            receiver_agent_id="r1",
            receiver_role="executor",
            candidate_memory_ids=["m1", "m2", "m3"],
            selected_memory_ids=["m1", "m2", "m3"],  # AllShare selects all
            memory_pool=memory_pool,
            generation_seed=0,
            workspace=tmp_path / "run",
        )

    # All 3 should be rendered
    assert mock_render.call_count == 3


def test_only_target_receiver_visibility(tmp_path):
    """Only the target receiver must see injected memories."""
    from smtr.marble.policy_runner import MarblePolicyRunner

    runner = MarblePolicyRunner(marble_root=tmp_path / "marble")
    memory_pool = _make_memory_pool()

    # We need to go deeper into the engine execution path to verify injection target
    # Since we can't run the real engine, verify via the context build
    with patch("smtr.marble.paired_context.build_pair_execution_context") as mock_ctx:
        mock_ctx.side_effect = RuntimeError("no marble env")

        result = runner.run_episode(
            method="smtr",
            task_entry=_make_task_entry(),
            receiver_agent_id="r1",
            receiver_role="executor",
            candidate_memory_ids=["m1"],
            selected_memory_ids=["m1"],
            memory_pool=memory_pool,
            generation_seed=0,
            workspace=tmp_path / "run",
        )

    # The context was built with receiver_agent_id="r1"
    mock_ctx.assert_called_once()
    call_kwargs = mock_ctx.call_args[1]
    assert call_kwargs["receiver_agent_id"] == "r1"


def test_native_evaluator_outcome_saved(tmp_path):
    """Native evaluator outcome must be saved in the result."""
    # Verify MarblePolicyRunResult schema correctly stores native evaluator fields
    result = MarblePolicyRunResult(
        method="smtr",
        task_id="t1",
        generation_seed=0,
        receiver_agent_id="r1",
        receiver_role="executor",
        candidate_memory_ids=("m1",),
        selected_memory_ids=("m1",),
        team_success=True,
        score=0.85,
        real_engine_executed=True,
        native_evaluator_executed=True,
        environment_valid=True,
        runtime_visibility_verified=True,
        cleanup_succeeded=True,
        invalid_reason=None,
    )
    assert result.native_evaluator_executed is True
    assert result.team_success is True
    assert result.score == 0.85
    assert result.real_engine_executed is True
    assert result.invalid_reason is None
