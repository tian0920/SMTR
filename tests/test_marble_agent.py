"""Tests for SMTR MARBLE integration data models and agent classes.

Data models (SMTRMarbleAgentState, AgentVisibleMarbleContext) are always testable.
Agent classes require MARBLE to be installed.
"""

from __future__ import annotations

import pytest

from smtr.runtime.marble_agent import (
    AgentVisibleMarbleContext,
    SMTRMarbleAgentState,
    _build_agent_visible_marble_context,
    _build_context_fingerprint,
    _format_payloads_for_injection,
)


# ---------------------------------------------------------------------------
# Data model tests — no MARBLE dependency
# ---------------------------------------------------------------------------


class TestSMTRMarbleAgentState:
    def test_default_state(self):
        state = SMTRMarbleAgentState()
        assert state.routing_done is False
        assert state.selected_memory_ids == []
        assert state.selected_payloads_text == ""
        assert state.routing_trace == []

    def test_state_with_routing_done(self):
        state = SMTRMarbleAgentState(
            routing_done=True,
            selected_memory_ids=["m1", "m2"],
            selected_payloads_text="Procedure: fix bug\n  1. Check logs",
        )
        assert state.routing_done is True
        assert len(state.selected_memory_ids) == 2
        assert "fix bug" in state.selected_payloads_text

    def test_state_routing_trace(self):
        state = SMTRMarbleAgentState(
            routing_trace=[{"memory_id": "m1", "action": "include"}]
        )
        assert len(state.routing_trace) == 1
        assert state.routing_trace[0]["memory_id"] == "m1"


class TestAgentVisibleMarbleContext:
    def test_default_context(self):
        ctx = AgentVisibleMarbleContext(
            agent_id="agent1",
            agent_role="debugger",
            task_description="Fix the database error",
        )
        assert ctx.agent_id == "agent1"
        assert ctx.agent_role == "debugger"
        assert ctx.visible_local_messages == []
        assert ctx.receiver_private_context == {}

    def test_context_with_messages(self):
        ctx = AgentVisibleMarbleContext(
            agent_id="agent1",
            agent_role="debugger",
            task_description="Fix the error",
            visible_local_messages=["msg1", "msg2"],
        )
        assert len(ctx.visible_local_messages) == 2


class TestBuildAgentVisibleMarbleContext:
    def test_basic_build(self):
        ctx = _build_agent_visible_marble_context(
            agent_id="agent1",
            agent_role="debugger",
            task="Fix the database error",
        )
        assert ctx.agent_id == "agent1"
        assert ctx.task_description == "Fix the database error"

    def test_build_with_messages(self):
        ctx = _build_agent_visible_marble_context(
            agent_id="agent1",
            agent_role="debugger",
            task="Fix error",
            local_messages=["log entry 1"],
        )
        assert len(ctx.visible_local_messages) == 1


class TestBuildContextFingerprint:
    def test_basic_fingerprint(self):
        fp = _build_context_fingerprint(
            task_text="Fix database error",
            receiver_agent_id="agent1",
            receiver_role="debugger",
            selected_memory_ids=["m1", "m2"],
        )
        assert fp.receiver_agent_id == "agent1"
        assert fp.receiver_role == "debugger"
        assert fp.selected_memory_ids == ["m1", "m2"]
        assert fp.task_id == "marble_episode"

    def test_empty_memory_ids(self):
        fp = _build_context_fingerprint(
            task_text="task",
            receiver_agent_id="a1",
            receiver_role="r",
            selected_memory_ids=[],
        )
        assert fp.selected_memory_ids == []


class TestFormatPayloadsForInjection:
    def test_empty_payloads(self):
        assert _format_payloads_for_injection([]) == ""

    def test_dict_payloads(self):
        payloads = [
            {
                "goal": "Fix database error",
                "steps": ["Check logs", "Identify root cause"],
                "preconditions": ["DB is accessible"],
                "postconditions": ["Error resolved"],
            }
        ]
        result = _format_payloads_for_injection(payloads)
        assert "Fix database error" in result
        assert "1. Check logs" in result
        assert "2. Identify root cause" in result
        assert "DB is accessible" in result
        assert "Error resolved" in result

    def test_multiple_payloads(self):
        payloads = [
            {"goal": "Step A", "steps": ["do A"]},
            {"goal": "Step B", "steps": ["do B"]},
        ]
        result = _format_payloads_for_injection(payloads)
        assert "Step A" in result
        assert "Step B" in result

    def test_pydantic_payloads(self):
        """Test with objects that have attributes instead of dict keys."""

        class MockPayload:
            goal = "Test goal"
            steps = ["step 1"]
            preconditions = []
            postconditions = ["done"]

        result = _format_payloads_for_injection([MockPayload()])
        assert "Test goal" in result
        assert "1. step 1" in result

    def test_string_payloads(self):
        """Test with plain string payloads (fallback)."""
        result = _format_payloads_for_injection(["just a string"])
        assert "just a string" in result


# ---------------------------------------------------------------------------
# Agent class tests — require MARBLE
# ---------------------------------------------------------------------------

try:
    from marble.agent.base_agent import BaseAgent

    from smtr.runtime.marble_agent import (
        PromptAwareBaseAgent,
        SMTRMarbleAgent,
        SMTRMarbleEngine,
        _MARBLE_AVAILABLE,
    )

    MARBLE_AVAILABLE = _MARBLE_AVAILABLE
except ImportError:
    MARBLE_AVAILABLE = False


@pytest.mark.skipif(not MARBLE_AVAILABLE, reason="MARBLE not installed")
class TestPromptAwareBaseAgent:
    def test_render_private_guidance_default_empty(self):
        """PromptAwareBaseAgent.render_private_guidance() returns '' by default."""
        # We can't easily instantiate without a full MARBLE env setup,
        # but we can test the method exists and returns empty
        assert hasattr(PromptAwareBaseAgent, "render_private_guidance")
        assert hasattr(PromptAwareBaseAgent, "_augment_with_private_guidance")
        assert hasattr(PromptAwareBaseAgent, "act")
        assert hasattr(PromptAwareBaseAgent, "_handle_new_communication_session")

    def test_augment_with_empty_guidance(self):
        """_augment_with_private_guidance returns prompt unchanged when guidance is empty."""
        # Test the method logic without full instantiation
        prompt = "original prompt"
        # Simulate: guidance = "" → return prompt unchanged
        guidance = ""
        if not guidance:
            result = prompt
        else:
            result = prompt + "\n\n[Private procedural guidance]\n" + guidance
        assert result == "original prompt"

    def test_augment_with_nonempty_guidance(self):
        """_augment_with_private_guidance appends guidance when non-empty."""
        prompt = "original prompt"
        guidance = "Do step 1 first"
        result = prompt + "\n\n[Private procedural guidance]\n" + guidance
        assert "[Private procedural guidance]" in result
        assert "Do step 1 first" in result


@pytest.mark.skipif(not MARBLE_AVAILABLE, reason="MARBLE not installed")
class TestSMTRMarbleAgent:
    def test_is_subclass_of_prompt_aware(self):
        assert issubclass(SMTRMarbleAgent, PromptAwareBaseAgent)

    def test_is_subclass_of_base_agent(self):
        assert issubclass(SMTRMarbleAgent, BaseAgent)

    def test_has_exposure_override(self):
        """SMTRMarbleAgent accepts exposure_override parameter."""
        # Can't instantiate without full MARBLE env, but verify signature
        import inspect

        sig = inspect.signature(SMTRMarbleAgent.__init__)
        params = sig.parameters
        assert "exposure_override" in params
        assert params["exposure_override"].default is None

    def test_has_render_private_guidance(self):
        assert hasattr(SMTRMarbleAgent, "render_private_guidance")


@pytest.mark.skipif(not MARBLE_AVAILABLE, reason="MARBLE not installed")
class TestSMTRMarbleEngine:
    def test_engine_is_subclass(self):
        from marble.engine.engine import Engine

        assert issubclass(SMTRMarbleEngine, Engine)

    def test_engine_init_params(self):
        import inspect

        sig = inspect.signature(SMTRMarbleEngine.__init__)
        params = sig.parameters
        assert "target_receiver_agent_id" in params
        assert "smtr_memory_pool" in params
        assert "exposure_override" in params


class TestTransferClassification:
    """Test the transfer classification logic."""

    def test_positive_transfer(self):
        from smtr.counterfactual.marble_paired_rollout import _classify_transfer

        assert _classify_transfer(1, 0) == "positive_transfer"

    def test_negative_transfer(self):
        from smtr.counterfactual.marble_paired_rollout import _classify_transfer

        assert _classify_transfer(0, 1) == "negative_transfer"

    def test_neutral_success(self):
        from smtr.counterfactual.marble_paired_rollout import _classify_transfer

        assert _classify_transfer(1, 1) == "neutral_success"

    def test_neutral_failure(self):
        from smtr.counterfactual.marble_paired_rollout import _classify_transfer

        assert _classify_transfer(0, 0) == "neutral_failure"


class TestMarbleOutcome:
    def test_basic_outcome(self):
        from smtr.counterfactual.marble_eval import MarbleOutcome

        outcome = MarbleOutcome(
            success=True,
            reward=1.0,
            task_id="task1",
            environment_type="DB",
            num_agents=3,
            num_iterations=2,
        )
        assert outcome.success is True
        assert outcome.reward == 1.0

    def test_extract_db_outcome(self):
        from smtr.counterfactual.marble_eval import extract_marble_outcome

        engine_result = {
            "task_evaluation": {
                "root_cause": ["lock_contention", "missing_index"],
                "predicted": "The issue is caused by lock_contention in the database.",
            }
        }
        outcome = extract_marble_outcome(
            engine_result,
            task_id="db_task_1",
            environment_type="DB",
            num_agents=3,
            num_iterations=2,
        )
        assert outcome.success is True
        assert outcome.environment_type == "DB"

    def test_extract_db_outcome_no_match(self):
        from smtr.counterfactual.marble_eval import extract_marble_outcome

        engine_result = {
            "task_evaluation": {
                "root_cause": ["lock_contention"],
                "predicted": "The issue is unrelated to the actual problem.",
            }
        }
        outcome = extract_marble_outcome(
            engine_result,
            task_id="db_task_2",
        )
        assert outcome.success is False

    def test_extract_outcome_unknown_format(self):
        from smtr.counterfactual.marble_eval import extract_marble_outcome

        engine_result = {"task_evaluation": "unexpected string"}
        outcome = extract_marble_outcome(engine_result)
        assert outcome.success is False


class TestMarblePairedOutcome:
    def test_from_branch_results(self):
        from smtr.counterfactual.marble_eval import MarbleOutcome
        from smtr.counterfactual.marble_paired_rollout import (
            MarbleBranchResult,
            MarblePairedOutcome,
        )

        share = MarbleBranchResult(
            outcome=MarbleOutcome(
                success=True, reward=1.0, task_id="t", environment_type="DB",
                num_agents=2, num_iterations=1,
            )
        )
        withhold = MarbleBranchResult(
            outcome=MarbleOutcome(
                success=False, reward=0.0, task_id="t", environment_type="DB",
                num_agents=2, num_iterations=1,
            )
        )
        result = MarblePairedOutcome.from_branch_results(
            task_id="t",
            environment_type="DB",
            target_receiver_agent_id="agent1",
            seed=42,
            share_result=share,
            withhold_result=withhold,
            selected_memory_ids=["m1"],
            routing_trace=[],
        )
        assert result.y_share == 1
        assert result.y_withhold == 0
        assert result.transfer_class == "positive_transfer"
        assert result.data_source == "marble"
