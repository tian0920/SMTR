"""Tests for τ³-bench integration (SMTRTauAgent, data models, eval wrapper).

All τ³-dependent tests are skipped if τ³-bench is not installed.
Data model and evaluation wrapper tests run unconditionally.
"""

from __future__ import annotations

import pytest

from smtr.counterfactual.schemas import EvaluationGroupMetadata
from smtr.counterfactual.tau_eval import TauOutcome, extract_outcome, summarize_outcomes
from smtr.runtime.tau3_agent import (
    _TAU3_AVAILABLE,
    AgentVisibleTauContext,
    SMTRTauAgentState,
    _build_agent_visible_context,
)

# ---------------------------------------------------------------------------
# SMTRTauAgentState tests
# ---------------------------------------------------------------------------


class TestSMTRTauAgentState:
    def test_default_state(self):
        state = SMTRTauAgentState()
        assert state.routing_done is False
        assert state.selected_memory_ids == []
        assert state.selected_payloads == []
        assert state.routing_trace == []
        assert state.turn_count == 0

    def test_routing_frozen_after_first_turn(self):
        state = SMTRTauAgentState()
        assert state.routing_done is False

        # Simulate routing completion
        state.routing_done = True
        state.selected_memory_ids = ["mem_1", "mem_2"]
        state.selected_payloads = [
            {"goal": "handle refund", "steps": ["check order", "process refund"]},
            {"goal": "verify identity", "steps": ["ask name", "check ID"]},
        ]
        state.routing_trace = [{"memory_id": "mem_1", "decision": "share"}]

        assert state.routing_done is True
        assert len(state.selected_memory_ids) == 2
        assert len(state.selected_payloads) == 2
        assert state.turn_count == 0

    def test_turn_count_increments(self):
        state = SMTRTauAgentState()
        state.turn_count += 1
        state.turn_count += 1
        assert state.turn_count == 2

    def test_payloads_are_serialized_dicts(self):
        state = SMTRTauAgentState(
            selected_payloads=[
                {"goal": "test", "steps": ["step1"], "preconditions": [], "postconditions": []}
            ]
        )
        assert isinstance(state.selected_payloads[0], dict)
        assert state.selected_payloads[0]["goal"] == "test"


# ---------------------------------------------------------------------------
# AgentVisibleTauContext tests
# ---------------------------------------------------------------------------


class TestAgentVisibleTauContext:
    def test_default_context(self):
        ctx = AgentVisibleTauContext()
        assert ctx.user_message == ""
        assert ctx.conversation_history == []
        assert ctx.domain_policy == ""
        assert ctx.tools == []
        assert ctx.task_public_metadata == {}

    def test_context_with_data(self):
        ctx = AgentVisibleTauContext(
            user_message="I need a refund",
            conversation_history=[{"role": "user", "content": "Hi"}],
            domain_policy="Retail return policy...",
            tools=[{"name": "get_order"}],
            task_public_metadata={"order_id": "12345"},
        )
        assert ctx.user_message == "I need a refund"
        assert len(ctx.conversation_history) == 1
        assert "Retail" in ctx.domain_policy

    def test_excludes_evaluation_internals(self):
        """Verify the context model has no fields for evaluation internals."""
        field_names = set(AgentVisibleTauContext.model_fields.keys())
        # These must NOT be present
        assert "evaluation_criteria" not in field_names
        assert "gold_db_state" not in field_names
        assert "reward_basis" not in field_names
        assert "reward_labels" not in field_names
        # These SHOULD be present
        assert "user_message" in field_names
        assert "domain_policy" in field_names
        assert "tools" in field_names

    def test_build_agent_visible_context_strips_internals(self):
        """Test the helper function that builds context from τ³ message."""

        class MockMessage:
            content = "I want to cancel my order"

        ctx = _build_agent_visible_context(
            message=MockMessage(),
            domain_policy="Cancel policy: ...",
            tools=[],
            task_metadata={"task_id": "test_001"},
        )
        assert ctx.user_message == "I want to cancel my order"
        assert ctx.domain_policy == "Cancel policy: ..."
        assert ctx.task_public_metadata == {"task_id": "test_001"}


# ---------------------------------------------------------------------------
# TauOutcome and evaluation wrapper tests
# ---------------------------------------------------------------------------


class TestTauOutcome:
    def test_extract_from_dict(self):
        sim_run = {
            "task_id": "retail_042",
            "reward_info": {"reward": 1.0, "db_check": {"passed": True}},
        }
        outcome = extract_outcome(sim_run, domain="retail")
        assert outcome.success is True
        assert outcome.reward == 1.0
        assert outcome.task_id == "retail_042"
        assert outcome.domain == "retail"

    def test_extract_zero_reward(self):
        sim_run = {"task_id": "retail_001", "reward_info": {"reward": 0.0}}
        outcome = extract_outcome(sim_run)
        assert outcome.success is False
        assert outcome.reward == 0.0

    def test_extract_no_reward_info(self):
        sim_run = {"task_id": "retail_001"}
        outcome = extract_outcome(sim_run)
        assert outcome.success is False
        assert outcome.reward == 0.0
        assert "no reward_info" in outcome.metadata.get("note", "")

    def test_extract_from_object(self):
        class MockSimRun:
            task_id = "retail_099"
            reward_info = None

            class RewardInfo:
                reward = 0.5

                def model_dump(self):
                    return {"reward": 0.5}

            reward_info = RewardInfo()

        outcome = extract_outcome(MockSimRun(), domain="retail")
        assert outcome.success is True
        assert outcome.reward == 0.5

    def test_summarize_outcomes(self):
        outcomes = [
            TauOutcome(success=True, reward=1.0, task_id="t1", domain="retail"),
            TauOutcome(success=False, reward=0.0, task_id="t2", domain="retail"),
            TauOutcome(success=True, reward=1.0, task_id="t3", domain="retail"),
        ]
        summary = summarize_outcomes(outcomes)
        assert summary["count"] == 3
        assert summary["successes"] == 2
        assert abs(summary["success_rate"] - 2 / 3) < 1e-6
        assert abs(summary["mean_reward"] - 2 / 3) < 1e-6

    def test_summarize_empty(self):
        summary = summarize_outcomes([])
        assert summary["count"] == 0
        assert summary["success_rate"] == 0.0


# ---------------------------------------------------------------------------
# data_source field tests
# ---------------------------------------------------------------------------


class TestDataSourceField:
    def test_default_is_toy(self):
        meta = EvaluationGroupMetadata()
        assert meta.data_source == "toy"

    def test_can_set_tau_bench(self):
        meta = EvaluationGroupMetadata(data_source="tau_bench")
        assert meta.data_source == "tau_bench"

    def test_can_set_imported(self):
        meta = EvaluationGroupMetadata(data_source="imported")
        assert meta.data_source == "imported"


# ---------------------------------------------------------------------------
# SMTRTauAgent tests (require τ³-bench)
# ---------------------------------------------------------------------------


class TestSMTRTauAgent:
    @pytest.mark.skipif(not _TAU3_AVAILABLE, reason="τ³-bench not installed")
    def test_agent_creation(self):
        from smtr.runtime.tau3_agent import SMTRTauAgent

        agent = SMTRTauAgent(
            tools=[],
            domain_policy="Test policy",
            llm="gpt-4.1",
        )
        assert agent.domain_policy == "Test policy"

    @pytest.mark.skipif(not _TAU3_AVAILABLE, reason="τ³-bench not installed")
    def test_agent_get_init_state(self):
        from smtr.runtime.tau3_agent import SMTRTauAgent

        agent = SMTRTauAgent(
            tools=[],
            domain_policy="Test policy",
            llm="gpt-4.1",
        )
        state = agent.get_init_state()
        assert isinstance(state, SMTRTauAgentState)
        assert state.routing_done is False

    @pytest.mark.skipif(not _TAU3_AVAILABLE, reason="τ³-bench not installed")
    def test_agent_with_memory_pool(self):
        from smtr.memory.pool import SharedMemoryPool
        from smtr.memory.schemas import MemoryRoutingCard, ProcedurePayload
        from smtr.runtime.tau3_agent import SMTRTauAgent

        card = MemoryRoutingCard(
            memory_id="mem_test",
            goal_summary="Handle refund",
            task_tags=["refund"],
        )
        payload = ProcedurePayload(
            memory_id="mem_test",
            goal="Handle refund",
            steps=["Check order", "Process refund"],
        )
        pool = SharedMemoryPool(routing_cards=[card], payloads=[payload])

        agent = SMTRTauAgent(
            tools=[],
            domain_policy="Test policy",
            llm="gpt-4.1",
            memory_pool=pool,
        )
        assert agent._memory_pool is not None

    def test_agent_raises_without_tau3(self):
        """When τ³ is not installed, SMTRTauAgent should raise ImportError."""
        if _TAU3_AVAILABLE:
            pytest.skip("τ³-bench is installed, cannot test ImportError path")
        from smtr.runtime.tau3_agent import SMTRTauAgent

        with pytest.raises(ImportError, match="τ³-bench"):
            SMTRTauAgent(tools=[], domain_policy="", llm="test")


class TestTau3BranchResult:
    """Tests for Tau3BranchResult data model."""

    def test_default_branch_result(self):
        from smtr.counterfactual.tau3_paired_rollout import Tau3BranchResult

        outcome = TauOutcome(
            success=True, reward=1.0, task_id="t1", domain="retail"
        )
        result = Tau3BranchResult(outcome=outcome)
        assert result.termination_reason == "unknown"
        assert result.num_messages == 0
        assert result.error is None

    def test_branch_result_with_error(self):
        from smtr.counterfactual.tau3_paired_rollout import Tau3BranchResult

        outcome = TauOutcome(
            success=False, reward=0.0, task_id="t1", domain="retail"
        )
        result = Tau3BranchResult(
            outcome=outcome,
            termination_reason="infrastructure_error",
            error="LLM timeout",
        )
        assert result.error == "LLM timeout"


class TestTau3PairedOutcome:
    """Tests for Tau3PairedOutcome data model."""

    def test_from_branch_results_positive(self):
        from smtr.counterfactual.tau3_paired_rollout import (
            Tau3BranchResult,
            Tau3PairedOutcome,
        )

        share = Tau3BranchResult(
            outcome=TauOutcome(
                success=True, reward=1.0, task_id="t1", domain="retail"
            )
        )
        withhold = Tau3BranchResult(
            outcome=TauOutcome(
                success=False, reward=0.0, task_id="t1", domain="retail"
            )
        )
        paired = Tau3PairedOutcome.from_branch_results(
            task_id="t1",
            domain="retail",
            seed=42,
            share_result=share,
            withhold_result=withhold,
            selected_memory_ids=["m1"],
        )
        assert paired.y_share == 1
        assert paired.y_withhold == 0
        assert paired.transfer_class == "positive"
        assert paired.data_source == "tau_bench"

    def test_from_branch_results_negative(self):
        from smtr.counterfactual.tau3_paired_rollout import (
            Tau3BranchResult,
            Tau3PairedOutcome,
        )

        share = Tau3BranchResult(
            outcome=TauOutcome(
                success=False, reward=0.0, task_id="t2", domain="retail"
            )
        )
        withhold = Tau3BranchResult(
            outcome=TauOutcome(
                success=True, reward=1.0, task_id="t2", domain="retail"
            )
        )
        paired = Tau3PairedOutcome.from_branch_results(
            task_id="t2",
            domain="retail",
            seed=42,
            share_result=share,
            withhold_result=withhold,
        )
        assert paired.y_share == 0
        assert paired.y_withhold == 1
        assert paired.transfer_class == "negative"

    def test_from_branch_results_neutral_success(self):
        from smtr.counterfactual.tau3_paired_rollout import (
            Tau3BranchResult,
            Tau3PairedOutcome,
        )

        share = Tau3BranchResult(
            outcome=TauOutcome(
                success=True, reward=1.0, task_id="t3", domain="retail"
            )
        )
        withhold = Tau3BranchResult(
            outcome=TauOutcome(
                success=True, reward=1.0, task_id="t3", domain="retail"
            )
        )
        paired = Tau3PairedOutcome.from_branch_results(
            task_id="t3",
            domain="retail",
            seed=42,
            share_result=share,
            withhold_result=withhold,
        )
        assert paired.transfer_class == "neutral_success"

    def test_from_branch_results_neutral_failure(self):
        from smtr.counterfactual.tau3_paired_rollout import (
            Tau3BranchResult,
            Tau3PairedOutcome,
        )

        share = Tau3BranchResult(
            outcome=TauOutcome(
                success=False, reward=0.0, task_id="t4", domain="retail"
            )
        )
        withhold = Tau3BranchResult(
            outcome=TauOutcome(
                success=False, reward=0.0, task_id="t4", domain="retail"
            )
        )
        paired = Tau3PairedOutcome.from_branch_results(
            task_id="t4",
            domain="retail",
            seed=42,
            share_result=share,
            withhold_result=withhold,
        )
        assert paired.transfer_class == "neutral_failure"


class TestTau3PairedRolloutConfig:
    """Tests for Tau3PairedRolloutConfig."""

    def test_default_config(self):
        from smtr.counterfactual.tau3_paired_rollout import Tau3PairedRolloutConfig

        config = Tau3PairedRolloutConfig()
        assert config.domain == "retail"
        assert config.agent_llm == "gpt-4.1"
        assert config.max_steps == 200

    def test_custom_config(self):
        from smtr.counterfactual.tau3_paired_rollout import Tau3PairedRolloutConfig

        config = Tau3PairedRolloutConfig(
            domain="airline", agent_llm="claude-3", max_steps=100
        )
        assert config.domain == "airline"
        assert config.agent_llm == "claude-3"
        assert config.max_steps == 100


class TestSummarizePairedOutcomes:
    """Tests for summarize_paired_outcomes."""

    def test_summarize_empty(self):
        from smtr.counterfactual.tau3_paired_rollout import summarize_paired_outcomes

        result = summarize_paired_outcomes([])
        assert result["count"] == 0

    def test_summarize_mixed(self):
        from smtr.counterfactual.tau3_paired_rollout import (
            Tau3BranchResult,
            Tau3PairedOutcome,
            summarize_paired_outcomes,
        )

        outcomes = []
        # positive: share=1, withhold=0
        outcomes.append(
            Tau3PairedOutcome.from_branch_results(
                task_id="t1", domain="retail", seed=42,
                share_result=Tau3BranchResult(
                    outcome=TauOutcome(success=True, reward=1.0, task_id="t1", domain="retail")
                ),
                withhold_result=Tau3BranchResult(
                    outcome=TauOutcome(success=False, reward=0.0, task_id="t1", domain="retail")
                ),
            )
        )
        # neutral_success: share=1, withhold=1
        outcomes.append(
            Tau3PairedOutcome.from_branch_results(
                task_id="t2", domain="retail", seed=42,
                share_result=Tau3BranchResult(
                    outcome=TauOutcome(success=True, reward=1.0, task_id="t2", domain="retail")
                ),
                withhold_result=Tau3BranchResult(
                    outcome=TauOutcome(success=True, reward=1.0, task_id="t2", domain="retail")
                ),
            )
        )

        summary = summarize_paired_outcomes(outcomes)
        assert summary["count"] == 2
        assert summary["transfer_class_distribution"]["positive"] == 1
        assert summary["transfer_class_distribution"]["neutral_success"] == 1
        assert summary["share_success_rate"] == 1.0  # both share=1
        assert summary["withhold_success_rate"] == 0.5  # one withhold=1


class TestTau3PairedRolloutRunner:
    """Tests for Tau3PairedRolloutRunner."""

    def test_runner_requires_tau3(self):
        """Runner should raise ImportError when τ³ is not installed."""
        from smtr.counterfactual.tau3_paired_rollout import (
            TAU3_AVAILABLE,
            Tau3PairedRolloutRunner,
        )

        if TAU3_AVAILABLE:
            pytest.skip("τ³-bench is installed, cannot test ImportError path")

        with pytest.raises(ImportError, match="τ³-bench"):
            Tau3PairedRolloutRunner()
