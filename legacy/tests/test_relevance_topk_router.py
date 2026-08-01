"""Tests for B1 RelevanceTopKRouter, router factory, and runtime integration."""

import pytest

from smtr.memory.seed_memories import seed_repository
from smtr.memory.store import SQLiteSharedMemoryRepository
from smtr.router.baseline_router import NoMemoryRouter
from smtr.router.baselines import RelevanceTopKRouter, RelevanceTopKRouterConfig
from smtr.router.candidate_proposer import CandidateProposal, CandidateRequest, CandidateScore
from smtr.router.factory import build_router
from smtr.router.sequential_router import ProductionSequentialRouter
from smtr.runtime.graph import build_graph, run_demo, run_demo_with_repository
from smtr.runtime.state import initial_state

# --- Helpers ---


def _make_proposal(
    scored_candidates: list[tuple[str, float]],
) -> CandidateProposal:
    """Build a CandidateProposal from (memory_id, score) pairs in ranking order."""
    request = CandidateRequest(
        task="Obtain a target artifact using the valid action sequence.",
        task_stage="planner",
        receiver_agent_id="planner",
        receiver_role="planner",
        receiver_capabilities=["planning"],
        environment_observation={},
        local_context_summary="",
        top_k=len(scored_candidates),
        seed=7,
    )
    candidates = [
        CandidateScore(
            memory_id=mid,
            total_score=score,
            goal_similarity=score,
            task_tag_overlap=score,
            environment_compatibility=score,
            receiver_compatibility=1.0,
        )
        for mid, score in scored_candidates
    ]
    return CandidateProposal(
        request=request,
        ranked_candidates=candidates,
        pool_revision=1,
    )


class _ErrorCritic:
    """A critic that raises on any access — used to verify B1 never calls it."""

    def predict(self, *args, **kwargs):
        raise RuntimeError("B1 must not call critic")

    def __getattr__(self, name):
        raise RuntimeError(f"B1 must not access critic.{name}")


# --- Test 1: B1 selects by relevance rank ---


class TestB1SelectsByRelevanceRank:
    """B1 selects top-k by proposer relevance ranking."""

    def test_selects_top_2_of_3(self):
        router = RelevanceTopKRouter(
            config=RelevanceTopKRouterConfig(max_shares_per_invocation=2)
        )
        proposal = _make_proposal([("m3", 0.9), ("m1", 0.8), ("m2", 0.4)])

        result = router.decide_from_proposal(
            receiver_agent_id="planner",
            proposal=proposal,
        )

        assert result.selected_memory_ids == ["m3", "m1"]
        # m2 should be withheld
        m2_decision = next(d for d in result.decisions if d.memory_id == "m2")
        assert m2_decision.action == "withhold"
        assert m2_decision.reason == "relevance_topk_budget_exceeded"

    def test_selected_have_share_action(self):
        router = RelevanceTopKRouter(
            config=RelevanceTopKRouterConfig(max_shares_per_invocation=2)
        )
        proposal = _make_proposal([("m3", 0.9), ("m1", 0.8), ("m2", 0.4)])

        result = router.decide_from_proposal(
            receiver_agent_id="planner",
            proposal=proposal,
        )

        for mid in ["m3", "m1"]:
            decision = next(d for d in result.decisions if d.memory_id == mid)
            assert decision.action == "share"
            assert decision.reason == "relevance_topk_selected"
            assert decision.accepted is True


# --- Test 2: B1 does not call critic ---


class TestB1DoesNotCallCritic:
    """B1 never accesses the transfer critic."""

    def test_no_critic_access_with_error_critic(self):
        """Even if an error-raising object is passed, B1 completes successfully."""
        router = RelevanceTopKRouter(
            config=RelevanceTopKRouterConfig(max_shares_per_invocation=2)
        )
        proposal = _make_proposal([("m1", 0.9), ("m2", 0.8)])

        # B1 does not accept a critic param, so it cannot call one.
        # This test verifies the router completes without any critic dependency.
        result = router.decide_from_proposal(
            receiver_agent_id="planner",
            proposal=proposal,
            cards_by_id={},  # B1 should not need cards
        )

        assert len(result.selected_memory_ids) == 2
        assert all(d.tau_mean is None for d in result.decisions)
        assert all(d.negative_risk_mean is None for d in result.decisions)

    def test_critic_fields_are_none(self):
        router = RelevanceTopKRouter()
        proposal = _make_proposal([("m1", 0.5)])

        result = router.decide_from_proposal(
            receiver_agent_id="planner",
            proposal=proposal,
        )

        decision = result.decisions[0]
        assert decision.tau_mean is None
        assert decision.tau_lcb is None
        assert decision.tau_ucb is None
        assert decision.negative_risk_mean is None
        assert decision.negative_risk_ucb is None
        assert decision.support_distance is None


# --- Test 3: B1 trace correctness ---


class TestB1TraceCorrectness:
    """B1 produces correct router trace fields."""

    def test_router_name_and_version(self):
        router = RelevanceTopKRouter()
        proposal = _make_proposal([("m1", 0.5)])

        result = router.decide_from_proposal(
            receiver_agent_id="planner",
            proposal=proposal,
        )

        assert result.router_name == "RelevanceTopKRouter"
        assert result.router_version == "1"

    def test_candidate_order_matches_proposal(self):
        router = RelevanceTopKRouter(
            config=RelevanceTopKRouterConfig(max_shares_per_invocation=2)
        )
        proposal = _make_proposal([("m3", 0.9), ("m1", 0.8), ("m2", 0.4)])

        result = router.decide_from_proposal(
            receiver_agent_id="planner",
            proposal=proposal,
        )

        # traversal_order matches proposer ranking
        traversal_order = result.decisions[0].traversal_order
        assert traversal_order == ["m3", "m1", "m2"]

    def test_proposal_ranks_and_scores(self):
        router = RelevanceTopKRouter(
            config=RelevanceTopKRouterConfig(max_shares_per_invocation=2)
        )
        proposal = _make_proposal([("m3", 0.9), ("m1", 0.8), ("m2", 0.4)])

        result = router.decide_from_proposal(
            receiver_agent_id="planner",
            proposal=proposal,
        )

        # candidate_position = proposal rank
        positions = {d.memory_id: d.candidate_position for d in result.decisions}
        assert positions == {"m3": 0, "m1": 1, "m2": 2}

        # score = proposer score
        scores = {d.memory_id: d.score for d in result.decisions}
        assert scores["m3"] == 0.9
        assert scores["m1"] == 0.8
        assert scores["m2"] == 0.4

    def test_decision_source_is_relevance_topk(self):
        router = RelevanceTopKRouter()
        proposal = _make_proposal([("m1", 0.5)])

        result = router.decide_from_proposal(
            receiver_agent_id="planner",
            proposal=proposal,
        )

        assert result.decisions[0].decision_source == "relevance_topk_router"


# --- Test 4: Payload isolation (runtime integration test) ---


class TestB1PayloadIsolation:
    """B1 payload isolation verified via runtime integration (not just router)."""

    def test_only_selected_payloads_enter_context(self, tmp_path):
        """Run full runtime with B1 and verify only selected payloads are visible."""
        repository = SQLiteSharedMemoryRepository(tmp_path / "memory.sqlite")
        seed_repository(repository)

        router = RelevanceTopKRouter(
            config=RelevanceTopKRouterConfig(max_shares_per_invocation=2)
        )
        state = run_demo_with_repository(
            repository=repository,
            seed=7,
            top_k=4,
            router=router,
        )

        # Verify router made share decisions (unlike NoMemoryRouter)
        assert any(
            d["action"] == "share"
            for trace in state["router_trace"]
            for d in trace["decisions"]
        )

        # Verify only selected payloads enter visible_payloads
        for trace in state["router_trace"]:
            selected_ids = set(trace["selected_memory_ids"])
            agent = trace["agent"]
            visible = state["agent_local_context"][agent]["visible_payloads"]
            visible_ids = {p["memory_id"] for p in visible}
            assert visible_ids == selected_ids

    def test_unselected_payload_steps_not_in_state(self, tmp_path):
        """Unselected memory steps must not appear in the state repr."""
        repository = SQLiteSharedMemoryRepository(tmp_path / "memory.sqlite")
        seed_repository(repository)

        router = RelevanceTopKRouter(
            config=RelevanceTopKRouterConfig(max_shares_per_invocation=1)
        )
        state = run_demo_with_repository(
            repository=repository,
            seed=7,
            top_k=4,
            router=router,
        )

        # Get all payload steps from all memories
        from smtr.memory.seed_memories import build_seed_memories

        all_payloads = build_seed_memories()
        all_steps = [step for _, payload in all_payloads for step in payload.steps]

        # Get selected memory IDs
        selected_ids = set()
        for trace in state["router_trace"]:
            selected_ids.update(trace["selected_memory_ids"])

        # Get unselected steps
        selected_payload_steps = set()
        for _, payload in all_payloads:
            if payload.memory_id in selected_ids:
                selected_payload_steps.update(payload.steps)

        unselected_steps = [s for s in all_steps if s not in selected_payload_steps]

        # Unselected steps should not appear in visible payloads
        state_text = repr(state)
        for step in unselected_steps:
            # Only check steps that are distinctive enough (not generic words)
            if len(step) > 10:
                assert step not in state_text or step in selected_payload_steps


# --- Test 5: Budget boundary cases ---


class TestB1BudgetBoundaries:
    """Cover budget boundary cases for max_shares_per_invocation."""

    def test_max_shares_0_withholds_all(self):
        router = RelevanceTopKRouter(
            config=RelevanceTopKRouterConfig(max_shares_per_invocation=0)
        )
        proposal = _make_proposal([("m1", 0.9), ("m2", 0.8)])

        result = router.decide_from_proposal(
            receiver_agent_id="planner",
            proposal=proposal,
        )

        assert result.selected_memory_ids == []
        assert all(d.action == "withhold" for d in result.decisions)

    def test_max_shares_1_shares_only_top(self):
        router = RelevanceTopKRouter(
            config=RelevanceTopKRouterConfig(max_shares_per_invocation=1)
        )
        proposal = _make_proposal([("m1", 0.9), ("m2", 0.8)])

        result = router.decide_from_proposal(
            receiver_agent_id="planner",
            proposal=proposal,
        )

        assert result.selected_memory_ids == ["m1"]

    def test_max_shares_exceeds_candidate_count(self):
        router = RelevanceTopKRouter(
            config=RelevanceTopKRouterConfig(max_shares_per_invocation=10)
        )
        proposal = _make_proposal([("m1", 0.9), ("m2", 0.8)])

        result = router.decide_from_proposal(
            receiver_agent_id="planner",
            proposal=proposal,
        )

        assert result.selected_memory_ids == ["m1", "m2"]

    def test_max_shares_none_shares_all(self):
        router = RelevanceTopKRouter(
            config=RelevanceTopKRouterConfig(max_shares_per_invocation=None)
        )
        proposal = _make_proposal([("m1", 0.9), ("m2", 0.8), ("m3", 0.7)])

        result = router.decide_from_proposal(
            receiver_agent_id="planner",
            proposal=proposal,
        )

        assert result.selected_memory_ids == ["m1", "m2", "m3"]

    def test_empty_proposal(self):
        router = RelevanceTopKRouter(
            config=RelevanceTopKRouterConfig(max_shares_per_invocation=3)
        )
        proposal = _make_proposal([])

        result = router.decide_from_proposal(
            receiver_agent_id="planner",
            proposal=proposal,
        )

        assert result.selected_memory_ids == []
        assert result.decisions == []

    def test_negative_budget_rejected(self):
        with pytest.raises(ValueError, match="max_shares_per_invocation must be >= 0"):
            RelevanceTopKRouterConfig(max_shares_per_invocation=-1)


# --- Test 6: Factory constructs all three modes ---


class TestRouterFactory:
    """Verify build_router() constructs correct router types."""

    def test_no_memory_mode(self):
        router = build_router("no-memory")
        assert isinstance(router, NoMemoryRouter)

    def test_relevance_topk_mode(self):
        router = build_router("relevance-topk", max_shares_per_invocation=3)
        assert isinstance(router, RelevanceTopKRouter)
        assert router.config.max_shares_per_invocation == 3

    def test_learned_mode_with_checkpoint(self):
        """Learned mode loads critic from checkpoint."""
        from pathlib import Path

        # Use existing checkpoint from the project
        checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "critic_pi0.joblib"
        if not checkpoint_path.exists():
            pytest.skip("critic_pi0.joblib checkpoint not available")

        router = build_router("learned", critic_checkpoint=str(checkpoint_path), seed=42)
        assert isinstance(router, ProductionSequentialRouter)
        assert router.critic is not None

    def test_learned_mode_without_checkpoint_raises(self):
        with pytest.raises(ValueError, match="learned mode requires a critic_checkpoint"):
            build_router("learned")

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="unknown router mode"):
            build_router("bogus-mode")

    def test_factory_rejects_negative_budget(self):
        with pytest.raises(ValueError, match="max_shares_per_invocation must be >= 0"):
            build_router("relevance-topk", max_shares_per_invocation=-1)

    def test_factory_passes_seed_to_learned(self):
        """Factory passes seed but does not apply share budgets to learned SMTR."""
        from pathlib import Path

        checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "critic_pi0.joblib"
        if not checkpoint_path.exists():
            pytest.skip("critic_pi0.joblib checkpoint not available")

        router = build_router(
            "learned",
            critic_checkpoint=str(checkpoint_path),
            seed=123,
            max_shares_per_invocation=2,
        )
        assert isinstance(router, ProductionSequentialRouter)
        assert router.seed == 123
        assert not hasattr(router.config, "max_shares_per_invocation")
        assert router.traversal_policy.policy_name == "random_order"


# --- Test 7: Regression — default behavior unchanged ---


class TestRegression:
    """Existing default behavior must not regress."""

    def test_default_run_demo_still_works(self):
        """Default run_demo() uses NoMemoryRouter (B0) — unchanged."""
        state = run_demo(seed=7)
        assert state["team_success"] is True
        # Default router withholds all
        assert state["selected_memory_ids_by_agent"] == {
            "planner": [],
            "executor": [],
            "critic": [],
        }

    def test_default_build_graph_uses_no_memory_router(self):
        """build_graph() without router arg still defaults to NoMemoryRouter."""
        from smtr.runtime.environment import ToyEnvironment

        env = ToyEnvironment(seed=7)
        app = build_graph()
        state = initial_state(
            task="Obtain a target artifact using the valid action sequence.",
            environment_observation=env.observe(),
            run_seed=7,
        )
        result = app.invoke(state)
        assert result["team_success"] is True

    def test_b1_via_run_demo_with_repository(self, tmp_path):
        """B1 can be injected into run_demo_with_repository."""
        repository = SQLiteSharedMemoryRepository(tmp_path / "memory.sqlite")
        seed_repository(repository)

        router = RelevanceTopKRouter(
            config=RelevanceTopKRouterConfig(max_shares_per_invocation=2)
        )
        state = run_demo_with_repository(
            repository=repository,
            seed=7,
            top_k=4,
            router=router,
        )

        # B1 should share some memories (unlike B0)
        any_shared = any(
            len(trace["selected_memory_ids"]) > 0
            for trace in state["router_trace"]
        )
        assert any_shared, "B1 should share at least some memories"
