"""Formal baseline distinctness (清单 Writer-Agnostic 18.11).

SemanticTop1 must pick the most semantically relevant memory even when the
receiver cannot satisfy its requirements, while ReceiverCompatibleTop1 must
prefer the slightly less relevant memory whose explicit requirements are
fully satisfied. Additionally, any two main-table methods can produce
different actions on some constructed candidate set.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from smtr.core.types import AgentProfile, MemoryRoutingCard, ReceiverState
from smtr.router.baselines import (
    GlobalTransferCriticRouter,
    NoMemoryRouter,
    ReceiverCompatibleTop1Router,
    SemanticTop1Router,
    SMTRNoCompatibilityInteractionRouter,
    SMTRNoRiskRouter,
)
from smtr.router.exposure_router import SMTRExposureRouter


def _card(
    memory_id: str,
    *,
    goal: str,
    tags: tuple,
    required_tools: tuple = (),
    required_capabilities: tuple = (),
    execution_role_tags: tuple = (),
    environment_constraints: tuple = (),
) -> MemoryRoutingCard:
    return MemoryRoutingCard(
        memory_id=memory_id,
        goal_summary=goal,
        task_tags=tags,
        required_tools=required_tools,
        required_capabilities=required_capabilities,
        execution_role_tags=execution_role_tags,
        environment_constraints=environment_constraints,
    )


def _semantic_but_incompatible() -> MemoryRoutingCard:
    """Highest semantic relevance, but the receiver satisfies none of the
    explicit requirements."""
    return _card(
        "mem_sem",
        goal="diagnose database issue",
        tags=("database",),
        required_tools=("admin_console",),
        required_capabilities=("cluster_admin",),
        execution_role_tags=("planner",),
        environment_constraints=("maintenance window",),
    )


def _lower_semantic_fully_satisfied() -> MemoryRoutingCard:
    """Lower semantic relevance, but every requirement is trivially
    satisfied (no explicit requirements)."""
    return _card("mem_compat", goal="unrelated topic", tags=("other",))


def _receiver_state() -> ReceiverState:
    return ReceiverState(
        task_id="t1",
        scenario="database",
        task_instruction="diagnose database issue",
        receiver=AgentProfile(
            agent_id="r1",
            role="executor",
            capabilities=("sql",),
            tool_names=("sql_tool",),
        ),
        environment_signature=("read-only SQL",),
    )


def _actions(router, cards) -> dict[str, str]:
    return {d.memory_id: d.action for d in router.decide(_receiver_state(), cards)}


def _critic(feature_block: str, predictions: dict[str, tuple[float, float]]):
    """Mock critic whose tau/eta depend on the candidate memory id."""
    critic = MagicMock()
    critic.feature_block = feature_block
    critic.epsilon_star = 0.2

    def _predict(item):
        tau, eta = predictions[item.candidate_card.memory_id]
        return SimpleNamespace(tau_hat=tau, eta_hat=eta)

    def _predict_calibrated(item):
        tau, eta = predictions[item.candidate_card.memory_id]
        return SimpleNamespace(tau_hat=tau, eta_hat=eta, eta_hat_calibrated=eta)

    critic.predict.side_effect = _predict
    critic.predict_calibrated.side_effect = _predict_calibrated
    return critic


class TestHeuristicBaselineDistinctness:
    def test_semantic_top1_ignores_receiver_compatibility(self):
        cards = [_semantic_but_incompatible(), _lower_semantic_fully_satisfied()]
        assert _actions(SemanticTop1Router(), cards)["mem_sem"] == "share"

    def test_receiver_compatible_top1_prefers_satisfied_requirements(self):
        cards = [_semantic_but_incompatible(), _lower_semantic_fully_satisfied()]
        assert _actions(ReceiverCompatibleTop1Router(), cards)["mem_compat"] == "share"

    def test_no_memory_differs_from_top1_baselines(self):
        cards = [_card("m0", goal="diagnose database issue", tags=("database",))]
        assert _actions(NoMemoryRouter(), cards) != _actions(SemanticTop1Router(), cards)
        assert _actions(NoMemoryRouter(), cards) != _actions(ReceiverCompatibleTop1Router(), cards)


class TestCriticBaselineDistinctness:
    def test_receiver_compatible_top1_differs_from_global_transfer_critic(self):
        cards = [
            _card("mem_heur", goal="diagnose database issue", tags=("database",)),
            _card("mem_critic", goal="unrelated topic", tags=("other",)),
        ]
        # Global critic learned that the heuristic favourite is harmful.
        global_router = GlobalTransferCriticRouter(
            critic=_critic("global_transfer", {
                "mem_heur": (-0.4, 0.05),
                "mem_critic": (0.3, 0.05),
            })
        )
        assert _actions(ReceiverCompatibleTop1Router(), cards) != _actions(global_router, cards)

    def test_global_transfer_critic_differs_from_smtr(self):
        cards = [_card("mem1", goal="diagnose database issue", tags=("database",))]
        # Same raw scores, but the full critic's calibrated risk exceeds the
        # budget while the global critic stays safe: the receiver-conditioned
        # risk gate flips the decision.
        global_router = GlobalTransferCriticRouter(
            critic=_critic("global_transfer", {"mem1": (0.4, 0.05)}))
        smtr_router = SMTRExposureRouter(
            critic=_critic("full", {"mem1": (0.4, 0.6)}))
        assert _actions(global_router, cards) != _actions(smtr_router, cards)

    def test_smtr_differs_from_smtr_no_risk(self):
        cards = [_card("mem1", goal="diagnose database issue", tags=("database",))]
        smtr = SMTRExposureRouter(critic=_critic("full", {"mem1": (0.4, 0.6)}))
        no_risk = SMTRNoRiskRouter(critic=_critic("full", {"mem1": (0.4, 0.6)}))
        assert _actions(smtr, cards) != _actions(no_risk, cards)

    def test_smtr_differs_from_smtr_no_compatibility_interaction(self):
        cards = [_card("mem1", goal="diagnose database issue", tags=("database",))]
        smtr = SMTRExposureRouter(critic=_critic("full", {"mem1": (0.4, 0.05)}))
        no_compat = SMTRNoCompatibilityInteractionRouter(
            critic=_critic("no_compatibility_interaction", {"mem1": (-0.3, 0.05)}))
        assert _actions(smtr, cards) != _actions(no_compat, cards)


class TestMainTablePairwiseDistinct:
    def test_no_two_methods_identical_on_constructed_suite(self):
        cards_suite = [
            [_semantic_but_incompatible(), _lower_semantic_fully_satisfied()],
            [_card("mem1", goal="diagnose database issue", tags=("database",))],
            [_card("mem2", goal="unrelated", tags=("misc",))],
        ]
        methods = {
            "b0_no_memory": lambda: NoMemoryRouter(),
            "semantic_top1": lambda: SemanticTop1Router(),
            "receiver_compatible_top1": lambda: ReceiverCompatibleTop1Router(),
            "global_transfer_critic": lambda: GlobalTransferCriticRouter(
                critic=_critic("global_transfer", {
                    "mem_sem": (0.4, 0.05), "mem_compat": (-0.2, 0.05),
                    "mem1": (0.4, 0.05), "mem2": (-0.2, 0.05),
                })),
            "smtr_no_compatibility_interaction": lambda: SMTRNoCompatibilityInteractionRouter(
                critic=_critic("no_compatibility_interaction", {
                    "mem_sem": (-0.1, 0.05), "mem_compat": (0.2, 0.05),
                    "mem1": (-0.1, 0.05), "mem2": (0.2, 0.05),
                })),
            "smtr_no_risk": lambda: SMTRNoRiskRouter(
                critic=_critic("full", {
                    "mem_sem": (-0.5, 0.9), "mem_compat": (0.2, 0.9),
                    "mem1": (0.4, 0.9), "mem2": (-0.2, 0.9),
                })),
            "smtr": lambda: SMTRExposureRouter(
                critic=_critic("full", {
                    "mem_sem": (0.5, 0.05), "mem_compat": (0.1, 0.6),
                    "mem1": (0.4, 0.6), "mem2": (-0.2, 0.05),
                })),
        }
        action_profiles = {}
        for name, factory in methods.items():
            router = factory()
            profile = tuple(
                tuple(sorted(_actions(router, cards).items())) for cards in cards_suite
            )
            action_profiles[name] = profile
        for a in methods:
            for b in methods:
                if a != b:
                    assert action_profiles[a] != action_profiles[b], (
                        f"methods {a} and {b} are behaviorally identical "
                        "on the distinctness suite"
                    )
