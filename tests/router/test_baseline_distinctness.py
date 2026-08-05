"""Test 2: formal baseline distinctness (清单 P0-2).

Asserts that any two main-table methods can produce different actions on
some constructed candidate set, so no two baselines are behaviorally
identical.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from smtr.core.types import AgentProfile, MemoryRoutingCard, ReceiverState
from smtr.router.baselines import (
    GlobalTransferCriticRouter,
    NoMemoryRouter,
    RoleAwareTop1Router,
    SemanticTop1Router,
    SMTRNoPairInteractionRouter,
    SMTRNoRiskRouter,
)
from smtr.router.exposure_router import SMTRExposureRouter


def _card(memory_id: str, *, goal: str, tags: tuple, writer_role: str = "planner") -> MemoryRoutingCard:
    return MemoryRoutingCard(
        memory_id=memory_id,
        goal_summary=goal,
        task_tags=tags,
        writer=AgentProfile(agent_id=f"w_{memory_id}", role=writer_role),
        source_task_id="src",
        source_scenario="database",
    )


def _receiver_state() -> ReceiverState:
    return ReceiverState(
        task_id="t1",
        scenario="database",
        task_instruction="diagnose database issue",
        receiver=AgentProfile(agent_id="r1", role="executor"),
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
    def test_semantic_top1_differs_from_role_aware_top1(self):
        # mem_sem is semantically similar but written by a different role;
        # mem_role is semantically unrelated but role/capability matched.
        cards = [
            _card("mem_sem", goal="diagnose database issue", tags=("database",), writer_role="critic"),
            _card("mem_role", goal="unrelated topic", tags=("other",), writer_role="executor"),
        ]
        semantic = _actions(SemanticTop1Router(), cards)
        role_aware = _actions(RoleAwareTop1Router(), cards)
        assert semantic != role_aware
        assert semantic["mem_sem"] == "share"
        assert role_aware["mem_role"] == "share"

    def test_no_memory_differs_from_top1_baselines(self):
        cards = [_card("m0", goal="diagnose database issue", tags=("database",))]
        assert _actions(NoMemoryRouter(), cards) != _actions(SemanticTop1Router(), cards)
        assert _actions(NoMemoryRouter(), cards) != _actions(RoleAwareTop1Router(), cards)


class TestCriticBaselineDistinctness:
    def test_role_aware_top1_differs_from_global_transfer_critic(self):
        cards = [
            _card("mem_heur", goal="diagnose database issue", tags=("database",)),
            _card("mem_critic", goal="unrelated topic", tags=("other",)),
        ]
        # Global critic learned that the heuristic favourite is harmful.
        global_router = GlobalTransferCriticRouter(
            critic=_critic("memory_task_only", {
                "mem_heur": (-0.4, 0.05),
                "mem_critic": (0.3, 0.05),
            })
        )
        assert _actions(RoleAwareTop1Router(), cards) != _actions(global_router, cards)

    def test_global_transfer_critic_differs_from_smtr(self):
        cards = [_card("mem1", goal="diagnose database issue", tags=("database",))]
        # Same raw scores, but the full critic's calibrated risk exceeds the
        # budget while the global critic stays safe: the receiver-conditioned
        # risk gate flips the decision.
        global_router = GlobalTransferCriticRouter(
            critic=_critic("memory_task_only", {"mem1": (0.4, 0.05)}))
        smtr_router = SMTRExposureRouter(
            critic=_critic("full", {"mem1": (0.4, 0.6)}))
        assert _actions(global_router, cards) != _actions(smtr_router, cards)

    def test_smtr_differs_from_smtr_no_risk(self):
        cards = [_card("mem1", goal="diagnose database issue", tags=("database",))]
        smtr = SMTRExposureRouter(critic=_critic("full", {"mem1": (0.4, 0.6)}))
        no_risk = SMTRNoRiskRouter(critic=_critic("full", {"mem1": (0.4, 0.6)}))
        assert _actions(smtr, cards) != _actions(no_risk, cards)

    def test_smtr_differs_from_smtr_no_pair_interaction(self):
        cards = [_card("mem1", goal="diagnose database issue", tags=("database",))]
        smtr = SMTRExposureRouter(critic=_critic("full", {"mem1": (0.4, 0.05)}))
        no_pair = SMTRNoPairInteractionRouter(
            critic=_critic("no_pair_interaction", {"mem1": (-0.3, 0.05)}))
        assert _actions(smtr, cards) != _actions(no_pair, cards)


class TestMainTablePairwiseDistinct:
    def test_no_two_methods_identical_on_constructed_suite(self):
        cards_suite = [
            [
                _card("mem_sem", goal="diagnose database issue", tags=("database",), writer_role="critic"),
                _card("mem_role", goal="unrelated topic", tags=("other",), writer_role="executor"),
            ],
            [_card("mem1", goal="diagnose database issue", tags=("database",))],
            [_card("mem2", goal="unrelated", tags=("misc",), writer_role="critic")],
        ]
        methods = {
            "b0_no_memory": lambda: NoMemoryRouter(),
            "semantic_top1": lambda: SemanticTop1Router(),
            "role_aware_top1": lambda: RoleAwareTop1Router(),
            "global_transfer_critic": lambda: GlobalTransferCriticRouter(
                critic=_critic("memory_task_only", {
                    "mem_sem": (-0.4, 0.05), "mem_role": (0.3, 0.05),
                    "mem1": (0.4, 0.05), "mem2": (-0.2, 0.05),
                })),
            "smtr_no_pair_interaction": lambda: SMTRNoPairInteractionRouter(
                critic=_critic("no_pair_interaction", {
                    "mem_sem": (0.5, 0.05), "mem_role": (0.1, 0.05),
                    "mem1": (-0.1, 0.05), "mem2": (0.2, 0.05),
                })),
            "smtr_no_risk": lambda: SMTRNoRiskRouter(
                critic=_critic("full", {
                    "mem_sem": (0.5, 0.9), "mem_role": (0.1, 0.9),
                    "mem1": (0.4, 0.9), "mem2": (-0.2, 0.9),
                })),
            "smtr": lambda: SMTRExposureRouter(
                critic=_critic("full", {
                    "mem_sem": (0.5, 0.05), "mem_role": (0.1, 0.6),
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
