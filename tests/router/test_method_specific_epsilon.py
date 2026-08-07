"""Each method consumes its own checkpoint epsilon_star (清单 P1-2 5.4).

The full / global_transfer / no_compatibility_interaction critics carry
distinct validation-selected epsilons; every router resolves its effective
risk budget from its own checkpoint, and a checkpoint without epsilon_star
fails instead of falling back to a hard-coded default.
"""

from __future__ import annotations

import pytest

from smtr.router.baselines import (
    GlobalTransferCriticRouter,
    SMTRNoCompatibilityInteractionRouter,
)
from smtr.router.exposure_router import SMTRExposureRouter
from smtr.router.transfer_critic import FourOutcomeTransferCritic


def _critic(feature_block: str, epsilon_star: float | None):
    critic = FourOutcomeTransferCritic(feature_block=feature_block)
    critic.epsilon_star = epsilon_star
    return critic


def test_each_router_uses_its_own_checkpoint_epsilon():
    full = _critic("full", 0.10)
    global_transfer = _critic("global_transfer", 0.25)
    no_compat = _critic("no_compatibility_interaction", 0.18)

    smtr_router = SMTRExposureRouter(critic=full)
    global_router = GlobalTransferCriticRouter(critic=global_transfer)
    no_compat_router = SMTRNoCompatibilityInteractionRouter(critic=no_compat)

    assert smtr_router._effective_risk_budget() == pytest.approx(0.10)
    assert global_router._router._effective_risk_budget() == pytest.approx(
        0.25
    )
    assert no_compat_router._router._effective_risk_budget() == pytest.approx(
        0.18
    )


def test_missing_epsilon_star_fails_closed():
    critic = _critic("full", None)
    router = SMTRExposureRouter(critic=critic)
    with pytest.raises(
        ValueError,
        match="Checkpoint does not contain validation-selected epsilon_star",
    ):
        router._effective_risk_budget()


def test_debug_override_wins_over_checkpoint_epsilon():
    critic = _critic("full", 0.10)
    router = SMTRExposureRouter(
        critic=critic,
        negative_risk_budget=0.05,
        allow_risk_budget_override=True,
    )
    assert router._effective_risk_budget() == pytest.approx(0.05)
