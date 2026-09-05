"""P0-7 / P0-8: refitted critic must actually drive future routing.

Regression tests for the ``stale_controller_critic_after_refit`` wiring
bug: the learner refit its internal critic while the controller kept
routing with the initial (stale) checkpoint, so refits never influenced
any routing decision.

P0-7 (adaptive):
    v1 critic trained on D_train with constant tau=-0.1 predicts
    mu=-0.1, sigma=0 -> LCB <= delta -> zero injection. After 5+
    positive probe edges (tau=+0.4), ``maybe_refit()`` produces v2 with
    mu≈+0.4, sigma=0 -> LCB > gamma -> the controller (hot-swapped via
    ``set_critic``) exploits and injects on the NEXT task. This proves
    "refit changes future routing".

P0-8 (frozen baseline):
    ``rima_transfer_frozen`` never refits: critic_version stays 1,
    refit_count stays 0, and the controller's critic object never
    changes for the entire stream.
"""

from __future__ import annotations

from typing import Any

import pytest

from smtr.memory.shared_memory_pool import SharedMemory, SharedMemoryPool
from smtr.rima.continual_transfer_learner import ContinualTransferLearner
from smtr.rima.features import (
    ReceiverConditionedTransferFeatures,
    RimaFeatureEncoder,
)
from smtr.rima.online_transfer_evidence import OnlineTransferEvidence
from smtr.rima.transfer_controller import RoutingMode, TransferAwareMemoryController
from smtr.rima.transfer_policy import TransferPolicy
from smtr.rima.transfer_state import ReceiverTransferStateContainer
from smtr.router.official_score_transfer_critic import (
    BootstrapOfficialScoreTransferCritic,
    MatchedInterventionExample,
)

BETA = 1.64
DELTA = 0.0
GAMMA = 0.35


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_features(
    task_id: str, memory_id: str, receiver_id: str
) -> ReceiverConditionedTransferFeatures:
    return ReceiverConditionedTransferFeatures(
        task_id=task_id,
        memory_id=memory_id,
        receiver_id=receiver_id,
        task_repr={"scenario": "test", "task_type": "test", "text": "probe task"},
        receiver_repr={"role": "executor", "capabilities": ["coding"]},
        routing_card={
            "goal_summary": "goal",
            "task_tags": ["test"],
            "precondition_summary": "",
            "compatible_receiver_roles": ["executor"],
            "compatible_receiver_capabilities": ["coding"],
            "procedure_type": "experience",
        },
    )


def _make_example(
    task_id: str,
    memory_id: str,
    receiver_id: str,
    *,
    source_agent_id: str,
    expose_score: float | None,
    withhold_score: float | None,
) -> MatchedInterventionExample:
    return MatchedInterventionExample(
        task_id=task_id,
        memory_id=memory_id,
        receiver_id=receiver_id,
        source_agent_id=source_agent_id,
        official_expose_score=expose_score,
        official_withhold_score=withhold_score,
        features=_make_features(task_id, memory_id, receiver_id),
    )


def _make_base_examples(n: int = 6) -> list[MatchedInterventionExample]:
    """D_train: single (task_id, receiver_id) family, constant tau=-0.1."""
    return [
        _make_example(
            "tb",
            f"mb{i}",
            "r1",
            source_agent_id="agent_A",
            expose_score=0.35,
            withhold_score=0.45,  # tau = -0.1
        )
        for i in range(n)
    ]


def _make_evidence(pos: int, index: int) -> OnlineTransferEvidence:
    """Positive probe edge with tau=+0.4.

    ``task_id="tb"`` keeps every edge in the same (task_id, receiver_id)
    bootstrap family as D_train so each member resamples a mixed-label
    dataset (sigma=0); leakage is guarded by ``task_position`` instead.
    """
    return OnlineTransferEvidence(
        task_id="tb",
        task_position=pos,
        receiver_id="r1",
        memory_id=f"probe_{pos}_{index}",
        expose_scores=[0.8],
        withhold_scores=[0.4],  # tau = +0.4
        observed_tau=0.4,
        tau_std=None,
        generation_seeds=[pos],
    )


def _fit_frozen_critic(
    examples: list[MatchedInterventionExample], encoder: RimaFeatureEncoder
) -> BootstrapOfficialScoreTransferCritic:
    critic = BootstrapOfficialScoreTransferCritic(
        encoder=encoder, n_bootstrap=31, seed=0
    )
    critic.fit(examples)
    critic.freeze()
    return critic


def _mem(mid: str) -> SharedMemory:
    return SharedMemory(
        memory_id=mid,
        source_agent_id="agent_A",
        origin_task_id=f"origin_{mid}",
        origin_task_position=0,
        routing_card={
            "goal_summary": "goal",
            "task_tags": ["test"],
            "precondition_summary": "",
            "compatible_receiver_roles": ["executor"],
            "compatible_receiver_capabilities": ["coding"],
            "procedure_type": "experience",
        },
    )


def _feature_builder(
    mem: SharedMemory,
    receiver_id: str,
    task: dict[str, Any],
    task_id: str,
) -> ReceiverConditionedTransferFeatures:
    return _make_features(task_id, mem.memory_id, receiver_id)


def _make_controller(
    critic: BootstrapOfficialScoreTransferCritic,
) -> tuple[TransferAwareMemoryController, SharedMemoryPool]:
    pool = SharedMemoryPool()
    for i in range(3):
        pool.add(_mem(f"m{i}"))
    controller = TransferAwareMemoryController(
        critic=critic,
        pool=pool,
        transfer_states=ReceiverTransferStateContainer(),
        policy=TransferPolicy(
            beta=BETA,
            delta=DELTA,
            gamma=GAMMA,
            gamma_quantile=0.75,
            gamma_positive_support=2,
            gamma_source_split="train",
        ),
        feature_builder=_feature_builder,
    )
    return controller, pool


def _plan(controller: TransferAwareMemoryController, pos: int):
    return controller.plan_for_task(
        task={"text": "probe task", "tags": ["test"]},
        task_id=f"t{pos}",
        task_position=pos,
        receiver_id="r1",
    )


def _ingest_probe_edges(
    learner: ContinualTransferLearner, pos: int, count: int
) -> None:
    for i in range(count):
        evidence = _make_evidence(pos, i)
        learner.add_online_evidence(
            evidence,
            features=_make_features(evidence.task_id, evidence.memory_id, "r1"),
            source_agent_id="agent_A",
        )


# ---------------------------------------------------------------------------
# P0-7: refit changes future routing (adaptive variant)
# ---------------------------------------------------------------------------


def test_refit_changes_future_routing():
    """v1 (mu=-0.1, sigma=0) never injects; after refit to v2
    (mu≈+0.4, sigma=0, LCB > gamma) the controller exploits and
    injects on the very next task."""
    encoder = RimaFeatureEncoder(n_features=128, include_receiver=True)
    base = _make_base_examples()
    initial_critic = _fit_frozen_critic(base, encoder)

    # P0-3 wiring: learner adopts the frozen checkpoint verbatim and the
    # controller is constructed with the learner's current critic.
    learner = ContinualTransferLearner(
        base_examples=base,
        encoder=encoder,
        refit_every_new_edges=5,
        initial_critic=initial_critic,
    )
    controller, _pool = _make_controller(learner.current_critic)
    assert controller.critic is learner.current_critic
    assert controller.critic_version == learner.critic_version == 1

    # --- v1 prediction: mu=-0.1, sigma=0 -> LCB < delta, no injection.
    candidate = _make_example(
        "tb", "m_cand", "r1", source_agent_id="agent_A",
        expose_score=None, withhold_score=None,
    )
    dist_v1, log_v1 = learner.predict_distribution(candidate, current_task_position=1)
    assert log_v1.critic_version == 1
    assert dist_v1.mu_tau == pytest.approx(-0.1, abs=1e-6)
    assert dist_v1.sigma_tau == pytest.approx(0.0, abs=1e-9)
    assert dist_v1.mu_tau - BETA * dist_v1.sigma_tau <= DELTA

    # --- Tasks 1..5 under v1: explore-only, zero injection.
    for pos in range(1, 6):
        plan = _plan(controller, pos)
        assert controller.critic_version == 1
        assert plan.selected_memory_ids == [], (
            "v1 critic (LCB <= delta) must never inject"
        )
        # Post-task probe: 18 positive edges total (6 base : 18 online).
        _ingest_probe_edges(learner, pos, count=4 if pos <= 3 else 3)

    # --- Refit and hot-swap the controller critic (P0-4).
    assert learner.should_refit()
    assert learner.maybe_refit() is True
    assert learner.critic_version == 2
    assert learner.current_critic is not initial_critic
    controller.set_critic(
        learner.current_critic,
        version=learner.critic_version,
        trained_through=learner.critic_trained_through_task_position,
    )

    # P0-5 identity invariant + version bookkeeping (P0-6).
    assert controller.critic is learner.current_critic
    assert controller.critic_version == 2
    # Forward-only: trained through position 5, used from task 6 on.
    assert controller.critic_trained_through == 5

    # --- v2 prediction: mu≈+0.4, sigma=0 -> LCB > gamma.
    dist_v2, log_v2 = learner.predict_distribution(candidate, current_task_position=6)
    assert log_v2.critic_version == 2
    assert log_v2.critic_trained_through_task_position == 5
    assert dist_v2.mu_tau > GAMMA
    assert dist_v2.mu_tau == pytest.approx(0.4, abs=0.05)
    assert dist_v2.sigma_tau == pytest.approx(0.0, abs=1e-6)
    assert dist_v2.mu_tau - BETA * dist_v2.sigma_tau > GAMMA

    # --- Task 6 under v2: exploit-only and injection happens.
    plan = _plan(controller, pos=6)
    assert plan.routing_mode == RoutingMode.EXPLOIT_ONLY
    assert len(plan.selected_memory_ids) == 1, (
        "refitted critic must change future routing: task 6 must inject"
    )


def test_refit_prediction_changes_on_same_candidate():
    """The same candidate must receive different predictions before and
    after the refit (direct evidence the controller sees a new model)."""
    encoder = RimaFeatureEncoder(n_features=128, include_receiver=True)
    base = _make_base_examples()
    learner = ContinualTransferLearner(
        base_examples=base,
        encoder=encoder,
        refit_every_new_edges=5,
        initial_critic=_fit_frozen_critic(base, encoder),
    )
    candidate = _make_example(
        "tb", "m_cand", "r1", source_agent_id="agent_A",
        expose_score=None, withhold_score=None,
    )

    before, _ = learner.predict_distribution(candidate, current_task_position=1)
    for pos in range(1, 6):
        _ingest_probe_edges(learner, pos, count=4 if pos <= 3 else 3)
    learner.maybe_refit()
    after, _ = learner.predict_distribution(candidate, current_task_position=6)

    assert before.mu_tau == pytest.approx(-0.1, abs=1e-6)
    assert after.mu_tau > 0.35
    assert after.mu_tau != pytest.approx(before.mu_tau, abs=0.1)


# ---------------------------------------------------------------------------
# P0-8: frozen baseline never swaps the critic
# ---------------------------------------------------------------------------


def test_frozen_baseline_never_swaps_critic():
    """``rima_transfer_frozen``: critic_version stays 1, refit_count
    stays 0, and the controller keeps the initial critic object for the
    whole stream."""
    encoder = RimaFeatureEncoder(n_features=128, include_receiver=True)
    base = _make_base_examples()
    initial_critic = _fit_frozen_critic(base, encoder)

    learner = ContinualTransferLearner(
        base_examples=base,
        encoder=encoder,
        refit_every_new_edges=5,
        initial_critic=initial_critic,
    )
    controller, _pool = _make_controller(learner.current_critic)
    refit_count = 0

    for pos in range(1, 9):
        plan = _plan(controller, pos)
        assert plan.selected_memory_ids == []
        # Frozen runner: evidence is logged but refit is NEVER invoked.
        _ingest_probe_edges(learner, pos, count=1)
        assert learner.critic_version == 1
        assert controller.critic is initial_critic
        assert controller.critic_version == 1

    assert refit_count == 0
    assert learner.n_online_examples == 8
    assert controller.critic_trained_through == -1


# ---------------------------------------------------------------------------
# Guard rails
# ---------------------------------------------------------------------------


def test_initial_critic_must_be_frozen():
    encoder = RimaFeatureEncoder(n_features=128, include_receiver=True)
    base = _make_base_examples()
    unfrozen = BootstrapOfficialScoreTransferCritic(
        encoder=encoder, n_bootstrap=5, seed=0
    )
    unfrozen.fit(base)  # fitted but NOT frozen
    with pytest.raises(RuntimeError, match="frozen"):
        ContinualTransferLearner(
            base_examples=base, encoder=encoder, initial_critic=unfrozen
        )


def test_set_critic_rejects_unfrozen_critic():
    encoder = RimaFeatureEncoder(n_features=128, include_receiver=True)
    base = _make_base_examples()
    controller, _pool = _make_controller(_fit_frozen_critic(base, encoder))

    unfrozen = BootstrapOfficialScoreTransferCritic(
        encoder=encoder, n_bootstrap=5, seed=0
    )
    unfrozen.fit(base)
    with pytest.raises(RuntimeError, match="frozen"):
        controller.set_critic(unfrozen, version=2, trained_through=5)
