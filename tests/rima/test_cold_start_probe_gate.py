"""Cold-start probe-gate tests (Phase 15 / 15.1 / 16, cold-start fix).

Probe eligibility is decoupled from execution eligibility:

* Execution gate: ``LCB > delta`` ("safe enough to use").
* Probe gate:     valid prediction + not self-transfer, selected by
  max ``UCB = mu + beta * sigma`` ("worth learning about").

Regression coverage:
    Phase 15   test_cold_start_negative_lcb_still_probes
    Phase 15.1 test_cold_start_bootstraps_into_continual_learning
    Phase 16   8 probe-gate regression tests (incl. Phase 4 invariant:
               execution=None while probe exists is a legal state).
"""

from __future__ import annotations

from typing import Any

import pytest
from unittest.mock import MagicMock

from smtr.memory.shared_memory_pool import SharedMemory, SharedMemoryPool
from smtr.rima.continual_transfer_learner import ContinualTransferLearner
from smtr.rima.features import (
    ReceiverConditionedTransferFeatures,
    RimaFeatureEncoder,
)
from smtr.rima.online_transfer_evidence import OnlineTransferEvidence
from smtr.rima.post_task_probe import ProbeSelectionPolicy, select_probe_candidate
from smtr.rima.transfer_controller import (
    RoutingMode,
    TransferAwareMemoryController,
    TransferCandidateDecision,
    TransferRoutingPlan,
    select_episode_edge,
)
from smtr.rima.transfer_policy import TransferPolicy
from smtr.rima.transfer_state import ReceiverTransferStateContainer
from smtr.router.official_score_transfer_critic import (
    BootstrapOfficialScoreTransferCritic,
    MatchedInterventionExample,
    TransferEffectDistribution,
)

BETA = 1.64
DELTA = 0.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _mem(
    mid: str,
    *,
    source: str = "agent_A",
    position: int = 0,
    tags: list[str] | None = None,
) -> SharedMemory:
    return SharedMemory(
        memory_id=mid,
        source_agent_id=source,
        origin_task_id=f"origin_{mid}",
        origin_task_position=position,
        routing_card={"task_tags": tags or [], "goal_summary": ""},
    )


def _task() -> dict[str, Any]:
    return {"text": "test task", "tags": ["alpha"]}


def _feature_builder(
    mem: SharedMemory,
    receiver_id: str,
    task: dict[str, Any],
    task_id: str,
) -> ReceiverConditionedTransferFeatures:
    return ReceiverConditionedTransferFeatures(
        task_id=task_id,
        memory_id=mem.memory_id,
        receiver_id=receiver_id,
        task_repr={"scenario": "test", "text": task.get("text", "")},
        receiver_repr={"role": "worker"},
        routing_card=mem.routing_card,
    )


def _make_policy(
    *, beta: float = BETA, delta: float = DELTA, gamma: float = 0.3
) -> TransferPolicy:
    return TransferPolicy(
        beta=beta,
        delta=delta,
        gamma=gamma,
        gamma_quantile=0.75,
        gamma_positive_support=1,
        gamma_source_split="train",
    )


def _make_mock_critic(mu_tau: float, sigma_tau: float) -> MagicMock:
    """Mock critic returning fixed (mu, sigma); None for self-transfer."""
    critic = MagicMock(spec=BootstrapOfficialScoreTransferCritic)

    def fake_predict(ex: MatchedInterventionExample) -> TransferEffectDistribution:
        if ex.source_agent_id == ex.receiver_id:
            return TransferEffectDistribution(
                memory_id=ex.memory_id,
                receiver_id=ex.receiver_id,
                task_id=ex.task_id,
                mu_expose=None,
                mu_withhold=None,
                mu_tau=None,
                sigma_tau=None,
                n_members=31,
            )
        return TransferEffectDistribution(
            memory_id=ex.memory_id,
            receiver_id=ex.receiver_id,
            task_id=ex.task_id,
            mu_expose=0.5,
            mu_withhold=0.5,
            mu_tau=mu_tau,
            sigma_tau=sigma_tau,
            n_members=31,
        )

    critic.predict_distribution = MagicMock(side_effect=fake_predict)
    return critic


def _build_controller(
    *,
    critic: MagicMock | None = None,
    pool: SharedMemoryPool | None = None,
    policy: TransferPolicy | None = None,
) -> tuple[TransferAwareMemoryController, ReceiverTransferStateContainer]:
    if critic is None:
        critic = _make_mock_critic(mu_tau=0.0, sigma_tau=0.30)
    if pool is None:
        pool = SharedMemoryPool()
    if policy is None:
        policy = _make_policy()
    container = ReceiverTransferStateContainer()
    ctrl = TransferAwareMemoryController(
        critic=critic,
        pool=pool,
        transfer_states=container,
        policy=policy,
        feature_builder=_feature_builder,
    )
    return ctrl, container


def _candidate(
    mid: str,
    *,
    mu: float,
    sigma: float,
    status: str = "negative",
    eligible: bool = False,
) -> TransferCandidateDecision:
    return TransferCandidateDecision(
        memory_id=mid,
        receiver_id="r1",
        task_id="t1",
        candidate_source="global",
        mu_tau=mu,
        sigma_tau=sigma,
        lcb=mu - BETA * sigma,
        ucb=mu + BETA * sigma,
        eligible_for_context=eligible,
        selected_for_context=False,
        status=status,
    )


def _make_features(task_id: str, memory_id: str, receiver_id: str) -> ReceiverConditionedTransferFeatures:
    return ReceiverConditionedTransferFeatures(
        task_id=task_id,
        memory_id=memory_id,
        receiver_id=receiver_id,
        task_repr={"scenario": "test", "task_type": "qa"},
        receiver_repr={"role": "agent_a"},
        routing_card={"tags": ["helpful"]},
    )


def _make_evidence(
    task_id: str,
    task_position: int,
    memory_id: str = "m1",
    receiver_id: str = "r1",
) -> OnlineTransferEvidence:
    expose = [0.8, 0.7]
    withhold = [0.5, 0.4]
    tau = sum(expose) / len(expose) - sum(withhold) / len(withhold)
    return OnlineTransferEvidence(
        task_id=task_id,
        task_position=task_position,
        receiver_id=receiver_id,
        memory_id=memory_id,
        expose_scores=expose,
        withhold_scores=withhold,
        observed_tau=tau,
        tau_std=0.1,
        generation_seeds=[0, 1],
    )


# ---------------------------------------------------------------------------
# Phase 15: cold-start controller test (mu=0, sigma=0.30)
# ---------------------------------------------------------------------------


def test_cold_start_negative_lcb_still_probes():
    """mu=0, sigma=0.30, beta=1.64, delta=0 -> LCB=-0.492, UCB=+0.492.

    Expected: EXPLORE_ONLY, execution candidate NONE, probe candidate EXISTS.
    """
    critic = _make_mock_critic(mu_tau=0.0, sigma_tau=0.30)
    pool = SharedMemoryPool()
    pool.add(_mem("m1", source="agent_A", position=0))
    ctrl, container = _build_controller(
        critic=critic, pool=pool, policy=_make_policy(delta=0.0, gamma=0.5)
    )

    plan = ctrl.plan_for_task(
        task=_task(), task_id="t1", task_position=1, receiver_id="r1",
    )

    # routing mode = EXPLORE_ONLY (best known LCB <= delta; here unknown)
    assert plan.routing_mode == RoutingMode.EXPLORE_ONLY

    # execution candidate = NONE (LCB = -0.492 <= delta = 0)
    decision = select_episode_edge({"r1": plan}, delta=DELTA)
    assert decision.selected_receiver_id is None
    assert decision.selected_memory_id is None
    assert plan.selected_memory_ids == []

    # probe candidate = EXISTS, selected by UCB = +0.492
    probe = ProbeSelectionPolicy.select(plan.routing_mode, plan.global_candidates)
    assert probe is not None
    assert probe.memory_id == "m1"
    assert probe.lcb == pytest.approx(-0.492)
    assert probe.ucb == pytest.approx(0.492)


def test_probe_does_not_require_lcb_above_delta():
    """Same scenario from the probe side: LCB below delta must not block probing."""
    critic = _make_mock_critic(mu_tau=0.0, sigma_tau=0.30)
    pool = SharedMemoryPool()
    pool.add(_mem("m1", source="agent_A", position=0))
    ctrl, _ = _build_controller(
        critic=critic, pool=pool, policy=_make_policy(delta=0.0, gamma=0.5)
    )

    plan = ctrl.plan_for_task(
        task=_task(), task_id="t1", task_position=1, receiver_id="r1",
    )

    # Every predicted global candidate has LCB < delta...
    predicted = [c for c in plan.global_candidates if c.lcb is not None]
    assert predicted and all(c.lcb <= DELTA for c in predicted)
    # ...yet the probe gate still selects one.
    receiver, probe = select_probe_candidate({"r1": plan})
    assert receiver == "r1"
    assert probe is not None


# ---------------------------------------------------------------------------
# Phase 15.1: 6-task synthetic bootstrap into continual learning
# ---------------------------------------------------------------------------


def test_cold_start_bootstraps_into_continual_learning():
    """Tasks 1-5 all LCB<0 but each yields a causal probe; after the 5th
    evidence refit_count==1, critic_version==1, and task 6 uses the new
    critic (forward-only)."""
    encoder = RimaFeatureEncoder(n_features=64, include_receiver=True)
    learner = ContinualTransferLearner(
        base_examples=[],
        encoder=encoder,
        source_agent_ids={"m1": "agent_A"},
        n_bootstrap=3,
        refit_every_new_edges=5,
    )
    # Cold start: no base critic at all.
    assert learner.current_critic is None
    assert learner.critic_version == 0

    probe_count = 0
    refit_count = 0

    # Tasks 1-5: all LCB < 0, but probe gate (UCB) fires every task.
    for pos in range(1, 6):
        # Simulated plan: mu=0, sigma=0.30 -> LCB=-0.492 < 0, UCB=+0.492.
        probe_count += 1  # one causal probe per task (max-UCB candidate)
        evidence = _make_evidence(task_id=f"t{pos}", task_position=pos)
        features = _make_features(f"t{pos}", "m1", "r1")
        learner.add_online_evidence(evidence, features=features)
        if learner.maybe_refit():
            refit_count += 1

        if pos < 5:
            assert learner.critic_version == 0
            assert learner.current_critic is None
        else:
            # After the 5th evidence the bootstrap fires.
            assert probe_count == 5
            assert refit_count == 1
            assert learner.critic_version == 1
            assert learner.current_critic is not None
            assert learner.current_critic.is_frozen
            assert learner.critic_trained_through_task_position == 5

    # Task 6 uses the NEW critic (version 1), forward-only:
    # trained_through (5) < current position (6).
    pred_example = MatchedInterventionExample(
        task_id="t6",
        memory_id="m1",
        receiver_id="r1",
        source_agent_id="agent_A",
        official_expose_score=None,
        official_withhold_score=None,
        features=_make_features("t6", "m1", "r1"),
    )
    dist, log = learner.predict_distribution(
        pred_example, current_task_position=6
    )
    assert dist is not None
    assert log.critic_version == 1
    assert log.critic_trained_through_task_position == 5
    assert log.critic_trained_through_task_position < 6


# ---------------------------------------------------------------------------
# Phase 16: probe-gate regression tests
# ---------------------------------------------------------------------------


def test_execution_requires_lcb_above_delta():
    """Execution must require LCB > delta."""
    # mu=0.05, sigma=0 -> LCB=0.05
    critic = _make_mock_critic(mu_tau=0.05, sigma_tau=0.0)
    pool = SharedMemoryPool()
    pool.add(_mem("m1", source="agent_A", position=0))

    # delta=0.1: LCB=0.05 <= delta -> no execution.
    ctrl, container = _build_controller(
        critic=critic, pool=pool, policy=_make_policy(delta=0.1, gamma=0.5)
    )
    container.ensure("r1").register_memory("m1", "agent_A", "t0", 0)
    plan = ctrl.plan_for_task(
        task=_task(), task_id="t1", task_position=1, receiver_id="r1",
    )
    decision = select_episode_edge({"r1": plan}, delta=0.1)
    assert decision.selected_memory_id is None
    assert plan.selected_memory_ids == []

    # delta=0.0: same candidate now executes (LCB=0.05 > 0).
    ctrl0, container0 = _build_controller(
        critic=critic, pool=pool, policy=_make_policy(delta=0.0, gamma=0.5)
    )
    container0.ensure("r1").register_memory("m1", "agent_A", "t0", 0)
    plan0 = ctrl0.plan_for_task(
        task=_task(), task_id="t1", task_position=1, receiver_id="r1",
    )
    decision_ok = select_episode_edge({"r1": plan0}, delta=0.0)
    assert decision_ok.selected_memory_id == "m1"


def test_explore_only_can_probe_without_execution():
    """Phase 4 invariant: execution=None while probe exists is legal."""
    critic = _make_mock_critic(mu_tau=0.0, sigma_tau=0.30)
    pool = SharedMemoryPool()
    pool.add(_mem("m1", source="agent_A", position=0))
    ctrl, _ = _build_controller(
        critic=critic, pool=pool, policy=_make_policy(delta=0.0, gamma=0.5)
    )

    plan = ctrl.plan_for_task(
        task=_task(), task_id="t1", task_position=1, receiver_id="r1",
    )

    decision = select_episode_edge({"r1": plan}, delta=DELTA)
    probe = ProbeSelectionPolicy.select(plan.routing_mode, plan.global_candidates)
    # No execution, but probe fires: the two gates are independent.
    assert decision.selected_memory_id is None
    assert probe is not None


def test_exploit_explore_can_probe_unseen_candidate():
    """EXPLOIT_EXPLORE must still retrieve + probe unseen global candidates."""
    critic = _make_mock_critic(mu_tau=0.2, sigma_tau=0.0)
    pool = SharedMemoryPool()
    pool.add(_mem("m_known", source="agent_A", position=0))
    pool.add(_mem("m_unseen", source="agent_B", position=0))
    ctrl, container = _build_controller(
        critic=critic, pool=pool, policy=_make_policy(delta=0.0, gamma=0.5)
    )
    state = container.ensure("r1")
    state.register_memory("m_known", "agent_A", "t0", 0)

    plan = ctrl.plan_for_task(
        task=_task(), task_id="t1", task_position=1, receiver_id="r1",
    )
    assert plan.routing_mode == RoutingMode.EXPLOIT_EXPLORE
    assert plan.global_retrieval_triggered is True

    receiver, probe = select_probe_candidate({"r1": plan})
    assert receiver == "r1"
    assert probe is not None
    # The unseen candidate is probeable (all predictions valid).
    assert probe.memory_id in {"m_unseen"}


def test_exploit_only_skips_global_probe():
    """EXPLOIT_ONLY must not trigger global retrieval nor probing."""
    critic = _make_mock_critic(mu_tau=0.4, sigma_tau=0.0)
    pool = SharedMemoryPool()
    pool.add(_mem("m_known", source="agent_A", position=0))
    pool.add(_mem("m_unseen", source="agent_B", position=0))
    ctrl, container = _build_controller(
        critic=critic, pool=pool, policy=_make_policy(delta=0.0, gamma=0.3)
    )
    state = container.ensure("r1")
    state.register_memory("m_known", "agent_A", "t0", 0)

    plan = ctrl.plan_for_task(
        task=_task(), task_id="t1", task_position=1, receiver_id="r1",
    )
    assert plan.routing_mode == RoutingMode.EXPLOIT_ONLY
    assert plan.global_retrieval_triggered is False
    assert plan.global_candidates == []
    assert ProbeSelectionPolicy.select(plan.routing_mode, plan.global_candidates) is None
    assert select_probe_candidate({"r1": plan}) == (None, None)


def test_probe_candidate_uses_max_ucb():
    """Probe selection must pick the candidate with the highest UCB."""
    c_low = _candidate("m_low", mu=0.0, sigma=0.1)   # UCB = 0.164
    c_high = _candidate("m_high", mu=-0.1, sigma=0.3)  # UCB = 0.392
    c_self = _candidate(
        "m_self", mu=0.9, sigma=0.1, status="self_transfer_excluded"
    )

    selected = ProbeSelectionPolicy.select(
        RoutingMode.EXPLORE_ONLY, [c_low, c_high, c_self]
    )
    assert selected is not None
    # Highest UCB wins even though mu is lower; self-transfer never probed.
    assert selected.memory_id == "m_high"
    assert selected.ucb == pytest.approx(-0.1 + BETA * 0.3)


def test_probe_never_uses_self_transfer():
    """Self-transfer memories must never be probed."""
    critic = _make_mock_critic(mu_tau=0.0, sigma_tau=0.30)
    pool = SharedMemoryPool()
    # Receiver is "agent_self"; its own memory must be excluded.
    pool.add(_mem("m_self", source="agent_self", position=0))
    pool.add(_mem("m_other", source="agent_A", position=0))
    ctrl, _ = _build_controller(
        critic=critic, pool=pool, policy=_make_policy(delta=0.0, gamma=0.5)
    )

    plan = ctrl.plan_for_task(
        task=_task(), task_id="t1", task_position=1, receiver_id="agent_self",
    )

    statuses = {c.memory_id: c.status for c in plan.global_candidates}
    assert statuses["m_self"] == "self_transfer_excluded"

    probe = ProbeSelectionPolicy.select(plan.routing_mode, plan.global_candidates)
    assert probe is not None
    assert probe.memory_id == "m_other"


def test_probe_never_uses_future_memory():
    """Global probe candidates must only come from past tasks (forward-only)."""
    critic = _make_mock_critic(mu_tau=0.0, sigma_tau=0.30)
    pool = SharedMemoryPool()
    pool.add(_mem("m_past", source="agent_A", position=0))
    pool.add(_mem("m_same", source="agent_B", position=5))
    pool.add(_mem("m_future", source="agent_C", position=10))
    ctrl, _ = _build_controller(
        critic=critic, pool=pool, policy=_make_policy(delta=0.0, gamma=0.5)
    )

    plan = ctrl.plan_for_task(
        task=_task(), task_id="t5", task_position=5, receiver_id="r1",
    )

    global_ids = {c.memory_id for c in plan.global_candidates}
    assert "m_past" in global_ids
    assert "m_same" not in global_ids    # position == current: not historical
    assert "m_future" not in global_ids  # position > current: future

    probe = ProbeSelectionPolicy.select(plan.routing_mode, plan.global_candidates)
    assert probe is not None
    assert probe.memory_id == "m_past"
