"""Tests for TransferAwareMemoryController (Commit 5 / §17-25, §43-44)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from smtr.memory.shared_memory_pool import SharedMemory, SharedMemoryPool
from smtr.rima.features import ReceiverConditionedTransferFeatures
from smtr.rima.transfer_controller import (
    RoutingMode,
    TransferAwareMemoryController,
    TransferCandidateDecision,
    TransferRoutingPlan,
)
from smtr.rima.transfer_policy import TransferPolicy
from smtr.rima.transfer_state import (
    ReceiverTransferState,
    ReceiverTransferStateContainer,
)
from smtr.router.official_score_transfer_critic import (
    BootstrapOfficialScoreTransferCritic,
    MatchedInterventionExample,
    PotentialOutcomeMember,
    TransferEffectDistribution,
)


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


def _make_policy(*, beta: float = 1.64, delta: float = 0.0, gamma: float = 0.3) -> TransferPolicy:
    return TransferPolicy(beta=beta, delta=delta, gamma=gamma, gamma_quantile=0.75, gamma_positive_support=1, gamma_source_split="train")


def _make_mock_critic(mu_tau: float = 0.5, sigma_tau: float = 0.1) -> MagicMock:
    """Mock critic that returns fixed (mu, sigma) for predict_distribution."""
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
            mu_expose=0.8,
            mu_withhold=0.3,
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
        critic = _make_mock_critic()
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


# ---------------------------------------------------------------------------
# §43: three routing modes
# ---------------------------------------------------------------------------

class TestRoutingModes:
    def test_empty_state_triggers_explore_only(self):
        ctrl, _ = _build_controller()
        plan = ctrl.plan_for_task(
            task=_task(), task_id="t1", task_position=1, receiver_id="r1",
        )
        assert plan.routing_mode == RoutingMode.EXPLORE_ONLY
        assert plan.best_known_lcb is None

    def test_best_lcb_below_delta_triggers_explore_only(self):
        # mu=0.05, sigma=0.0 => LCB=0.05, delta=0.1 → below delta
        critic = _make_mock_critic(mu_tau=0.05, sigma_tau=0.0)
        policy = _make_policy(delta=0.1, gamma=0.5)
        pool = SharedMemoryPool()
        pool.add(_mem("m1", source="agent_A", position=0))
        ctrl, container = _build_controller(critic=critic, pool=pool, policy=policy)

        # Pre-register known memory
        state = container.ensure("r1")
        state.register_memory("m1", "agent_A", "t0", 0)

        plan = ctrl.plan_for_task(
            task=_task(), task_id="t1", task_position=1, receiver_id="r1",
        )
        assert plan.routing_mode == RoutingMode.EXPLORE_ONLY

    def test_positive_but_below_gamma_triggers_exploit_explore(self):
        # mu=0.2, sigma=0.0 => LCB=0.2, delta=0, gamma=0.5 → 0 < 0.2 < 0.5
        critic = _make_mock_critic(mu_tau=0.2, sigma_tau=0.0)
        policy = _make_policy(delta=0.0, gamma=0.5)
        pool = SharedMemoryPool()
        pool.add(_mem("m1", source="agent_A", position=0))
        ctrl, container = _build_controller(critic=critic, pool=pool, policy=policy)

        state = container.ensure("r1")
        state.register_memory("m1", "agent_A", "t0", 0)

        plan = ctrl.plan_for_task(
            task=_task(), task_id="t1", task_position=1, receiver_id="r1",
        )
        assert plan.routing_mode == RoutingMode.EXPLOIT_EXPLORE

    def test_best_lcb_equal_gamma_triggers_exploit_only(self):
        # mu=0.3, sigma=0.0 => LCB=0.3, gamma=0.3 → exactly gamma
        critic = _make_mock_critic(mu_tau=0.3, sigma_tau=0.0)
        policy = _make_policy(delta=0.0, gamma=0.3)
        pool = SharedMemoryPool()
        pool.add(_mem("m1", source="agent_A", position=0))
        ctrl, container = _build_controller(critic=critic, pool=pool, policy=policy)

        state = container.ensure("r1")
        state.register_memory("m1", "agent_A", "t0", 0)

        plan = ctrl.plan_for_task(
            task=_task(), task_id="t1", task_position=1, receiver_id="r1",
        )
        assert plan.routing_mode == RoutingMode.EXPLOIT_ONLY

    def test_best_lcb_above_gamma_triggers_exploit_only(self):
        # mu=0.8, sigma=0.1 => LCB=0.8 - 1.64*0.1 = 0.636, gamma=0.3
        critic = _make_mock_critic(mu_tau=0.8, sigma_tau=0.1)
        policy = _make_policy(delta=0.0, gamma=0.3)
        pool = SharedMemoryPool()
        pool.add(_mem("m1", source="agent_A", position=0))
        ctrl, container = _build_controller(critic=critic, pool=pool, policy=policy)

        state = container.ensure("r1")
        state.register_memory("m1", "agent_A", "t0", 0)

        plan = ctrl.plan_for_task(
            task=_task(), task_id="t1", task_position=1, receiver_id="r1",
        )
        assert plan.routing_mode == RoutingMode.EXPLOIT_ONLY

    def test_exploit_only_does_not_call_global_retrieval(self):
        """Most important invariant: EXPLOIT_ONLY never touches global pool."""
        critic = _make_mock_critic(mu_tau=0.8, sigma_tau=0.0)
        policy = _make_policy(delta=0.0, gamma=0.3)
        pool = MagicMock(spec=SharedMemoryPool)
        pool.rank_subset.return_value = [_mem("m1", source="agent_A", position=0)]

        ctrl, container = _build_controller(critic=critic, pool=pool, policy=policy)
        state = container.ensure("r1")
        state.register_memory("m1", "agent_A", "t0", 0)

        plan = ctrl.plan_for_task(
            task=_task(), task_id="t1", task_position=1, receiver_id="r1",
        )
        assert plan.routing_mode == RoutingMode.EXPLOIT_ONLY
        assert plan.global_retrieval_triggered is False
        pool.retrieve_unseen.assert_not_called()


# ---------------------------------------------------------------------------
# §44: unseen-global tests
# ---------------------------------------------------------------------------

class TestUnseenGlobalExploration:
    def test_global_exploration_excludes_known_ids(self):
        """retrieve_unseen must be called with exclude_memory_ids = known."""
        critic = _make_mock_critic(mu_tau=0.1, sigma_tau=0.0)  # LCB=0.1, below gamma=0.5
        policy = _make_policy(delta=0.0, gamma=0.5)
        pool = MagicMock(spec=SharedMemoryPool)
        pool.rank_subset.return_value = [_mem("m1", source="agent_A", position=0)]
        pool.retrieve_unseen.return_value = [_mem("m2", source="agent_A", position=0)]

        ctrl, container = _build_controller(critic=critic, pool=pool, policy=policy)
        state = container.ensure("r1")
        state.register_memory("m1", "agent_A", "t0", 0)

        ctrl.plan_for_task(
            task=_task(), task_id="t1", task_position=1, receiver_id="r1",
        )
        pool.retrieve_unseen.assert_called_once()
        call_kwargs = pool.retrieve_unseen.call_args
        exclude = call_kwargs.kwargs.get("exclude_memory_ids", call_kwargs[1].get("exclude_memory_ids"))
        assert "m1" in exclude

    def test_global_exploration_is_historical_only(self):
        pool = SharedMemoryPool()
        pool.add(_mem("m1", source="agent_A", position=0))
        pool.add(_mem("m_future", source="agent_A", position=5))

        critic = _make_mock_critic(mu_tau=0.1, sigma_tau=0.0)
        policy = _make_policy(delta=0.0, gamma=0.5)
        ctrl, container = _build_controller(critic=critic, pool=pool, policy=policy)

        plan = ctrl.plan_for_task(
            task=_task(), task_id="t1", task_position=2, receiver_id="r1",
        )
        # Only m1 should appear (m_future has position 5 >= 2)
        all_ids = {d.memory_id for d in plan.global_candidates}
        assert "m_future" not in all_ids

    def test_current_task_memory_cannot_be_explored(self):
        pool = SharedMemoryPool()
        pool.add(_mem("m1", source="agent_A", position=0))
        pool.add(_mem("m_current", source="agent_A", position=1))

        critic = _make_mock_critic(mu_tau=0.1, sigma_tau=0.0)
        policy = _make_policy(delta=0.0, gamma=0.5)
        ctrl, container = _build_controller(critic=critic, pool=pool, policy=policy)

        plan = ctrl.plan_for_task(
            task=_task(), task_id="t1", task_position=1, receiver_id="r1",
        )
        all_ids = {d.memory_id for d in plan.global_candidates}
        assert "m_current" not in all_ids

    def test_global_exploration_does_not_reconsider_known_negative_as_unknown(self):
        """Negative known memory must NOT appear in global unseen."""
        pool = SharedMemoryPool()
        pool.add(_mem("m1", source="agent_A", position=0))
        pool.add(_mem("m2", source="agent_A", position=0))

        critic = _make_mock_critic(mu_tau=0.1, sigma_tau=0.0)
        policy = _make_policy(delta=0.0, gamma=0.5)
        ctrl, container = _build_controller(critic=critic, pool=pool, policy=policy)

        # Both m1 and m2 are known (explored before)
        state = container.ensure("r1")
        state.register_memory("m1", "agent_A", "t0", 0)
        state.register_memory("m2", "agent_A", "t0", 0)

        plan = ctrl.plan_for_task(
            task=_task(), task_id="t1", task_position=1, receiver_id="r1",
        )
        global_ids = {d.memory_id for d in plan.global_candidates}
        assert "m1" not in global_ids
        assert "m2" not in global_ids


# ---------------------------------------------------------------------------
# Self-transfer
# ---------------------------------------------------------------------------

class TestSelfTransfer:
    def test_self_transfer_never_enters_context(self):
        pool = SharedMemoryPool()
        pool.add(_mem("m_self", source="r1", position=0))  # source == receiver

        critic = _make_mock_critic(mu_tau=0.9, sigma_tau=0.0)
        policy = _make_policy(delta=0.0, gamma=0.3)
        ctrl, _ = _build_controller(critic=critic, pool=pool, policy=policy)

        plan = ctrl.plan_for_task(
            task=_task(), task_id="t1", task_position=1, receiver_id="r1",
        )
        assert plan.selected_memory_ids == []
        # Check self-transfer decision exists
        self_decisions = [
            d for d in plan.global_candidates if d.status == "self_transfer_excluded"
        ]
        assert len(self_decisions) == 1
        assert self_decisions[0].memory_id == "m_self"


# ---------------------------------------------------------------------------
# Context budget
# ---------------------------------------------------------------------------

class TestContextBudget:
    def test_only_one_memory_per_receiver_is_injected(self):
        pool = SharedMemoryPool()
        pool.add(_mem("m1", source="agent_A", position=0))
        pool.add(_mem("m2", source="agent_B", position=0))

        critic = _make_mock_critic(mu_tau=0.8, sigma_tau=0.0)
        policy = _make_policy(delta=0.0, gamma=0.3)
        ctrl, _ = _build_controller(critic=critic, pool=pool, policy=policy)

        plan = ctrl.plan_for_task(
            task=_task(), task_id="t1", task_position=1, receiver_id="r1",
        )
        assert len(plan.selected_memory_ids) <= 1

    def test_highest_lcb_memory_is_selected(self):
        """When multiple candidates exist, the highest LCB wins."""
        pool = SharedMemoryPool()
        pool.add(_mem("m1", source="agent_A", position=0))
        pool.add(_mem("m2", source="agent_B", position=0))

        # Make critic return different predictions based on memory_id
        critic = MagicMock(spec=BootstrapOfficialScoreTransferCritic)

        def fake_predict(ex: MatchedInterventionExample) -> TransferEffectDistribution:
            mu = 0.9 if ex.memory_id == "m1" else 0.5
            return TransferEffectDistribution(
                memory_id=ex.memory_id,
                receiver_id=ex.receiver_id,
                task_id=ex.task_id,
                mu_expose=mu,
                mu_withhold=0.2,
                mu_tau=mu - 0.2,
                sigma_tau=0.0,
                n_members=31,
            )

        critic.predict_distribution = MagicMock(side_effect=fake_predict)

        policy = _make_policy(delta=0.0, gamma=0.3)
        ctrl, _ = _build_controller(critic=critic, pool=pool, policy=policy)

        plan = ctrl.plan_for_task(
            task=_task(), task_id="t1", task_position=1, receiver_id="r1",
        )
        assert plan.selected_memory_ids == ["m1"]


# ---------------------------------------------------------------------------
# State updates
# ---------------------------------------------------------------------------

class TestStateUpdates:
    def test_global_candidates_enter_transfer_state(self):
        pool = SharedMemoryPool()
        pool.add(_mem("m1", source="agent_A", position=0))

        critic = _make_mock_critic(mu_tau=0.1, sigma_tau=0.0)
        policy = _make_policy(delta=0.0, gamma=0.5)
        ctrl, container = _build_controller(critic=critic, pool=pool, policy=policy)

        ctrl.plan_for_task(
            task=_task(), task_id="t1", task_position=1, receiver_id="r1",
        )
        state = container.get("r1")
        assert state is not None
        assert state.contains("m1")

    def test_negative_candidate_still_in_state(self):
        """Negative LCB memory enters K_r (not discarded)."""
        # mu=-0.5, sigma=0.0 => LCB=-0.5 (negative)
        critic = _make_mock_critic(mu_tau=-0.5, sigma_tau=0.0)
        policy = _make_policy(delta=0.0, gamma=0.5)
        pool = SharedMemoryPool()
        pool.add(_mem("m1", source="agent_A", position=0))

        ctrl, container = _build_controller(critic=critic, pool=pool, policy=policy)

        ctrl.plan_for_task(
            task=_task(), task_id="t1", task_position=1, receiver_id="r1",
        )
        state = container.get("r1")
        assert state is not None
        assert state.contains("m1")
        assert not any(d.selected_for_context for d in (
            ctrl.plan_for_task(
                task=_task(), task_id="t2", task_position=2, receiver_id="r1",
            ).known_candidates
        ) if d.memory_id == "m1" and d.lcb is not None and d.lcb <= 0)
