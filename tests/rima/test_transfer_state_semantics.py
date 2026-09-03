"""Phase 6: strict PREDICTED_ONLY vs CAUSAL_OBSERVED state semantics.

These tests pin the statistical meaning of the transfer state:

* Registering/predicting a memory creates a PREDICTED_ONLY entry and
  must NOT count toward the causal state |K^causal|.
* The causal state grows ONLY when a successful probe records a
  matched causal observation (record_causal_observation upgrades the
  existing entry in place — never duplicates it).
* Zero probes over an entire run must imply zero online causal growth,
  regardless of how many memories are registered and predicted.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from smtr.memory.shared_memory_pool import SharedMemory, SharedMemoryPool
from smtr.rima.features import ReceiverConditionedTransferFeatures
from smtr.rima.transfer_controller import TransferAwareMemoryController
from smtr.rima.transfer_policy import TransferPolicy
from smtr.rima.transfer_state import (
    ReceiverTransferState,
    ReceiverTransferStateContainer,
)
from smtr.router.official_score_transfer_critic import (
    BootstrapOfficialScoreTransferCritic,
    MatchedInterventionExample,
    TransferEffectDistribution,
)


def _make_state(receiver_id: str = "r1") -> ReceiverTransferState:
    return ReceiverTransferState(receiver_id=receiver_id)


def test_predicted_registration_does_not_increase_causal_state():
    """register_memory + record_prediction must stay PREDICTED_ONLY."""
    state = _make_state()
    state.register_memory(
        memory_id="m1", source_agent_id="agent_A",
        task_id="t1", task_position=0,
    )
    state.record_prediction(
        memory_id="m1", task_id="t1", task_position=0,
        mu_tau=0.2, sigma_tau=0.3, lcb=-0.292,
        status="negative", candidate_source="global",
    )

    assert len(state) == 1
    assert len(state.predicted_only_entries()) == 1
    assert len(state.causal_observed_entries()) == 0


def test_causal_state_increases_only_after_successful_probe():
    """Causal count grows only via record_causal_observation (upgrade, no dup)."""
    state = _make_state()
    for mid in ("m1", "m2", "m3"):
        state.register_memory(
            memory_id=mid, source_agent_id="agent_A",
            task_id="t1", task_position=0,
        )
        state.record_prediction(
            memory_id=mid, task_id="t1", task_position=0,
            mu_tau=0.1, sigma_tau=0.3, lcb=-0.392,
            status="negative", candidate_source="global",
        )

    assert len(state.causal_observed_entries()) == 0

    # One successful probe upgrades m2 in place.
    state.record_causal_observation(memory_id="m2", task_id="t1", observed_tau=0.4)

    assert len(state) == 3  # no duplicate entry created
    assert len(state.causal_observed_entries()) == 1
    assert len(state.predicted_only_entries()) == 2
    entry = state.get_entry("m2")
    assert entry is not None
    assert entry.times_causally_probed == 1
    assert entry.observed_tau_n == 1

    # Repeated probes on the same edge accumulate stats, not entries.
    state.record_causal_observation(memory_id="m2", task_id="t2", observed_tau=0.6)
    assert len(state) == 3
    assert len(state.causal_observed_entries()) == 1
    assert state.get_entry("m2").observed_tau_n == 2


# ---------------------------------------------------------------------------
# Loop-level: zero probes must imply zero causal growth
# ---------------------------------------------------------------------------


def _mem(mid: str, source: str = "agent_A", position: int = 0) -> SharedMemory:
    return SharedMemory(
        memory_id=mid,
        source_agent_id=source,
        origin_task_id=f"origin_{mid}",
        origin_task_position=position,
        routing_card={"task_tags": [], "goal_summary": ""},
    )


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


def _cold_start_critic() -> MagicMock:
    """Cold-start critic: mu=0, sigma=0.30 -> LCB always negative."""
    critic = MagicMock(spec=BootstrapOfficialScoreTransferCritic)

    def fake_predict(ex: MatchedInterventionExample) -> TransferEffectDistribution:
        return TransferEffectDistribution(
            memory_id=ex.memory_id,
            receiver_id=ex.receiver_id,
            task_id=ex.task_id,
            mu_expose=0.3,
            mu_withhold=0.3,
            mu_tau=0.0,
            sigma_tau=0.30,
            n_members=31,
        )

    critic.predict_distribution = MagicMock(side_effect=fake_predict)
    return critic


def test_zero_probe_implies_zero_online_causal_growth():
    """Full routing loop with NO probes: state fills with PREDICTED_ONLY
    entries but the causal state must remain exactly zero."""
    pool = SharedMemoryPool()
    for i in range(4):
        pool.add(_mem(f"m{i}", position=0))

    container = ReceiverTransferStateContainer()
    ctrl = TransferAwareMemoryController(
        critic=_cold_start_critic(),
        pool=pool,
        transfer_states=container,
        policy=TransferPolicy(
            beta=1.64, delta=0.0, gamma=0.35,
            gamma_quantile=0.75, gamma_positive_support=2,
            gamma_source_split="train",
        ),
        feature_builder=_feature_builder,
    )

    for pos in range(1, 9):
        plan = ctrl.plan_for_task(
            task={"text": f"task {pos}", "tags": []},
            task_id=f"t{pos}", task_position=pos, receiver_id="r1",
        )
        assert plan.selected_memory_ids == [], (
            "cold-start critic must never select an execution edge"
        )
        # Deliberately NO record_causal_observation anywhere in the loop.

    state = container.get("r1")
    assert state is not None
    assert len(state) > 0, "memories should have been registered"
    assert len(state.predicted_only_entries()) == len(state)
    assert len(state.causal_observed_entries()) == 0
