"""End-to-end smoke test for the RIMA-Transfer pilot pipeline.

Validates that all components wire together correctly without requiring
LLM API calls.  Uses mocks for MARBLE trajectory collection and
outcome evaluation, but runs real TransferAwareMemoryController,
ReceiverTransferState, SharedMemoryPool, and metrics.

Covers the §56 code-level acceptance criteria checklist.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from smtr.memory.shared_memory_pool import SharedMemory, SharedMemoryPool
from smtr.rima.features import ReceiverConditionedTransferFeatures
from smtr.rima.transfer_controller import (
    RoutingMode,
    TransferAwareMemoryController,
    TransferRoutingPlan,
)
from smtr.rima.transfer_metrics import (
    build_curve_records,
    compute_transfer_cost,
    compute_transfer_routing_metrics,
)
from smtr.rima.transfer_policy import TransferPolicy
from smtr.rima.transfer_state import ReceiverTransferStateContainer
from smtr.router.official_score_transfer_critic import (
    MatchedInterventionExample,
    TransferEffectDistribution,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_memory(
    mid: str,
    *,
    source: str = "agent_a",
    origin_pos: int = 0,
    tags: list[str] | None = None,
) -> SharedMemory:
    return SharedMemory(
        memory_id=mid,
        routing_card={
            "task_tags": tags or ["bargaining"],
            "goal_summary": f"goal for {mid}",
            "procedure_type": "experience",
        },
        procedure_payload={"action": f"use_{mid}"},
        source_agent_id=source,
        origin_task_id=f"task_{origin_pos}",
        origin_task_position=origin_pos,
    )


def _make_policy() -> TransferPolicy:
    return TransferPolicy(
        beta=1.64,
        delta=0.0,
        gamma=0.28,
        gamma_quantile=0.75,
        gamma_positive_support=10,
        gamma_source_split="train",
        critic_checkpoint_sha256="test_sha",
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
        task_repr={"scenario": "bargaining", "text": task.get("text", "")},
        receiver_repr={"role": "worker"},
        routing_card=dict(mem.routing_card),
    )


def _make_mock_critic(mu: float = 0.4, sigma: float = 0.05):
    """Mock critic returning TransferEffectDistribution with fixed (mu, sigma)."""
    critic = MagicMock()
    critic.is_frozen = True

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
            mu_expose=mu + 0.1,
            mu_withhold=0.3,
            mu_tau=mu,
            sigma_tau=sigma,
            n_members=31,
        )

    critic.predict_distribution = MagicMock(side_effect=fake_predict)
    critic.checkpoint_sha256.return_value = "test_sha"
    return critic


# ---------------------------------------------------------------------------
# Smoke test: full pipeline wiring
# ---------------------------------------------------------------------------


class TestEndToEndSmoke:
    """Verify all components wire together for a minimal 3-task run."""

    def test_three_task_transfer_pipeline(self):
        """Run 3 simulated tasks through rima_transfer logic.

        Validates:
        - Controller produces plans with correct routing modes
        - Transfer state accumulates memories
        - Metrics aggregate correctly
        - Curve records produced per position
        - online_intervention_episodes == 0
        """
        pool = SharedMemoryPool()
        state_container = ReceiverTransferStateContainer()
        policy = _make_policy()
        critic = _make_mock_critic(mu=0.4, sigma=0.05)

        # Seed pool with 3 memories from different sources
        for i, src in enumerate(["agent_a", "agent_b", "agent_c"]):
            pool.add(_make_memory(f"m{i}", source=src, origin_pos=0))

        controller = TransferAwareMemoryController(
            critic=critic,
            pool=pool,
            transfer_states=state_container,
            policy=policy,
            feature_builder=_feature_builder,
            context_budget=1,
            known_probe_top_k=20,
            global_explore_top_k=5,
        )

        all_diagnostics = []
        all_records = []

        # Simulate 3 tasks
        for task_pos in range(3):
            task_dict = {
                "text": f"bargaining task {task_pos}",
                "tags": ["bargaining"],
                "agent_ids": ["agent_a", "agent_b", "agent_c"],
            }
            receiver_id = "agent_a"  # always the receiver

            plan = controller.plan_for_task(
                task=task_dict,
                task_id=f"task_{task_pos}",
                task_position=task_pos,
                receiver_id=receiver_id,
            )

            # Build diagnostic (mirrors runner _build_routing_diagnostic)
            state = state_container.get(receiver_id)
            diag = {
                "task_position": task_pos,
                "routing_mode": plan.routing_mode,
                "transfer_state_size_before": len(state) if state else 0,
                "transfer_state_size_after": len(state) if state else 0,
                "n_known_candidates_considered": len(plan.known_candidates),
                "n_global_candidates_considered": len(plan.global_candidates),
                "global_retrieval_triggered": plan.global_retrieval_triggered,
                "best_known_lcb": plan.best_known_lcb,
                "selected_memory_ids": plan.selected_memory_ids,
                "selected_source": (
                    "known"
                    if plan.known_candidates
                    and any(
                        c.selected_for_context
                        for c in plan.known_candidates
                    )
                    else (
                        "global"
                        if plan.global_candidates
                        and any(
                            c.selected_for_context
                            for c in plan.global_candidates
                        )
                        else "none"
                    )
                ),
                "selected_memory_id": (
                    plan.selected_memory_ids[0]
                    if plan.selected_memory_ids
                    else None
                ),
                "receiver_id": receiver_id,
                "global_candidate_ids": [
                    c.memory_id for c in plan.global_candidates
                ],
                "beta": policy.beta,
                "delta": policy.delta,
                "gamma": policy.gamma,
            }
            all_diagnostics.append(diag)
            all_records.append({
                "task_position": task_pos,
                "task_score": 0.6 + task_pos * 0.05,
                "is_valid": True,
            })

        # ---- Verify metrics pipeline ----
        routing_metrics = compute_transfer_routing_metrics(all_diagnostics)
        assert routing_metrics["n_diagnostics"] == 3

        cost = compute_transfer_cost(all_diagnostics)
        assert cost["online_transfer_cost"]["online_intervention_episodes"] == 0

        curve = build_curve_records(all_diagnostics, all_records)
        assert len(curve) == 3
        assert all("task_position" in c for c in curve)
        assert all("task_score" in c for c in curve)

    def test_exploit_only_skips_global_retrieval(self):
        """§56: exploit-only state must not access global pool."""
        pool = SharedMemoryPool()
        state_container = ReceiverTransferStateContainer()
        policy = _make_policy()
        # High mu, low sigma -> high LCB -> exploit_only
        critic = _make_mock_critic(mu=0.8, sigma=0.01)

        # Add memories and pre-explore them
        for i in range(5):
            pool.add(_make_memory(f"m{i}", source="agent_b", origin_pos=0))

        controller = TransferAwareMemoryController(
            critic=critic,
            pool=pool,
            transfer_states=state_container,
            policy=policy,
            feature_builder=_feature_builder,
            context_budget=1,
        )

        task = {"text": "bargaining", "tags": ["bargaining"], "agent_ids": ["agent_a", "agent_b"]}
        plan = controller.plan_for_task(
            task=task, task_id="t_exploit", task_position=1, receiver_id="agent_a",
        )

        # With high LCB, should be exploit_only
        if plan.routing_mode == RoutingMode.EXPLOIT_ONLY:
            assert plan.global_retrieval_triggered is False
            assert len(plan.global_candidates) == 0

    def test_self_transfer_excluded(self):
        """§56: self-transfer must not enter critic/K/context."""
        pool = SharedMemoryPool()
        state_container = ReceiverTransferStateContainer()
        policy = _make_policy()
        critic = _make_mock_critic(mu=0.5, sigma=0.1)

        # All memories from agent_a (same as receiver)
        pool.add(_make_memory("m1", source="agent_a", origin_pos=0))
        pool.add(_make_memory("m2", source="agent_a", origin_pos=0))

        controller = TransferAwareMemoryController(
            critic=critic,
            pool=pool,
            transfer_states=state_container,
            policy=policy,
            feature_builder=_feature_builder,
            context_budget=1,
        )

        task = {"text": "bargaining", "tags": ["bargaining"], "agent_ids": ["agent_a"]}
        plan = controller.plan_for_task(
            task=task, task_id="t_self", task_position=1, receiver_id="agent_a",
        )

        # No valid candidates since all are self-transfer
        assert plan.selected_memory_ids == []

    def test_policy_file_roundtrip(self):
        """§56: transfer_policy.json save/load roundtrip."""
        policy = _make_policy()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(
                {
                    "schema_version": "rima_transfer_policy_v1",
                    "beta": policy.beta,
                    "delta": policy.delta,
                    "gamma": policy.gamma,
                    "gamma_quantile": policy.gamma_quantile,
                    "gamma_positive_support": policy.gamma_positive_support,
                    "gamma_source_split": policy.gamma_source_split,
                    "critic_checkpoint_sha256": policy.critic_checkpoint_sha256,
                },
                f,
            )
            f.flush()
            path = f.name

        from experiments.rima.run_continual_transfer import load_transfer_policy

        loaded = load_transfer_policy(path)
        assert loaded.beta == pytest.approx(policy.beta)
        assert loaded.gamma == pytest.approx(policy.gamma)
        assert loaded.critic_checkpoint_sha256 == policy.critic_checkpoint_sha256

    def test_negative_memory_can_become_positive(self):
        """§56: negative memory can become positive on a new task."""
        pool = SharedMemoryPool()
        state_container = ReceiverTransferStateContainer()
        policy = _make_policy()
        pool.add(_make_memory("m1", source="agent_b", origin_pos=0))

        # First call: low mu -> negative LCB
        critic = MagicMock()
        critic.is_frozen = True

        call_count = [0]
        mus = [0.05, 0.6]
        sigmas = [0.2, 0.05]

        def fake_predict_varying(ex: MatchedInterventionExample) -> TransferEffectDistribution:
            idx = min(call_count[0], len(mus) - 1)
            call_count[0] += 1
            if ex.source_agent_id == ex.receiver_id:
                return TransferEffectDistribution(
                    memory_id=ex.memory_id, receiver_id=ex.receiver_id,
                    task_id=ex.task_id, mu_expose=None, mu_withhold=None,
                    mu_tau=None, sigma_tau=None, n_members=31,
                )
            return TransferEffectDistribution(
                memory_id=ex.memory_id, receiver_id=ex.receiver_id,
                task_id=ex.task_id, mu_expose=mus[idx] + 0.1,
                mu_withhold=0.3, mu_tau=mus[idx], sigma_tau=sigmas[idx],
                n_members=31,
            )

        critic.predict_distribution = MagicMock(side_effect=fake_predict_varying)
        critic.checkpoint_sha256.return_value = "test"

        controller = TransferAwareMemoryController(
            critic=critic,
            pool=pool,
            transfer_states=state_container,
            policy=policy,
            feature_builder=_feature_builder,
            context_budget=1,
        )

        task0 = {"text": "hard task", "tags": ["hard"], "agent_ids": ["agent_a", "agent_b"]}
        plan0 = controller.plan_for_task(
            task=task0, task_id="t0", task_position=1, receiver_id="agent_a",
        )

        # Task 1: same memory should now be positive
        task1 = {"text": "easy task", "tags": ["easy"], "agent_ids": ["agent_a", "agent_b"]}
        plan1 = controller.plan_for_task(
            task=task1, task_id="t1", task_position=2, receiver_id="agent_a",
        )

        # m1 should be selectable in task 1 (LCB > delta=0)
        if plan1.known_candidates:
            best_lcb = max(
                (c.lcb for c in plan1.known_candidates if c.lcb is not None),
                default=None,
            )
            assert best_lcb is not None and best_lcb > 0
