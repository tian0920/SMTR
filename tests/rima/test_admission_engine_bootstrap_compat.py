"""Regression test: admission engine bootstrap critic compatibility.

Bug: ``RimaAdmissionEngine.__init__`` accessed ``critic._mu1`` / ``critic._mu0``
directly, which fails for ``BootstrapOfficialScoreTransferCritic`` (it stores
models in ``self.members``).

Fix: guard check now handles both critic types.
"""

from __future__ import annotations

import pytest

from smtr.memory.shared_memory_pool import SharedMemoryPool
from smtr.rima.admission_engine import RimaAdmissionEngine
from smtr.rima.features import ReceiverConditionedTransferFeatures, RimaFeatureEncoder
from smtr.router.official_score_transfer_critic import (
    BootstrapOfficialScoreTransferCritic,
    MatchedInterventionExample,
    OfficialScoreTransferCritic,
)


def _make_features(i: int) -> ReceiverConditionedTransferFeatures:
    return ReceiverConditionedTransferFeatures(
        task_id=f"task_{i // 2}",
        memory_id=f"mem_{i}",
        receiver_id=f"agent_{i % 3 + 1}",
        task_repr={"scenario": "bargaining", "task_type": "test"},
        receiver_repr={"role": "solver", "capabilities": ["reasoning"]},
        routing_card={"task_tags": ["logic"], "goal_summary": "solve"},
    )


def _make_examples(n: int = 10) -> list[MatchedInterventionExample]:
    """Minimal synthetic training examples."""
    examples: list[MatchedInterventionExample] = []
    for i in range(n):
        examples.append(
            MatchedInterventionExample(
                task_id=f"task_{i // 2}",
                memory_id=f"mem_{i}",
                receiver_id=f"agent_{i % 3 + 1}",
                source_agent_id=f"agent_{(i + 1) % 3 + 1}",
                official_expose_score=0.8 + i * 0.01,
                official_withhold_score=0.7 + i * 0.01,
                features=_make_features(i),
            )
        )
    return examples


def _feature_builder(memory, receiver_id, task):
    return ReceiverConditionedTransferFeatures(
        task_id="t0", memory_id=memory.memory_id,
        receiver_id=receiver_id,
        task_repr={}, receiver_repr={}, routing_card={},
    )


class TestBootstrapCriticAdmissionCompat:
    """RimaAdmissionEngine must accept BootstrapOfficialScoreTransferCritic."""

    def test_bootstrap_critic_accepted(self) -> None:
        encoder = RimaFeatureEncoder()
        critic = BootstrapOfficialScoreTransferCritic(
            encoder=encoder, n_bootstrap=3, seed=0,
        )
        critic.fit(_make_examples())
        critic.freeze()
        pool = SharedMemoryPool()
        # Must NOT raise AttributeError on _mu1.
        engine = RimaAdmissionEngine(
            critic=critic, pool=pool,
            feature_builder=_feature_builder,
        )
        assert engine.critic is critic

    def test_unfitted_bootstrap_critic_rejected(self) -> None:
        encoder = RimaFeatureEncoder()
        critic = BootstrapOfficialScoreTransferCritic(
            encoder=encoder, n_bootstrap=3, seed=0,
        )
        # Not fitted — members is empty.
        pool = SharedMemoryPool()
        with pytest.raises(RuntimeError, match="fitted"):
            RimaAdmissionEngine(
                critic=critic, pool=pool,
                feature_builder=_feature_builder,
            )

    def test_point_critic_still_works(self) -> None:
        encoder = RimaFeatureEncoder()
        critic = OfficialScoreTransferCritic(encoder=encoder)
        critic.fit(_make_examples())
        critic.freeze()
        pool = SharedMemoryPool()
        engine = RimaAdmissionEngine(
            critic=critic, pool=pool,
            feature_builder=_feature_builder,
        )
        assert engine.critic is critic
