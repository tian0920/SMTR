"""Tests for continual transfer learner (§15).

Covers:
    1. Initialization with base examples
    2. Online evidence accumulation
    3. Refit threshold (should_refit / maybe_refit)
    4. Full refit (base + online → new critic)
    5. critic_version increment
    6. Forward-only leakage guard
    7. CriticPredictionLog metadata
    8. Frozen baseline preserved (rima_transfer_frozen)
    9. force_refit override
   10. Edge cases (no base, self-transfer source)
"""

from __future__ import annotations

import pytest

from smtr.rima.continual_transfer_learner import (
    RIMA_TRANSFER_ADAPTIVE,
    RIMA_TRANSFER_FROZEN,
    ContinualTransferLearner,
    CriticPredictionLog,
)
from smtr.rima.features import (
    ReceiverConditionedTransferFeatures,
    RimaFeatureEncoder,
)
from smtr.rima.online_transfer_evidence import OnlineTransferEvidence
from smtr.router.official_score_transfer_critic import (
    MatchedInterventionExample,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_encoder() -> RimaFeatureEncoder:
    return RimaFeatureEncoder(n_features=64, include_receiver=True)


def _make_features(
    task_id: str = "t1",
    memory_id: str = "m1",
    receiver_id: str = "r1",
) -> ReceiverConditionedTransferFeatures:
    return ReceiverConditionedTransferFeatures(
        task_id=task_id,
        memory_id=memory_id,
        receiver_id=receiver_id,
        task_repr={"scenario": "test", "task_type": "qa"},
        receiver_repr={"role": "agent_a"},
        routing_card={"tags": ["helpful"]},
    )


def _make_example(
    task_id: str = "t1",
    memory_id: str = "m1",
    receiver_id: str = "r1",
    source_agent_id: str = "agent_x",
    expose: float = 0.8,
    withhold: float = 0.5,
) -> MatchedInterventionExample:
    return MatchedInterventionExample(
        task_id=task_id,
        memory_id=memory_id,
        receiver_id=receiver_id,
        source_agent_id=source_agent_id,
        official_expose_score=expose,
        official_withhold_score=withhold,
        features=_make_features(task_id, memory_id, receiver_id),
    )


def _make_evidence(
    task_id: str = "t5",
    task_position: int = 5,
    memory_id: str = "m1",
    receiver_id: str = "r1",
    expose_scores: list[float] | None = None,
    withhold_scores: list[float] | None = None,
) -> OnlineTransferEvidence:
    e = expose_scores or [0.8, 0.7]
    w = withhold_scores or [0.5, 0.4]
    tau = sum(e) / len(e) - sum(w) / len(w)
    return OnlineTransferEvidence(
        task_id=task_id,
        task_position=task_position,
        receiver_id=receiver_id,
        memory_id=memory_id,
        expose_scores=e,
        withhold_scores=w,
        observed_tau=tau,
        tau_std=0.1,
        generation_seeds=[0, 1],
    )


def _make_base_examples(n_tasks: int = 3) -> list[MatchedInterventionExample]:
    """Create base training examples across multiple tasks."""
    examples = []
    for i in range(n_tasks):
        for j in range(2):
            examples.append(
                _make_example(
                    task_id=f"base_t{i}",
                    memory_id=f"m{j}",
                    receiver_id="r1",
                    source_agent_id=f"agent_{j}",
                    expose=0.7 + 0.05 * j,
                    withhold=0.5,
                )
            )
    return examples


# ---------------------------------------------------------------------------
# Test: method name constants
# ---------------------------------------------------------------------------


def test_method_name_constants():
    """Method name constants must be defined correctly."""
    assert RIMA_TRANSFER_FROZEN == "rima_transfer_frozen"
    assert RIMA_TRANSFER_ADAPTIVE == "rima_transfer_adaptive"


# ---------------------------------------------------------------------------
# Test: initialization
# ---------------------------------------------------------------------------


def test_init_with_base_examples():
    """Learner must fit and freeze initial critic on base examples."""
    encoder = _make_encoder()
    base = _make_base_examples()

    learner = ContinualTransferLearner(
        base_examples=base,
        encoder=encoder,
        source_agent_ids={"m0": "agent_0", "m1": "agent_1"},
        n_bootstrap=3,
    )

    assert learner.current_critic is not None
    assert learner.current_critic.is_frozen
    assert learner.critic_version == 1
    assert learner.n_base_examples == len(base)
    assert learner.n_online_examples == 0
    assert learner.edges_since_last_refit == 0


def test_init_without_base_examples():
    """Learner without base examples must have no initial critic."""
    encoder = _make_encoder()

    learner = ContinualTransferLearner(
        base_examples=[],
        encoder=encoder,
        n_bootstrap=3,
    )

    assert learner.current_critic is None
    assert learner.critic_version == 0
    assert learner.n_base_examples == 0


# ---------------------------------------------------------------------------
# Test: online evidence accumulation
# ---------------------------------------------------------------------------


def test_add_online_evidence():
    """add_online_evidence must convert and accumulate."""
    encoder = _make_encoder()
    base = _make_base_examples()

    learner = ContinualTransferLearner(
        base_examples=base,
        encoder=encoder,
        source_agent_ids={"m1": "agent_0"},
        n_bootstrap=3,
    )

    evidence = _make_evidence(task_id="t5", task_position=5)
    features = _make_features("t5", "m1", "r1")

    example = learner.add_online_evidence(evidence, features=features)

    assert isinstance(example, MatchedInterventionExample)
    assert example.task_id == "t5"
    assert example.memory_id == "m1"
    assert example.receiver_id == "r1"
    # Mean of [0.8, 0.7] = 0.75
    assert example.official_expose_score == pytest.approx(0.75)
    # Mean of [0.5, 0.4] = 0.45
    assert example.official_withhold_score == pytest.approx(0.45)

    assert learner.n_online_examples == 1
    assert learner.edges_since_last_refit == 1


def test_add_online_evidence_source_agent_lookup():
    """Source agent must be looked up from mapping when not provided."""
    encoder = _make_encoder()
    learner = ContinualTransferLearner(
        base_examples=_make_base_examples(),
        encoder=encoder,
        source_agent_ids={"m1": "agent_source"},
        n_bootstrap=3,
    )

    evidence = _make_evidence(memory_id="m1")
    features = _make_features("t5", "m1", "r1")
    example = learner.add_online_evidence(evidence, features=features)

    assert example.source_agent_id == "agent_source"


def test_add_online_evidence_explicit_source():
    """Explicit source_agent_id must override mapping."""
    encoder = _make_encoder()
    learner = ContinualTransferLearner(
        base_examples=_make_base_examples(),
        encoder=encoder,
        source_agent_ids={"m1": "agent_mapped"},
        n_bootstrap=3,
    )

    evidence = _make_evidence(memory_id="m1")
    features = _make_features("t5", "m1", "r1")
    example = learner.add_online_evidence(
        evidence, features=features, source_agent_id="agent_explicit"
    )

    assert example.source_agent_id == "agent_explicit"


# ---------------------------------------------------------------------------
# Test: refit threshold
# ---------------------------------------------------------------------------


def test_should_refit_threshold():
    """should_refit triggers at the configured threshold."""
    encoder = _make_encoder()
    learner = ContinualTransferLearner(
        base_examples=_make_base_examples(),
        encoder=encoder,
        n_bootstrap=3,
        refit_every_new_edges=3,
    )

    # Add evidence one by one
    for i in range(2):
        evidence = _make_evidence(
            task_id=f"t{i}", task_position=i, memory_id=f"m{i}"
        )
        features = _make_features(f"t{i}", f"m{i}", "r1")
        learner.add_online_evidence(evidence, features=features)
        assert not learner.should_refit()

    # Third evidence triggers the threshold
    evidence = _make_evidence(task_id="t2", task_position=2, memory_id="m2")
    features = _make_features("t2", "m2", "r1")
    learner.add_online_evidence(evidence, features=features)
    assert learner.should_refit()


def test_maybe_refit_returns_false_below_threshold():
    """maybe_refit must return False when below threshold."""
    encoder = _make_encoder()
    learner = ContinualTransferLearner(
        base_examples=_make_base_examples(),
        encoder=encoder,
        n_bootstrap=3,
        refit_every_new_edges=5,
    )

    evidence = _make_evidence(task_id="t1", task_position=1)
    features = _make_features("t1", "m1", "r1")
    learner.add_online_evidence(evidence, features=features)

    result = learner.maybe_refit()
    assert result is False
    assert learner.critic_version == 1  # Unchanged


def test_maybe_refit_returns_true_at_threshold():
    """maybe_refit must return True and update critic at threshold."""
    encoder = _make_encoder()
    learner = ContinualTransferLearner(
        base_examples=_make_base_examples(),
        encoder=encoder,
        source_agent_ids={f"m{i}": f"agent_{i}" for i in range(5)},
        n_bootstrap=3,
        refit_every_new_edges=3,
    )

    old_version = learner.critic_version

    for i in range(3):
        evidence = _make_evidence(
            task_id=f"online_t{i}", task_position=i + 10, memory_id=f"m{i}"
        )
        features = _make_features(f"online_t{i}", f"m{i}", "r1")
        learner.add_online_evidence(evidence, features=features)

    result = learner.maybe_refit()
    assert result is True
    assert learner.critic_version == old_version + 1
    assert learner.edges_since_last_refit == 0


# ---------------------------------------------------------------------------
# Test: refit uses base + online
# ---------------------------------------------------------------------------


def test_refit_uses_base_plus_online():
    """After refit, critic must be trained on base + online examples."""
    encoder = _make_encoder()
    base = _make_base_examples(n_tasks=2)

    learner = ContinualTransferLearner(
        base_examples=base,
        encoder=encoder,
        source_agent_ids={"m0": "a0", "m1": "a1", "m_online": "a2"},
        n_bootstrap=3,
        refit_every_new_edges=2,
    )

    # Add 2 online examples
    for i in range(2):
        evidence = _make_evidence(
            task_id=f"online_{i}",
            task_position=100 + i,
            memory_id="m_online",
        )
        features = _make_features(f"online_{i}", "m_online", "r1")
        learner.add_online_evidence(evidence, features=features)

    assert learner.maybe_refit() is True
    assert learner.critic_version == 2

    # Critic should be frozen after refit
    assert learner.current_critic is not None
    assert learner.current_critic.is_frozen

    # Training stats should reflect combined data
    stats = learner.current_critic._training_stats
    # base has 4 examples (2 tasks × 2), online has 2 = 6 total
    assert stats["n_examples_total"] == 6


# ---------------------------------------------------------------------------
# Test: critic_version increment
# ---------------------------------------------------------------------------


def test_critic_version_increments():
    """Each refit must increment critic_version."""
    encoder = _make_encoder()
    learner = ContinualTransferLearner(
        base_examples=_make_base_examples(),
        encoder=encoder,
        source_agent_ids={f"m{i}": f"a{i}" for i in range(10)},
        n_bootstrap=3,
        refit_every_new_edges=1,
    )

    assert learner.critic_version == 1

    # Each evidence triggers a refit (threshold = 1)
    for i in range(3):
        evidence = _make_evidence(
            task_id=f"t_{i}", task_position=i + 10, memory_id=f"m{i}"
        )
        features = _make_features(f"t_{i}", f"m{i}", "r1")
        learner.add_online_evidence(evidence, features=features)
        learner.maybe_refit()

    assert learner.critic_version == 4  # 1 initial + 3 refits


# ---------------------------------------------------------------------------
# Test: forward-only leakage guard
# ---------------------------------------------------------------------------


def test_forward_only_leakage_guard():
    """predict_distribution must reject if critic was trained on >= current position."""
    encoder = _make_encoder()
    learner = ContinualTransferLearner(
        base_examples=_make_base_examples(),
        encoder=encoder,
        source_agent_ids={"m0": "a0"},
        n_bootstrap=3,
        refit_every_new_edges=1,
    )

    # Add online evidence at position 10 and refit
    evidence = _make_evidence(task_id="t10", task_position=10, memory_id="m0")
    features = _make_features("t10", "m0", "r1")
    learner.add_online_evidence(evidence, features=features)
    learner.maybe_refit()

    # critic is now trained through position 10
    assert learner.critic_trained_through_task_position == 10

    # Predicting at position 10 should fail (must be strictly >)
    pred_example = _make_example(task_id="t10", memory_id="m0")
    with pytest.raises(AssertionError, match="Leakage"):
        learner.predict_distribution(pred_example, current_task_position=10)

    # Predicting at position 11 should succeed
    pred_example_11 = _make_example(task_id="t11", memory_id="m0")
    dist, log = learner.predict_distribution(
        pred_example_11, current_task_position=11
    )
    assert dist is not None
    assert log.critic_version == 2


def test_predict_before_any_refit_succeeds():
    """Prediction with initial critic should work (trained_through = -1)."""
    encoder = _make_encoder()
    learner = ContinualTransferLearner(
        base_examples=_make_base_examples(),
        encoder=encoder,
        n_bootstrap=3,
    )

    pred_example = _make_example(task_id="t1", memory_id="m0")
    dist, log = learner.predict_distribution(
        pred_example, current_task_position=0
    )
    assert dist is not None
    assert log.critic_version == 1
    assert log.critic_trained_through_task_position == -1


# ---------------------------------------------------------------------------
# Test: CriticPredictionLog
# ---------------------------------------------------------------------------


def test_prediction_log_metadata():
    """CriticPredictionLog must carry version and trained-through info."""
    log = CriticPredictionLog(
        critic_version=3,
        critic_trained_through_task_position=17,
    )
    assert log.critic_version == 3
    assert log.critic_trained_through_task_position == 17

    # Frozen dataclass
    with pytest.raises(AttributeError):
        log.critic_version = 4  # type: ignore


def test_prediction_log_after_refit():
    """Prediction log must reflect updated version after refit."""
    encoder = _make_encoder()
    learner = ContinualTransferLearner(
        base_examples=_make_base_examples(),
        encoder=encoder,
        source_agent_ids={"m0": "a0"},
        n_bootstrap=3,
        refit_every_new_edges=1,
    )

    # Add evidence at position 5 and refit
    evidence = _make_evidence(task_id="t5", task_position=5, memory_id="m0")
    features = _make_features("t5", "m0", "r1")
    learner.add_online_evidence(evidence, features=features)
    learner.maybe_refit()

    # Predict at position 6
    pred_example = _make_example(task_id="t6", memory_id="m0")
    _, log = learner.predict_distribution(
        pred_example, current_task_position=6
    )

    assert log.critic_version == 2
    assert log.critic_trained_through_task_position == 5


# ---------------------------------------------------------------------------
# Test: force_refit
# ---------------------------------------------------------------------------


def test_force_refit_below_threshold():
    """force_refit must trigger refit even below threshold."""
    encoder = _make_encoder()
    learner = ContinualTransferLearner(
        base_examples=_make_base_examples(),
        encoder=encoder,
        source_agent_ids={"m0": "a0"},
        n_bootstrap=3,
        refit_every_new_edges=100,  # Very high threshold
    )

    assert learner.critic_version == 1

    # Add just 1 evidence (well below threshold of 100)
    evidence = _make_evidence(task_id="t1", task_position=1, memory_id="m0")
    features = _make_features("t1", "m0", "r1")
    learner.add_online_evidence(evidence, features=features)

    assert not learner.should_refit()
    assert learner.maybe_refit() is False

    # Force refit
    learner.force_refit()
    assert learner.critic_version == 2
    assert learner.edges_since_last_refit == 0


# ---------------------------------------------------------------------------
# Test: no critic available
# ---------------------------------------------------------------------------


def test_predict_without_critic_raises():
    """Predicting without any critic must raise RuntimeError."""
    encoder = _make_encoder()
    learner = ContinualTransferLearner(
        base_examples=[],
        encoder=encoder,
        n_bootstrap=3,
    )

    pred_example = _make_example()
    with pytest.raises(RuntimeError, match="No critic"):
        learner.predict_distribution(pred_example, current_task_position=0)


# ---------------------------------------------------------------------------
# Test: initial base examples are independent copy
# ---------------------------------------------------------------------------


def test_base_examples_independent_copy():
    """Learner must not mutate the input base_examples list."""
    encoder = _make_encoder()
    base = _make_base_examples()
    original_len = len(base)

    learner = ContinualTransferLearner(
        base_examples=base,
        encoder=encoder,
        n_bootstrap=3,
    )

    # Mutating the original list must not affect the learner
    base.clear()
    assert learner.n_base_examples == original_len
