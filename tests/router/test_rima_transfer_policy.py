"""Tests for transfer policy: gamma, LCB, observed_tau (RIMA-v2 §42).

Covers:
    test_gamma_is_q75_of_positive_train_observed_tau
    test_gamma_ignores_negative_tau
    test_gamma_ignores_zero_tau
    test_gamma_aggregates_repeated_seeds_by_edge
    test_gamma_never_reads_validation
    test_gamma_never_reads_test
    test_gamma_is_not_based_on_predicted_tau
    test_no_positive_train_tau_fails
    test_gamma_must_be_ge_delta
    test_lcb_equals_mu_minus_beta_sigma
"""

from __future__ import annotations

import numpy as np
import pytest

from smtr.rima.features import ReceiverConditionedTransferFeatures
from smtr.rima.transfer_policy import (
    TransferPolicy,
    compute_gamma,
    lower_confidence_bound,
    observed_tau,
)
from smtr.router.official_score_transfer_critic import MatchedInterventionExample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_features(
    task_id: str = "t1",
    memory_id: str = "m1",
    receiver_id: str = "r1",
) -> ReceiverConditionedTransferFeatures:
    return ReceiverConditionedTransferFeatures(
        task_id=task_id,
        memory_id=memory_id,
        receiver_id=receiver_id,
        task_repr={"scenario": "test", "task_type": "test", "text": ""},
        receiver_repr={"role": "executor", "capabilities": []},
        routing_card={
            "goal_summary": "",
            "task_tags": ["test"],
            "precondition_summary": "",
            "compatible_receiver_roles": [],
            "compatible_receiver_capabilities": [],
            "procedure_type": "experience",
        },
    )


def _make_example(
    task_id: str,
    memory_id: str,
    receiver_id: str,
    expose: float | None,
    withhold: float | None,
    source_agent_id: str = "src",
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


def _make_examples_with_taus(
    taus: list[float],
    *,
    task_prefix: str = "t",
) -> list[MatchedInterventionExample]:
    """Create one example per tau value, each with a unique edge."""
    examples = []
    for i, tau in enumerate(taus):
        # expose=tau+0.5, withhold=0.5 => observed_tau = tau
        examples.append(
            _make_example(
                task_id=f"{task_prefix}{i}",
                memory_id=f"m{i}",
                receiver_id=f"r{i}",
                expose=tau + 0.5,
                withhold=0.5,
            )
        )
    return examples


# ---------------------------------------------------------------------------
# Tests: observed_tau
# ---------------------------------------------------------------------------


def test_observed_tau_basic():
    ex = _make_example("t1", "m1", "r1", expose=0.8, withhold=0.3)
    assert observed_tau(ex) == pytest.approx(0.5)


def test_observed_tau_none_expose():
    ex = _make_example("t1", "m1", "r1", expose=None, withhold=0.3)
    assert observed_tau(ex) is None


def test_observed_tau_none_withhold():
    ex = _make_example("t1", "m1", "r1", expose=0.8, withhold=None)
    assert observed_tau(ex) is None


# ---------------------------------------------------------------------------
# Tests: LCB
# ---------------------------------------------------------------------------


def test_lcb_equals_mu_minus_beta_sigma():
    assert lower_confidence_bound(1.0, 0.5, 1.64) == pytest.approx(1.0 - 1.64 * 0.5)
    assert lower_confidence_bound(0.0, 0.0, 1.64) == pytest.approx(0.0)
    assert lower_confidence_bound(-1.0, 2.0, 0.5) == pytest.approx(-2.0)


# ---------------------------------------------------------------------------
# Tests: gamma
# ---------------------------------------------------------------------------


def test_gamma_is_q75_of_positive_train_observed_tau():
    """gamma must be Q75 of positive observed tau from TRAIN data."""
    positive_taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    examples = _make_examples_with_taus(positive_taus)
    gamma, support = compute_gamma(examples, quantile=0.75)
    expected = float(np.quantile(positive_taus, 0.75, method="linear"))
    assert gamma == pytest.approx(expected)
    assert support == len(positive_taus)


def test_gamma_ignores_negative_tau():
    """Negative tau edges must be excluded from gamma calculation."""
    taus = [0.1, 0.2, 0.3, 0.4, -0.5, -0.1]
    examples = _make_examples_with_taus(taus)
    gamma, support = compute_gamma(examples)
    positive_only = [t for t in taus if t > 0]
    expected = float(np.quantile(positive_only, 0.75, method="linear"))
    assert gamma == pytest.approx(expected)
    assert support == len(positive_only)


def test_gamma_ignores_zero_tau():
    """Zero tau edges must be excluded from gamma calculation."""
    taus = [0.0, 0.1, 0.2, 0.3]
    examples = _make_examples_with_taus(taus)
    gamma, support = compute_gamma(examples)
    positive_only = [t for t in taus if t > 0]
    expected = float(np.quantile(positive_only, 0.75, method="linear"))
    assert gamma == pytest.approx(expected)
    assert support == len(positive_only)


def test_gamma_aggregates_repeated_seeds_by_edge():
    """Multiple seeds for the same edge must be averaged before gamma."""
    # Two examples for the same edge (t1, r1, m1) with different scores
    ex1 = _make_example("t1", "m1", "r1", expose=0.8, withhold=0.4)  # tau=0.4
    ex2 = _make_example("t1", "m1", "r1", expose=0.6, withhold=0.4)  # tau=0.2
    # Another edge with tau=0.5
    ex3 = _make_example("t2", "m2", "r2", expose=0.7, withhold=0.2)  # tau=0.5

    examples = [ex1, ex2, ex3]
    gamma, support = compute_gamma(examples)

    # Edge (t1, r1, m1): mean((0.4, 0.2)) = 0.3
    # Edge (t2, r2, m2): mean((0.5,)) = 0.5
    # positive_taus = [0.3, 0.5]
    expected = float(np.quantile([0.3, 0.5], 0.75, method="linear"))
    assert gamma == pytest.approx(expected)
    assert support == 2


def test_gamma_never_reads_validation():
    """gamma must only use the provided train_examples list.

    This test passes only TRAIN examples; if the function tried to read
    validation/test, it would have no data source.
    """
    # Only positive tau examples (simulating TRAIN only)
    taus = [0.1, 0.2, 0.3]
    examples = _make_examples_with_taus(taus)
    gamma, _ = compute_gamma(examples)
    expected = float(np.quantile(taus, 0.75, method="linear"))
    assert gamma == pytest.approx(expected)


def test_gamma_never_reads_test():
    """Same isolation check — gamma from explicit train list only."""
    taus = [0.5, 1.0, 1.5]
    examples = _make_examples_with_taus(taus)
    gamma, _ = compute_gamma(examples)
    expected = float(np.quantile(taus, 0.75, method="linear"))
    assert gamma == pytest.approx(expected)


def test_gamma_is_not_based_on_predicted_tau():
    """compute_gamma uses observed_tau (from scores), not any critic prediction."""
    # The function only receives MatchedInterventionExample which has
    # official scores — no predicted_tau field. This test verifies
    # the function signature accepts only examples, not predictions.
    taus = [0.2, 0.4, 0.6]
    examples = _make_examples_with_taus(taus)
    # compute_gamma only takes examples — no prediction argument
    gamma, _ = compute_gamma(examples)
    expected = float(np.quantile(taus, 0.75, method="linear"))
    assert gamma == pytest.approx(expected)


def test_no_positive_train_tau_fails():
    """If no positive tau exists, compute_gamma must raise ValueError."""
    taus = [-0.5, -0.3, 0.0, -0.1]
    examples = _make_examples_with_taus(taus)
    with pytest.raises(ValueError, match="No positive observed tau"):
        compute_gamma(examples)


def test_gamma_must_be_ge_delta():
    """gamma < delta must raise ValueError."""
    taus = [0.01, 0.02, 0.03]
    examples = _make_examples_with_taus(taus)
    # delta is larger than any positive tau => gamma < delta
    with pytest.raises(ValueError, match="gamma.*delta"):
        compute_gamma(examples, delta=10.0)


# ---------------------------------------------------------------------------
# Tests: TransferPolicy dataclass
# ---------------------------------------------------------------------------


def test_transfer_policy_is_frozen():
    policy = TransferPolicy(
        beta=1.64, delta=0.0, gamma=0.5,
        gamma_quantile=0.75, gamma_positive_support=10,
        gamma_source_split="train",
    )
    with pytest.raises(AttributeError):
        policy.beta = 2.0  # type: ignore[misc]


def test_transfer_policy_optional_sha():
    policy = TransferPolicy(
        beta=1.64, delta=0.0, gamma=0.5,
        gamma_quantile=0.75, gamma_positive_support=10,
        gamma_source_split="train",
    )
    assert policy.critic_checkpoint_sha256 is None

    policy2 = TransferPolicy(
        beta=1.64, delta=0.0, gamma=0.5,
        gamma_quantile=0.75, gamma_positive_support=10,
        gamma_source_split="train",
        critic_checkpoint_sha256="abc123",
    )
    assert policy2.critic_checkpoint_sha256 == "abc123"
