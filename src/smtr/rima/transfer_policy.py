"""Transfer policy module (RIMA-v2 §9-11).

Defines the routing policy parameters and computation functions for the
continual transfer controller:

* LCB (Lower Confidence Bound): ``mu - beta * sigma``
* UCB (Upper Confidence Bound): ``mu + beta * sigma``
* gamma: Q75 of positive observed train tau
* delta: minimum LCB threshold for exploitation

LCB gates *exploitation* (execution safety: ``LCB > delta``), while UCB
drives *exploration* (probe candidate selection). Probe eligibility must
NOT require ``LCB > delta`` — that would deadlock cold start.

β = 1.64 is the **conservative uncertainty coefficient** (§16.1).
We do NOT claim "95% confidence" unless empirical coverage supports it.

gamma must be computed ONLY from TRAIN split observed tau — never from
validation, test, or predicted tau. gamma is NOT updated online (§16.5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from smtr.router.official_score_transfer_critic import MatchedInterventionExample

__all__ = [
    "TransferPolicy",
    "lower_confidence_bound",
    "upper_confidence_bound",
    "observed_tau",
    "compute_gamma",
]


@dataclass(frozen=True)
class TransferPolicy:
    """Frozen routing policy parameters.

    Attributes:
        beta: conservative uncertainty coefficient (default 1.64, §16.1).
        delta: minimum LCB threshold for exploitation (default 0.0).
        gamma: Q75 of positive observed train tau.
        gamma_quantile: quantile used to compute gamma (default 0.75).
        gamma_positive_support: number of positive-tau edges used.
        gamma_source_split: split name used for gamma ("train").
        critic_checkpoint_sha256: SHA256 of the critic checkpoint.
    """

    beta: float
    delta: float
    gamma: float

    gamma_quantile: float
    gamma_positive_support: int
    gamma_source_split: str

    critic_checkpoint_sha256: str | None = None


def lower_confidence_bound(
    mu: float,
    sigma: float,
    beta: float,
) -> float:
    """Compute Lower Confidence Bound: ``mu - beta * sigma``."""
    return mu - beta * sigma


def upper_confidence_bound(
    mu: float,
    sigma: float,
    beta: float,
) -> float:
    """Compute Upper Confidence Bound: ``mu + beta * sigma``.

    Used for exploration/probe candidate selection only. Reuses the same
    ``beta`` as LCB — no new hyperparameter.
    """
    return mu + beta * sigma


def observed_tau(ex: "MatchedInterventionExample") -> float | None:
    """Compute observed tau from a matched intervention example.

    Returns None if either score is missing (fail-closed).
    """
    if ex.official_expose_score is None:
        return None
    if ex.official_withhold_score is None:
        return None
    return ex.official_expose_score - ex.official_withhold_score


def compute_gamma(
    train_examples: list["MatchedInterventionExample"],
    *,
    quantile: float = 0.75,
    delta: float = 0.0,
) -> tuple[float, int]:
    """Compute gamma = Q75 of positive observed train tau.

    Aggregates by treatment edge (task_id, receiver_id, memory_id) to
    handle multiple seeds per edge, then takes the mean per edge.

    Args:
        train_examples: TRAIN split examples only.
        quantile: quantile for gamma (default 0.75).
        delta: delta threshold; gamma must be >= delta.

    Returns:
        (gamma, positive_support) tuple.

    Raises:
        ValueError: if no positive observed tau in train data,
            or if gamma < delta.
    """
    # Aggregate by edge: (task_id, receiver_id, memory_id)
    edge_taus: dict[tuple[str, str, str], list[float]] = {}
    for ex in train_examples:
        tau = observed_tau(ex)
        if tau is None:
            continue
        key = (ex.task_id, ex.receiver_id, ex.memory_id)
        edge_taus.setdefault(key, []).append(tau)

    # Mean tau per edge
    edge_means = [float(np.mean(vals)) for vals in edge_taus.values()]

    # Filter positive
    positive_taus = [t for t in edge_means if t > 0]
    if not positive_taus:
        raise ValueError(
            "No positive observed tau in TRAIN split — cannot compute gamma. "
            "Training data may be insufficient or all effects are non-positive."
        )

    gamma = float(np.quantile(positive_taus, quantile, method="linear"))
    if gamma < delta:
        raise ValueError(
            f"gamma ({gamma:.6f}) < delta ({delta}). "
            "This violates the invariant gamma >= delta."
        )

    return gamma, len(positive_taus)
