"""Off-policy correction for transfer effect estimation (B-04).

This module provides importance weighting and correction mechanisms for
paired intervention records collected under different routing policies.
When the data collection policy (pi_old) differs from the evaluation policy
(pi_new), importance weights correct for the distribution shift.

Key concepts:
- Importance weight: w = pi_new(a|o,S) / pi_old(a|o,S)
- Corrected tau: tau_corrected = w * tau_observed
- Weight clipping: prevent extreme corrections from dominating
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from pydantic import BaseModel, ConfigDict

from smtr.counterfactual.schemas import PairedInterventionRecord


class WeightingScheme(str, Enum):
    """Supported importance weighting schemes."""

    RATIO = "ratio"
    """Standard importance ratio: pi_new / pi_old."""

    CLIPPED_RATIO = "clipped_ratio"
    """Clipped importance ratio with configurable bounds."""

    SELF_NORMALIZED = "self_normalized"
    """Self-normalized importance weights (sum to 1)."""


class OffPolicyConfig(BaseModel):
    """Configuration for off-policy correction."""

    model_config = ConfigDict(frozen=True)

    max_weight: float = 10.0
    """Maximum allowed importance weight (clipping upper bound)."""

    min_weight: float = 0.01
    """Minimum allowed importance weight (clipping lower bound)."""

    default_selection_probability: float = 0.5
    """Default selection probability when not recorded."""

    default_prefix_probability: float = 0.5
    """Default prefix sampling probability when not recorded."""

    weighting_scheme: WeightingScheme = WeightingScheme.CLIPPED_RATIO
    """Weighting scheme to use."""

    require_recorded_probabilities: bool = False
    """If True, raise error when probabilities are missing from records."""


@dataclass
class CorrectionResult:
    """Result of off-policy correction for a single record."""

    record_id: str
    raw_weight: float
    clipped_weight: float
    tau_observed: float
    tau_corrected: float
    transfer_class: str
    is_clipped: bool


@dataclass
class CorrectionSummary:
    """Summary of off-policy correction across multiple records."""

    n_records: int
    n_clipped: int
    mean_weight: float
    median_weight: float
    max_weight_applied: float
    min_weight_applied: float
    tau_mean_uncorrected: float
    tau_mean_corrected: float
    effective_sample_size: float
    weights: list[float] = field(default_factory=list)


class OffPolicyCorrector:
    """Computes importance weights and corrects transfer effect estimates.

    The corrector handles records collected under different routing policies
    by computing importance weights that adjust for the distribution shift
    between the data collection policy and the evaluation policy.

    Usage:
        corrector = OffPolicyCorrector(config=OffPolicyConfig())
        result = corrector.correct_record(record, target_prob=0.3)
        summary = corrector.correct_batch(records, target_prob=0.3)
    """

    def __init__(
        self,
        *,
        config: OffPolicyConfig | None = None,
    ) -> None:
        self.config = config or OffPolicyConfig()

    def compute_weight(
        self,
        *,
        target_probability: float,
        behavior_probability: float,
    ) -> tuple[float, float]:
        """Compute importance weight w = target_prob / behavior_prob.

        Returns:
            Tuple of (raw_weight, clipped_weight)
        """
        # Avoid division by zero
        if behavior_probability <= 0.0:
            behavior_probability = self.config.min_weight
        if target_probability <= 0.0:
            target_probability = self.config.min_weight

        raw_weight = target_probability / behavior_probability

        if self.config.weighting_scheme == WeightingScheme.CLIPPED_RATIO:
            clipped = float(
                np.clip(raw_weight, self.config.min_weight, self.config.max_weight)
            )
        elif self.config.weighting_scheme == WeightingScheme.RATIO:
            clipped = raw_weight
        else:
            # Self-normalized: computed at batch level, return raw for now
            clipped = raw_weight

        return float(raw_weight), float(clipped)

    def correct_record(
        self,
        record: PairedInterventionRecord,
        *,
        target_probability: float | None = None,
    ) -> CorrectionResult:
        """Compute off-policy correction for a single paired record.

        Args:
            record: Paired intervention record with observed outcomes
            target_probability: Probability of the action under the target policy.
                If None, uses the record's recorded probability or default.

        Returns:
            CorrectionResult with raw and corrected tau estimates
        """
        # Get behavior probability (from data collection)
        behavior_prob = self._get_behavior_probability(record)

        # Get target probability (for evaluation policy)
        if target_probability is None:
            target_prob = self._get_default_target_probability(record)
        else:
            target_prob = target_probability

        # Compute importance weight
        raw_weight, clipped_weight = self.compute_weight(
            target_probability=target_prob,
            behavior_probability=behavior_prob,
        )

        # Compute observed tau (marginal effect: y_share - y_withhold)
        tau_observed = float(record.y_share - record.y_withhold)

        # Apply correction
        tau_corrected = clipped_weight * tau_observed

        is_clipped = abs(raw_weight - clipped_weight) > 1e-9

        return CorrectionResult(
            record_id=record.record_id,
            raw_weight=raw_weight,
            clipped_weight=clipped_weight,
            tau_observed=tau_observed,
            tau_corrected=tau_corrected,
            transfer_class=record.transfer_class,
            is_clipped=is_clipped,
        )

    def correct_batch(
        self,
        records: list[PairedInterventionRecord],
        *,
        target_probability: float | None = None,
    ) -> CorrectionSummary:
        """Compute off-policy correction for a batch of records.

        For self-normalized weighting, computes weights and normalizes them
        to sum to 1 across the batch.

        Returns:
            CorrectionSummary with aggregate statistics
        """
        if not records:
            return CorrectionSummary(
                n_records=0,
                n_clipped=0,
                mean_weight=0.0,
                median_weight=0.0,
                max_weight_applied=0.0,
                min_weight_applied=0.0,
                tau_mean_uncorrected=0.0,
                tau_mean_corrected=0.0,
                effective_sample_size=0.0,
            )

        results = [
            self.correct_record(record, target_probability=target_probability)
            for record in records
        ]

        weights = [r.clipped_weight for r in results]

        # Self-normalized: normalize weights to sum to n_records
        if self.config.weighting_scheme == WeightingScheme.SELF_NORMALIZED:
            total = sum(weights)
            if total > 0:
                norm_factor = len(weights) / total
                weights = [w * norm_factor for w in weights]
                # Recompute corrected tau with normalized weights
                for i, result in enumerate(results):
                    tau_observed = result.tau_observed
                    result.clipped_weight = weights[i]
                    result.tau_corrected = weights[i] * tau_observed

        tau_observed_values = [r.tau_observed for r in results]
        tau_corrected_values = [r.tau_corrected for r in results]

        # Effective sample size: (sum w)^2 / sum(w^2)
        weights_array = np.array(weights)
        sum_w = float(weights_array.sum())
        sum_w2 = float((weights_array**2).sum())
        ess = (sum_w**2 / sum_w2) if sum_w2 > 0 else 0.0

        return CorrectionSummary(
            n_records=len(records),
            n_clipped=sum(1 for r in results if r.is_clipped),
            mean_weight=float(np.mean(weights)),
            median_weight=float(np.median(weights)),
            max_weight_applied=float(np.max(weights)),
            min_weight_applied=float(np.min(weights)),
            tau_mean_uncorrected=float(np.mean(tau_observed_values)),
            tau_mean_corrected=float(np.mean(tau_corrected_values)),
            effective_sample_size=ess,
            weights=weights,
        )

    def compute_ess(self, weights: list[float]) -> float:
        """Compute effective sample size from a list of weights.

        ESS = (sum w)^2 / sum(w^2)
        Higher ESS indicates more uniform weights (less distribution shift).
        """
        if not weights:
            return 0.0
        w = np.array(weights)
        sum_w = float(w.sum())
        sum_w2 = float((w**2).sum())
        if sum_w2 == 0:
            return 0.0
        return (sum_w**2) / sum_w2

    def _get_behavior_probability(self, record: PairedInterventionRecord) -> float:
        """Extract the behavior probability from a record."""
        if record.target_selection_probability is not None:
            return float(record.target_selection_probability)
        if self.config.require_recorded_probabilities:
            raise ValueError(
                f"Record {record.record_id} has no target_selection_probability"
            )
        return self.config.default_selection_probability

    def _get_default_target_probability(
        self, record: PairedInterventionRecord
    ) -> float:
        """Get default target probability for a record."""
        return self.config.default_selection_probability


class PolicyRatioEstimator:
    """Estimates the ratio between two policies for a given context.

    This is used when we need to estimate pi_new(a|o,S) / pi_old(a|o,S)
    but don't have direct access to the new policy's probabilities.
    """

    def __init__(
        self,
        *,
        epsilon: float = 0.1,
        n_candidates: int = 5,
    ) -> None:
        self.epsilon = epsilon
        self.n_candidates = n_candidates

    def estimate_ratio(
        self,
        *,
        old_probability: float,
        new_rank: int | None = None,
        n_candidates: int | None = None,
    ) -> float:
        """Estimate the policy ratio pi_new / pi_old.

        For an epsilon-greedy policy:
        - If the action is the greedy choice: prob = (1-eps) + eps/k
        - If the action is non-greedy: prob = eps/k

        Args:
            old_probability: Probability under the old policy
            new_rank: Rank of the action under new policy (0 = best)
            n_candidates: Number of candidate actions

        Returns:
            Estimated ratio pi_new / pi_old
        """
        k = n_candidates or self.n_candidates
        eps = self.epsilon

        if new_rank is None or new_rank == 0:
            # Greedy action
            new_prob = (1.0 - eps) + eps / k
        else:
            # Non-greedy action
            new_prob = eps / k

        if old_probability <= 0:
            old_probability = 1e-6

        return new_prob / old_probability

    def uniform_probability(self, n_candidates: int | None = None) -> float:
        """Return the probability under a uniform random policy."""
        k = n_candidates or self.n_candidates
        return 1.0 / k if k > 0 else 0.0
