"""Online policy refresh and active data acquisition (B-03).

This module provides mechanisms for updating the transfer critic online
as new paired intervention records become available. It includes:

- OnlinePolicyRefresher: Periodic critic retraining with new records
- ActiveDataAcquisition: Identify high-uncertainty regions for data collection
- PolicyVersionManager: Track policy versions during refresh

The system supports safe atomic updates where the critic is replaced
only after successful retraining, preventing partial state corruption.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from pydantic import BaseModel, ConfigDict

from smtr.counterfactual.schemas import PairedInterventionRecord
from smtr.router.off_policy_correction import OffPolicyConfig, OffPolicyCorrector
from smtr.router.transfer_critic import FourOutcomeTransferCritic
from smtr.router.transfer_features import (
    HashingTransferFeatureEncoder,
    TransferPredictionInput,
)


class RefreshTrigger(str, Enum):
    """Triggers for policy refresh."""

    RECORD_COUNT = "record_count"
    """Trigger when new record count exceeds threshold."""

    UNCERTAINTY = "uncertainty"
    """Trigger when prediction uncertainty exceeds threshold."""

    TIME_ELAPSED = "time_elapsed"
    """Trigger when time since last refresh exceeds threshold."""

    MANUAL = "manual"
    """Manual trigger only."""


class RefreshConfig(BaseModel):
    """Configuration for online policy refresh."""

    model_config = ConfigDict(frozen=True)

    trigger_mode: RefreshTrigger = RefreshTrigger.RECORD_COUNT
    """Primary trigger mode for refresh."""

    min_new_records: int = 10
    """Minimum new records before triggering refresh (for record_count mode)."""

    uncertainty_threshold: float = 0.4
    """Uncertainty threshold for triggering refresh (for uncertainty mode)."""

    max_refresh_attempts: int = 3
    """Maximum number of refresh attempts before giving up."""

    n_bootstrap: int = 31
    """Number of bootstrap samples for critic retraining."""

    retain_old_critic_on_failure: bool = True
    """If True, keep old critic if retraining fails."""

    enable_off_policy_correction: bool = True
    """If True, apply off-policy correction during retraining."""


@dataclass
class RefreshState:
    """Tracks the state of the policy refresh system."""

    current_version: int = 0
    last_refresh_version: int = 0
    total_records_seen: int = 0
    new_records_since_refresh: int = 0
    refresh_count: int = 0
    failed_refresh_count: int = 0
    last_uncertainty_score: float = 0.0


class OnlinePolicyRefresher:
    """Manages online critic retraining with new paired records.

    The refresher monitors incoming data and triggers critic retraining
    when configured thresholds are met. It supports off-policy correction
    to account for records collected under different policies.

    Usage:
        refresher = OnlinePolicyRefresher(
            initial_critic=critic,
            config=RefreshConfig(),
        )
        # Add new records
        refresher.add_records(new_records)
        # Check if refresh is needed
        if refresher.should_refresh():
            new_critic = refresher.refresh()
    """

    def __init__(
        self,
        *,
        initial_critic: FourOutcomeTransferCritic | None = None,
        config: RefreshConfig | None = None,
        off_policy_config: OffPolicyConfig | None = None,
    ) -> None:
        self.critic = initial_critic or FourOutcomeTransferCritic()
        self.config = config or RefreshConfig()
        self.off_policy_corrector = OffPolicyCorrector(config=off_policy_config)
        self.state = RefreshState()
        self._pending_records: list[PairedInterventionRecord] = []
        self._all_records: list[PairedInterventionRecord] = []

    def add_records(self, records: list[PairedInterventionRecord]) -> int:
        """Add new paired records to the pending buffer.

        Returns:
            Number of records added
        """
        self._pending_records.extend(records)
        self._all_records.extend(records)
        self.state.new_records_since_refresh += len(records)
        self.state.total_records_seen += len(records)
        return len(records)

    def should_refresh(self) -> bool:
        """Check if a refresh should be triggered based on current state."""
        if self.config.trigger_mode == RefreshTrigger.MANUAL:
            return False

        if self.config.trigger_mode == RefreshTrigger.RECORD_COUNT:
            return self.state.new_records_since_refresh >= self.config.min_new_records

        if self.config.trigger_mode == RefreshTrigger.UNCERTAINTY:
            return self.state.last_uncertainty_score > self.config.uncertainty_threshold

        return False

    def refresh(
        self,
        *,
        seed: int = 0,
        force: bool = False,
    ) -> tuple[FourOutcomeTransferCritic, bool]:
        """Perform a critic refresh/retrain.

        Args:
            seed: Random seed for reproducibility
            force: Force refresh even if trigger conditions aren't met

        Returns:
            Tuple of (new_critic, success)
        """
        if not force and not self.should_refresh():
            return self.critic, False

        # Combine all records for retraining
        all_records = self._all_records
        if not all_records:
            return self.critic, False

        # Apply off-policy correction if enabled
        if self.config.enable_off_policy_correction and len(all_records) > 0:
            # Use corrected weights to filter/reweight records
            summary = self.off_policy_corrector.correct_batch(all_records)
            # Filter records with very low weights (outliers from old policies)
            threshold = summary.mean_weight * 0.1
            filtered_records = [
                r for r, w in zip(all_records, summary.weights, strict=False)
                if w >= threshold
            ]
        else:
            filtered_records = all_records

        if len(filtered_records) < 2:
            return self.critic, False

        # Retrain critic
        try:
            encoder = self.critic.encoder or HashingTransferFeatureEncoder()
            new_critic = FourOutcomeTransferCritic(encoder=encoder)
            new_critic.fit(
                filtered_records,
                seed=seed,
                n_bootstrap=self.config.n_bootstrap,
            )
            # Atomic update
            self.critic = new_critic
            self.state.refresh_count += 1
            self.state.last_refresh_version = self.state.current_version
            self.state.current_version += 1
            self.state.new_records_since_refresh = 0
            self._pending_records.clear()
            return new_critic, True
        except Exception:
            self.state.failed_refresh_count += 1
            if self.config.retain_old_critic_on_failure:
                return self.critic, False
            raise

    def estimate_uncertainty(
        self,
        samples: list[TransferPredictionInput] | None = None,
    ) -> float:
        """Estimate current prediction uncertainty.

        Returns mean uncertainty (tau_ucb - tau_lcb) across samples.
        """
        if samples is None or not samples:
            return self.state.last_uncertainty_score

        uncertainties = []
        for sample in samples:
            try:
                estimate = self.critic.predict(sample)
                uncertainty = estimate.tau_ucb - estimate.tau_lcb
                uncertainties.append(uncertainty)
            except Exception:
                continue

        if not uncertainties:
            return 0.0

        mean_uncertainty = float(np.mean(uncertainties))
        self.state.last_uncertainty_score = mean_uncertainty
        return mean_uncertainty

    def get_version_info(self) -> dict:
        """Get current version information."""
        return {
            "current_version": self.state.current_version,
            "total_records_seen": self.state.total_records_seen,
            "refresh_count": self.state.refresh_count,
            "failed_refresh_count": self.state.failed_refresh_count,
            "new_records_since_refresh": self.state.new_records_since_refresh,
        }


class ActiveAcquisitionConfig(BaseModel):
    """Configuration for active data acquisition."""

    model_config = ConfigDict(frozen=True)

    max_candidates_per_round: int = 5
    """Maximum candidates to suggest per acquisition round."""

    uncertainty_quantile: float = 0.8
    """Quantile threshold for high-uncertainty regions."""

    exploration_epsilon: float = 0.1
    """Probability of random exploration vs exploitation."""


@dataclass
class AcquisitionSuggestion:
    """A suggested data point to collect."""

    context_description: str
    priority_score: float
    reason: str
    estimated_uncertainty: float


class ActiveDataAcquisition:
    """Identifies high-uncertainty regions for targeted data collection.

    The acquisition system analyzes the current critic's prediction
    uncertainty and suggests contexts where new data would be most
    valuable for improving the critic's estimates.

    Usage:
        acquisition = ActiveDataAcquisition(critic=critic)
        suggestions = acquisition.suggest_acquisitions(candidate_inputs)
    """

    def __init__(
        self,
        *,
        critic: FourOutcomeTransferCritic,
        config: ActiveAcquisitionConfig | None = None,
    ) -> None:
        self.critic = critic
        self.config = config or ActiveAcquisitionConfig()

    def score_candidates(
        self,
        candidates: list[TransferPredictionInput],
    ) -> list[tuple[TransferPredictionInput, float]]:
        """Score candidates by acquisition value (uncertainty).

        Returns:
            List of (input, score) tuples sorted by score descending
        """
        scored = []
        for candidate in candidates:
            try:
                estimate = self.critic.predict(candidate)
                uncertainty = estimate.tau_ucb - estimate.tau_lcb
                # Higher uncertainty = higher acquisition value
                scored.append((candidate, uncertainty))
            except Exception:
                scored.append((candidate, 0.0))

        # Sort by uncertainty descending
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def suggest_acquisitions(
        self,
        candidates: list[TransferPredictionInput],
        *,
        context_descriptions: list[str] | None = None,
    ) -> list[AcquisitionSuggestion]:
        """Suggest which data points to collect.

        Args:
            candidates: Candidate prediction inputs to evaluate
            context_descriptions: Optional descriptions for each candidate

        Returns:
            List of acquisition suggestions, sorted by priority
        """
        if context_descriptions is None:
            context_descriptions = [f"context_{i}" for i in range(len(candidates))]

        scored = self.score_candidates(candidates)

        # Filter by uncertainty quantile threshold
        if scored:
            uncertainties = [s for _, s in scored]
            threshold = float(np.quantile(uncertainties, self.config.uncertainty_quantile))
        else:
            threshold = 0.0

        suggestions = []
        max_candidates = self.config.max_candidates_per_round
        for i, (_candidate, uncertainty) in enumerate(scored[:max_candidates]):
            if uncertainty >= threshold or uncertainty > 0.3:
                desc = context_descriptions[i] if i < len(context_descriptions) else f"context_{i}"
                suggestions.append(
                    AcquisitionSuggestion(
                        context_description=desc,
                        priority_score=uncertainty,
                        reason="high_uncertainty" if uncertainty > 0.3 else "boundary_region",
                        estimated_uncertainty=uncertainty,
                    )
                )

        return suggestions

    def find_boundary_regions(
        self,
        candidates: list[TransferPredictionInput],
        *,
        tau_threshold: float = 0.0,
        margin: float = 0.1,
    ) -> list[TransferPredictionInput]:
        """Find candidates near the decision boundary.

        Candidates near tau ≈ threshold are most informative for
        improving the router's decision accuracy.

        Returns:
            List of candidates near the decision boundary
        """
        boundary_candidates = []
        for candidate in candidates:
            try:
                estimate = self.critic.predict(candidate)
                # Near boundary if tau_mean is close to threshold
                if abs(estimate.tau_mean - tau_threshold) <= margin:
                    boundary_candidates.append(candidate)
            except Exception:
                continue

        return boundary_candidates
