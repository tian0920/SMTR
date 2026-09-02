"""Uncertainty and gamma statistical audit for RIMA-v2 (§16).

This module provides:

1. **Coverage audit** (§16.2): validates bootstrap σ by computing
   empirical LCB coverage and sigma–error correlation.

2. **Uncertainty calibration** (§16.3): if ``corr(σ, |μ − τ_obs|) ≤ 0``,
   emit ``UNCERTAINTY_UNCALIBRATED`` warning — do NOT silently adjust β.

3. **Gamma report** (§16.6): detailed distribution of positive observed
   tau from train data, with per-scenario diagnostics.

β = 1.64 is referred to as the **conservative uncertainty coefficient**
(§16.1); we never claim "95% confidence" without empirical coverage
evidence.

γ remains fixed at Q75(τ_obs_train > 0) and is NOT updated online
(§16.5).
"""

from __future__ import annotations

import statistics
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from smtr.router.official_score_transfer_critic import (
        BootstrapOfficialScoreTransferCritic,
        MatchedInterventionExample,
    )

__all__ = [
    "BETA_LABEL",
    "UNCERTAINTY_UNCALIBRATED",
    "UncertaintyAuditReport",
    "GammaReport",
    "audit_coverage",
    "compute_gamma_report",
]

#: §16.1 — β label: never claim "95% confidence" without empirical evidence.
BETA_LABEL = "conservative uncertainty coefficient"

#: §16.3 — warning code for uncalibrated uncertainty.
UNCERTAINTY_UNCALIBRATED = "UNCERTAINTY_UNCALIBRATED"


# ---------------------------------------------------------------------------
# §16.2 — Coverage audit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UncertaintyAuditReport:
    """Result of bootstrap uncertainty validation (§16.2 / §16.3).

    Attributes:
        lcb_empirical_coverage: P[tau_obs >= mu - beta*sigma] on
            calibration data.
        sigma_abs_error_correlation: Pearson corr(sigma, |mu - tau_obs|).
            If ≤ 0, sigma is NOT a reliable epistemic uncertainty estimate.
        mean_sigma: mean of bootstrap sigma across calibration edges.
        median_sigma: median of bootstrap sigma across calibration edges.
        n_calibration_edges: number of edges used for calibration.
        uncertainty_calibrated: True if sigma–error correlation > 0.
    """

    lcb_empirical_coverage: float
    sigma_abs_error_correlation: float | None
    mean_sigma: float
    median_sigma: float
    n_calibration_edges: int
    uncertainty_calibrated: bool


def audit_coverage(
    critic: "BootstrapOfficialScoreTransferCritic",
    calibration_examples: list["MatchedInterventionExample"],
    *,
    beta: float = 1.64,
) -> UncertaintyAuditReport:
    """Audit bootstrap uncertainty on held-out calibration data (§16.2).

    For each calibration edge:
    1. Predict (mu, sigma) from the critic.
    2. Compute observed tau from the matched scores.
    3. Check LCB coverage: ``tau_obs >= mu - beta * sigma``.
    4. Compute ``|mu - tau_obs|`` for correlation with sigma.

    Args:
        critic: fitted bootstrap critic.
        calibration_examples: held-out matched interventions.
        beta: the conservative uncertainty coefficient (default 1.64).

    Returns:
        UncertaintyAuditReport with coverage and calibration metrics.
    """
    covered = 0
    total = 0
    sigmas: list[float] = []
    abs_errors: list[float] = []

    for ex in calibration_examples:
        # Skip self-transfer and invalid
        if ex.source_agent_id == ex.receiver_id:
            continue
        if (
            ex.official_expose_score is None
            or ex.official_withhold_score is None
        ):
            continue

        dist = critic.predict_distribution(ex)
        if dist.mu_tau is None or dist.sigma_tau is None:
            continue

        tau_obs = ex.official_expose_score - ex.official_withhold_score
        lcb = dist.mu_tau - beta * dist.sigma_tau

        if tau_obs >= lcb:
            covered += 1
        total += 1

        sigmas.append(dist.sigma_tau)
        abs_errors.append(abs(dist.mu_tau - tau_obs))

    if total == 0:
        return UncertaintyAuditReport(
            lcb_empirical_coverage=0.0,
            sigma_abs_error_correlation=None,
            mean_sigma=0.0,
            median_sigma=0.0,
            n_calibration_edges=0,
            uncertainty_calibrated=False,
        )

    coverage = covered / total

    # Sigma–error correlation (§16.3)
    sigma_error_corr: float | None = None
    if len(sigmas) >= 3:
        sigma_arr = np.array(sigmas)
        error_arr = np.array(abs_errors)
        # Avoid division by zero if all sigmas are identical
        if sigma_arr.std() > 0 and error_arr.std() > 0:
            sigma_error_corr = float(
                np.corrcoef(sigma_arr, error_arr)[0, 1]
            )
        else:
            sigma_error_corr = 0.0

    calibrated = (
        sigma_error_corr is not None and sigma_error_corr > 0
    )

    # Emit warning if uncalibrated (§16.3)
    if not calibrated:
        warnings.warn(
            f"{UNCERTAINTY_UNCALIBRATED}: sigma–error correlation = "
            f"{sigma_error_corr}; bootstrap sigma is NOT a reliable "
            f"epistemic uncertainty estimate. Do NOT silently adjust beta.",
            stacklevel=2,
        )

    return UncertaintyAuditReport(
        lcb_empirical_coverage=coverage,
        sigma_abs_error_correlation=sigma_error_corr,
        mean_sigma=float(np.mean(sigmas)),
        median_sigma=float(np.median(sigmas)),
        n_calibration_edges=total,
        uncertainty_calibrated=calibrated,
    )


# ---------------------------------------------------------------------------
# §16.6 — Gamma report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GammaReport:
    """Detailed gamma distribution report (§16.6).

    Attributes:
        gamma: Q75 of positive observed train tau.
        positive_tau_count: number of positive-tau edges.
        positive_tau_mean: mean of positive observed tau.
        positive_tau_median: median of positive observed tau.
        positive_tau_q25: 25th percentile.
        positive_tau_q50: 50th percentile (same as median).
        positive_tau_q75: 75th percentile (= gamma).
        positive_tau_q90: 90th percentile.
        per_scenario: optional per-scenario breakdown.
    """

    gamma: float
    positive_tau_count: int
    positive_tau_mean: float
    positive_tau_median: float
    positive_tau_q25: float
    positive_tau_q50: float
    positive_tau_q75: float
    positive_tau_q90: float

    per_scenario: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        result: dict[str, Any] = {
            "gamma": self.gamma,
            "positive_tau_count": self.positive_tau_count,
            "positive_tau_mean": self.positive_tau_mean,
            "positive_tau_median": self.positive_tau_median,
            "positive_tau_q25": self.positive_tau_q25,
            "positive_tau_q50": self.positive_tau_q50,
            "positive_tau_q75": self.positive_tau_q75,
            "positive_tau_q90": self.positive_tau_q90,
        }
        if self.per_scenario:
            result["per_scenario"] = dict(self.per_scenario)
        return result


def compute_gamma_report(
    train_examples: list["MatchedInterventionExample"],
    *,
    quantile: float = 0.75,
    scenario_key: str = "scenario",
) -> GammaReport:
    """Compute gamma report with full distribution diagnostics (§16.6).

    Aggregates by treatment edge ``(task_id, receiver_id, memory_id)``
    to handle multiple seeds per edge, then takes the mean per edge.

    Args:
        train_examples: TRAIN split matched interventions only.
        quantile: quantile for gamma (default 0.75, §16.5).
        scenario_key: key in task_repr to extract scenario name.

    Returns:
        GammaReport with global and per-scenario diagnostics.

    Raises:
        ValueError: if no positive observed tau in train data.
    """
    # Aggregate by edge
    edge_taus: dict[tuple[str, str, str], list[float]] = {}
    edge_scenario: dict[tuple[str, str, str], str] = {}

    for ex in train_examples:
        if ex.official_expose_score is None:
            continue
        if ex.official_withhold_score is None:
            continue
        tau = ex.official_expose_score - ex.official_withhold_score
        key = (ex.task_id, ex.receiver_id, ex.memory_id)
        edge_taus.setdefault(key, []).append(tau)
        # Track scenario from features
        scenario = ex.features.task_repr.get(scenario_key, "unknown")
        edge_scenario[key] = scenario

    # Mean tau per edge
    edge_means: dict[tuple[str, str, str], float] = {
        k: float(np.mean(v)) for k, v in edge_taus.items()
    }

    # Filter positive
    positive_taus = [t for t in edge_means.values() if t > 0]
    if not positive_taus:
        raise ValueError(
            "No positive observed tau in TRAIN split — cannot compute gamma. "
            "Training data may be insufficient or all effects are non-positive."
        )

    arr = np.array(positive_taus)

    # Per-scenario breakdown
    per_scenario: dict[str, list[float]] = {}
    for edge_key, mean_tau in edge_means.items():
        if mean_tau <= 0:
            continue
        scenario = edge_scenario.get(edge_key, "unknown")
        per_scenario.setdefault(scenario, []).append(mean_tau)

    scenario_diagnostics: dict[str, dict[str, Any]] = {}
    for scenario, taus in sorted(per_scenario.items()):
        s_arr = np.array(taus)
        scenario_diagnostics[scenario] = {
            "count": len(taus),
            "mean": float(s_arr.mean()),
            "median": float(np.median(s_arr)),
            "q75": float(np.quantile(s_arr, 0.75, method="linear")),
        }

    return GammaReport(
        gamma=float(np.quantile(arr, quantile, method="linear")),
        positive_tau_count=len(positive_taus),
        positive_tau_mean=float(arr.mean()),
        positive_tau_median=float(np.median(arr)),
        positive_tau_q25=float(np.quantile(arr, 0.25, method="linear")),
        positive_tau_q50=float(np.quantile(arr, 0.50, method="linear")),
        positive_tau_q75=float(np.quantile(arr, 0.75, method="linear")),
        positive_tau_q90=float(np.quantile(arr, 0.90, method="linear")),
        per_scenario=scenario_diagnostics,
    )
