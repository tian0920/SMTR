"""Noisy outcome wrapper for TCI robustness validation.

Wraps reward observations with Gaussian noise to test whether TCI's
validation decisions degrade gracefully under noisy reward estimation.

Design:
  - Default OFF (noise_sigma=0.0 → transparent passthrough)
  - Does NOT modify TCI decision logic
  - Only replaces the reward observation that TCI reads

Formula:
    r_hat = r + epsilon,  where epsilon ~ N(0, sigma)

Supported sigma values: 0.0, 0.1, 0.2, 0.3
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class NoisyOutcomeWrapper:
    """Wraps a reward observation with optional Gaussian noise.

    Parameters
    ----------
    noise_sigma:
        Standard deviation of the Gaussian noise. 0.0 = no noise.
    seed:
        Random seed for reproducibility.  Each call to :meth:`observe`
        advances the internal RNG state deterministically.
    """

    noise_sigma: float = 0.0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.noise_sigma < 0:
            raise ValueError(f"noise_sigma must be >= 0, got {self.noise_sigma}")

    def observe(self, true_reward: float, rng: np.random.RandomState | None = None) -> float:
        """Return a noisy observation of the true reward.

        Parameters
        ----------
        true_reward:
            The ground-truth reward value.
        rng:
            Optional external RNG.  When ``None``, a new one is created
            from ``self.seed`` (deterministic per-call).
        """
        if self.noise_sigma == 0.0:
            return true_reward
        _rng = rng or np.random.RandomState(self.seed)
        epsilon = _rng.normal(0.0, self.noise_sigma)
        return float(true_reward + epsilon)

    def observe_tau(
        self,
        true_tau: float,
        rng: np.random.RandomState | None = None,
    ) -> float:
        """Return a noisy observation of the treatment effect tau.

        tau = share_reward - withhold_reward.
        With noise, tau_hat = tau + epsilon_share - epsilon_withhold.
        """
        if self.noise_sigma == 0.0:
            return true_tau
        _rng = rng or np.random.RandomState(self.seed)
        eps_share = _rng.normal(0.0, self.noise_sigma)
        eps_withhold = _rng.normal(0.0, self.noise_sigma)
        return float(true_tau + eps_share - eps_withhold)


# Standard noise levels for robustness experiments
NOISE_LEVELS: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3)


def apply_noise_to_paired_outcome(
    record: dict[str, Any],
    noise_sigma: float,
    rng: np.random.RandomState,
) -> tuple[bool, bool]:
    """Apply noisy observation to a paired record's share/withhold outcomes.

    Returns (share_success_noisy, withhold_success_noisy) where the
    boolean is the noisy observation of the binary team_success.

    For binary outcomes, noise is modelled as:
      - true success (1.0) → observed as 1.0 + N(0, sigma) → threshold at 0.5
      - true failure (0.0) → observed as 0.0 + N(0, sigma) → threshold at 0.5

    This preserves the interpretation: noise can flip the observed outcome
    when the noise magnitude exceeds 0.5.
    """
    if noise_sigma == 0.0:
        share_ok = bool(record.get("share", {}).get("team_success", False))
        withhold_ok = bool(record.get("withhold", {}).get("team_success", False))
        return share_ok, withhold_ok

    share_true = float(bool(record.get("share", {}).get("team_success", False)))
    withhold_true = float(bool(record.get("withhold", {}).get("team_success", False)))

    share_obs = share_true + rng.normal(0.0, noise_sigma)
    withhold_obs = withhold_true + rng.normal(0.0, noise_sigma)

    return share_obs >= 0.5, withhold_obs >= 0.5
