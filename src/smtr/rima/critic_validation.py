"""Critic validation metrics for RIMA (Phase 18).

The formal critic must be validated beyond classification accuracy. For a
continuous potential-outcome critic we report, against held-out matched
interventions:

* MAE / RMSE of tau_hat vs observed_delta
* Pearson / Spearman correlation
* sign accuracy
* positive-effect precision / recall, negative-effect recall
* receiver-conditioned extra: same-memory receiver ranking accuracy —
  whether the critic predicts ``tau(m, r1) > tau(m, r2)`` correctly when
  the observed deltas order the same way.

Invalid pairs (observed delta None) are excluded and counted, never
treated as zero (fail-closed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np

__all__ = ["CriticValidationReport", "validate_critic"]


@dataclass
class CriticValidationReport:
    """Validation metrics for the continuous critic."""

    n_pairs_total: int = 0
    n_pairs_valid: int = 0
    n_pairs_invalid: int = 0
    mae: float | None = None
    rmse: float | None = None
    pearson: float | None = None
    spearman: float | None = None
    sign_accuracy: float | None = None
    positive_precision: float | None = None
    positive_recall: float | None = None
    negative_recall: float | None = None
    receiver_ranking_accuracy: float | None = None
    n_ranking_pairs: int = 0
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = {
            "n_pairs_total": self.n_pairs_total,
            "n_pairs_valid": self.n_pairs_valid,
            "n_pairs_invalid": self.n_pairs_invalid,
            "mae": self.mae,
            "rmse": self.rmse,
            "pearson": self.pearson,
            "spearman": self.spearman,
            "sign_accuracy": self.sign_accuracy,
            "positive_precision": self.positive_precision,
            "positive_recall": self.positive_recall,
            "negative_recall": self.negative_recall,
            "receiver_ranking_accuracy": self.receiver_ranking_accuracy,
            "n_ranking_pairs": self.n_ranking_pairs,
        }
        out.update(self.extras)
        return out


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    xr = np.argsort(np.argsort(x)).astype(float)
    yr = np.argsort(np.argsort(y)).astype(float)
    return _pearson(xr, yr)


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def validate_critic(
    pairs: list[dict[str, Any]],
    *,
    sign_threshold: float = 0.0,
) -> CriticValidationReport:
    """Compute validation metrics from (predicted_tau, observed_delta) pairs.

    Each pair dict must contain:
        ``predicted_tau``: critic tau_hat (or None);
        ``observed_delta``: official-score delta (or None => invalid);
        optional ``memory_id`` / ``receiver_id`` for ranking accuracy.
    """
    report = CriticValidationReport(n_pairs_total=len(pairs))

    valid = [
        p
        for p in pairs
        if p.get("predicted_tau") is not None and p.get("observed_delta") is not None
    ]
    report.n_pairs_valid = len(valid)
    report.n_pairs_invalid = len(pairs) - len(valid)
    if not valid:
        return report

    pred = np.array([p["predicted_tau"] for p in valid], dtype=float)
    obs = np.array([p["observed_delta"] for p in valid], dtype=float)

    err = pred - obs
    report.mae = float(np.abs(err).mean())
    report.rmse = float(np.sqrt((err**2).mean()))
    report.pearson = _pearson(pred, obs)
    report.spearman = _spearman(pred, obs)

    pred_sign = np.sign(pred)
    obs_sign = np.sign(obs)
    nonzero = obs_sign != 0
    if nonzero.any():
        report.sign_accuracy = float((pred_sign[nonzero] == obs_sign[nonzero]).mean())

    pred_pos = pred > sign_threshold
    obs_pos = obs > sign_threshold
    obs_neg = obs < -sign_threshold
    if obs_pos.any():
        report.positive_recall = float(pred_pos[obs_pos].mean())
        if pred_pos.any():
            report.positive_precision = float(obs_pos[pred_pos].mean())
    if obs_neg.any():
        report.negative_recall = float((pred < -sign_threshold)[obs_neg].mean())

    # Same-memory receiver ranking accuracy.
    by_memory: dict[str, list[dict[str, Any]]] = {}
    for p in valid:
        if p.get("memory_id"):
            by_memory.setdefault(p["memory_id"], []).append(p)
    ranking_correct = 0
    ranking_total = 0
    for _mid, group in by_memory.items():
        if len(group) < 2:
            continue
        for a, b in combinations(group, 2):
            if a.get("receiver_id") == b.get("receiver_id"):
                continue
            d_obs = a["observed_delta"] - b["observed_delta"]
            d_pred = a["predicted_tau"] - b["predicted_tau"]
            if d_obs == 0 or d_pred == 0:
                continue
            ranking_total += 1
            if (d_obs > 0) == (d_pred > 0):
                ranking_correct += 1
    report.n_ranking_pairs = ranking_total
    if ranking_total:
        report.receiver_ranking_accuracy = ranking_correct / ranking_total
    return report
