"""Uncertainty summarization for Robust-SMTR."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from smtr.robust.estimates import RobustTransferEstimate
from smtr.router.transfer_critic import MemberPredictionBatch


def summarize_member_predictions(
    member_predictions: MemberPredictionBatch | Sequence[Sequence[float]],
    confidence_level: float,
) -> RobustTransferEstimate:
    """Summarize ensemble member predictions into mean and confidence bounds."""
    if not 0.5 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0.5, 1)")
    raw = (
        member_predictions.probabilities
        if isinstance(member_predictions, MemberPredictionBatch)
        else member_predictions
    )
    probs = np.asarray(raw, dtype=float)
    if probs.ndim != 2 or probs.shape[0] == 0 or probs.shape[1] != 4:
        raise ValueError("member_predictions must have shape (n_members, 4)")
    if not np.all(np.isfinite(probs)):
        raise ValueError("member_predictions must be finite")
    row_sums = probs.sum(axis=1)
    if np.any(row_sums <= 0):
        raise ValueError("member prediction rows must have positive mass")
    probs = probs / row_sums[:, None]

    alpha = 1.0 - confidence_level
    lower_q = alpha / 2.0
    upper_q = 1.0 - alpha / 2.0
    q_mean = probs.mean(axis=0)
    tau = probs[:, 2] - probs[:, 1]
    risk = probs[:, 1]
    return RobustTransferEstimate(
        tau_mean=float(q_mean[2] - q_mean[1]),
        tau_lcb=float(np.quantile(tau, lower_q)),
        tau_ucb=float(np.quantile(tau, upper_q)),
        negative_risk_mean=float(risk.mean()),
        negative_risk_lcb=float(np.quantile(risk, lower_q)),
        negative_risk_ucb=float(np.quantile(risk, upper_q)),
        confidence_level=float(confidence_level),
    )
