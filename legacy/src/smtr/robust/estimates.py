"""Estimate objects for Robust-SMTR."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RobustTransferEstimate:
    tau_mean: float
    tau_lcb: float
    tau_ucb: float
    negative_risk_mean: float
    negative_risk_lcb: float
    negative_risk_ucb: float
    confidence_level: float
