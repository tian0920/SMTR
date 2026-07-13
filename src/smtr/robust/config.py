"""Configuration for the optional Robust-SMTR extension."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RobustSMTRGateConfig:
    negative_risk_budget: float = 0.2
    confidence_level: float = 0.9

    def __post_init__(self) -> None:
        if not 0.0 <= self.negative_risk_budget <= 1.0:
            raise ValueError("negative_risk_budget must be in [0, 1]")
        if not 0.5 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0.5, 1)")
