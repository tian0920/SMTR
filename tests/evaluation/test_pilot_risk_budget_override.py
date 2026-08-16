"""Pilot risk-budget overrides require the explicit opt-in.

SMTR-v1 removed the epsilon* risk threshold from the routing decision.
The risk-budget override mechanism no longer exists.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(
    reason="SMTR-v1 removed epsilon* risk threshold from the routing decision"
)
class TestPilotRiskBudgetOverride:
    """Placeholder: original tests targeted a removed mechanism."""
