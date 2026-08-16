"""Default risk budget resolves from the checkpoint epsilon_star.

SMTR-v1 removed the epsilon* risk threshold from the routing decision.
The checkpoint epsilon_star is retained only for diagnostic reporting.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(
    reason="SMTR-v1 removed epsilon* risk threshold from the routing decision"
)
class TestDefaultRiskBudget:
    """Placeholder: original tests targeted a removed mechanism."""
