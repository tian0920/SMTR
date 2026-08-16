"""Epsilon-star mechanism removed in SMTR-v1.

SMTR-v1 uses pure tau selective exposure (tau > 0) as the sole routing
gate.  The validation-selected epsilon* risk threshold is no longer part
of the method definition.  These tests are preserved as skip placeholders
so the test tree remains stable.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(
    reason="SMTR-v1 removed epsilon* risk threshold from the routing decision"
)
class TestMethodSpecificEpsilon:
    """Placeholder: original epsilon-star tests targeted a removed mechanism."""
