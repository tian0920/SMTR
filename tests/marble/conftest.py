"""Environment gating for real-engine MARBLE integration tests.

Tests marked ``marble_integration`` declare that they require a live
Docker environment plus model credentials (see their ``requires_docker``
and ``requires_model_credentials`` markers). They are skipped unless the
operator explicitly opts in with ``SMTR_RUN_MARBLE_INTEGRATION=1``, so
the regular test suite never executes real MARBLE engine runs.
"""

from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("SMTR_RUN_MARBLE_INTEGRATION") == "1":
        return
    skip = pytest.mark.skip(
        reason=(
            "real MARBLE engine integration test requires Docker and model "
            "credentials; set SMTR_RUN_MARBLE_INTEGRATION=1 to run"
        )
    )
    for item in items:
        if item.get_closest_marker("marble_integration") is not None:
            item.add_marker(skip)
