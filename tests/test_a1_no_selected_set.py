"""Tests for A1 feature ablation (context_plus_candidate removed in SMTR-v1).

The ``context_plus_candidate`` feature block was removed as part of the
SMTR-v1 simplification.  The current feature blocks are ``full``,
``no_compatibility_interaction`` and ``global_transfer``; the
no-compatibility-interaction ablation is tested via
``test_feature_ablation_modes.py`` and the baselines router suite.
"""

import pytest


@pytest.mark.skip(reason="context_plus_candidate feature block removed in SMTR-v1")
class TestA1FeatureBlock:
    """Placeholder: original A1 tests targeted a removed feature block."""


@pytest.mark.skip(reason="context_plus_candidate feature block removed in SMTR-v1")
class TestA1Checkpoint:
    """Placeholder: A1 checkpoint tests targeted a removed feature block."""
