"""Formal Protocol §3/§11: experiment_mode and coverage_mode consistency.

Both arguments must agree; inconsistent states fail closed so no
ambiguous configuration can enter training.
"""

from __future__ import annotations

import pytest

from smtr.evaluation.experiment_protocol import (
    validate_mode_consistency,
)


class TestModeConsistency:
    """experiment_mode and coverage_mode must agree (清单 §3)."""

    def test_both_formal_passes(self):
        mode = validate_mode_consistency(
            experiment_mode="formal", coverage_mode="formal"
        )
        assert mode == "formal"

    def test_both_pilot_passes(self):
        mode = validate_mode_consistency(
            experiment_mode="pilot", coverage_mode="pilot"
        )
        assert mode == "pilot"

    def test_formal_pilot_rejected(self):
        with pytest.raises(ValueError, match="must agree"):
            validate_mode_consistency(
                experiment_mode="formal", coverage_mode="pilot"
            )

    def test_pilot_formal_rejected(self):
        with pytest.raises(ValueError, match="must agree"):
            validate_mode_consistency(
                experiment_mode="pilot", coverage_mode="formal"
            )

    def test_only_experiment_mode_set(self):
        mode = validate_mode_consistency(
            experiment_mode="formal", coverage_mode=None
        )
        assert mode == "formal"

    def test_only_coverage_mode_set(self):
        mode = validate_mode_consistency(
            experiment_mode=None, coverage_mode="pilot"
        )
        assert mode == "pilot"

    def test_neither_set_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            validate_mode_consistency(
                experiment_mode=None, coverage_mode=None
            )

    def test_unsupported_mode_rejected(self):
        with pytest.raises(ValueError, match="unsupported mode"):
            validate_mode_consistency(
                experiment_mode="smoke", coverage_mode=None
            )
