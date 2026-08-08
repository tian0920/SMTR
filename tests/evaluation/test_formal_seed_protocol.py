"""Formal Protocol §1/§10: generation-seed protocol validator tests.

Formal evaluations require **exactly** seeds ``[0, 1, 2, 3, 4]``; pilot
evaluations require **exactly** seeds ``[0, 1, 2]``. Any deviation
(fewer, more, arbitrary values, duplicates below the count) fails closed.
"""

from __future__ import annotations

import pytest

from smtr.evaluation.experiment_protocol import (
    FORMAL_SEEDS,
    MINIMUM_UNIQUE_SEEDS,
    PILOT_SEEDS,
    SEED_PROTOCOL_NAME,
    validate_generation_seed_protocol,
)


class TestFormalSeedProtocolExact:
    """Formal mode requires exactly (0, 1, 2, 3, 4)."""

    def test_formal_accepts_exact_seeds(self):
        seeds = validate_generation_seed_protocol(
            generation_seeds=[4, 2, 0, 3, 1], experiment_mode="formal")
        assert seeds == FORMAL_SEEDS

    def test_formal_accepts_duplicates_of_exact_seeds(self):
        seeds = validate_generation_seed_protocol(
            generation_seeds=[0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
            experiment_mode="formal",
        )
        assert seeds == FORMAL_SEEDS

    def test_formal_rejects_three_seeds(self):
        with pytest.raises(ValueError, match="requires exactly seeds"):
            validate_generation_seed_protocol(
                generation_seeds=[0, 1, 2], experiment_mode="formal")

    def test_formal_rejects_four_seeds(self):
        with pytest.raises(ValueError, match="requires exactly seeds"):
            validate_generation_seed_protocol(
                generation_seeds=[0, 1, 2, 3], experiment_mode="formal")

    def test_formal_rejects_six_seeds(self):
        with pytest.raises(ValueError, match="requires exactly seeds"):
            validate_generation_seed_protocol(
                generation_seeds=[0, 1, 2, 3, 4, 5],
                experiment_mode="formal",
            )

    def test_formal_rejects_arbitrary_five_seeds(self):
        with pytest.raises(ValueError, match="requires exactly seeds"):
            validate_generation_seed_protocol(
                generation_seeds=[5, 6, 7, 8, 9], experiment_mode="formal")


class TestPilotSeedProtocolExact:
    """Pilot mode requires exactly (0, 1, 2)."""

    def test_pilot_accepts_exact_seeds(self):
        seeds = validate_generation_seed_protocol(
            generation_seeds=[2, 0, 1], experiment_mode="pilot")
        assert seeds == PILOT_SEEDS

    def test_pilot_accepts_duplicates_of_exact_seeds(self):
        seeds = validate_generation_seed_protocol(
            generation_seeds=[0, 0, 1, 1, 2, 2], experiment_mode="pilot")
        assert seeds == PILOT_SEEDS

    def test_pilot_rejects_two_seeds(self):
        with pytest.raises(ValueError, match="requires exactly seeds"):
            validate_generation_seed_protocol(
                generation_seeds=[0, 1], experiment_mode="pilot")

    def test_pilot_rejects_four_seeds(self):
        with pytest.raises(ValueError, match="requires exactly seeds"):
            validate_generation_seed_protocol(
                generation_seeds=[0, 1, 2, 3], experiment_mode="pilot")

    def test_pilot_rejects_arbitrary_three_seeds(self):
        with pytest.raises(ValueError, match="requires exactly seeds"):
            validate_generation_seed_protocol(
                generation_seeds=[3, 4, 5], experiment_mode="pilot")


class TestSeedProtocolConstants:
    def test_formal_seeds_constant(self):
        assert FORMAL_SEEDS == (0, 1, 2, 3, 4)

    def test_pilot_seeds_constant(self):
        assert PILOT_SEEDS == (0, 1, 2)

    def test_minimum_table_matches_seeds(self):
        assert MINIMUM_UNIQUE_SEEDS == {"formal": 5, "pilot": 3}

    def test_seed_protocol_names(self):
        assert SEED_PROTOCOL_NAME["formal"] == "formal_v1"
        assert SEED_PROTOCOL_NAME["pilot"] == "pilot_v1"

    def test_unsupported_mode_rejected(self):
        with pytest.raises(ValueError, match="unsupported experiment_mode"):
            validate_generation_seed_protocol(
                generation_seeds=[0, 1, 2, 3, 4], experiment_mode="smoke")
