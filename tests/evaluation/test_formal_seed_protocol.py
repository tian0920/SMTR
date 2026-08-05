"""R6 Test 7: generation-seed protocol validator (清单 P1-1).

Formal evaluations require at least five unique generation seeds and
pilots at least three; duplicate seeds never count toward the minimum.
"""

from __future__ import annotations

import pytest

from smtr.evaluation.experiment_protocol import (
    MINIMUM_UNIQUE_SEEDS,
    validate_generation_seed_protocol,
)


class TestFormalSeedProtocol:
    def test_formal_rejects_three_seeds(self):
        with pytest.raises(ValueError, match="at least 5 unique generation seeds"):
            validate_generation_seed_protocol(
                generation_seeds=[0, 1, 2], experiment_mode="formal")

    def test_formal_rejects_four_seeds(self):
        with pytest.raises(ValueError, match="at least 5 unique generation seeds"):
            validate_generation_seed_protocol(
                generation_seeds=[0, 1, 2, 3], experiment_mode="formal")

    def test_formal_accepts_five_seeds(self):
        seeds = validate_generation_seed_protocol(
            generation_seeds=[4, 2, 0, 3, 1], experiment_mode="formal")
        assert seeds == (0, 1, 2, 3, 4)

    def test_duplicates_do_not_count_toward_formal_minimum(self):
        with pytest.raises(ValueError, match="received 3"):
            validate_generation_seed_protocol(
                generation_seeds=[0, 0, 1, 1, 2, 2], experiment_mode="formal")

    def test_pilot_rejects_two_seeds(self):
        with pytest.raises(ValueError, match="at least 3 unique generation seeds"):
            validate_generation_seed_protocol(
                generation_seeds=[0, 1], experiment_mode="pilot")

    def test_pilot_accepts_three_seeds(self):
        seeds = validate_generation_seed_protocol(
            generation_seeds=[2, 0, 1], experiment_mode="pilot")
        assert seeds == (0, 1, 2)

    def test_pilot_deduplicates_seeds(self):
        seeds = validate_generation_seed_protocol(
            generation_seeds=[0, 0, 1, 1, 2, 2], experiment_mode="pilot")
        assert seeds == (0, 1, 2)

    def test_unsupported_mode_rejected(self):
        with pytest.raises(ValueError, match="unsupported experiment_mode"):
            validate_generation_seed_protocol(
                generation_seeds=[0, 1, 2, 3, 4], experiment_mode="smoke")

    def test_minimum_table_matches_validator(self):
        assert MINIMUM_UNIQUE_SEEDS == {"formal": 5, "pilot": 3}
