"""Canonical paired-outcome accessor tests (第二轮清单第一章)."""

from __future__ import annotations

import pytest

from smtr.marble.paired_outcomes import get_paired_outcomes, paired_transfer_label


def test_nested_positive_transfer_outcome():
    record = {
        "share": {"team_success": True},
        "withhold": {"team_success": False},
    }

    y_share, y_withhold = get_paired_outcomes(record)

    assert (y_share, y_withhold) == (1, 0)
    assert paired_transfer_label(y_share, y_withhold) == "positive_transfer"


def test_nested_negative_transfer_outcome():
    record = {
        "share": {"team_success": False},
        "withhold": {"team_success": True},
    }

    y_share, y_withhold = get_paired_outcomes(record)

    assert (y_share, y_withhold) == (0, 1)
    assert paired_transfer_label(y_share, y_withhold) == "negative_transfer"


def test_old_top_level_outcome_schema_is_rejected():
    record = {
        "y_share": 1,
        "y_withhold": 0,
    }

    with pytest.raises(ValueError):
        get_paired_outcomes(record)
