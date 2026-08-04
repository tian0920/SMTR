"""Canonical paired-outcome accessor (第二轮清单第一章).

Every paired record stores its two potential outcomes in the nested
canonical fields::

    record["share"]["team_success"]
    record["withhold"]["team_success"]

All training, evaluation, calibration and analysis code must read the
outcomes exclusively through :func:`get_paired_outcomes` and derive the
transfer label through :func:`paired_transfer_label`. The legacy
top-level ``y_share`` / ``y_withhold`` schema is rejected instead of
silently tolerated, so stale fields can never corrupt labels, empirical
tau/eta, flip statistics or risk-utility curves.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# (y_share, y_withhold) -> four-outcome transfer label.
OUTCOME_TO_LABEL: dict[tuple[int, int], str] = {
    (0, 0): "neutral_failure",
    (0, 1): "negative_transfer",
    (1, 0): "positive_transfer",
    (1, 1): "neutral_success",
}

# Inverse mapping: transfer label -> implied (y_share, y_withhold).
LABEL_TO_OUTCOMES: dict[str, tuple[int, int]] = {
    label: outcome for outcome, label in OUTCOME_TO_LABEL.items()
}


def get_paired_outcomes(record: Mapping[str, Any]) -> tuple[int, int]:
    """Return (y_share, y_withhold) from the canonical paired schema.

    Raises:
        ValueError: when the record lacks the nested canonical fields.
            Legacy top-level ``y_share``/``y_withhold`` records are
            deliberately rejected so outdated schemas fail fast instead
            of silently producing wrong labels.
    """
    try:
        y_share = int(bool(record["share"]["team_success"]))
        y_withhold = int(bool(record["withhold"]["team_success"]))
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Paired record must contain "
            "share.team_success and withhold.team_success."
        ) from exc

    return y_share, y_withhold


def paired_transfer_label(y_share: int, y_withhold: int) -> str:
    """Four-outcome transfer label for one (y_share, y_withhold) pair."""
    return OUTCOME_TO_LABEL[(y_share, y_withhold)]


def paired_record_label(record: Mapping[str, Any]) -> str:
    """Transfer label of a paired record via the canonical accessor."""
    y_share, y_withhold = get_paired_outcomes(record)
    return paired_transfer_label(y_share, y_withhold)
