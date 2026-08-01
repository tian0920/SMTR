"""Four-outcome transfer labels."""

from __future__ import annotations

from typing import Literal

FourOutcomeLabel = Literal[
    "positive_transfer",
    "negative_transfer",
    "neutral_success",
    "neutral_failure",
]

ALL_LABELS: tuple[FourOutcomeLabel, ...] = (
    "positive_transfer",
    "negative_transfer",
    "neutral_success",
    "neutral_failure",
)


def compute_label(y_share_team: bool, y_withhold_team: bool) -> FourOutcomeLabel:
    if y_share_team and not y_withhold_team:
        return "positive_transfer"
    if not y_share_team and y_withhold_team:
        return "negative_transfer"
    if y_share_team and y_withhold_team:
        return "neutral_success"
    return "neutral_failure"
