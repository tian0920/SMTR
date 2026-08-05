"""Local-outcome reporting (清单 P0-15).

SMTR v1 has no reliable receiver-local evaluator, so local/team divergence
is never claimed and local metrics are reported as ``null`` (never 0):

> SMTR controls cross-agent memory exposure using team-level transfer
> outcomes.

If a reliable local outcome source appears later it must be added as an
extension; it is intentionally out of scope for this version.
"""

from __future__ import annotations

from typing import Any


def local_outcome_report() -> dict[str, Any]:
    """Fixed null report for receiver-local outcomes (清单 P0-15)."""
    return {
        "local_outcome_available": False,
        "local_positive_team_negative_count": None,
        "local_negative_team_positive_count": None,
    }
