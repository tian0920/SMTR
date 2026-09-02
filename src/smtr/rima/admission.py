"""RIMA admission decisions and decision-source invariants (Phase 4).

Formal invariant:

* Every formal admission decision carries ``decision_source``.
* The ONLY legal value in the formal path is ``"frozen_transfer_critic"``.
* ``decision_source == "observed_delta"`` must make the formal main
  runner FAIL (counterfactual outcome is unavailable at admission time).

The online expose/withhold prototype (generation B) used observed deltas
to admit memories; that remains available as an oracle upper bound but is
hard-blocked from the canonical admission path here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from smtr.rima import FORBIDDEN_DECISION_SOURCES, FORMAL_DECISION_SOURCE

__all__ = [
    "AdmissionDecision",
    "AdmissionStatus",
    "assert_formal_decision_source",
    "ObservedDeltaAdmissionError",
]


class ObservedDeltaAdmissionError(RuntimeError):
    """Raised when a forbidden decision source enters the formal path."""


class AdmissionStatus:
    """Admission statuses used by the canonical runner."""

    ADMITTED = "admitted"
    REJECTED = "rejected"
    SELF_TRANSFER_EXCLUDED = "self_transfer_excluded"
    INVALID_PREDICTION = "invalid_prediction"


@dataclass(frozen=True)
class AdmissionDecision:
    """One (memory, receiver, task) admission decision.

    Attributes:
        memory_id: candidate memory.
        receiver_id: receiver agent.
        task_id: task at admission time.
        status: one of :class:`AdmissionStatus` values.
        tau_hat: critic prediction (None when invalid/excluded).
        decision_source: MUST be ``frozen_transfer_critic`` in the formal
            path; anything else is rejected by the invariant guard.
        decided_at: UTC timestamp.
        metadata: diagnostic-only fields (e.g. observed_delta recorded
            AFTER admission for mechanism evaluation — never used as input).
    """

    memory_id: str
    receiver_id: str
    task_id: str
    status: str
    tau_hat: float | None = None
    decision_source: str = FORMAL_DECISION_SOURCE
    decided_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = field(default_factory=dict)

    @property
    def admitted(self) -> bool:
        return self.status == AdmissionStatus.ADMITTED


def assert_formal_decision_source(decision_source: str) -> None:
    """Fail-closed guard for the formal admission path (Phase 4.2).

    Raises:
        ObservedDeltaAdmissionError: when the decision source is forbidden
            (e.g. ``observed_delta``, ``team_success``, ``oracle``).
    """
    if decision_source == FORMAL_DECISION_SOURCE:
        return
    if decision_source in FORBIDDEN_DECISION_SOURCES:
        raise ObservedDeltaAdmissionError(
            f"Forbidden decision_source={decision_source!r} in the formal "
            f"RIMA admission path. Only {FORMAL_DECISION_SOURCE!r} is legal. "
            f"Observed expose/withhold deltas are mechanism-evidence/oracle "
            f"upper bound only."
        )
    raise ObservedDeltaAdmissionError(
        f"Unknown decision_source={decision_source!r}; the formal RIMA path "
        f"only allows {FORMAL_DECISION_SOURCE!r}."
    )
