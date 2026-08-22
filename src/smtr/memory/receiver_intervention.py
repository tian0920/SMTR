"""Receiver-conditioned intervention evaluator (Receiver=3 protocol).

Evaluates memory utility per-receiver by running expose/withhold
counterfactual rollouts for each receiver independently.

The key insight: Δ(m, r) = expose(m, r) - withhold(m, r)
can differ across receivers even for the same memory.

Silent-zero policy: any failure to obtain a real counterfactual
outcome raises ``MissingCounterfactualOutcomeError`` instead of
falling back to (0.0, 0.0) which would silently map to rejection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable


class MissingCounterfactualOutcomeError(Exception):
    """Raised when a counterfactual outcome cannot be obtained.

    This prevents silent-zero rejection: missing data must never
    be conflated with measured Δ ≤ 0.
    """

    def __init__(
        self,
        *,
        memory_id: str,
        receiver_id: str,
        episode_id: int | str = "unknown",
        reason: str = "unknown",
    ) -> None:
        self.memory_id = memory_id
        self.receiver_id = receiver_id
        self.episode_id = episode_id
        self.reason = reason
        super().__init__(
            f"Missing counterfactual outcome: "
            f"memory={memory_id} receiver={receiver_id} "
            f"episode={episode_id} reason={reason}"
        )


def _validate_finite(value: Any, label: str) -> float:
    """Ensure *value* is a finite float; raise TypeError/ValueError otherwise."""
    if value is None:
        raise ValueError(f"{label} is None")
    f = float(value)
    if not math.isfinite(f):
        raise ValueError(f"{label} is not finite: {value!r}")
    return f


@dataclass(frozen=True)
class ReceiverOutcome:
    """Outcome of a single receiver's expose/withhold pair."""

    receiver_id: str
    expose_reward: float
    withhold_reward: float
    delta: float
    decision: str  # "validated" | "rejected"


@dataclass(frozen=True)
class MultiReceiverInterventionResult:
    """Aggregated result of evaluating one memory across multiple receivers."""

    memory_id: str
    receiver_outcomes: tuple[ReceiverOutcome, ...]
    n_validated: int
    n_rejected: int
    mean_delta: float
    any_positive: bool
    all_positive: bool

    @property
    def validated_receivers(self) -> list[str]:
        return [o.receiver_id for o in self.receiver_outcomes if o.decision == "validated"]

    @property
    def rejected_receivers(self) -> list[str]:
        return [o.receiver_id for o in self.receiver_outcomes if o.decision == "rejected"]

    @property
    def disagreement_rate(self) -> float:
        """P(decision_i != decision_j) for all receiver pairs."""
        n = len(self.receiver_outcomes)
        if n < 2:
            return 0.0
        decisions = [o.decision for o in self.receiver_outcomes]
        n_pairs = n * (n - 1) // 2
        n_disagree = sum(
            1 for i in range(n) for j in range(i + 1, n)
            if decisions[i] != decisions[j]
        )
        return n_disagree / n_pairs


class ReceiverInterventionEvaluator:
    """Evaluates memory utility per-receiver using counterfactual rollouts.

    For each (memory, receiver) pair, computes:
        delta(receiver) = expose(memory, receiver) - withhold(memory, receiver)

    Decision rule (threshold-free):
        delta > 0  → validated for this receiver
        delta ≤ 0  → rejected for this receiver
    """

    def __init__(
        self,
        *,
        outcome_fn: Callable[..., tuple[float, float]] | None = None,
    ) -> None:
        """Initialize with an optional outcome function.

        Parameters
        ----------
        outcome_fn:
            Callable(memory, receiver_id, receiver_state) -> (expose_reward, withhold_reward).
            If None, uses a lookup-based evaluator from paired records.
        """
        self._outcome_fn = outcome_fn
        self._results: list[MultiReceiverInterventionResult] = []

    def evaluate(
        self,
        *,
        memory_id: str,
        receiver_ids: list[str],
        receiver_states: dict[str, Any] | None = None,
        paired_outcomes: dict[str, tuple[float, float]] | None = None,
        episode_id: int | str = "unknown",
    ) -> MultiReceiverInterventionResult:
        """Evaluate one memory across multiple receivers.

        Parameters
        ----------
        memory_id:
            The candidate memory to evaluate.
        receiver_ids:
            List of receiver agent IDs.
        receiver_states:
            Optional per-receiver state for context.
        paired_outcomes:
            Pre-computed (expose, withhold) per receiver_id.
            Used for offline evaluation when real rollouts are unavailable.
        episode_id:
            Episode identifier for error reporting.

        Returns
        -------
        MultiReceiverInterventionResult with per-receiver decisions.

        Raises
        ------
        MissingCounterfactualOutcomeError
            If a real counterfactual outcome cannot be obtained for
            any (memory, receiver) pair.  Silent-zero is forbidden.
        """
        outcomes: list[ReceiverOutcome] = []

        for rid in receiver_ids:
            raw: tuple[Any, Any] | None = None

            if paired_outcomes is not None and rid in paired_outcomes:
                raw = paired_outcomes[rid]
            elif self._outcome_fn is not None:
                state = (receiver_states or {}).get(rid)
                raw = self._outcome_fn(memory_id, rid, state)

            if raw is None:
                raise MissingCounterfactualOutcomeError(
                    memory_id=memory_id,
                    receiver_id=rid,
                    episode_id=episode_id,
                    reason="no outcome source available "
                           "(outcome_fn is None and receiver not in paired_outcomes)",
                )

            try:
                expose = _validate_finite(raw[0], "expose_reward")
                withhold = _validate_finite(raw[1], "withhold_reward")
            except (ValueError, TypeError, IndexError) as exc:
                raise MissingCounterfactualOutcomeError(
                    memory_id=memory_id,
                    receiver_id=rid,
                    episode_id=episode_id,
                    reason=f"malformed outcome: {exc}",
                ) from exc

            delta = expose - withhold
            decision = "validated" if delta > 0 else "rejected"
            outcomes.append(ReceiverOutcome(
                receiver_id=rid,
                expose_reward=expose,
                withhold_reward=withhold,
                delta=delta,
                decision=decision,
            ))

        deltas = [o.delta for o in outcomes]
        n_val = sum(1 for o in outcomes if o.decision == "validated")
        result = MultiReceiverInterventionResult(
            memory_id=memory_id,
            receiver_outcomes=tuple(outcomes),
            n_validated=n_val,
            n_rejected=len(outcomes) - n_val,
            mean_delta=sum(deltas) / len(deltas) if deltas else 0.0,
            any_positive=n_val > 0,
            all_positive=n_val == len(outcomes),
        )
        self._results.append(result)
        return result

    def evaluate_from_paired_records(
        self,
        *,
        memory_id: str,
        records: list[dict],
        receiver_ids: list[str],
        episode_id: int | str = "unknown",
    ) -> MultiReceiverInterventionResult:
        """Evaluate using existing paired records (offline mode).

        Groups records by receiver_agent_id and extracts share/withhold
        team_success as (expose, withhold) outcomes.
        """
        paired_outcomes: dict[str, tuple[float, float]] = {}
        for r in records:
            rid = r.get("receiver_agent_id", "")
            if rid not in receiver_ids:
                continue
            share_ok = bool(r.get("share", {}).get("team_success", False))
            withhold_ok = bool(r.get("withhold", {}).get("team_success", False))
            paired_outcomes[rid] = (float(share_ok), float(withhold_ok))

        return self.evaluate(
            memory_id=memory_id,
            receiver_ids=receiver_ids,
            paired_outcomes=paired_outcomes,
            episode_id=episode_id,
        )

    @property
    def results(self) -> list[MultiReceiverInterventionResult]:
        return list(self._results)

    def summary(self) -> dict[str, Any]:
        """Aggregate statistics across all evaluations."""
        if not self._results:
            return {"total": 0}
        total_memories = len(self._results)
        total_validated = sum(r.n_validated for r in self._results)
        total_rejected = sum(r.n_rejected for r in self._results)
        total_receivers = sum(len(r.receiver_outcomes) for r in self._results)
        disagreement_rates = [r.disagreement_rate for r in self._results if len(r.receiver_outcomes) > 1]
        return {
            "total_memories": total_memories,
            "total_receiver_evaluations": total_receivers,
            "total_validated": total_validated,
            "total_rejected": total_rejected,
            "validation_rate": total_validated / max(total_receivers, 1),
            "mean_disagreement_rate": (
                sum(disagreement_rates) / len(disagreement_rates)
                if disagreement_rates else 0.0
            ),
            "any_positive_count": sum(1 for r in self._results if r.any_positive),
            "all_positive_count": sum(1 for r in self._results if r.all_positive),
        }
