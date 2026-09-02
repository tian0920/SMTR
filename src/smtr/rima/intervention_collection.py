"""Matched intervention collection for RIMA (Phase 4).

The online expose/withhold evaluator is demoted: its ONLY legal roles are

1. training intervention collection,
2. validation intervention collection,
3. test mechanism evaluation,
4. oracle upper bound.

Its observed deltas MUST NOT drive formal admission. This module provides
the canonical collector contract (purpose-tagged) and converts historical
intervention records into :class:`smtr.router.official_score_transfer_critic.
MatchedInterventionExample` training inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from smtr.rima.features import (
    ReceiverConditionedTransferFeatures,
)
from smtr.router.official_score_transfer_critic import (
    MatchedInterventionExample,
)

__all__ = [
    "InterventionPurpose",
    "MatchedInterventionCollector",
    "build_training_examples",
]


class InterventionPurpose:
    """Legal purposes for running matched expose/withhold interventions."""

    TRAINING_COLLECTION = "training_intervention_collection"
    VALIDATION_COLLECTION = "validation_intervention_collection"
    MECHANISM_EVAL = "test_mechanism_evaluation"
    ORACLE_UPPER_BOUND = "oracle_upper_bound"

    ALL = frozenset(
        {
            TRAINING_COLLECTION,
            VALIDATION_COLLECTION,
            MECHANISM_EVAL,
            ORACLE_UPPER_BOUND,
        }
    )


@dataclass(frozen=True)
class MatchedInterventionCollector:
    """Purpose-tagged wrapper over an intervention evaluator.

    Guarantees (by construction) that the collected records are labeled
    with their intended purpose; the canonical runner never feeds these
    observed deltas into admission.
    """

    purpose: str
    evaluator: Any

    def __post_init__(self) -> None:
        if self.purpose not in InterventionPurpose.ALL:
            raise ValueError(
                f"Illegal intervention purpose {self.purpose!r}; "
                f"allowed: {sorted(InterventionPurpose.ALL)}"
            )

    def collect(
        self,
        candidate: Any,
        receiver_id: str,
        task: Any,
        *,
        seed: int = 0,
        **kwargs: Any,
    ) -> Any:
        """Run one matched intervention and return the raw record."""
        return self.evaluator.validate(
            candidate, receiver_id, task, seed=seed, **kwargs
        )


def build_training_examples(
    records: list[Any],
    *,
    source_agent_ids: dict[str, str],
    feature_builder: Any,
) -> list[MatchedInterventionExample]:
    """Convert intervention records into critic training examples.

    Args:
        records: intervention records with normalized expose/withhold
            scores (e.g. ``OnlineValidationRecord``-like objects).
        source_agent_ids: mapping ``memory_id -> source_agent_id`` used to
            exclude self-transfer pairs during critic training.
        feature_builder: callable ``(record) -> ReceiverConditionedTransferFeatures``
            producing routing-card-only features.

    Returns:
        List of :class:`MatchedInterventionExample`. Invalid records keep
        ``None`` scores (fail-closed; excluded inside ``critic.fit``).
    """
    examples: list[MatchedInterventionExample] = []
    for rec in records:
        memory_id = rec.memory_id
        expose = getattr(rec, "normalized_expose_score", None)
        withhold = getattr(rec, "normalized_withhold_score", None)
        # Fail-closed: invalid branch -> None (never a silent zero).
        if not getattr(rec, "expose_metric_valid", True):
            expose = None
        if not getattr(rec, "withhold_metric_valid", True):
            withhold = None
        features = feature_builder(rec)
        examples.append(
            MatchedInterventionExample(
                task_id=rec.task_id,
                memory_id=memory_id,
                receiver_id=rec.receiver_id,
                source_agent_id=source_agent_ids.get(memory_id, ""),
                official_expose_score=expose,
                official_withhold_score=withhold,
                features=features,
            )
        )
    return examples
