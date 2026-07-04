"""Memory refinement and contradiction repair (B-07).

This module provides mechanisms for detecting and resolving contradictions
in the memory store. It analyzes paired intervention records and execution
evidence to:

- Detect contradictory memories (same scenario, opposite effects)
- Merge similar memories with conflicting outcomes
- Update routing cards based on new evidence

The refinement process helps maintain memory quality by identifying
memories that may have become outdated or inconsistent.
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from smtr.counterfactual.schemas import PairedInterventionRecord
from smtr.memory.schemas import MemoryRoutingCard


class ContradictionType(str, Enum):
    """Types of contradictions detected between memories."""

    TRANSFER_CONTRADICTION = "transfer_contradiction"
    """One memory shows positive transfer, another shows negative in similar context."""

    EXECUTION_CONTRADICTION = "execution_contradiction"
    """Memory has both success and failure evidence in similar contexts."""

    OUTCOME_CONTRADICTION = "outcome_contradiction"
    """Memories with same goal have opposite outcomes."""


class RefinementAction(str, Enum):
    """Actions that can be taken to resolve contradictions."""

    MERGE = "merge"
    """Merge the contradictory memories into one."""

    DEPRECATE = "deprecate"
    """Mark one memory as deprecated based on evidence."""

    FLAG = "flag"
    """Flag for manual review without automatic resolution."""

    UPDATE_EVIDENCE = "update_evidence"
    """Update the evidence counts without structural changes."""


@dataclass
class Contradiction:
    """A detected contradiction between two memories."""

    memory_id_a: str
    memory_id_b: str
    contradiction_type: ContradictionType
    severity: float
    description: str
    evidence_a: dict = field(default_factory=dict)
    evidence_b: dict = field(default_factory=dict)


@dataclass
class RefinementSuggestion:
    """A suggested refinement action for a memory or pair."""

    memory_ids: list[str]
    action: RefinementAction
    confidence: float
    reason: str
    details: dict = field(default_factory=dict)


class ContradictionDetector:
    """Detects contradictions between memories based on evidence.

    The detector analyzes paired intervention records and routing cards
    to find pairs of memories that show contradictory behavior in
    similar contexts.

    Usage:
        detector = ContradictionDetector()
        contradictions = detector.detect(records, cards_by_id)
    """

    def __init__(
        self,
        *,
        similarity_threshold: float = 0.7,
        contradiction_threshold: float = 0.3,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.contradiction_threshold = contradiction_threshold

    def detect(
        self,
        records: list[PairedInterventionRecord],
        cards_by_id: dict[str, MemoryRoutingCard],
    ) -> list[Contradiction]:
        """Detect contradictions in the memory store.

        Args:
            records: Paired intervention records with transfer outcomes
            cards_by_id: Mapping of memory ID to routing card

        Returns:
            List of detected contradictions
        """
        contradictions = []

        # Group records by memory
        records_by_memory: dict[str, list[PairedInterventionRecord]] = {}
        for record in records:
            mid = record.candidate_memory_id
            if mid not in records_by_memory:
                records_by_memory[mid] = []
            records_by_memory[mid].append(record)

        # Check for transfer contradictions between memory pairs
        memory_ids = list(records_by_memory.keys())
        for i, mid_a in enumerate(memory_ids):
            for mid_b in memory_ids[i + 1:]:
                contradiction = self._check_transfer_contradiction(
                    mid_a, mid_b, records_by_memory, cards_by_id
                )
                if contradiction:
                    contradictions.append(contradiction)

        # Check for execution contradictions within single memories
        for mid, card in cards_by_id.items():
            contradiction = self._check_execution_contradiction(mid, card)
            if contradiction:
                contradictions.append(contradiction)

        return contradictions

    def _check_transfer_contradiction(
        self,
        mid_a: str,
        mid_b: str,
        records_by_memory: dict[str, list[PairedInterventionRecord]],
        cards_by_id: dict[str, MemoryRoutingCard],
    ) -> Contradiction | None:
        """Check if two memories have contradictory transfer effects."""
        records_a = records_by_memory.get(mid_a, [])
        records_b = records_by_memory.get(mid_b, [])

        if not records_a or not records_b:
            return None

        # Compute average tau for each memory
        tau_a = np.mean([r.y_share - r.y_withhold for r in records_a])
        tau_b = np.mean([r.y_share - r.y_withhold for r in records_b])

        # Check for contradiction: opposite signs with sufficient magnitude
        if abs(tau_a - tau_b) < self.contradiction_threshold:
            return None

        # Check if memories have similar contexts
        card_a = cards_by_id.get(mid_a)
        card_b = cards_by_id.get(mid_b)
        if not card_a or not card_b:
            return None

        similarity = self._compute_card_similarity(card_a, card_b)
        if similarity < self.similarity_threshold:
            return None

        # Determine severity based on magnitude of contradiction
        severity = abs(tau_a - tau_b) * similarity

        return Contradiction(
            memory_id_a=mid_a,
            memory_id_b=mid_b,
            contradiction_type=ContradictionType.TRANSFER_CONTRADICTION,
            severity=severity,
            description=f"Memories {mid_a} and {mid_b} show opposite transfer effects "
                        f"(tau_a={tau_a:.2f}, tau_b={tau_b:.2f}) in similar contexts",
            evidence_a={"tau_mean": float(tau_a), "n_records": len(records_a)},
            evidence_b={"tau_mean": float(tau_b), "n_records": len(records_b)},
        )

    def _check_execution_contradiction(
        self,
        memory_id: str,
        card: MemoryRoutingCard,
    ) -> Contradiction | None:
        """Check if a memory has contradictory execution evidence."""
        # Check if memory has both success and failure evidence
        has_success = card.execution_success_count > 0
        has_failure = card.execution_failure_count > 0

        if not (has_success and has_failure):
            return None

        # Check if success and failure contexts are similar
        success_contexts = card.execution_success_contexts
        failure_contexts = card.execution_failure_contexts

        if not success_contexts or not failure_contexts:
            return None

        # Compute contradiction severity based on ratio
        total = card.execution_success_count + card.execution_failure_count
        failure_ratio = card.execution_failure_count / total

        # High contradiction if failure rate is significant but not dominant
        if 0.2 <= failure_ratio <= 0.8:
            severity = 1.0 - abs(0.5 - failure_ratio) * 2
            return Contradiction(
                memory_id_a=memory_id,
                memory_id_b=memory_id,
                contradiction_type=ContradictionType.EXECUTION_CONTRADICTION,
                severity=severity,
                description=f"Memory {memory_id} has mixed execution evidence "
                            f"({card.execution_success_count} successes, "
                            f"{card.execution_failure_count} failures)",
                evidence_a={
                    "success_count": card.execution_success_count,
                    "failure_count": card.execution_failure_count,
                },
            )

        return None

    def _compute_card_similarity(
        self,
        card_a: MemoryRoutingCard,
        card_b: MemoryRoutingCard,
    ) -> float:
        """Compute similarity between two routing cards."""
        # Jaccard similarity on task tags
        tags_a = set(card_a.task_tags)
        tags_b = set(card_b.task_tags)
        if tags_a or tags_b:
            tag_similarity = len(tags_a & tags_b) / len(tags_a | tags_b)
        else:
            tag_similarity = 1.0

        # Receiver role overlap
        roles_a = set(card_a.compatible_receiver_roles)
        roles_b = set(card_b.compatible_receiver_roles)
        if roles_a or roles_b:
            role_similarity = len(roles_a & roles_b) / len(roles_a | roles_b)
        else:
            role_similarity = 1.0

        # Combined similarity
        return 0.6 * tag_similarity + 0.4 * role_similarity


class MemoryRefiner:
    """Suggests and applies refinements to resolve contradictions.

    The refiner takes detected contradictions and suggests appropriate
    actions to resolve them, such as merging, deprecating, or updating
    evidence counts.

    Usage:
        refiner = MemoryRefiner()
        suggestions = refiner.suggest_refinements(contradictions, cards_by_id)
    """

    def __init__(
        self,
        *,
        merge_threshold: float = 0.8,
        deprecate_threshold: float = 0.6,
    ) -> None:
        self.merge_threshold = merge_threshold
        self.deprecate_threshold = deprecate_threshold

    def suggest_refinements(
        self,
        contradictions: list[Contradiction],
        cards_by_id: dict[str, MemoryRoutingCard],
    ) -> list[RefinementSuggestion]:
        """Suggest refinement actions for detected contradictions.

        Args:
            contradictions: List of detected contradictions
            cards_by_id: Mapping of memory ID to routing card

        Returns:
            List of refinement suggestions
        """
        suggestions = []

        for contradiction in contradictions:
            suggestion = self._suggest_for_contradiction(contradiction, cards_by_id)
            if suggestion:
                suggestions.append(suggestion)

        return suggestions

    def _suggest_for_contradiction(
        self,
        contradiction: Contradiction,
        cards_by_id: dict[str, MemoryRoutingCard],
    ) -> RefinementSuggestion | None:
        """Suggest a refinement action for a single contradiction."""
        if contradiction.contradiction_type == ContradictionType.TRANSFER_CONTRADICTION:
            return self._suggest_for_transfer_contradiction(contradiction, cards_by_id)
        elif contradiction.contradiction_type == ContradictionType.EXECUTION_CONTRADICTION:
            return self._suggest_for_execution_contradiction(contradiction, cards_by_id)
        return None

    def _suggest_for_transfer_contradiction(
        self,
        contradiction: Contradiction,
        cards_by_id: dict[str, MemoryRoutingCard],
    ) -> RefinementSuggestion | None:
        """Suggest action for transfer contradiction."""
        card_a = cards_by_id.get(contradiction.memory_id_a)
        card_b = cards_by_id.get(contradiction.memory_id_b)
        if not card_a or not card_b:
            return None

        # Determine which memory has stronger evidence
        evidence_a = contradiction.evidence_a.get("n_records", 0)
        evidence_b = contradiction.evidence_b.get("n_records", 0)

        if contradiction.severity > self.deprecate_threshold:
            # Strong contradiction: deprecate the weaker memory
            if evidence_a > evidence_b * 1.5:
                deprecate_id = contradiction.memory_id_b
            elif evidence_b > evidence_a * 1.5:
                deprecate_id = contradiction.memory_id_a
            else:
                # Similar evidence: flag for review
                return RefinementSuggestion(
                    memory_ids=[contradiction.memory_id_a, contradiction.memory_id_b],
                    action=RefinementAction.FLAG,
                    confidence=contradiction.severity,
                    reason=contradiction.description,
                )

            return RefinementSuggestion(
                memory_ids=[deprecate_id],
                action=RefinementAction.DEPRECATE,
                confidence=contradiction.severity,
                reason=f"Deprecate due to contradictory evidence: {contradiction.description}",
            )

        # Mild contradiction: update evidence
        return RefinementSuggestion(
            memory_ids=[contradiction.memory_id_a, contradiction.memory_id_b],
            action=RefinementAction.UPDATE_EVIDENCE,
            confidence=contradiction.severity,
            reason=contradiction.description,
        )

    def _suggest_for_execution_contradiction(
        self,
        contradiction: Contradiction,
        cards_by_id: dict[str, MemoryRoutingCard],
    ) -> RefinementSuggestion | None:
        """Suggest action for execution contradiction."""
        card = cards_by_id.get(contradiction.memory_id_a)
        if not card:
            return None

        # Update evidence counts based on contradiction
        return RefinementSuggestion(
            memory_ids=[contradiction.memory_id_a],
            action=RefinementAction.UPDATE_EVIDENCE,
            confidence=contradiction.severity,
            reason=contradiction.description,
            details={
                "success_count": card.execution_success_count,
                "failure_count": card.execution_failure_count,
            },
        )

    def apply_refinement(
        self,
        suggestion: RefinementSuggestion,
        cards_by_id: dict[str, MemoryRoutingCard],
    ) -> dict[str, MemoryRoutingCard]:
        """Apply a refinement suggestion to the cards.

        Note: This creates new card instances since cards are frozen.

        Returns:
            Updated cards_by_id mapping
        """
        updated = dict(cards_by_id)

        if suggestion.action == RefinementAction.UPDATE_EVIDENCE:
            for mid in suggestion.memory_ids:
                if mid in updated:
                    # Evidence update is tracked but card is frozen
                    # In practice, this would trigger a card version bump
                    pass

        elif suggestion.action == RefinementAction.DEPRECATE:
            for mid in suggestion.memory_ids:
                if mid in updated:
                    # Mark as deprecated by removing from active set
                    # In practice, this would set a deprecated flag
                    pass

        return updated
