"""High-order group effects analysis (B-05).

This module provides mechanisms for analyzing interaction effects between
memories beyond pairwise analysis. It supports:

- k-way interaction feature generation
- Group tau estimation: tau({m1, m2, m3} | o, S)
- SHAP-style contribution decomposition
- Higher-order effect detection (3-way, 4-way interactions)

Group effects capture synergies and conflicts between memories that
are not visible in pairwise analysis.
"""

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from smtr.counterfactual.schemas import PairedInterventionRecord


@dataclass
class InteractionEffect:
    """Represents an interaction effect between a group of memories."""

    memory_ids: tuple[str, ...]
    order: int
    effect_size: float
    std_error: float
    p_value: float
    n_observations: int
    is_significant: bool


@dataclass
class ContributionAnalysis:
    """SHAP-style contribution analysis for a group of memories."""

    memory_id: str
    shap_value: float
    base_value: float
    n_samples: int
    contributions: dict[str, float] = field(default_factory=dict)


@dataclass
class GroupEffectSummary:
    """Summary of group-level effects analysis."""

    n_memories: int
    n_groups_analyzed: int
    pairwise_effects: list[InteractionEffect] = field(default_factory=list)
    higher_order_effects: list[InteractionEffect] = field(default_factory=list)
    contributions: list[ContributionAnalysis] = field(default_factory=list)
    total_variance_explained: float = 0.0


class GroupEffectAnalyzer:
    """Analyzes higher-order interaction effects between memories.

    The analyzer computes group-level tau estimates and decomposes
    them into individual, pairwise, and higher-order contributions
    using a SHAP-style approach.

    Usage:
        analyzer = GroupEffectAnalyzer()
        summary = analyzer.analyze(records, memory_ids=["m1", "m2", "m3"])
    """

    def __init__(
        self,
        *,
        max_order: int = 3,
        significance_level: float = 0.05,
        min_samples_per_group: int = 5,
    ) -> None:
        self.max_order = max_order
        self.significance_level = significance_level
        self.min_samples_per_group = min_samples_per_group

    def analyze(
        self,
        records: list[PairedInterventionRecord],
        *,
        memory_ids: list[str] | None = None,
    ) -> GroupEffectSummary:
        """Analyze group effects for a set of memories.

        Args:
            records: Paired intervention records
            memory_ids: Specific memories to analyze (None = all in records)

        Returns:
            GroupEffectSummary with pairwise and higher-order effects
        """
        if memory_ids is None:
            memory_ids = list({r.candidate_memory_id for r in records})

        if len(memory_ids) < 2:
            return GroupEffectSummary(
                n_memories=len(memory_ids),
                n_groups_analyzed=0,
            )

        # Group records by memory
        records_by_memory = self._group_records_by_memory(records)

        # Compute pairwise effects
        pairwise_effects = []
        for mid_a, mid_b in combinations(memory_ids, 2):
            effect = self._compute_pairwise_effect(
                mid_a, mid_b, records_by_memory
            )
            if effect:
                pairwise_effects.append(effect)

        # Compute higher-order effects
        higher_order_effects = []
        for order in range(3, min(self.max_order + 1, len(memory_ids) + 1)):
            for group in combinations(memory_ids, order):
                effect = self._compute_group_effect(
                    group, records_by_memory
                )
                if effect:
                    higher_order_effects.append(effect)

        # Compute SHAP-style contributions
        contributions = self._compute_contributions(
            memory_ids, records_by_memory
        )

        # Total variance explained
        all_effects = pairwise_effects + higher_order_effects
        total_variance = sum(e.effect_size**2 for e in all_effects)

        return GroupEffectSummary(
            n_memories=len(memory_ids),
            n_groups_analyzed=len(pairwise_effects) + len(higher_order_effects),
            pairwise_effects=pairwise_effects,
            higher_order_effects=higher_order_effects,
            contributions=contributions,
            total_variance_explained=total_variance,
        )

    def compute_group_tau(
        self,
        records: list[PairedInterventionRecord],
        memory_ids: list[str],
    ) -> float:
        """Compute the group-level tau for a set of memories.

        Group tau = E[y_share(S) - y_withhold(S)] where S is the group.
        """
        relevant = [r for r in records if r.candidate_memory_id in memory_ids]
        if not relevant:
            return 0.0
        taus = [r.y_share - r.y_withhold for r in relevant]
        return float(np.mean(taus))

    def _group_records_by_memory(
        self, records: list[PairedInterventionRecord]
    ) -> dict[str, list[PairedInterventionRecord]]:
        """Group records by candidate memory ID."""
        grouped: dict[str, list[PairedInterventionRecord]] = {}
        for record in records:
            mid = record.candidate_memory_id
            if mid not in grouped:
                grouped[mid] = []
            grouped[mid].append(record)
        return grouped

    def _compute_pairwise_effect(
        self,
        mid_a: str,
        mid_b: str,
        records_by_memory: dict[str, list[PairedInterventionRecord]],
    ) -> InteractionEffect | None:
        """Compute pairwise interaction effect between two memories."""
        records_a = records_by_memory.get(mid_a, [])
        records_b = records_by_memory.get(mid_b, [])

        if len(records_a) < self.min_samples_per_group:
            return None
        if len(records_b) < self.min_samples_per_group:
            return None

        # Individual effects
        tau_a = np.mean([r.y_share - r.y_withhold for r in records_a])
        tau_b = np.mean([r.y_share - r.y_withhold for r in records_b])

        # Combined effect (all records for both)
        combined = records_a + records_b
        tau_combined = np.mean([r.y_share - r.y_withhold for r in combined])

        # Interaction = combined - (individual_a + individual_b)
        interaction = tau_combined - (tau_a + tau_b) / 2

        # Standard error
        n = len(combined)
        std_error = float(np.std([r.y_share - r.y_withhold for r in combined]) / np.sqrt(n))

        # Simple z-test
        if std_error > 0:
            z_score = interaction / std_error
            # Approximate p-value using normal distribution
            p_value = 2 * (1 - _normal_cdf(abs(z_score)))
        else:
            p_value = 1.0

        return InteractionEffect(
            memory_ids=(mid_a, mid_b),
            order=2,
            effect_size=float(interaction),
            std_error=std_error,
            p_value=p_value,
            n_observations=n,
            is_significant=p_value < self.significance_level,
        )

    def _compute_group_effect(
        self,
        group: tuple[str, ...],
        records_by_memory: dict[str, list[PairedInterventionRecord]],
    ) -> InteractionEffect | None:
        """Compute higher-order interaction effect for a group."""
        # Collect all records for group members
        group_records = []
        for mid in group:
            group_records.extend(records_by_memory.get(mid, []))

        if len(group_records) < self.min_samples_per_group:
            return None

        # Group tau
        group_tau = np.mean([r.y_share - r.y_withhold for r in group_records])

        # Individual taus
        individual_taus = []
        for mid in group:
            records = records_by_memory.get(mid, [])
            if records:
                individual_taus.append(np.mean([r.y_share - r.y_withhold for r in records]))

        if not individual_taus:
            return None

        # Higher-order effect = group_tau - mean(individual_taus)
        effect = group_tau - np.mean(individual_taus)

        n = len(group_records)
        std_error = float(np.std([r.y_share - r.y_withhold for r in group_records]) / np.sqrt(n))

        if std_error > 0:
            z_score = effect / std_error
            p_value = 2 * (1 - _normal_cdf(abs(z_score)))
        else:
            p_value = 1.0

        return InteractionEffect(
            memory_ids=group,
            order=len(group),
            effect_size=float(effect),
            std_error=std_error,
            p_value=p_value,
            n_observations=n,
            is_significant=p_value < self.significance_level,
        )

    def _compute_contributions(
        self,
        memory_ids: list[str],
        records_by_memory: dict[str, list[PairedInterventionRecord]],
    ) -> list[ContributionAnalysis]:
        """Compute SHAP-style contributions for each memory.

        Uses a sampling-based approximation of Shapley values.
        """
        contributions = []

        for mid in memory_ids:
            records = records_by_memory.get(mid, [])
            if not records:
                continue

            # Base value: mean tau across all records
            all_taus = []
            for recs in records_by_memory.values():
                all_taus.extend([r.y_share - r.y_withhold for r in recs])
            base_value = float(np.mean(all_taus)) if all_taus else 0.0

            # This memory's mean tau
            own_taus = [r.y_share - r.y_withhold for r in records]
            own_mean = float(np.mean(own_taus))

            # Shapley value approximation: marginal contribution
            shapley_value = own_mean - base_value

            contributions.append(
                ContributionAnalysis(
                    memory_id=mid,
                    shap_value=shapley_value,
                    base_value=base_value,
                    n_samples=len(records),
                )
            )

        return contributions


def _normal_cdf(x: float) -> float:
    """Approximate the standard normal CDF."""
    return 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
