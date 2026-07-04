"""Offline-only interaction-boundary prefix sampler (S2 / A-07).

This sampler is used *only* during offline counterfactual collection. Instead of
balancing prefix size (like :class:`StratifiedEligiblePrefixSampler`), it prefers
prefixes that are likely to *interact* with the target and flip the transfer
effect ``tau(m|o, S) != tau(m|o, empty)``.

Two complementary scoring components are supported:

* **Structural** (A-07.1 / A-07.2): reuses the A-01 pairwise interaction signals
  (environment conflict, forbidden-vs-required conflict, precondition/postcondition
  overlap, role / capability / task-tag relevance) computed purely from routing
  cards.
* **Critic-based** (A-07.4 / A-07.5 / A-07.6): an optional, mechanism-agnostic
  scorer that probes the *current* critic for prediction disagreement between the
  empty prefix and the candidate prefix, ensemble uncertainty, and ``tau_hat``
  near zero.

A-07.3 (target vs prefix action *strategy* opposition) is intentionally **not**
implemented: ``strategy`` is deliberately excluded from the routing-card schema to
prevent mechanism leakage (identical constraint to A-01.7), so it cannot be scored
from card-visible fields.
"""

import random
from collections import Counter
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict

from smtr.router.candidate_proposer import CandidateScore
from smtr.router.transfer_features import _pair_interaction_signals

# (target_memory_id, prefix_memory_id) -> non-negative interaction score
CriticPrefixScorer = Callable[[str, str], float]


class InteractionBoundaryConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_prefix_size: int = 2
    allow_conflict_prefixes: bool = True
    structural_weight: float = 1.0
    critic_weight: float = 1.0
    min_score_epsilon: float = 1e-9


class InteractionBoundaryPrefixSampler:
    policy_name = "InteractionBoundaryPrefixSampler"
    policy_version = "1"

    def __init__(
        self,
        config: InteractionBoundaryConfig | None = None,
        *,
        cards_by_id: dict,
        prefix_size_counts: Counter | None = None,
    ) -> None:
        self.config = config or InteractionBoundaryConfig()
        # Accept either MemoryRoutingCard or RoutingFeatureSnapshot objects; both
        # expose the fields required by ``_pair_interaction_signals``.
        self.cards = dict(cards_by_id)
        self.prefix_size_counts = (
            prefix_size_counts if prefix_size_counts is not None else Counter()
        )
        self.critic_scorer: CriticPrefixScorer | None = None
        self.fallback_count = 0

    def eligible_candidates(
        self,
        *,
        candidate_order: list[str],
        target_index: int,
        candidate_scores: dict[str, CandidateScore],
    ) -> list[str]:
        eligible: list[str] = []
        seen: set[str] = set()
        for memory_id in candidate_order[:target_index]:
            if memory_id in seen:
                continue
            seen.add(memory_id)
            score = candidate_scores[memory_id]
            if score.explicit_environment_conflict and not self.config.allow_conflict_prefixes:
                continue
            if score.receiver_compatibility <= 0:
                continue
            if memory_id not in self.cards:
                continue
            eligible.append(memory_id)
        return eligible

    def structural_interaction_score(self, target_id: str, prefix_id: str) -> float:
        signals = _pair_interaction_signals(self.cards[target_id], self.cards[prefix_id])
        # Conflicts are the strongest effect-flip drivers (A-07.1 / A-07.2); relevance
        # overlaps only indicate the prefix is on-topic enough to interact at all.
        return (
            2.0 * signals["env_conflict"]
            + 2.0 * signals["forbidden_conflict"]
            + 1.0 * signals["precond_postcond_overlap"]
            + 0.5 * signals["postcond_postcond_overlap"]
            + 0.25
            * (
                signals["role_overlap"]
                + signals["capability_overlap"]
                + signals["task_tag_overlap"]
            )
        )

    def prefix_score(self, target_id: str, prefix_id: str) -> float:
        score = self.config.structural_weight * self.structural_interaction_score(
            target_id, prefix_id
        )
        if self.critic_scorer is not None:
            score += self.config.critic_weight * float(self.critic_scorer(target_id, prefix_id))
        return score

    def sample(
        self,
        *,
        candidate_order: list[str],
        target_index: int,
        candidate_scores: dict[str, CandidateScore],
        seed: int,
    ) -> list[str]:
        eligible = self.eligible_candidates(
            candidate_order=candidate_order,
            target_index=target_index,
            candidate_scores=candidate_scores,
        )
        max_k = min(self.config.max_prefix_size, len(eligible))
        if max_k == 0:
            self.prefix_size_counts[0] += 1
            return []

        target_id = candidate_order[target_index]
        rng = random.Random(seed)
        scored: list[tuple[float, str]] = []
        for memory_id in eligible:
            base = self.prefix_score(target_id, memory_id)
            jitter = rng.random() * self.config.min_score_epsilon
            scored.append((base + jitter, memory_id))
        scored.sort(key=lambda item: (-item[0], item[1]))

        interacting = [mid for score, mid in scored if score > self.config.min_score_epsilon]
        if not interacting:
            # No interaction signal at all: fall back to empty prefix rather than
            # inject an unrelated (noise) prefix.
            self.fallback_count += 1
            self.prefix_size_counts[0] += 1
            return []

        k = min(len(interacting), max_k)
        chosen = {mid for _, mid in scored[:k]}
        selected = [
            memory_id for memory_id in candidate_order[:target_index] if memory_id in chosen
        ]
        self.prefix_size_counts[len(selected)] += 1
        return selected

    def sampling_probability(self, actual_prefix_size: int, eligible_count: int) -> float | None:
        # Deterministic top-k selection is not a simple closed-form probability.
        del actual_prefix_size, eligible_count
        return None
