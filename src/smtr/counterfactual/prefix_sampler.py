import random
from collections import Counter
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from smtr.router.candidate_proposer import CandidateScore


class PrefixSamplingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: Literal["empty", "uniform", "stratified"] = "stratified"
    max_prefix_size: int = 2
    include_prob: float = 0.5
    allow_conflict_prefixes: bool = False


class PrefixSamplingPolicy(Protocol):
    policy_name: str
    policy_version: str

    def sample(
        self,
        *,
        candidate_order: list[str],
        target_index: int,
        candidate_scores: dict[str, CandidateScore],
        seed: int,
    ) -> list[str]: ...


class StratifiedEligiblePrefixSampler:
    policy_name = "StratifiedEligiblePrefixSampler"
    policy_version = "1"

    def __init__(
        self,
        config: PrefixSamplingConfig | None = None,
        prefix_size_counts: Counter[int] | None = None,
        pinned_memory_ids: set[str] | None = None,
    ) -> None:
        self.config = config or PrefixSamplingConfig()
        self.prefix_size_counts = (
            prefix_size_counts if prefix_size_counts is not None else Counter()
        )
        self.pinned_memory_ids = pinned_memory_ids
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
            if self.pinned_memory_ids is not None and memory_id not in self.pinned_memory_ids:
                continue
            if (
                score.explicit_environment_conflict
                and not self.config.allow_conflict_prefixes
            ):
                continue
            if score.receiver_compatibility <= 0:
                continue
            eligible.append(memory_id)
        return eligible

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
        if self.config.mode == "empty" or max_k == 0:
            self.prefix_size_counts[0] += 1
            return []

        rng = random.Random(seed)
        if self.config.mode == "uniform":
            k = rng.randint(0, max_k)
        else:
            feasible = list(range(max_k + 1))
            requested = min(
                range(self.config.max_prefix_size + 1),
                key=lambda size: (self.prefix_size_counts[size], size),
            )
            if requested > max_k:
                self.fallback_count += 1
                k = min(feasible, key=lambda size: (self.prefix_size_counts[size], size))
            else:
                k = requested

        chosen = set(rng.sample(eligible, k))
        selected = [
            memory_id for memory_id in candidate_order[:target_index] if memory_id in chosen
        ]
        self.prefix_size_counts[len(selected)] += 1
        return selected

    def sampling_probability(self, actual_prefix_size: int, eligible_count: int) -> float | None:
        if self.config.mode == "empty":
            return 1.0 if actual_prefix_size == 0 else 0.0
        if self.config.mode == "uniform":
            max_k = min(self.config.max_prefix_size, eligible_count)
            return 1.0 / (max_k + 1) if max_k >= 0 else None
        return None


class TargetSelectionPolicy(Protocol):
    policy_name: str
    policy_version: str

    def select(
        self,
        *,
        candidate_ids: list[str],
        seed: int,
        designated_target_id: str | None = None,
    ) -> tuple[str, float | None]: ...


class UniformCandidateTargetPolicy:
    policy_name = "UniformCandidateTargetPolicy"
    policy_version = "1"

    def select(
        self,
        *,
        candidate_ids: list[str],
        seed: int,
        designated_target_id: str | None = None,
    ) -> tuple[str, float | None]:
        del designated_target_id
        if not candidate_ids:
            raise ValueError("candidate_ids must not be empty")
        return random.Random(seed).choice(candidate_ids), 1.0 / len(candidate_ids)


class ScenarioDesignatedTargetPolicy:
    policy_name = "ScenarioDesignatedTargetPolicy"
    policy_version = "1"

    def select(
        self,
        *,
        candidate_ids: list[str],
        seed: int,
        designated_target_id: str | None = None,
    ) -> tuple[str, float | None]:
        del seed
        if designated_target_id is None:
            raise ValueError("designated_target_id is required")
        if designated_target_id not in candidate_ids:
            raise ValueError(f"designated target is not in candidate set: {designated_target_id}")
        return designated_target_id, 1.0
