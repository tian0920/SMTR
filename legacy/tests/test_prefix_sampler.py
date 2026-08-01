from collections import Counter

from smtr.counterfactual.prefix_sampler import (
    PrefixSamplingConfig,
    StratifiedEligiblePrefixSampler,
)
from smtr.router.candidate_proposer import CandidateScore


def _scores(conflict: bool = False) -> dict[str, CandidateScore]:
    return {
        "a": CandidateScore(memory_id="a", receiver_compatibility=1.0),
        "b": CandidateScore(
            memory_id="b",
            receiver_compatibility=1.0,
            explicit_environment_conflict=conflict,
        ),
        "c": CandidateScore(memory_id="c", receiver_compatibility=1.0),
    }


def test_empty_prefix_mode_always_returns_empty() -> None:
    sampler = StratifiedEligiblePrefixSampler(PrefixSamplingConfig(mode="empty"))

    assert sampler.sample(
        candidate_order=["a", "b", "c"],
        target_index=2,
        candidate_scores=_scores(),
        seed=1,
    ) == []


def test_uniform_prefix_mode_is_deterministic() -> None:
    sampler = StratifiedEligiblePrefixSampler(PrefixSamplingConfig(mode="uniform"))
    kwargs = {
        "candidate_order": ["a", "b", "c"],
        "target_index": 2,
        "candidate_scores": _scores(),
        "seed": 3,
    }

    assert sampler.sample(**kwargs) == StratifiedEligiblePrefixSampler(
        PrefixSamplingConfig(mode="uniform")
    ).sample(**kwargs)


def test_stratified_prefix_sampler_covers_cardinalities() -> None:
    counts: Counter[int] = Counter()
    sampler = StratifiedEligiblePrefixSampler(
        PrefixSamplingConfig(mode="stratified", max_prefix_size=2),
        prefix_size_counts=counts,
    )

    for seed in range(6):
        sampler.sample(
            candidate_order=["a", "b", "c"],
            target_index=2,
            candidate_scores=_scores(),
            seed=seed,
        )

    assert set(counts) == {0, 1, 2}


def test_prefix_excludes_conflicts_by_default_and_can_allow_them() -> None:
    default = StratifiedEligiblePrefixSampler(PrefixSamplingConfig(mode="uniform"))
    allowed = StratifiedEligiblePrefixSampler(
        PrefixSamplingConfig(mode="uniform", allow_conflict_prefixes=True)
    )

    assert "b" not in default.eligible_candidates(
        candidate_order=["a", "b", "c"],
        target_index=2,
        candidate_scores=_scores(conflict=True),
    )
    assert "b" in allowed.eligible_candidates(
        candidate_order=["a", "b", "c"],
        target_index=2,
        candidate_scores=_scores(conflict=True),
    )
