"""Tests for sequential traversal policies."""

from smtr.router.traversal import (
    RandomTraversal,
    traversal_permutation_indices,
)


def test_random_order_is_seed_reproducible_and_does_not_mutate_input() -> None:
    candidates = ["m1", "m2", "m3", "m4"]
    original = list(candidates)
    policy = RandomTraversal()
    assert policy.order(candidates, seed=7) == policy.order(candidates, seed=7)
    assert candidates == original


def test_permutation_indices_are_recorded_relative_to_proposer_order() -> None:
    assert traversal_permutation_indices(
        ["m1", "m2", "m3"],
        ["m3", "m1", "m2"],
    ) == (2, 0, 1)
