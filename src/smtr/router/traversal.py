"""Traversal policies for sequential SMTR candidate evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from random import Random
from typing import Protocol, TypeVar

T = TypeVar("T")


class TraversalPolicy(Protocol):
    """Orders proposer-ranked candidates for one-pass sequential routing."""

    policy_name: str

    def order(self, candidates: Sequence[T], *, seed: int) -> tuple[T, ...]:
        """Return the traversal order without mutating ``candidates``."""
        ...


@dataclass(frozen=True)
class RandomTraversal:
    policy_name: str = "random_order"

    def order(self, candidates: Sequence[T], *, seed: int) -> tuple[T, ...]:
        ordered = list(candidates)
        Random(seed).shuffle(ordered)
        return tuple(ordered)


def traversal_permutation_indices(
    proposal_order: Sequence[T],
    traversal_order: Sequence[T],
) -> tuple[int, ...]:
    """Return traversal positions as proposer-rank indices."""
    rank_by_item = {item: index for index, item in enumerate(proposal_order)}
    try:
        return tuple(rank_by_item[item] for item in traversal_order)
    except KeyError as exc:
        raise ValueError("traversal order contains an item outside proposal order") from exc
