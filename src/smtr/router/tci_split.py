"""TCI pair group-based split for held-out generalization evaluation.

Split unit is the group key:
    (task_id, receiver_agent_id, candidate_memory_id)

No pair-level split: the same (task, receiver, memory) group
must never appear in both train and test/valid.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from smtr.router.tci_dataset import TCIPair


@dataclass(frozen=True)
class TCISplit:
    """Train / valid / test split of TCI pairs.

    Invariant: groups in train, valid, test are disjoint.
    """

    train_pairs: list[TCIPair]
    valid_pairs: list[TCIPair]
    test_pairs: list[TCIPair]

    @property
    def n_train(self) -> int:
        return len(self.train_pairs)

    @property
    def n_valid(self) -> int:
        return len(self.valid_pairs)

    @property
    def n_test(self) -> int:
        return len(self.test_pairs)

    def group_keys(self, pairs: list[TCIPair]) -> set[tuple[str, str, str]]:
        """Extract unique group keys from a list of pairs."""
        return {
            (p.task_id, p.receiver_agent_id, p.candidate_memory_id)
            for p in pairs
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_train": self.n_train,
            "n_valid": self.n_valid,
            "n_test": self.n_test,
        }


def _group_key(pair: TCIPair) -> tuple[str, str, str]:
    return (pair.task_id, pair.receiver_agent_id, pair.candidate_memory_id)


def split_tci_pairs(
    pairs: list[TCIPair],
    *,
    seed: int = 42,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
) -> TCISplit:
    """Split TCI pairs by (task, receiver, memory) groups.

    Algorithm:
      1. Group pairs by (task_id, receiver_agent_id, candidate_memory_id).
      2. Shuffle group keys deterministically with the given seed.
      3. Allocate groups to train / valid / test by ratio.
      4. Expand groups back to pair lists.

    No pair-level split: the same group is fully assigned to one split.

    Parameters
    ----------
    pairs : list of TCIPair
    seed : random seed for deterministic shuffle
    train_ratio : fraction of groups for training
    valid_ratio : fraction of groups for validation
                  (test_ratio = 1 - train_ratio - valid_ratio)

    Returns
    -------
    TCISplit with disjoint train / valid / test groups.
    """
    if not pairs:
        return TCISplit(train_pairs=[], valid_pairs=[], test_pairs=[])

    # Step 1: aggregate by group key.
    groups: dict[tuple[str, str, str], list[TCIPair]] = defaultdict(list)
    for pair in pairs:
        groups[_group_key(pair)].append(pair)

    # Step 2: shuffle group keys deterministically.
    keys = list(groups.keys())
    rng = _Random(seed)
    rng.shuffle(keys)

    # Step 3: split keys by ratio.
    n = len(keys)
    n_train = max(1, int(round(n * train_ratio)))
    n_valid = max(0, int(round(n * valid_ratio)))
    # Ensure at least 1 key in test if possible.
    if n_train + n_valid >= n and n >= 3:
        n_train = max(1, n - 2)
        n_valid = 1
    n_test = n - n_train - n_valid

    train_keys = set(keys[:n_train])
    valid_keys = set(keys[n_train : n_train + n_valid])
    test_keys = set(keys[n_train + n_valid :])

    # Step 4: expand to pair lists.
    train_pairs: list[TCIPair] = []
    valid_pairs: list[TCIPair] = []
    test_pairs: list[TCIPair] = []

    for key in keys:
        if key in train_keys:
            train_pairs.extend(groups[key])
        elif key in valid_keys:
            valid_pairs.extend(groups[key])
        else:
            test_pairs.extend(groups[key])

    return TCISplit(
        train_pairs=train_pairs,
        valid_pairs=valid_pairs,
        test_pairs=test_pairs,
    )


class _Random:
    """Minimal deterministic shuffle using numpy RandomState."""

    def __init__(self, seed: int) -> None:
        self._rng = __import__("numpy").random.RandomState(seed)

    def shuffle(self, lst: list[Any]) -> None:
        """In-place Fisher-Yates shuffle."""
        n = len(lst)
        indices = self._rng.permutation(n)
        lst[:] = [lst[i] for i in indices]
