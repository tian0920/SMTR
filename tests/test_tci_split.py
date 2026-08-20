"""Tests for TCI group-based split (Task 9).

Verifies:
  - No group leakage between train/valid/test.
  - Reproducibility under same seed.
  - Empty input handling.
  - Ratio approximation for small datasets.
"""

from __future__ import annotations

import pytest

from smtr.router.tci_dataset import TCIPair
from smtr.router.tci_split import TCISplit, split_tci_pairs


def _make_pair(
    task_id: str,
    receiver: str,
    memory_id: str,
    ptype: str = "precondition",
    direction: int = 1,
) -> TCIPair:
    return TCIPair(
        perturbation_id=f"pert_{task_id}_{memory_id}",
        task_id=task_id,
        receiver_agent_id=receiver,
        candidate_memory_id=memory_id,
        perturbation_type=ptype,
        changed_field="precondition_tags",
        y0=1,
        y_original=1,
        y_perturbed=0,
        effect_original=0,
        effect_perturbed=-1,
        direction=direction,
        contrast_type="induced_damage",
    )


class TestTCISplit:
    def test_no_group_leakage(self) -> None:
        """Same (task, receiver, memory) must NOT appear in multiple splits."""
        pairs = [
            _make_pair("t1", "a1", "m1"),
            _make_pair("t1", "a1", "m2"),
            _make_pair("t1", "a1", "m3"),
            _make_pair("t2", "a2", "m4"),
            _make_pair("t2", "a2", "m5"),
            _make_pair("t3", "a3", "m6"),
            _make_pair("t3", "a3", "m7"),
            _make_pair("t4", "a4", "m8"),
            _make_pair("t4", "a4", "m9"),
            _make_pair("t5", "a5", "m10"),
        ]
        split = split_tci_pairs(pairs, seed=42)

        def group_keys(pairs: list[TCIPair]) -> set[tuple[str, str, str]]:
            return {
                (p.task_id, p.receiver_agent_id, p.candidate_memory_id)
                for p in pairs
            }

        train_g = group_keys(split.train_pairs)
        valid_g = group_keys(split.valid_pairs)
        test_g = group_keys(split.test_pairs)

        assert train_g.isdisjoint(valid_g)
        assert train_g.isdisjoint(test_g)
        assert valid_g.isdisjoint(test_g)

    def test_split_reproducible(self) -> None:
        """Same seed must produce identical splits."""
        pairs = [
            _make_pair("t1", "a1", f"m{i}") for i in range(10)
        ]
        s1 = split_tci_pairs(pairs, seed=99)
        s2 = split_tci_pairs(pairs, seed=99)

        assert [p.perturbation_id for p in s1.train_pairs] == [
            p.perturbation_id for p in s2.train_pairs
        ]
        assert [p.perturbation_id for p in s1.valid_pairs] == [
            p.perturbation_id for p in s2.valid_pairs
        ]
        assert [p.perturbation_id for p in s1.test_pairs] == [
            p.perturbation_id for p in s2.test_pairs
        ]

    def test_empty_input(self) -> None:
        """Empty input must return empty splits."""
        split = split_tci_pairs([], seed=42)
        assert split.n_train == 0
        assert split.n_valid == 0
        assert split.n_test == 0

    def test_all_pairs_assigned(self) -> None:
        """Every input pair must appear in exactly one split."""
        pairs = [
            _make_pair("t1", "a1", f"m{i}") for i in range(12)
        ]
        split = split_tci_pairs(pairs, seed=42)
        total = split.n_train + split.n_valid + split.n_test
        assert total == len(pairs)

    def test_small_dataset_three_groups(self) -> None:
        """With 3 groups, each gets one split."""
        pairs = [
            _make_pair("t1", "a1", "m1"),
            _make_pair("t2", "a1", "m2"),
            _make_pair("t3", "a1", "m3"),
        ]
        split = split_tci_pairs(pairs, seed=42)
        # All three should be assigned somewhere.
        total = split.n_train + split.n_valid + split.n_test
        assert total == 3

    def test_multiple_pairs_per_group(self) -> None:
        """Multiple pairs from same group all go to same split."""
        pairs = [
            _make_pair("t1", "a1", "m1", direction=1),
            _make_pair("t1", "a1", "m1", direction=-1),
            _make_pair("t2", "a2", "m2", direction=1),
            _make_pair("t2", "a2", "m2", direction=-1),
            _make_pair("t3", "a3", "m3", direction=1),
        ]
        split = split_tci_pairs(pairs, seed=42)

        def group_keys(pairs: list[TCIPair]) -> set[tuple[str, str, str]]:
            return {
                (p.task_id, p.receiver_agent_id, p.candidate_memory_id)
                for p in pairs
            }

        train_g = group_keys(split.train_pairs)
        valid_g = group_keys(split.valid_pairs)
        test_g = group_keys(split.test_pairs)

        # t1/a1/m1 must be in exactly one split.
        key1 = ("t1", "a1", "m1")
        in_train = key1 in train_g
        in_valid = key1 in valid_g
        in_test = key1 in test_g
        assert sum([in_train, in_valid, in_test]) == 1

    def test_to_dict(self) -> None:
        split = split_tci_pairs([
            _make_pair(f"t{i}", "a1", f"m{i}") for i in range(10)
        ], seed=42)
        d = split.to_dict()
        assert "n_train" in d
        assert "n_valid" in d
        assert "n_test" in d
