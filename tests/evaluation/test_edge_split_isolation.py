"""清单 Test 3: treatment-edge split isolation (P0-4).

All seed records of one treatment edge ``(task, receiver, memory)`` must
live in exactly one split. The formal audit must fail fast when a seed of
an edge is placed in a different split.
"""

from __future__ import annotations

import pytest

from smtr.counterfactual.edge_keys import (
    group_records_by_edge,
    treatment_edge_key,
)
from smtr.evaluation.split_audit import audit_split_leakage


def _record(task: str, receiver: str, memory: str, seed: int) -> dict:
    return {
        "task_id": task,
        "receiver_agent_id": receiver,
        "candidate_memory_id": memory,
        "generation_seed": seed,
    }


def _splits(train=None, validation=None, test=None) -> dict[str, list[dict]]:
    return {
        "train": train or [],
        "validation": validation or [],
        "test": test or [],
    }


class TestEdgeKeyIdentity:
    def test_seed_does_not_change_edge_key(self):
        """Seeds are repeated trials of the same edge, not new edges."""
        a = _record("t1", "r1", "m1", 0)
        b = _record("t1", "r1", "m1", 7)
        assert treatment_edge_key(a) == treatment_edge_key(b)
        groups = group_records_by_edge([a, b])
        assert len(groups) == 1
        assert sorted(groups[("t1", "r1", "m1")]) == [0, 1]

    def test_receiver_or_memory_difference_is_a_different_edge(self):
        records = [
            _record("t1", "r1", "m1", 0),
            _record("t1", "r2", "m1", 0),
            _record("t1", "r1", "m2", 0),
        ]
        assert len(group_records_by_edge(records)) == 3


class TestEdgeSplitIsolation:
    def test_same_edge_seeds_in_different_splits_fails_audit(self):
        """清单验收: len(edge_observed_splits[edge]) == 1 for every edge.

        A record-level split puts two seeds of the same edge into different
        splits; the audit must reject it even though all other identifiers
        (task, source trajectory, edge_id) are consistent.
        """
        splits = _splits(
            train=[
                _record("t1", "r1", "m1", 0),
                _record("t1", "r1", "m2", 0),
            ],
            validation=[
                _record("t1", "r1", "m1", 1),
                _record("t2", "r9", "m9", 0),
            ],
            test=[_record("t3", "r9", "m9", 0)],
        )
        with pytest.raises(ValueError, match="edge_observed_splits"):
            audit_split_leakage(splits)

    def test_clean_split_by_edge_passes(self):
        splits = _splits(
            train=[
                _record("t1", "r1", "m1", 0),
                _record("t1", "r1", "m1", 1),
                _record("t1", "r1", "m1", 2),
            ],
            validation=[_record("t2", "r1", "m1", 0)],
            test=[_record("t3", "r2", "m2", 0)],
        )
        audit = audit_split_leakage(splits)
        assert audit["treatment_edge_overlap"] == []
        assert audit["treatment_edge_count_by_split"] == {
            "train": 1,
            "validation": 1,
            "test": 1,
        }

    def test_task_grouped_split_keeps_edges_isolated(self):
        """Task-group splitting keeps all edges of a task in one split."""
        splits = _splits(
            train=[
                _record("t1", "r1", "m1", s) for s in range(3)
            ] + [_record("t1", "r2", "m1", 0)],
            validation=[_record("t2", "r1", "m1", s) for s in range(2)],
            test=[_record("t3", "r1", "m1", 0)],
        )
        audit = audit_split_leakage(splits)
        assert audit["treatment_edge_overlap"] == []
        assert audit["treatment_edge_count_by_split"]["train"] == 2
