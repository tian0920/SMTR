"""清单 Test 5: edge-cluster bootstrap (P0-6).

Bootstrap must resample treatment edges, not seed records: all seeds of a
drawn edge enter the bootstrap member together, and an edge's seeds can
never be split across a bootstrap member.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from smtr.counterfactual.edge_keys import (
    edge_cluster_bootstrap_indices,
    group_records_by_edge,
)


def _record(task: str, receiver: str, memory: str, seed: int) -> dict:
    return {
        "task_id": task,
        "receiver_agent_id": receiver,
        "candidate_memory_id": memory,
        "generation_seed": seed,
    }


def _records() -> list[dict]:
    records = [_record("t1", "r1", "m1", s) for s in range(4)]  # edge A
    records += [_record("t1", "r1", "m2", s) for s in range(2)]  # edge B
    records += [_record("t2", "r1", "m1", 0)]  # edge C
    return records


class TestEdgeClusterBootstrap:
    def test_draw_size_consistent_with_drawn_edge_clusters(self):
        records = _records()
        groups = group_records_by_edge(records)
        rng = np.random.default_rng(0)
        for _ in range(20):
            indices = edge_cluster_bootstrap_indices(records, rng)
            counts = Counter(
                (records[i]["task_id"], records[i]["receiver_agent_id"],
                 records[i]["candidate_memory_id"])
                for i in indices
            )
            # draw size is exactly the sum of the drawn clusters' seed counts
            assert len(indices) == sum(
                counts[edge] for edge in groups
            )

    def test_uniform_cluster_size_keeps_draw_size_fixed(self):
        """With equal seed counts per edge, every member has n rows."""
        records = [_record(f"t{i}", "r1", "m1", s) for i in range(3) for s in range(2)]
        rng = np.random.default_rng(0)
        for _ in range(50):
            indices = edge_cluster_bootstrap_indices(records, rng)
            assert len(indices) == len(records)

    def test_same_edge_seeds_always_appear_together(self):
        """清单验收: seeds of one edge are never split across a member."""
        records = _records()
        groups = group_records_by_edge(records)
        rng = np.random.default_rng(1)
        for _ in range(100):
            indices = edge_cluster_bootstrap_indices(records, rng)
            counts = Counter(
                (records[i]["task_id"], records[i]["receiver_agent_id"],
                 records[i]["candidate_memory_id"])
                for i in indices
            )
            for edge, idxs in groups.items():
                # either the whole edge is present (multiples of seed count
                # are possible when the edge is drawn repeatedly) or absent
                assert counts[edge] % len(idxs) == 0, (
                    f"edge {edge} partially sampled: {counts[edge]} of "
                    f"{len(idxs)} seeds"
                )

    def test_edge_drawn_multiple_times_brings_all_seeds_each_time(self):
        records = _records()
        rng = np.random.default_rng(2)
        seen = False
        for _ in range(200):
            indices = edge_cluster_bootstrap_indices(records, rng)
            counts = Counter(
                (records[i]["task_id"], records[i]["candidate_memory_id"])
                for i in indices
            )
            if counts[("t1", "m1")] > 4:
                # edge A drawn twice -> all 4 seeds appear twice (8 rows)
                assert counts[("t1", "m1")] == 8
                seen = True
                break
        assert seen, "bootstrap never resampled an edge in 200 draws"

    def test_empty_records_return_empty_array(self):
        indices = edge_cluster_bootstrap_indices([], np.random.default_rng(0))
        assert len(indices) == 0
