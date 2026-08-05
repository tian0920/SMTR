"""清单 Test 4: edge-equal training weights (P0-5).

Every treatment edge contributes total weight 1 regardless of seed count,
so an edge with 10 seeds never outweighs an edge with 2 seeds.
"""

from __future__ import annotations

import numpy as np

from smtr.counterfactual.edge_keys import edge_equal_sample_weights


def _record(task: str, receiver: str, memory: str, seed: int) -> dict:
    return {
        "task_id": task,
        "receiver_agent_id": receiver,
        "candidate_memory_id": memory,
        "generation_seed": seed,
    }


class TestEdgeEqualWeighting:
    def test_weight_formula_one_over_seed_count(self):
        records = [_record("t1", "r1", "m1", s) for s in range(10)]
        records += [_record("t1", "r1", "m2", s) for s in range(2)]
        weights = edge_equal_sample_weights(records)
        assert np.allclose(weights[:10], 1.0 / 10)
        assert np.allclose(weights[10:], 1.0 / 2)

    def test_total_weight_per_edge_is_one(self):
        """清单验收: edges with many seeds get the same total weight."""
        records = [_record("t1", "r1", "m1", s) for s in range(10)]
        records += [_record("t1", "r1", "m2", s) for s in range(2)]
        records += [_record("t2", "r1", "m1", 0)]
        weights = edge_equal_sample_weights(records)
        # edge A (10 seeds), edge B (2 seeds), edge C (1 seed)
        assert np.isclose(weights[:10].sum(), 1.0)
        assert np.isclose(weights[10:12].sum(), 1.0)
        assert np.isclose(weights[12:].sum(), 1.0)

    def test_uniform_when_all_edges_have_one_seed(self):
        records = [_record(f"t{i}", "r1", "m1", 0) for i in range(4)]
        weights = edge_equal_sample_weights(records)
        assert np.allclose(weights, 1.0)
        assert weights.shape == (4,)

    def test_weights_sum_equals_number_of_edges(self):
        records = [_record("t1", "r1", "m1", s) for s in range(5)]
        records += [_record("t1", "r2", "m1", s) for s in range(3)]
        weights = edge_equal_sample_weights(records)
        assert np.isclose(weights.sum(), 2.0)
