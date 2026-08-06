"""Edge-equal sample weights under shared control (清单 Shared-Control 第10章).

Every treatment edge contributes total weight 1 regardless of how many
seed records it has, so high-seed edges never outweigh low-seed edges.
"""

from __future__ import annotations

import pytest

from smtr.counterfactual.edge_keys import (
    edge_equal_sample_weights,
    group_records_by_edge,
)


def _record(*, memory_id: str, seed: int) -> dict:
    return {
        "task_id": "t1",
        "receiver_agent_id": "r1",
        "candidate_memory_id": memory_id,
        "generation_seed": seed,
    }


def test_each_edge_total_weight_is_one():
    records = (
        [_record(memory_id="mA", seed=seed) for seed in range(5)]
        + [_record(memory_id="mB", seed=seed) for seed in range(2)]
        + [_record(memory_id="mC", seed=0)]
    )
    weights = edge_equal_sample_weights(records)
    assert weights.shape == (8,)

    groups = group_records_by_edge(records)
    assert len(groups) == 3
    for indices in groups.values():
        total = sum(weights[idx] for idx in indices)
        assert total == pytest.approx(1.0)

    # Per-record weight is exactly 1 / n_e on its own edge.
    assert weights[0] == pytest.approx(1.0 / 5)
    assert weights[5] == pytest.approx(1.0 / 2)
    assert weights[7] == pytest.approx(1.0)


def test_total_weight_equals_edge_count():
    records = [
        _record(memory_id=memory_id, seed=seed)
        for memory_id in ("mA", "mB", "mC", "mD")
        for seed in range(3)
    ]
    weights = edge_equal_sample_weights(records)
    assert weights.sum() == pytest.approx(4.0)
