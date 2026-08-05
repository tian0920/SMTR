"""清单 Test 5: per-receiver common seed support.

Formal receiver-policy replay requires identical seed support across all
candidate edges; pilot mode must use the intersection, never the union.
"""

from __future__ import annotations

import pytest

from smtr.marble.paired_evaluation import _receiver_episode_seed_support


def _edge_to_seeds() -> dict[tuple[str, str, str], set[int]]:
    return {
        ("t1", "r1", "memA"): {0, 1, 2, 3, 4},
        ("t1", "r1", "memB"): {0, 1, 2, 3, 4},
    }


def test_identical_support_passes_formal():
    seeds = _receiver_episode_seed_support(
        task_id="t1",
        receiver_agent_id="r1",
        candidate_memory_ids=["memA", "memB"],
        edge_to_seeds=_edge_to_seeds(),
        formal_mode=True,
    )
    assert seeds == [0, 1, 2, 3, 4]


def test_mismatched_support_fails_formal():
    edge_to_seeds = _edge_to_seeds()
    edge_to_seeds[("t1", "r1", "memB")] = {0, 1, 2, 3}
    with pytest.raises(ValueError, match="identical seed support"):
        _receiver_episode_seed_support(
            task_id="t1",
            receiver_agent_id="r1",
            candidate_memory_ids=["memA", "memB"],
            edge_to_seeds=edge_to_seeds,
            formal_mode=True,
        )


def test_pilot_uses_intersection_not_union():
    edge_to_seeds = _edge_to_seeds()
    edge_to_seeds[("t1", "r1", "memB")] = {0, 1, 2, 3}
    seeds = _receiver_episode_seed_support(
        task_id="t1",
        receiver_agent_id="r1",
        candidate_memory_ids=["memA", "memB"],
        edge_to_seeds=edge_to_seeds,
        formal_mode=False,
    )
    # No B/seed4 trace may ever be created: seed 4 is outside the support.
    assert seeds == [0, 1, 2, 3]
    assert 4 not in seeds


def test_missing_edge_formal_fails_pilot_skips():
    edge_to_seeds = _edge_to_seeds()
    with pytest.raises(ValueError, match="without valid paired outcomes"):
        _receiver_episode_seed_support(
            task_id="t1",
            receiver_agent_id="r1",
            candidate_memory_ids=["memA", "memC"],
            edge_to_seeds=edge_to_seeds,
            formal_mode=True,
        )
    assert _receiver_episode_seed_support(
        task_id="t1",
        receiver_agent_id="r1",
        candidate_memory_ids=["memA", "memC"],
        edge_to_seeds=edge_to_seeds,
        formal_mode=False,
    ) == []
