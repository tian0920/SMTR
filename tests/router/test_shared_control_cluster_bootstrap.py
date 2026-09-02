"""Control-family cluster bootstrap (清单 Shared-Control 第10章).

Bootstrap resamples whole (task, receiver) control families; rows that
share one no-memory control never split across a bootstrap member.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from smtr.core.types import (
    AgentProfile,
    CandidateExposureInput,
    MemoryRoutingCard,
    ReceiverState,
)
from smtr.counterfactual.edge_keys import (
    control_family_key,
    group_records_by_control_family,
)
from smtr.router.transfer_critic import (
    FourOutcomeTransferCritic,
    _cluster_bootstrap_with_full_coverage,
)


def _exposure(memory_id: str) -> CandidateExposureInput:
    writer = AgentProfile(agent_id="w1", role="planner", capabilities=("plan",))
    receiver = AgentProfile(agent_id="r1", role="executor", capabilities=("sql",))
    card = MemoryRoutingCard(
        memory_id=memory_id,
        goal_summary="Diagnose database issue",
        task_tags=("database",),
        environment_constraints=("read-only",),
        writer=writer,
        source_task_id="t1",
        source_scenario="database",
        compatible_receiver_roles=("executor",),
    )
    rs = ReceiverState(
        task_id="t2",
        scenario="database",
        task_instruction="Fix the slow query",
        receiver=receiver,
    )
    return CandidateExposureInput(receiver_state=rs, candidate_card=card)


def _records(task_id: str, receiver: str, *, rows: int) -> list[dict]:
    return [
        {
            "task_id": task_id,
            "receiver_agent_id": receiver,
            "candidate_memory_id": f"m{idx % 4}",
            "generation_seed": idx % 5,
        }
        for idx in range(rows)
    ]


def test_bootstrap_draws_keep_families_whole():
    records = _records("t1", "r1", rows=12) + _records("t2", "r2", rows=8)
    clusters = group_records_by_control_family(records)
    assert set(clusters) == {("t1", "r1"), ("t2", "r2")}
    family_sizes = {
        key: len(indices) for key, indices in clusters.items()
    }

    # Both families contain both classes, so every draw covers them.
    y = np.array([idx % 2 for idx in range(len(records))], dtype=int)
    rng = np.random.default_rng(7)
    for _ in range(50):
        draw = _cluster_bootstrap_with_full_coverage(
            y, clusters, {0, 1}, rng, max_attempts=1
        )
        assert draw is not None
        # Whole-cluster resampling: every family appears an integer number
        # of times (with replacement, so totals may exceed the row count).
        counts = Counter(
            control_family_key(records[idx]) for idx in draw.tolist()
        )
        for family, count in counts.items():
            assert count % family_sizes[family] == 0, family
        assert sum(counts.values()) == len(draw)


def test_fit_requires_clusters_to_partition_every_row():
    critic = FourOutcomeTransferCritic()
    with pytest.raises(
        ValueError, match="must partition every training record row"
    ):
        critic.fit(
            inputs=[_exposure("m1"), _exposure("m2")],
            labels=["positive_transfer", "negative_transfer"],
            bootstrap_clusters={("t1", "r1"): [0]},
        )


def test_cluster_partition_missing_row_fails():
    """Clusters that omit at least one row must raise."""
    critic = FourOutcomeTransferCritic()
    with pytest.raises(
        ValueError, match="must partition every training record row"
    ):
        critic.fit(
            inputs=[_exposure("m1"), _exposure("m2"), _exposure("m3")],
            labels=["positive_transfer", "negative_transfer", "positive_transfer"],
            bootstrap_clusters={("t1", "r1"): [0, 1]},  # row 2 missing
        )


def test_cluster_partition_duplicate_row_fails():
    """Clusters that contain the same row index twice must raise."""
    critic = FourOutcomeTransferCritic()
    with pytest.raises(ValueError, match="overlap"):
        critic.fit(
            inputs=[_exposure("m1"), _exposure("m2")],
            labels=["positive_transfer", "negative_transfer"],
            bootstrap_clusters={("t1", "r1"): [0, 1], ("t2", "r2"): [1]},
        )


def test_cluster_partition_out_of_range_fails():
    """Clusters with an index >= n_rows (or < 0) must raise."""
    critic = FourOutcomeTransferCritic()
    with pytest.raises(ValueError, match="out of range"):
        critic.fit(
            inputs=[_exposure("m1"), _exposure("m2")],
            labels=["positive_transfer", "negative_transfer"],
            bootstrap_clusters={("t1", "r1"): [0, 1, 5]},
        )


def test_cluster_partition_complete_passes():
    """A valid partition of all rows must not raise a partition error."""
    critic = FourOutcomeTransferCritic(seed=42)
    # 4 rows split into two clusters covering every row exactly once.
    critic.fit(
        inputs=[
            _exposure("m1"),
            _exposure("m2"),
            _exposure("m3"),
            _exposure("m4"),
        ],
        labels=[
            "positive_transfer",
            "negative_transfer",
            "positive_transfer",
            "negative_transfer",
        ],
        bootstrap_clusters={("t1", "r1"): [0, 1], ("t2", "r2"): [2, 3]},
    )
    assert critic._fitted
