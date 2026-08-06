"""Shared treatment-edge key definitions (清单 P0-3).

One treatment edge is ``e = (task, receiver, memory)``. Different
generation seeds are repeated random trials of the *same* edge, never
independent task samples. Every module that groups, splits, weights or
resamples paired records must use these key definitions so the field
order is defined exactly once project-wide.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

# Fixed field order: task_id, receiver_agent_id, candidate_memory_id.
TreatmentEdgeKey = tuple[str, str, str]

# Seed-level key adds generation_seed as the fourth element.
SeedPairKey = tuple[str, str, str, int]

# Shared no-memory control identity (清单 Shared-Control 第2章): one
# control execution per (task, receiver, seed) group, reused by every
# candidate share on that group.
ControlGroupKey = tuple[str, str, int]

# Control family spans all seeds of one (task, receiver) group and is
# the critic bootstrap cluster unit.
ControlFamilyKey = tuple[str, str]


def treatment_edge_key(record: dict[str, Any]) -> TreatmentEdgeKey:
    """Edge identity of one paired record: (task, receiver, memory)."""
    return (
        str(record["task_id"]),
        str(record["receiver_agent_id"]),
        str(record["candidate_memory_id"]),
    )


def seed_pair_key(record: dict[str, Any]) -> SeedPairKey:
    """Seed-level identity of one paired record."""
    return (
        str(record["task_id"]),
        str(record["receiver_agent_id"]),
        str(record["candidate_memory_id"]),
        int(record["generation_seed"]),
    )


def control_group_key(record: dict[str, Any]) -> ControlGroupKey:
    """Shared-control group identity: (task, receiver, seed)."""
    return (
        str(record["task_id"]),
        str(record["receiver_agent_id"]),
        int(record["generation_seed"]),
    )


def control_family_key(record: dict[str, Any]) -> ControlFamilyKey:
    """Control family identity: (task, receiver) across all seeds."""
    return (
        str(record["task_id"]),
        str(record["receiver_agent_id"]),
    )


def group_records_by_edge(
    records: list[dict[str, Any]],
) -> dict[TreatmentEdgeKey, list[int]]:
    """Map each treatment edge to the row indices of its seed records."""
    groups: dict[TreatmentEdgeKey, list[int]] = defaultdict(list)
    for idx, rec in enumerate(records):
        groups[treatment_edge_key(rec)].append(idx)
    return dict(groups)


def group_records_by_control_family(
    records: list[dict[str, Any]],
) -> dict[ControlFamilyKey, list[int]]:
    """Map each control family to the row indices of its records.

    One family covers all candidates, seeds and shared controls of a
    (task, receiver) group; bootstrap members resample whole families so
    the statistically dependent share/control pairs never split across
    a member (清单 Shared-Control 第10章).
    """
    groups: dict[ControlFamilyKey, list[int]] = defaultdict(list)
    for idx, rec in enumerate(records):
        groups[control_family_key(rec)].append(idx)
    return dict(groups)


def edge_equal_sample_weights(records: list[dict[str, Any]]) -> np.ndarray:
    """Per-record training weight ``1 / n_e`` (清单 P0-5).

    ``n_e`` is the number of valid seed records on the record's edge, so
    every edge contributes total weight 1 regardless of its seed count:
    edges with many seeds never outweigh edges with few.
    """
    edge_counts = Counter(treatment_edge_key(rec) for rec in records)
    return np.asarray(
        [1.0 / edge_counts[treatment_edge_key(rec)] for rec in records],
        dtype=float,
    )


def edge_cluster_bootstrap_indices(
    records: list[dict[str, Any]],
    rng: np.random.Generator,
) -> np.ndarray:
    """Edge-cluster bootstrap draw (清单 P0-6).

    Samples treatment edges with replacement; every draw of an edge
    contributes *all* of its seed records, so one edge's seeds can never
    be split across a bootstrap member and seeds are never treated as
    independent tasks.
    """
    groups = group_records_by_edge(records)
    edges = list(groups.keys())
    if not edges:
        return np.asarray([], dtype=int)
    chosen = rng.choice(len(edges), size=len(edges), replace=True)
    indices: list[int] = []
    for pos in chosen:
        indices.extend(groups[edges[pos]])
    indices_array = np.asarray(indices, dtype=int)
    rng.shuffle(indices_array)
    return indices_array
