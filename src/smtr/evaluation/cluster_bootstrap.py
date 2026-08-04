"""Cluster bootstrap confidence intervals for paired evaluation statistics.

Individual candidate records are not independent: records from the same
target task (and receiver episode) share environment, evaluator and writer
memory pool. Ordinary per-record bootstrap is therefore forbidden (清单
第十三章). All reported intervals must resample whole clusters:

* ``target_task_id``; or
* ``(target_task_id, receiver_agent_id)``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Sequence

import numpy as np

CLUSTER_TARGET_TASK = "target_task_id"
CLUSTER_TASK_RECEIVER = "target_task_id+receiver_agent_id"
ALLOWED_CLUSTER_UNITS = (CLUSTER_TARGET_TASK, CLUSTER_TASK_RECEIVER)


def cluster_key(unit: dict[str, Any], cluster_by: str) -> str:
    """Cluster identifier for one statistical unit."""
    task_id = str(unit.get("task_id", ""))
    if cluster_by == CLUSTER_TARGET_TASK:
        return task_id
    if cluster_by == CLUSTER_TASK_RECEIVER:
        return f"{task_id}::{unit.get('receiver_agent_id', '')}"
    raise ValueError(
        f"cluster_by must be one of {ALLOWED_CLUSTER_UNITS}, got {cluster_by!r}"
    )


def cluster_bootstrap_ci(
    units: Sequence[dict[str, Any]],
    *,
    statistic: Callable[[Sequence[dict[str, Any]]], float],
    cluster_by: str = CLUSTER_TARGET_TASK,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 7,
) -> dict[str, Any]:
    """Percentile cluster bootstrap confidence interval.

    Whole clusters are resampled with replacement; every unit of a resampled
    cluster enters the bootstrap sample together. ``confidence`` must be at
    least 0.95 per the audit requirement.
    """
    if confidence < 0.95:
        raise ValueError("confidence must be at least 0.95")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")

    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        clusters[cluster_key(unit, cluster_by)].append(unit)

    point_estimate = float(statistic(list(units)))
    cluster_ids = sorted(clusters)
    n_clusters = len(cluster_ids)
    if n_clusters == 0:
        return {
            "point_estimate": point_estimate,
            "ci_lower": point_estimate,
            "ci_upper": point_estimate,
            "confidence": confidence,
            "n_bootstrap": 0,
            "n_clusters": 0,
            "cluster_by": cluster_by,
        }

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_clusters, size=(n_bootstrap, n_clusters))
    boot_stats = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sample: list[dict[str, Any]] = []
        for idx in draws[b]:
            sample.extend(clusters[cluster_ids[int(idx)]])
        boot_stats[b] = statistic(sample)

    alpha = (1.0 - confidence) / 2.0
    return {
        "point_estimate": point_estimate,
        "ci_lower": float(np.quantile(boot_stats, alpha)),
        "ci_upper": float(np.quantile(boot_stats, 1.0 - alpha)),
        "confidence": confidence,
        "n_bootstrap": n_bootstrap,
        "n_clusters": n_clusters,
        "cluster_by": cluster_by,
    }
