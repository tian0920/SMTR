"""Paired group bootstrap comparisons for ablation experiments.

Bootstrap unit: base episode = (task_instance_id, generation_seed).
Multiple traversal seeds belong to the same group and are resampled together.

Reports paired 95% CI for method-pair differences in:
- success rate
- negative transfer rate
- positive transfer rate
- average selected count
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel


class PairedComparisonResult(BaseModel):
    """Result of a paired comparison between two methods."""

    method_a: str
    method_b: str
    n_base_episodes: int = 0
    n_traversal_runs: int = 0
    # Per-metric deltas (a - b)
    success_diff_mean: float = 0.0
    success_diff_ci_low: float = 0.0
    success_diff_ci_high: float = 0.0
    neg_transfer_diff_mean: float = 0.0
    neg_transfer_diff_ci_low: float = 0.0
    neg_transfer_diff_ci_high: float = 0.0
    pos_transfer_diff_mean: float = 0.0
    pos_transfer_diff_ci_low: float = 0.0
    pos_transfer_diff_ci_high: float = 0.0
    avg_selected_diff_mean: float = 0.0
    avg_selected_diff_ci_low: float = 0.0
    avg_selected_diff_ci_high: float = 0.0


def _group_runs_by_episode(
    runs: list[dict[str, Any]],
    method: str,
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    """Group runs by (task_instance_id, generation_seed) for a method."""
    method_runs = [r for r in runs if r.get("method") == method]
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for run in method_runs:
        key = (run.get("task_instance_id", ""), run.get("generation_seed", 0))
        groups.setdefault(key, []).append(run)
    return groups


def _episode_metric(
    group_runs: list[dict[str, Any]],
    metric: str,
) -> float:
    """Compute a per-episode metric by averaging across traversal seeds."""
    values = []
    for run in group_runs:
        if metric == "success":
            values.append(float(run.get("team_success", False)))
        elif metric == "selected_count":
            values.append(float(run.get("selected_count", 0)))
        elif metric == "negative_transfer":
            label = run.get("policy_level_transfer_label")
            values.append(1.0 if label == "negative_transfer" else 0.0)
        elif metric == "positive_transfer":
            label = run.get("policy_level_transfer_label")
            values.append(1.0 if label == "positive_transfer" else 0.0)
    return float(np.mean(values)) if values else 0.0


def compute_paired_comparisons(
    runs: list[dict[str, Any]],
    *,
    pairs: list[tuple[str, str]] | None = None,
    bootstrap_seed: int = 42,
    n_bootstrap: int = 1000,
) -> dict[str, PairedComparisonResult]:
    """Compute paired group bootstrap 95% CI for method-pair differences.

    Args:
        runs: All run record dicts from the experiment.
        pairs: List of (method_a, method_b) pairs to compare.
            Defaults to SMTR vs each formal baseline.
        bootstrap_seed: Random seed for bootstrap.
        n_bootstrap: Number of bootstrap iterations.

    Returns:
        Dict of "method_a vs method_b" -> PairedComparisonResult.
    """
    if pairs is None:
        pairs = [
            ("SMTR", "B1-Matched"),
            ("SMTR", "EffectOnly-SMTR"),
            ("SMTR", "Static-SMTR"),
            ("SMTR", "FactualSuccess-SMTR"),
        ]

    rng = np.random.default_rng(bootstrap_seed)
    results: dict[str, PairedComparisonResult] = {}

    for method_a, method_b in pairs:
        groups_a = _group_runs_by_episode(runs, method_a)
        groups_b = _group_runs_by_episode(runs, method_b)

        # Use intersection of episode keys (both methods must have run)
        common_keys = sorted(set(groups_a.keys()) & set(groups_b.keys()))
        if not common_keys:
            results[f"{method_a} vs {method_b}"] = PairedComparisonResult(
                method_a=method_a, method_b=method_b
            )
            continue

        n_groups = len(common_keys)
        n_trav = sum(len(groups_a[k]) for k in common_keys) / n_groups

        # Compute per-episode metrics for each group
        metrics_a: dict[str, list[float]] = {
            "success": [], "neg_transfer": [], "pos_transfer": [], "selected": []
        }
        metrics_b: dict[str, list[float]] = {
            "success": [], "neg_transfer": [], "pos_transfer": [], "selected": []
        }

        for key in common_keys:
            metrics_a["success"].append(_episode_metric(groups_a[key], "success"))
            metrics_b["success"].append(_episode_metric(groups_b[key], "success"))
            metrics_a["neg_transfer"].append(_episode_metric(groups_a[key], "negative_transfer"))
            metrics_b["neg_transfer"].append(_episode_metric(groups_b[key], "negative_transfer"))
            metrics_a["pos_transfer"].append(_episode_metric(groups_a[key], "positive_transfer"))
            metrics_b["pos_transfer"].append(_episode_metric(groups_b[key], "positive_transfer"))
            metrics_a["selected"].append(_episode_metric(groups_a[key], "selected_count"))
            metrics_b["selected"].append(_episode_metric(groups_b[key], "selected_count"))

        # Bootstrap
        boot_deltas: dict[str, list[float]] = {
            key: []
            for key in ["success", "neg_transfer", "pos_transfer", "selected"]
        }

        for _ in range(n_bootstrap):
            indices = rng.integers(0, n_groups, size=n_groups)
            for metric_key in boot_deltas:
                vals_a = [metrics_a[metric_key][i] for i in indices]
                vals_b = [metrics_b[metric_key][i] for i in indices]
                mean_a = float(np.mean(vals_a))
                mean_b = float(np.mean(vals_b))
                boot_deltas[metric_key].append(mean_a - mean_b)

        def _summarize(deltas: list[float]) -> tuple[float, float, float]:
            arr = np.array(deltas)
            return (
                float(np.mean(arr)),
                float(np.percentile(arr, 2.5)),
                float(np.percentile(arr, 97.5)),
            )

        s_mean, s_low, s_high = _summarize(boot_deltas["success"])
        nt_mean, nt_low, nt_high = _summarize(boot_deltas["neg_transfer"])
        pt_mean, pt_low, pt_high = _summarize(boot_deltas["pos_transfer"])
        sel_mean, sel_low, sel_high = _summarize(boot_deltas["selected"])

        pair_key = f"{method_a} vs {method_b}"
        results[pair_key] = PairedComparisonResult(
            method_a=method_a,
            method_b=method_b,
            n_base_episodes=n_groups,
            n_traversal_runs=int(round(n_trav)),
            success_diff_mean=s_mean,
            success_diff_ci_low=s_low,
            success_diff_ci_high=s_high,
            neg_transfer_diff_mean=nt_mean,
            neg_transfer_diff_ci_low=nt_low,
            neg_transfer_diff_ci_high=nt_high,
            pos_transfer_diff_mean=pt_mean,
            pos_transfer_diff_ci_low=pt_low,
            pos_transfer_diff_ci_high=pt_high,
            avg_selected_diff_mean=sel_mean,
            avg_selected_diff_ci_low=sel_low,
            avg_selected_diff_ci_high=sel_high,
        )

    return results
