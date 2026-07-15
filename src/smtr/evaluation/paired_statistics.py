"""Paired statistics for SMTR experiment runs."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

import numpy as np

from smtr.experiment.schemas import ComparisonRunRecord, MethodSummary

MetricName = Literal[
    "success",
    "positive_transfer",
    "negative_transfer",
    "net_transfer",
    "exposure",
    "opportunity_capture",
    "safety_preservation",
]


@dataclass(frozen=True)
class PairedDifference:
    method_a: str
    method_b: str
    metric: str
    point_estimate: float
    ci_low: float
    ci_high: float
    n_base_episodes: int
    valid_bootstrap_resamples: int = 0


def compute_method_summary(method_id: str, runs: list[ComparisonRunRecord]) -> MethodSummary:
    method_runs = [run for run in runs if run.method == method_id]
    if not method_runs:
        return MethodSummary(method=method_id)
    by_base = _aggregate_by_base(method_runs)
    values = list(by_base.values())
    success = [value["success"] for value in values]
    exposure = [value["exposure"] for value in values]
    pos = [value["positive_transfer"] for value in values if value["has_label"]]
    neg = [value["negative_transfer"] for value in values if value["has_label"]]
    all_withhold = [1.0 if value["exposure"] == 0 else 0.0 for value in values]
    return MethodSummary(
        method=method_id,
        episode_count=len(values),
        success_rate=_mean(success),
        avg_selected_size=_mean(exposure),
        median_selected_size=float(np.median(exposure)) if exposure else 0.0,
        all_withhold_rate=_mean(all_withhold),
        avg_candidate_count=_mean(
            [
                float(_mean([len(inv.candidate_memory_ids) for inv in run.invocations]))
                for run in method_runs
            ]
        ),
        mean_runtime=_mean([run.runtime_seconds for run in method_runs]),
        positive_transfer_rate=_mean(pos) if pos else None,
        negative_transfer_rate=_mean(neg) if neg else None,
        neutral_success_rate=_label_rate(method_runs, "neutral_success"),
        neutral_failure_rate=_label_rate(method_runs, "neutral_failure"),
    )


def compute_paired_difference(
    runs: list[ComparisonRunRecord],
    method_a: str,
    method_b: str,
    metric: MetricName,
    *,
    bootstrap_seed: int = 42,
    bootstrap_n: int = 1000,
) -> PairedDifference:
    a = _aggregate_method_by_base(runs, method_a)
    b = _aggregate_method_by_base(runs, method_b)
    base_ids = sorted(set(a) & set(b))
    if not base_ids:
        return PairedDifference(method_a, method_b, metric, 0.0, 0.0, 0.0, 0)
    if metric in {"opportunity_capture", "safety_preservation"}:
        return _compute_conditional_paired_difference(
            runs,
            method_a,
            method_b,
            metric,
            base_ids=base_ids,
            bootstrap_seed=bootstrap_seed,
            bootstrap_n=bootstrap_n,
        )
    diffs = np.array([a[base_id][metric] - b[base_id][metric] for base_id in base_ids])
    point = float(diffs.mean())
    rng = np.random.default_rng(bootstrap_seed)
    samples = []
    for _ in range(bootstrap_n):
        idx = rng.integers(0, len(diffs), size=len(diffs))
        samples.append(float(diffs[idx].mean()))
    return PairedDifference(
        method_a=method_a,
        method_b=method_b,
        metric=metric,
        point_estimate=point,
        ci_low=float(np.percentile(samples, 2.5)),
        ci_high=float(np.percentile(samples, 97.5)),
        n_base_episodes=len(base_ids),
        valid_bootstrap_resamples=len(samples),
    )


def compute_group_bootstrap_ci(
    runs: list[ComparisonRunRecord],
    *,
    method_pairs: list[tuple[str, str]],
    bootstrap_seed: int,
    bootstrap_n: int,
) -> dict[str, dict[str, float | int | str]]:
    result: dict[str, dict[str, float | int | str]] = {}
    for method_a, method_b in method_pairs:
        for metric in (
            "success",
            "positive_transfer",
            "negative_transfer",
            "net_transfer",
            "exposure",
            "opportunity_capture",
            "safety_preservation",
        ):
            paired = compute_paired_difference(
                runs,
                method_a,
                method_b,
                metric,  # type: ignore[arg-type]
                bootstrap_seed=bootstrap_seed,
                bootstrap_n=bootstrap_n,
            )
            result[f"{method_a} - {method_b}:{metric}"] = {
                "method_a": method_a,
                "method_b": method_b,
                "metric": metric,
                "point_estimate": paired.point_estimate,
                "ci_low": paired.ci_low,
                "ci_high": paired.ci_high,
                "n_base_episodes": paired.n_base_episodes,
                "valid_bootstrap_resamples": paired.valid_bootstrap_resamples,
                "n_positive_opportunities": sum(
                    1 for base_id in paired_base_ids(runs, method_a, method_b)
                    if _b0_success_by_base(runs).get(base_id) is False
                ),
                "n_negative_opportunities": sum(
                    1 for base_id in paired_base_ids(runs, method_a, method_b)
                    if _b0_success_by_base(runs).get(base_id) is True
                ),
            }
    return result


def _aggregate_by_base(runs: list[ComparisonRunRecord]) -> dict[str, dict[str, float | bool]]:
    grouped: dict[str, list[ComparisonRunRecord]] = defaultdict(list)
    for run in runs:
        grouped[run.base_episode_id].append(run)
    result: dict[str, dict[str, float | bool]] = {}
    for base_id, base_runs in grouped.items():
        labels = [
            run.policy_level_transfer_label
            for run in base_runs
            if run.policy_level_transfer_label
        ]
        positive_transfer = (
            _mean([1.0 if label == "positive_transfer" else 0.0 for label in labels])
            if labels
            else 0.0
        )
        negative_transfer = (
            _mean([1.0 if label == "negative_transfer" else 0.0 for label in labels])
            if labels
            else 0.0
        )
        b0_success = _b0_success_by_base(runs).get(base_id)
        result[base_id] = {
            "success": _mean([float(run.team_success) for run in base_runs]),
            "positive_transfer": positive_transfer,
            "negative_transfer": negative_transfer,
            "net_transfer": positive_transfer - negative_transfer,
            "opportunity_capture": (
                positive_transfer if b0_success is False else 0.0
            ),
            "safety_preservation": (
                1.0 - negative_transfer if b0_success is True else 0.0
            ),
            "exposure": _mean([float(run.total_memory_exposures) for run in base_runs]),
            "has_label": bool(labels),
        }
    return result


def _aggregate_method_by_base(
    runs: list[ComparisonRunRecord],
    method: str,
) -> dict[str, dict[str, float | bool]]:
    return _aggregate_by_base_for_all(
        [run for run in runs if run.method == method],
        all_runs=runs,
    )


def _aggregate_by_base_for_all(
    runs: list[ComparisonRunRecord],
    *,
    all_runs: list[ComparisonRunRecord],
) -> dict[str, dict[str, float | bool]]:
    grouped: dict[str, list[ComparisonRunRecord]] = defaultdict(list)
    for run in runs:
        grouped[run.base_episode_id].append(run)
    b0_success_map = _b0_success_by_base(all_runs)
    result: dict[str, dict[str, float | bool]] = {}
    for base_id, base_runs in grouped.items():
        labels = [
            run.policy_level_transfer_label
            for run in base_runs
            if run.policy_level_transfer_label
        ]
        positive_transfer = (
            _mean([1.0 if label == "positive_transfer" else 0.0 for label in labels])
            if labels
            else 0.0
        )
        negative_transfer = (
            _mean([1.0 if label == "negative_transfer" else 0.0 for label in labels])
            if labels
            else 0.0
        )
        b0_success = b0_success_map.get(base_id)
        result[base_id] = {
            "success": _mean([float(run.team_success) for run in base_runs]),
            "positive_transfer": positive_transfer,
            "negative_transfer": negative_transfer,
            "net_transfer": positive_transfer - negative_transfer,
            "opportunity_capture": (
                positive_transfer if b0_success is False else 0.0
            ),
            "safety_preservation": (
                1.0 - negative_transfer if b0_success is True else 0.0
            ),
            "exposure": _mean([float(run.total_memory_exposures) for run in base_runs]),
            "has_label": bool(labels),
        }
    return result


def _compute_conditional_paired_difference(
    runs: list[ComparisonRunRecord],
    method_a: str,
    method_b: str,
    metric: Literal["opportunity_capture", "safety_preservation"],
    *,
    base_ids: list[str],
    bootstrap_seed: int,
    bootstrap_n: int,
) -> PairedDifference:
    a = _aggregate_method_by_base(runs, method_a)
    b = _aggregate_method_by_base(runs, method_b)
    point_a = _conditional_metric(a, runs, metric, base_ids)
    point_b = _conditional_metric(b, runs, metric, base_ids)
    if point_a is None or point_b is None:
        return PairedDifference(method_a, method_b, metric, 0.0, 0.0, 0.0, len(base_ids), 0)
    point = point_a - point_b
    rng = np.random.default_rng(bootstrap_seed)
    samples = []
    for _ in range(bootstrap_n):
        idx = rng.integers(0, len(base_ids), size=len(base_ids))
        sampled_base_ids = [base_ids[int(i)] for i in idx]
        sample_a = _conditional_metric(a, runs, metric, sampled_base_ids)
        sample_b = _conditional_metric(b, runs, metric, sampled_base_ids)
        if sample_a is None or sample_b is None:
            continue
        samples.append(sample_a - sample_b)
    if not samples:
        return PairedDifference(method_a, method_b, metric, point, point, point, len(base_ids), 0)
    return PairedDifference(
        method_a=method_a,
        method_b=method_b,
        metric=metric,
        point_estimate=point,
        ci_low=float(np.percentile(samples, 2.5)),
        ci_high=float(np.percentile(samples, 97.5)),
        n_base_episodes=len(base_ids),
        valid_bootstrap_resamples=len(samples),
    )


def _conditional_metric(
    method_by_base: dict[str, dict[str, float | bool]],
    runs: list[ComparisonRunRecord],
    metric: Literal["opportunity_capture", "safety_preservation"],
    base_ids: list[str],
) -> float | None:
    b0_success = _b0_success_by_base(runs)
    if metric == "opportunity_capture":
        opportunity_ids = [base_id for base_id in base_ids if b0_success.get(base_id) is False]
        if not opportunity_ids:
            return None
        return float(
            sum(float(method_by_base[base_id]["positive_transfer"]) for base_id in opportunity_ids)
            / len(opportunity_ids)
        )
    opportunity_ids = [base_id for base_id in base_ids if b0_success.get(base_id) is True]
    if not opportunity_ids:
        return None
    return float(
        1.0
        - sum(float(method_by_base[base_id]["negative_transfer"]) for base_id in opportunity_ids)
        / len(opportunity_ids)
    )


def _label_rate(runs: list[ComparisonRunRecord], label: str) -> float | None:
    labels = [run.policy_level_transfer_label for run in runs if run.policy_level_transfer_label]
    if not labels:
        return None
    return sum(1 for item in labels if item == label) / len(labels)


def _mean(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def paired_base_ids(
    runs: list[ComparisonRunRecord],
    method_a: str,
    method_b: str,
) -> list[str]:
    a = _aggregate_method_by_base(runs, method_a)
    b = _aggregate_method_by_base(runs, method_b)
    return sorted(set(a) & set(b))


def _b0_success_by_base(runs: list[ComparisonRunRecord]) -> dict[str, bool]:
    return {run.base_episode_id: run.team_success for run in runs if run.method == "B0"}
