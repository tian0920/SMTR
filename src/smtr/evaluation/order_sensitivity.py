"""Minimal order-sensitivity diagnostic for formal SMTR."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

from smtr.experiment.runner import ComparisonRunner
from smtr.experiment.schemas import ComparisonRunRecord, ExperimentConfig
from smtr.experiment.writer import ExperimentWriter

ORDER_SENSITIVITY_METRICS = (
    "order_outcome_flip_rate",
    "selected_set_exact_match_rate",
    "mean_selected_set_jaccard",
)


def run_order_sensitivity(
    *,
    db_path: str,
    critic_checkpoint: str,
    output_dir: str,
    task_seeds: list[int],
    generation_seeds: list[int],
    scenario_replicates: int,
    scenario: str,
    candidate_count: int = 4,
    negative_risk_budget: float = 0.2,
    overwrite: bool = False,
    fail_fast: bool = True,
) -> dict[str, float]:
    """Run SMTR under all K=4 traversal permutations and return three metrics."""
    if candidate_count != 4:
        raise ValueError("minimal order sensitivity diagnostic requires K=4")
    if scenario != "prefix_sensitive":
        raise ValueError("minimal order sensitivity diagnostic is prefix-sensitive only")
    output_path = Path(output_dir)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output directory already exists: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    all_runs: list[ComparisonRunRecord] = []
    for permutation in itertools.permutations(range(candidate_count)):
        permutation_id = "rank_" + "_".join(str(index) for index in permutation)
        config = ExperimentConfig(
            db_path=db_path,
            critic_checkpoint=critic_checkpoint,
            task_seeds=task_seeds,
            generation_seeds=generation_seeds,
            traversal_seeds=[0],
            scenario_replicates=scenario_replicates,
            top_k=candidate_count,
            negative_risk_budget=negative_risk_budget,
            output_dir=str(output_path / "_permutations" / permutation_id),
            overwrite=True,
            fail_fast=fail_fast,
            scenario=scenario,
            methods=["SMTR"],
            explicit_permutation=list(permutation),
            permutation_id=permutation_id,
            permutation_application_policy="critic_traversal_only",
        )
        runner = ComparisonRunner(config)
        runner.run()
        writer = ExperimentWriter(config.output_dir, overwrite=True)
        all_runs.extend(writer.load_runs())

    metrics = compute_minimal_order_metrics(all_runs)
    (output_path / "order_sensitivity.json").write_text(
        _json_dumps(metrics) + "\n",
        encoding="utf-8",
    )
    return metrics


def compute_minimal_order_metrics(
    runs: list[ComparisonRunRecord],
) -> dict[str, float]:
    """Compute only the retained order-sensitivity metrics."""
    grouped: dict[str, list[ComparisonRunRecord]] = {}
    for run in runs:
        if run.method == "SMTR":
            grouped.setdefault(run.base_episode_id, []).append(run)
    if not grouped:
        return {metric: 0.0 for metric in ORDER_SENSITIVITY_METRICS}

    outcome_flips = []
    exact_matches = []
    jaccards = []
    for base_runs in grouped.values():
        outcomes = [int(run.team_success) for run in base_runs]
        outcome_flips.append(float(max(outcomes) != min(outcomes)))
        selected_sets = [
            frozenset(run.unique_selected_memory_ids or run.selected_memory_ids)
            for run in base_runs
        ]
        first = selected_sets[0] if selected_sets else frozenset()
        exact_matches.append(float(all(item == first for item in selected_sets)))
        for index, left in enumerate(selected_sets):
            for right in selected_sets[index + 1:]:
                union = left | right
                jaccards.append(1.0 if not union else len(left & right) / len(union))

    return {
        "order_outcome_flip_rate": _mean(outcome_flips),
        "selected_set_exact_match_rate": _mean(exact_matches),
        "mean_selected_set_jaccard": _mean(jaccards),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)
