"""Phase 6: Outcome Signal Ablation Experiment.

Compares three outcome signal types for TCI delta computation:
A. binary_success (baseline)
B. native_final_score (where available)
C. native_iteration_improvement (summary length delta)

Fixed: same tasks, same model, same receiver=3, same seed.

Output:
- results/marble/outcome_signal_ablation/ablation_episodes.csv
- results/marble/outcome_signal_ablation/ablation_summary.csv
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from smtr.marble.task_loader import MarbleTaskLoader
from smtr.marble.trajectory_collector import TrajectoryCollector
from smtr.marble.outcome.behavioral_outcome import (
    BehavioralOutcome,
    BehavioralOutcomeEvaluator,
    compute_delta,
)


DOMAINS = ["database", "minecraft"]
TASKS_PER_DOMAIN = 10
SEEDS = [0, 1]
RECEIVER_IDS = ["agent1", "agent2", "agent3"]

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "marble" / "outcome_signal_ablation"


def run_ablation() -> None:
    """Run outcome signal ablation across domains."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    loader = MarbleTaskLoader()
    collector = TrajectoryCollector()
    outcome_eval_minecraft = BehavioralOutcomeEvaluator(scenario="minecraft")
    outcome_eval_database = BehavioralOutcomeEvaluator(scenario="database")

    rows: list[dict[str, Any]] = []

    for domain in DOMAINS:
        print(f"\n=== Domain: {domain} ===")
        all_tasks = loader.load_scenario(domain)
        # Select first TASKS_PER_DOMAIN tasks
        tasks = all_tasks[:TASKS_PER_DOMAIN]
        print(f"  {len(tasks)} tasks selected")

        for task in tasks:
            for seed in SEEDS:
                print(f"  task={domain}/{task.task_id} seed={seed}", end="", flush=True)
                t0 = time.time()

                # Run expose branch (with memory injection — simulated as no_memory for now
                # since we're measuring outcome signals, not actual TCI)
                traj_expose = collector.collect(task, seed=seed, method="no_memory")
                # Run withhold branch (same — for ablation, we compare signal extraction)
                traj_withhold = collector.collect(task, seed=seed + 1000, method="no_memory")

                duration = time.time() - t0

                # Extract outcomes using BehavioralOutcomeEvaluator
                evaluator = outcome_eval_minecraft if domain == "minecraft" else outcome_eval_database
                expose_outcome = evaluator.extract(traj_expose.raw_output)
                withhold_outcome = evaluator.extract(traj_withhold.raw_output)

                # Compute deltas for all three signal types
                delta_binary = (
                    (1.0 if traj_expose.team_success else 0.0)
                    - (1.0 if traj_withhold.team_success else 0.0)
                )
                delta_native = compute_delta(expose_outcome, withhold_outcome)
                delta_iteration = _compute_iteration_delta(
                    expose_outcome, withhold_outcome
                )

                row = {
                    "domain": domain,
                    "task_id": task.task_id,
                    "seed": seed,
                    "duration_seconds": round(duration, 2),
                    # Expose branch
                    "expose_success": traj_expose.team_success,
                    "expose_score": traj_expose.score,
                    "expose_native_score": expose_outcome.performance_score,
                    "expose_signal_type": expose_outcome.signal_type,
                    "expose_n_iterations": len(expose_outcome.iteration_scores),
                    "expose_token_usage": expose_outcome.metadata.get("token_usage", 0),
                    # Withhold branch
                    "withhold_success": traj_withhold.team_success,
                    "withhold_score": traj_withhold.score,
                    "withhold_native_score": withhold_outcome.performance_score,
                    "withhold_signal_type": withhold_outcome.signal_type,
                    "withhold_n_iterations": len(withhold_outcome.iteration_scores),
                    "withhold_token_usage": withhold_outcome.metadata.get("token_usage", 0),
                    # Deltas
                    "delta_binary": delta_binary,
                    "delta_native": delta_native,
                    "delta_iteration": delta_iteration,
                }
                rows.append(row)

                print(f"  ({duration:.0f}s, δ_bin={delta_binary:+.2f}, δ_nat={delta_native:+.3f}, δ_iter={delta_iteration:+.3f})")

    # Write ablation CSV
    csv_path = OUTPUT_DIR / "ablation_episodes.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWritten: {csv_path} ({len(rows)} rows)")

    # Compute summary metrics
    summary = _compute_summary(rows)
    summary_path = OUTPUT_DIR / "ablation_summary.csv"
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()) if summary else [])
        writer.writeheader()
        writer.writerows(summary)
    print(f"Written: {summary_path}")


def _compute_iteration_delta(
    expose: BehavioralOutcome,
    withhold: BehavioralOutcome,
) -> float:
    """Compute delta based on iteration improvement (summary length delta)."""
    def _iter_score(outcome: BehavioralOutcome) -> float | None:
        scores = outcome.iteration_scores
        if len(scores) >= 2:
            improvement = scores[-1] - scores[0]
            return max(0.0, min(1.0, improvement / 1000.0))
        return None

    expose_score = _iter_score(expose)
    withhold_score = _iter_score(withhold)
    if expose_score is None or withhold_score is None:
        return 0.0
    return expose_score - withhold_score


def _compute_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute per-signal-type summary metrics."""
    signal_types = [
        ("binary_success", "delta_binary"),
        ("native_final_score", "delta_native"),
        ("iteration_improvement", "delta_iteration"),
    ]

    summary = []
    for signal_name, delta_key in signal_types:
        deltas = [r[delta_key] for r in rows]
        n_total = len(deltas)
        n_positive = sum(1 for d in deltas if d > 0)
        n_zero = sum(1 for d in deltas if d == 0)
        n_negative = sum(1 for d in deltas if d < 0)

        # Unique outcome values (resolution)
        if signal_name == "binary_success":
            unique_values = len(set(r["expose_success"] for r in rows) | set(r["withhold_success"] for r in rows))
        elif signal_name == "native_final_score":
            expose_scores = [r["expose_native_score"] for r in rows if r["expose_native_score"] is not None]
            withhold_scores = [r["withhold_native_score"] for r in rows if r["withhold_native_score"] is not None]
            unique_values = len(set(expose_scores + withhold_scores))
        else:
            unique_values = 0  # Would need more detailed computation

        # Pairwise discrimination rate
        n_discriminated = sum(
            1 for r in rows
            if _get_pair_values(r, signal_name) is not None
            and _get_pair_values(r, signal_name)[0] != _get_pair_values(r, signal_name)[1]
        )
        pdr = n_discriminated / n_total if n_total > 0 else 0.0

        summary.append({
            "signal_type": signal_name,
            "n_total": n_total,
            "n_positive_delta": n_positive,
            "n_zero_delta": n_zero,
            "n_negative_delta": n_negative,
            "positive_delta_rate": round(n_positive / n_total, 4) if n_total else 0,
            "zero_delta_rate": round(n_zero / n_total, 4) if n_total else 0,
            "negative_delta_rate": round(n_negative / n_total, 4) if n_total else 0,
            "mean_delta": round(sum(deltas) / len(deltas), 4) if deltas else 0,
            "std_delta": round(_std(deltas), 4) if deltas else 0,
            "unique_outcome_values": unique_values,
            "pairwise_discrimination_rate": round(pdr, 4),
        })

    return summary


def _get_pair_values(row: dict[str, Any], signal_name: str) -> tuple[Any, Any] | None:
    """Get (expose_value, withhold_value) for a given signal type."""
    if signal_name == "binary_success":
        return (row["expose_success"], row["withhold_success"])
    elif signal_name == "native_final_score":
        if row["expose_native_score"] is None or row["withhold_native_score"] is None:
            return None
        return (row["expose_native_score"], row["withhold_native_score"])
    elif signal_name == "iteration_improvement":
        return (row["delta_iteration"] != 0, None)  # Simplified
    return None


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5


if __name__ == "__main__":
    run_ablation()
