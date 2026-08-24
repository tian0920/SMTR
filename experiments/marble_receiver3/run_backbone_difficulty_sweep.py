"""Backbone difficulty sweep: test no_memory baseline across model tiers.

This script runs no_memory baseline for 3 model tiers to find a non-saturated
backbone. The selection criterion is:

    30% <= baseline_success_rate <= 80%

This avoids floor (too hard) and ceiling (too easy) effects.

**IMPORTANT**: This script only runs no_memory baseline. It does NOT run SMTR
or any memory-augmented methods. The backbone selection is based solely on
baseline difficulty, not on SMTR performance.

Models tested (from current provider):
- **Strong**: qwen3-30b-a3b (MoE, 30B total, 3B active)
- **Medium**: qwen3-14b (dense, 14B)
- **Smaller**: qwen3-8b (dense, 8B)

Usage::

    export DASHSCOPE_API_KEY="sk-..."
    export OPENAI_API_KEY="$DASHSCOPE_API_KEY"
    export DASHSCOPE_BASE_URL="https://llm-jhxtd03gjg0gd2o2.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
    export OPENAI_BASE_URL="$DASHSCOPE_BASE_URL"

    python experiments/marble_receiver3/run_backbone_difficulty_sweep.py

Output::

    results/marble/backbone_sweep/
    ├── sweep_results.csv
    ├── model_comparison.md
    └── recommended_backbone.md
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from smtr.marble.task_loader import MarbleTaskLoader, ALL_SCENARIOS
from smtr.marble.trajectory_collector import TrajectoryCollector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# Model tiers to test
MODEL_TIERS = [
    {
        "tier": "strong",
        "model_id": "openai/qwen3-30b-a3b",
        "description": "MoE, 30B total, 3B active",
    },
    {
        "tier": "medium",
        "model_id": "openai/qwen3-14b",
        "description": "Dense, 14B",
    },
    {
        "tier": "small",
        "model_id": "openai/qwen3-8b",
        "description": "Dense, 8B",
    },
]

# Sweep configuration
SWEEP_CONFIG = {
    "scenarios": list(ALL_SCENARIOS),
    "tasks_per_scenario": 10,
    "seeds": [0, 1, 2],
    "method": "no_memory",
    "max_iterations": 10,
    "timeout_seconds": 1200,
}


def run_single_episode(
    *,
    model_id: str,
    scenario: str,
    task_id: str,
    seed: int,
    collector: TrajectoryCollector,
) -> dict[str, Any]:
    """Run one episode with no_memory baseline."""
    start_time = time.time()

    try:
        # Load task
        loader = MarbleTaskLoader()
        task = loader.get_task(scenario, task_id)

        # Run with no_memory
        trajectory = collector.collect(
            task=task,
            method="no_memory",
            memory_injection=None,
            selected_memory_ids=None,
            generation_seed=seed,
        )

        duration = time.time() - start_time

        return {
            "model_id": model_id,
            "scenario": scenario,
            "task_id": task_id,
            "seed": seed,
            "method": "no_memory",
            "team_success": trajectory.team_success,
            "team_reward": trajectory.team_reward,
            "score": trajectory.score,
            "n_iterations": len(trajectory.iterations),
            "duration_seconds": duration,
            "exit_code": trajectory.exit_code,
            "error": None,
        }

    except Exception as e:
        duration = time.time() - start_time
        logger.exception(
            f"Failed: model={model_id}, scenario={scenario}, "
            f"task_id={task_id}, seed={seed}"
        )
        return {
            "model_id": model_id,
            "scenario": scenario,
            "task_id": task_id,
            "seed": seed,
            "method": "no_memory",
            "team_success": False,
            "team_reward": 0.0,
            "score": 0.0,
            "n_iterations": 0,
            "duration_seconds": duration,
            "exit_code": -1,
            "error": str(e),
        }


def run_sweep() -> list[dict[str, Any]]:
    """Run full sweep across all models, scenarios, tasks, and seeds."""
    results = []

    for model_config in MODEL_TIERS:
        model_id = model_config["model_id"]
        tier = model_config["tier"]

        logger.info(f"\n{'='*60}")
        logger.info(f"Testing model tier: {tier} ({model_id})")
        logger.info(f"{'='*60}\n")

        # Set model environment variable
        os.environ["MARBLE_LLM_MODEL"] = model_id

        # Initialize collector
        collector = TrajectoryCollector(
            marble_root=Path("/home/ecs-user/MARBLE"),
            output_root=Path("results/marble/backbone_sweep") / tier,
        )

        # Load tasks
        loader = MarbleTaskLoader()

        for scenario in SWEEP_CONFIG["scenarios"]:
            logger.info(f"\nScenario: {scenario}")

            tasks = loader.load_scenario(
                scenario, limit=SWEEP_CONFIG["tasks_per_scenario"]
            )

            for task in tasks:
                for seed in SWEEP_CONFIG["seeds"]:
                    logger.info(
                        f"  Running: {scenario}/{task.task_id} seed={seed}"
                    )

                    result = run_single_episode(
                        model_id=model_id,
                        scenario=scenario,
                        task_id=task.task_id,
                        seed=seed,
                        collector=collector,
                    )

                    results.append(result)

                    logger.info(
                        f"    → success={result['team_success']}, "
                        f"score={result['score']:.2f}, "
                        f"duration={result['duration_seconds']:.1f}s"
                    )

    return results


def analyze_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze sweep results and recommend backbone."""
    analysis = {}

    for model_config in MODEL_TIERS:
        model_id = model_config["model_id"]
        tier = model_config["tier"]

        model_results = [r for r in results if r["model_id"] == model_id]

        if not model_results:
            continue

        # Compute metrics
        n_total = len(model_results)
        n_success = sum(1 for r in model_results if r["team_success"])
        success_rate = n_success / n_total if n_total > 0 else 0.0

        scores = [r["score"] for r in model_results]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        durations = [r["duration_seconds"] for r in model_results]
        avg_duration = sum(durations) / len(durations) if durations else 0.0

        n_errors = sum(1 for r in model_results if r["error"] is not None)
        error_rate = n_errors / n_total if n_total > 0 else 0.0

        analysis[tier] = {
            "model_id": model_id,
            "description": model_config["description"],
            "n_episodes": n_total,
            "n_success": n_success,
            "success_rate": success_rate,
            "avg_score": avg_score,
            "avg_duration": avg_duration,
            "n_errors": n_errors,
            "error_rate": error_rate,
            "is_saturated": success_rate > 0.8,
            "is_too_hard": success_rate < 0.3,
            "is_suitable": 0.3 <= success_rate <= 0.8,
        }

    # Recommend backbone
    suitable_tiers = [
        tier for tier, stats in analysis.items() if stats["is_suitable"]
    ]

    if suitable_tiers:
        # Prefer medium tier if suitable (balance between cost and capability)
        recommended = "medium" if "medium" in suitable_tiers else suitable_tiers[0]
    else:
        # No suitable tier found — recommend strongest non-saturated model
        non_saturated = [
            tier for tier, stats in analysis.items() if not stats["is_saturated"]
        ]
        recommended = non_saturated[0] if non_saturated else "strong"

    analysis["recommended_tier"] = recommended
    analysis["recommended_model_id"] = analysis[recommended]["model_id"]

    return analysis


def write_results(
    results: list[dict[str, Any]],
    analysis: dict[str, Any],
    output_dir: Path,
) -> None:
    """Write results to CSV and markdown reports."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write CSV
    csv_path = output_dir / "sweep_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "model_id",
            "scenario",
            "task_id",
            "seed",
            "method",
            "team_success",
            "team_reward",
            "score",
            "n_iterations",
            "duration_seconds",
            "exit_code",
            "error",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"Wrote results to {csv_path}")

    # Write model comparison markdown
    comparison_path = output_dir / "model_comparison.md"
    with comparison_path.open("w", encoding="utf-8") as f:
        f.write("# Backbone Difficulty Sweep: Model Comparison\n\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Summary\n\n")
        f.write("| Tier | Model | Success Rate | Avg Score | Avg Duration | Saturated? | Suitable? |\n")
        f.write("|------|-------|--------------|-----------|--------------|------------|----------|\n")

        for tier in ["strong", "medium", "small"]:
            if tier not in analysis:
                continue
            stats = analysis[tier]
            saturated = "✅ YES" if stats["is_saturated"] else "❌ NO"
            suitable = "✅ YES" if stats["is_suitable"] else "❌ NO"
            f.write(
                f"| {tier} | {stats['model_id']} | "
                f"{stats['success_rate']:.1%} | "
                f"{stats['avg_score']:.2f} | "
                f"{stats['avg_duration']:.1f}s | "
                f"{saturated} | {suitable} |\n"
            )

        f.write(f"\n## Recommended Backbone\n\n")
        f.write(f"**Tier**: {analysis['recommended_tier']}\n\n")
        f.write(f"**Model**: {analysis['recommended_model_id']}\n\n")

        recommended_stats = analysis[analysis["recommended_tier"]]
        f.write(f"**Rationale**:\n")
        f.write(f"- Success rate: {recommended_stats['success_rate']:.1%} (target: 30%-80%)\n")
        f.write(f"- Not saturated: {not recommended_stats['is_saturated']}\n")
        f.write(f"- Not too hard: {not recommended_stats['is_too_hard']}\n")

    logger.info(f"Wrote comparison to {comparison_path}")

    # Write recommendation
    recommendation_path = output_dir / "recommended_backbone.md"
    with recommendation_path.open("w", encoding="utf-8") as f:
        f.write("# Recommended Backbone for MultiAgentBench\n\n")
        f.write(f"**Tier**: {analysis['recommended_tier']}\n\n")
        f.write(f"**Model ID**: `{analysis['recommended_model_id']}`\n\n")
        f.write(f"**Description**: {analysis[analysis['recommended_tier']]['description']}\n\n")

        f.write("## Selection Criteria\n\n")
        f.write("- ✅ Baseline success rate between 30% and 80%\n")
        f.write("- ✅ Avoids ceiling effect (too easy)\n")
        f.write("- ✅ Avoids floor effect (too hard)\n")
        f.write("- ❌ NOT selected based on SMTR performance\n\n")

        f.write("## Usage\n\n")
        f.write("```bash\n")
        f.write(f"export MARBLE_LLM_MODEL=\"{analysis['recommended_model_id']}\"\n")
        f.write("```\n")

    logger.info(f"Wrote recommendation to {recommendation_path}")


def main() -> None:
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("Backbone Difficulty Sweep")
    logger.info("=" * 60)
    logger.info(f"Scenarios: {SWEEP_CONFIG['scenarios']}")
    logger.info(f"Tasks per scenario: {SWEEP_CONFIG['tasks_per_scenario']}")
    logger.info(f"Seeds: {SWEEP_CONFIG['seeds']}")
    logger.info(f"Total episodes: {len(SWEEP_CONFIG['scenarios']) * SWEEP_CONFIG['tasks_per_scenario'] * len(SWEEP_CONFIG['seeds']) * len(MODEL_TIERS)}")
    logger.info("")

    # Run sweep
    results = run_sweep()

    # Analyze
    analysis = analyze_results(results)

    # Write results
    output_dir = Path("results/marble/backbone_sweep")
    write_results(results, analysis, output_dir)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SWEEP COMPLETE")
    logger.info("=" * 60)
    logger.info(f"\nRecommended backbone: {analysis['recommended_tier']} ({analysis['recommended_model_id']})")
    logger.info(f"\nResults written to: {output_dir}")


if __name__ == "__main__":
    main()
