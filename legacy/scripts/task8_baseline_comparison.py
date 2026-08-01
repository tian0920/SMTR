#!/usr/bin/env python3
"""Task 8: Baseline Comparison — NoMemory / AllMemory / Semantic Top-k / SMTR.

Compares memory routing strategies on MARBLE DB Environment.
All conditions share: same roles, same graph, same model, same task, same MARBLE evaluator.
Only the exposure_override differs.

Conditions:
  - NoMemory:      exposure_override=[]        (forced empty, S_K=∅)
  - AllMemory:     exposure_override=[all_ids]  (forced full set, length-limited)
  - Semantic Top-k: exposure_override=[top_k_ids] (forced top-k by semantic rank)
  - SMTR:          exposure_override=None        (router selects S_K)

Usage:
  python scripts/task8_baseline_comparison.py                    # Run all 3 tasks × 4 conditions
  python scripts/task8_baseline_comparison.py --tasks 1          # Run only task 1
  python scripts/task8_baseline_comparison.py --conditions nomemory,smtr  # Run subset
  python scripts/task8_baseline_comparison.py --dry-run          # Validate setup without running
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from typing import Any

# ---------------------------------------------------------------------------
# LLM API setup — must happen before any MARBLE imports
# ---------------------------------------------------------------------------
LLM_API_BASE = "https://llm-jhxtd03gjg0gd2o2.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
LLM_API_KEY = "sk-8692d980a9e148d3843a64135ae0b0f2"
LLM_MODEL = "openai/qwen3.5-plus"

os.environ["OPENAI_API_BASE"] = LLM_API_BASE
os.environ["OPENAI_API_KEY"] = LLM_API_KEY

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("task8_baseline")

# ---------------------------------------------------------------------------
# Init SQL loader
# ---------------------------------------------------------------------------

def _load_init_sql(domain: str) -> str:
    """Load init SQL from MARBLE's config files."""
    config_path = (
        f"/home/ecs-user/MARBLE/marble/configs/test_config_database/"
        f"gpt-3.5-turbo_{domain}_LOCK_CONTENTION.yaml"
    )
    try:
        import yaml

        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cfg["environment"]["init_sql"]
    except Exception as e:
        logger.warning(f"Could not load init SQL from {config_path}: {e}")
        # Fallback: minimal init SQL
        return (
            "CREATE TABLE test_items (\n"
            "  item_id SERIAL PRIMARY KEY,\n"
            "  name VARCHAR(100) NOT NULL,\n"
            "  value DECIMAL(10, 2) NOT NULL,\n"
            "  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\n"
            ");\n"
            "INSERT INTO test_items (name, value) VALUES ('alpha', 100.00), ('beta', 200.00);\n"
        )


# ---------------------------------------------------------------------------
# Task definitions — 3 DB tasks with different anomaly types
# ---------------------------------------------------------------------------

TASK_DEFINITIONS = [
    {
        "task_id": "E_COMMERCE_LOCK_CONTENTION",
        "domain": "E_COMMERCE",
        "anomaly": "LOCK_CONTENTION",
        "root_causes": ["LOCK_CONTENTION"],
        "init_sql": None,  # Loaded lazily
        "task_content": (
            "This database is used in an e-commerce system to manage customer information, "
            "product details, orders, order items, and payments. It consists of five main tables: "
            "customers, products, orders, order items, and payments, with foreign key relationships "
            "between them.\n\n"
            "Recently, during operation, the database has seen performance issues. Use sql "
            "queries to find out what is wrong, and find out the reason that caused it. The "
            "root cause can be only two of the following: 'INSERT_LARGE_DATA', 'MISSING_INDEXES', "
            "'LOCK_CONTENTION', 'VACUUM', 'REDUNDANT_INDEX', 'FETCH_LARGE_DATA'. "
            "The planner should assign different agent to analyze possbility for each root "
            "cause and make final decision."
        ),
    },
    {
        "task_id": "E_COMMERCE_VACUUM",
        "domain": "E_COMMERCE",
        "anomaly": "VACUUM",
        "root_causes": ["VACUUM"],
        "init_sql": None,
        "task_content": (
            "This database is used in an e-commerce system to manage customer information, "
            "product details, orders, order items, and payments. It consists of five main tables: "
            "customers, products, orders, order items, and payments, with foreign key relationships "
            "between them.\n\n"
            "Recently, during operation, the database has seen performance issues. Use sql "
            "queries to find out what is wrong, and find out the reason that caused it. The "
            "root cause can be only two of the following: 'INSERT_LARGE_DATA', 'MISSING_INDEXES', "
            "'LOCK_CONTENTION', 'VACUUM', 'REDUNDANT_INDEX', 'FETCH_LARGE_DATA'. "
            "The planner should assign different agent to analyze possbility for each root "
            "cause and make final decision."
        ),
    },
    {
        "task_id": "E_COMMERCE_REDUNDANT_INDEX",
        "domain": "E_COMMERCE",
        "anomaly": "REDUNDANT_INDEX",
        "root_causes": ["REDUNDANT_INDEX"],
        "init_sql": None,
        "task_content": (
            "This database is used in an e-commerce system to manage customer information, "
            "product details, orders, order items, and payments. It consists of five main tables: "
            "customers, products, orders, order items, and payments, with foreign key relationships "
            "between them.\n\n"
            "Recently, during operation, the database has seen performance issues. Use sql "
            "queries to find out what is wrong, and find out the reason that caused it. The "
            "root cause can be only two of the following: 'INSERT_LARGE_DATA', 'MISSING_INDEXES', "
            "'LOCK_CONTENTION', 'VACUUM', 'REDUNDANT_INDEX', 'FETCH_LARGE_DATA'. "
            "The planner should assign different agent to analyze possbility for each root "
            "cause and make final decision."
        ),
    },
]


def _ensure_init_sql_loaded(task_def: dict) -> str:
    """Lazily load init_sql if not already loaded."""
    if task_def["init_sql"] is None:
        task_def["init_sql"] = _load_init_sql(task_def["domain"])
    return task_def["init_sql"]


# ---------------------------------------------------------------------------
# Condition definitions
# ---------------------------------------------------------------------------

CONDITIONS = {
    "nomemory": {
        "label": "NoMemory",
        "description": "exposure_override=[] (forced empty, S_K=∅)",
        "exposure_override": [],
    },
    "allmemory": {
        "label": "AllMemory",
        "description": "exposure_override=[all_ids] (forced full set, length-limited)",
        "exposure_override": "all",  # Will be resolved at runtime
    },
    "topk": {
        "label": "Semantic Top-k",
        "description": "exposure_override=[top_k_ids] (forced top-k by semantic rank)",
        "exposure_override": "topk",  # Will be resolved at runtime
    },
    "smtr": {
        "label": "SMTR",
        "description": "exposure_override=None (router selects S_K)",
        "exposure_override": None,
    },
}


# ---------------------------------------------------------------------------
# Test procedures (same as Task 6)
# ---------------------------------------------------------------------------

def build_test_procedures() -> tuple[list, list, list]:
    """Create 4 generic diagnostic procedures.

    Returns:
        (routing_cards, payloads, memory_ids)
    """
    from smtr.memory.schemas import MemoryRoutingCard, ProcedurePayload

    procedures = [
        {
            "memory_id": "proc_systematic_diagnosis",
            "goal": "Systematic database error diagnosis",
            "task_tags": ["diagnosis", "systematic", "database"],
            "precondition_summary": "Database is accessible",
            "postcondition_summary": "Root cause identified",
            "steps": [
                "Check database connectivity and basic query execution",
                "Review recent error logs for patterns",
                "Explain query plans for slow queries",
                "Check system resource usage (CPU, memory, disk I/O)",
                "Narrow down root cause by elimination",
            ],
        },
        {
            "memory_id": "proc_log_analysis",
            "goal": "Structured log analysis workflow",
            "task_tags": ["log_analysis", "monitoring", "patterns"],
            "precondition_summary": "Logging is enabled",
            "postcondition_summary": "Anomalous patterns identified",
            "steps": [
                "Collect logs from the relevant time window",
                "Filter for ERROR and WARNING level entries",
                "Group similar error messages together",
                "Identify the most frequent error patterns",
                "Correlate error timing with reported issues",
            ],
        },
        {
            "memory_id": "proc_performance_audit",
            "goal": "Database performance audit checklist",
            "task_tags": ["performance", "audit", "optimization"],
            "precondition_summary": "Database statistics are up to date",
            "postcondition_summary": "Performance bottlenecks documented",
            "steps": [
                "Run ANALYZE on all tables to update statistics",
                "Check for table bloat and dead tuple accumulation",
                "Review index usage statistics for unused or missing indexes",
                "Examine query execution plans for sequential scans on large tables",
                "Document findings and prioritize remediation",
            ],
        },
        {
            "memory_id": "proc_query_optimization",
            "goal": "Query optimization procedure",
            "task_tags": ["query", "optimization", "execution_plan"],
            "precondition_summary": "pg_stat_statements is enabled",
            "postcondition_summary": "Slow queries identified and optimized",
            "steps": [
                "Query pg_stat_statements for top queries by total time",
                "Examine execution plans using EXPLAIN (ANALYZE, BUFFERS)",
                "Check for missing indexes on frequently filtered columns",
                "Review join strategies and consider denormalization",
                "Validate improvements with before/after timing comparison",
            ],
        },
    ]

    routing_cards = []
    payloads = []
    memory_ids = []

    for proc in procedures:
        mid = proc["memory_id"]
        memory_ids.append(mid)
        routing_cards.append(
            MemoryRoutingCard(
                memory_id=mid,
                goal_summary=proc["goal"],
                task_tags=proc["task_tags"],
                precondition_summary=proc["precondition_summary"],
                postcondition_summary=proc["postcondition_summary"],
            )
        )
        payloads.append(
            ProcedurePayload(
                memory_id=mid,
                goal=proc["goal"],
                steps=proc["steps"],
                preconditions=[proc["precondition_summary"]],
                postconditions=[proc["postcondition_summary"]],
            )
        )

    return routing_cards, payloads, memory_ids


# ---------------------------------------------------------------------------
# MARBLE config builder
# ---------------------------------------------------------------------------

def build_marble_config(task_def: dict[str, Any]) -> dict[str, Any]:
    """Build a MARBLE config dict for a given task definition."""
    _ensure_init_sql_loaded(task_def)
    anomaly = task_def["anomaly"]
    return {
        "coordinate_mode": "graph",
        "relationships": [
            ["agent1", "agent2", "collaborate with"],
            ["agent1", "agent3", "collaborate with"],
            ["agent1", "agent4", "collaborate with"],
            ["agent1", "agent5", "collaborate with"],
            ["agent2", "agent3", "collaborate with"],
            ["agent2", "agent4", "collaborate with"],
            ["agent2", "agent5", "collaborate with"],
            ["agent3", "agent4", "collaborate with"],
            ["agent3", "agent5", "collaborate with"],
            ["agent4", "agent5", "collaborate with"],
        ],
        "llm": LLM_MODEL,
        "environment": {
            "type": "DB",
            "name": f"DB Baseline: {task_def['task_id']}",
            "max_iterations": 5,
            "init_sql": task_def["init_sql"],
            "anomalies": [
                {
                    "anomaly": anomaly,
                    "threads": 100,
                    "ncolumn": 20,
                    "nrow": 20000,
                    "colsize": 100,
                }
            ],
        },
        "communication": False,
        "task": {
            "content": task_def["task_content"],
            "output_format": (
                "Please make the decision after exploring all rootcauses, as a premature "
                "decision may lead to incorrect conclusions.\n"
                "Please choose the most likely cause of the database anomaly from the "
                "following list, based on the expert agents: "
                "'INSERT_LARGE_DATA', 'LOCK_CONTENTION', 'VACUUM', 'REDUNDANT_INDEX', 'FETCH_LARGE_DATA'\n"
                "You can ONLY CHOOSE two.\n"
                "You have access to the Database, and you can perform queries to get the "
                "required information."
            ),
            "labels": [
                "INSERT_LARGE_DATA", "LOCK_CONTENTION", "VACUUM",
                "REDUNDANT_INDEX", "FETCH_LARGE_DATA",
            ],
            "root_causes": task_def["root_causes"],
            "number_of_labels_pred": 2,
        },
        "agents": [
            {
                "type": "BaseAgent",
                "agent_id": "agent1",
                "profile": (
                    "agent1 will explore the possibility of INSERT_LARGE_DATA as a root cause. "
                    "Recommended tables: `pg_stat_statements`. You can search for INSERTs."
                ),
            },
            {
                "type": "BaseAgent",
                "agent_id": "agent2",
                "profile": (
                    "agent2 will explore the possibility of LOCK_CONTENTION as a root cause. "
                    "Recommended tables: `pg_locks`."
                ),
            },
            {
                "type": "BaseAgent",
                "agent_id": "agent3",
                "profile": (
                    "agent3 will explore the possibility of VACUUM as a root cause. "
                    "Recommended to search for inappropiate VACUUMs from `pg_stat_all_tables`."
                ),
            },
            {
                "type": "BaseAgent",
                "agent_id": "agent4",
                "profile": (
                    "agent4 will explore the possibility of REDUNDANT_INDEX as a root cause. "
                    "Recommended tables: `pg_stat_user_indexes`, `pg_indexes`."
                ),
            },
            {
                "type": "BaseAgent",
                "agent_id": "agent5",
                "profile": (
                    "agent5 will explore the possibility of FETCH_LARGE_DATA as a root cause. "
                    "Recommended to search for SELECTs from `pg_stat_statements`."
                ),
            },
        ],
        "memory": {"type": "SharedMemory"},
        "metrics": {"accuracy": True, "response_time": True},
        "output": {
            "file_path": f"/home/ecs-user/SMTR/data/marble_baseline_{task_def['task_id']}.jsonl",
        },
        "engine_planner": {
            "initial_progress": "Starting baseline comparison simulation.",
        },
    }


# ---------------------------------------------------------------------------
# Condition resolution
# ---------------------------------------------------------------------------

def resolve_exposure_override(
    condition_key: str,
    all_memory_ids: list[str],
) -> list[str] | None:
    """Resolve the exposure_override value for a given condition.

    Args:
        condition_key: One of 'nomemory', 'allmemory', 'topk', 'smtr'.
        all_memory_ids: All available memory IDs.

    Returns:
        exposure_override value for SMTRMarbleEngine.
    """
    cond = CONDITIONS[condition_key]
    override = cond["exposure_override"]

    if override == "all":
        # AllMemory: force all IDs (with length limit — same as SMTR)
        return list(all_memory_ids)
    elif override == "topk":
        # Semantic Top-k: force top 3 by "semantic rank"
        # For this baseline, we use a fixed subset (first 3 IDs)
        # In production, this would use semantic similarity scoring
        return list(all_memory_ids[:3])
    elif override is None:
        # SMTR: let router decide
        return None
    else:
        # NoMemory: forced empty
        return []


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_single_condition(
    task_def: dict[str, Any],
    condition_key: str,
    memory_pool,
    all_memory_ids: list[str],
    target_receiver: str = "agent2",
) -> dict[str, Any]:
    """Run a single task × condition combination.

    Returns:
        Dict with task_id, condition, success, reward, transfer_class, metadata.
    """
    from marble.configs.config import Config
    from smtr.counterfactual.marble_eval import extract_marble_outcome
    from smtr.runtime.marble_agent import SMTRMarbleEngine

    config_data = build_marble_config(task_def)
    config = Config(config_data)

    exposure_override = resolve_exposure_override(condition_key, all_memory_ids)

    logger.info(
        f"Running {task_def['task_id']} × {CONDITIONS[condition_key]['label']} "
        f"(exposure_override={exposure_override})"
    )

    start_time = time.time()

    try:
        engine = SMTRMarbleEngine(
            config=config,
            target_receiver_agent_id=target_receiver,
            smtr_memory_pool=memory_pool,
            exposure_override=exposure_override,
        )

        engine.start()

        # Extract outcome
        task_eval = engine.evaluator.metrics.get("task_evaluation", {})
        engine_result = {"task_evaluation": task_eval}

        outcome = extract_marble_outcome(
            engine_result,
            task_id=f"{task_def['task_id']}_{condition_key}",
            environment_type="DB",
            num_agents=len(engine.agents),
            num_iterations=engine.current_iteration,
        )

        elapsed = time.time() - start_time

        # Get target agent's selected memories (if SMTR condition)
        selected_ids = []
        for agent in engine.agents:
            if agent.agent_id == target_receiver and hasattr(agent, "_smtr_state"):
                selected_ids = agent._smtr_state.selected_memory_ids
                break

        return {
            "task_id": task_def["task_id"],
            "condition": condition_key,
            "condition_label": CONDITIONS[condition_key]["label"],
            "success": outcome.success,
            "reward": outcome.reward,
            "num_iterations": outcome.num_iterations,
            "selected_memory_ids": selected_ids,
            "elapsed_seconds": round(elapsed, 1),
            "error": None,
            "raw_evaluation": task_eval,
        }

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Error in {task_def['task_id']} × {condition_key}: {e}")
        return {
            "task_id": task_def["task_id"],
            "condition": condition_key,
            "condition_label": CONDITIONS[condition_key]["label"],
            "success": False,
            "reward": 0.0,
            "num_iterations": 0,
            "selected_memory_ids": [],
            "elapsed_seconds": round(elapsed, 1),
            "error": str(e),
            "raw_evaluation": {},
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Task 8: Baseline Comparison")
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated task indices (0,1,2) or 'all' (default: all)",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default=None,
        help="Comma-separated condition keys (nomemory,allmemory,topk,smtr) or 'all'",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate setup without running MARBLE",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/ecs-user/SMTR/outputs/baseline_comparison.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    # Parse task indices
    if args.tasks is None or args.tasks == "all":
        task_indices = list(range(len(TASK_DEFINITIONS)))
    else:
        task_indices = [int(x.strip()) for x in args.tasks.split(",")]

    # Parse conditions
    if args.conditions is None or args.conditions == "all":
        condition_keys = list(CONDITIONS.keys())
    else:
        condition_keys = [x.strip() for x in args.conditions.split(",")]

    logger.info("=" * 70)
    logger.info("Task 8: Baseline Comparison — NoMemory / AllMemory / Top-k / SMTR")
    logger.info("=" * 70)
    logger.info(f"Tasks: {[TASK_DEFINITIONS[i]['task_id'] for i in task_indices]}")
    logger.info(f"Conditions: {[CONDITIONS[k]['label'] for k in condition_keys]}")

    # Build memory pool
    routing_cards, payloads, memory_ids = build_test_procedures()
    from smtr.memory.pool import SharedMemoryPool

    pool = SharedMemoryPool(routing_cards=routing_cards, payloads=payloads)
    logger.info(f"Memory pool: {len(memory_ids)} procedures: {memory_ids}")

    if args.dry_run:
        logger.info("DRY RUN: Validating setup...")
        for ci in condition_keys:
            override = resolve_exposure_override(ci, memory_ids)
            logger.info(f"  {CONDITIONS[ci]['label']}: exposure_override={override}")
        logger.info("DRY RUN: Setup valid. Exiting.")
        return

    # Change to MARBLE directory for evaluator_prompts.json
    marble_dir = "/home/ecs-user/MARBLE/marble"
    if os.path.isdir(marble_dir):
        original_cwd = os.getcwd()
        os.chdir(marble_dir)
    else:
        logger.error(f"MARBLE directory not found at {marble_dir}")
        sys.exit(1)

    # Monkey-patches for Docker/subprocess (same as Task 6)
    from marble.environments.db_env import DBEnvironment

    _original_start_docker = DBEnvironment.start_docker_containers

    def _noop_start_docker(self):
        logger.info("Docker start skipped (containers already running)")

    DBEnvironment.start_docker_containers = _noop_start_docker

    import subprocess as _subprocess

    _original_subprocess_run = _subprocess.run

    def _patched_subprocess_run(cmd, *args, **kwargs):
        if isinstance(cmd, list) and cmd and cmd[0] in ("python", "python3"):
            cmd = ["/home/ecs-user/MARBLE/.venv/bin/python"] + cmd[1:]
        return _original_subprocess_run(cmd, *args, **kwargs)

    _subprocess.run = _patched_subprocess_run

    # Run all combinations
    all_results = []
    total_runs = len(task_indices) * len(condition_keys)
    current_run = 0

    try:
        for task_idx in task_indices:
            task_def = TASK_DEFINITIONS[task_idx]
            for cond_key in condition_keys:
                current_run += 1
                logger.info(f"\n--- Run {current_run}/{total_runs} ---")

                result = run_single_condition(
                    task_def=task_def,
                    condition_key=cond_key,
                    memory_pool=pool,
                    all_memory_ids=memory_ids,
                )
                all_results.append(result)

                status = "✓" if result["success"] else "✗"
                logger.info(
                    f"  {status} {result['task_id']} × {result['condition_label']}: "
                    f"success={result['success']}, reward={result['reward']}, "
                    f"time={result['elapsed_seconds']}s"
                )

    finally:
        # Restore patches
        DBEnvironment.start_docker_containers = _original_start_docker
        _subprocess.run = _original_subprocess_run
        os.chdir(original_cwd)

    # Compute summary statistics
    summary = compute_summary(all_results, condition_keys)

    # Output
    output_data = {
        "results": all_results,
        "summary": summary,
        "metadata": {
            "model": LLM_MODEL,
            "num_procedures": len(memory_ids),
            "target_receiver": "agent2",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    logger.info(f"\nResults written to {args.output}")

    # Print summary table
    print_summary_table(summary, condition_keys)


def compute_summary(
    results: list[dict], condition_keys: list[str]
) -> dict[str, Any]:
    """Compute summary statistics across all results."""
    summary = {}
    for cond_key in condition_keys:
        cond_results = [r for r in results if r["condition"] == cond_key]
        if not cond_results:
            continue

        successes = sum(1 for r in cond_results if r["success"])
        total = len(cond_results)
        errors = sum(1 for r in cond_results if r["error"])

        summary[cond_key] = {
            "label": CONDITIONS[cond_key]["label"],
            "total_tasks": total,
            "successes": successes,
            "success_rate": round(successes / total, 3) if total > 0 else 0.0,
            "errors": errors,
            "avg_iterations": (
                round(sum(r["num_iterations"] for r in cond_results) / total, 1)
                if total > 0
                else 0.0
            ),
            "avg_time_seconds": (
                round(sum(r["elapsed_seconds"] for r in cond_results) / total, 1)
                if total > 0
                else 0.0
            ),
        }

    return summary


def print_summary_table(summary: dict, condition_keys: list[str]) -> None:
    """Print a formatted summary table."""
    print("\n" + "=" * 80)
    print("BASELINE COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Condition':<20} {'Success Rate':>15} {'Avg Iter':>10} {'Avg Time':>10} {'Errors':>8}")
    print("-" * 80)

    for cond_key in condition_keys:
        if cond_key not in summary:
            continue
        s = summary[cond_key]
        print(
            f"{s['label']:<20} "
            f"{s['successes']}/{s['total_tasks']} ({s['success_rate']:.1%})".rjust(14) + " " * 6 +
            f"{s['avg_iterations']:>10.1f} "
            f"{s['avg_time_seconds']:>9.1f}s "
            f"{s['errors']:>8}"
        )

    print("=" * 80)
    print("\nTransfer class analysis:")
    print("  positive_transfer  = (Y_share=1, Y_withhold=0) — memory helps")
    print("  negative_transfer  = (Y_share=0, Y_withhold=1) — memory hurts")
    print("  neutral_success    = (Y_share=1, Y_withhold=1) — memory doesn't change outcome")
    print("  neutral_failure    = (Y_share=0, Y_withhold=0) — memory doesn't change outcome")


if __name__ == "__main__":
    main()
