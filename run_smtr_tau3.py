#!/usr/bin/env python3
"""Direct τ³-bench ↔ SMTR integration runner.

Runs SMTRTauAgent on τ³ retail domain tasks using a direct Python script,
without requiring τ³ registry integration. This is the first integration
step — confirm the agent executes, router injects payloads, and official
evaluator scores results.

Usage:
    # Baseline (no SMTR memory) with API key:
    python3 run_smtr_tau3.py --domain retail --num-tasks 3 \\
        --agent-llm openai/qwen3.5-plus \\
        --llm-args '{"api_base": "https://..."}'

    # With SMTR memory pool:
    python3 run_smtr_tau3.py --domain retail --num-tasks 3 \\
        --agent-llm openai/qwen3.5-plus \\
        --memory-pool path/to/pool.json

    # Paired rollout (share vs withhold):
    python3 run_smtr_tau3.py --domain retail --num-tasks 3 \\
        --agent-llm openai/qwen3.5-plus \\
        --paired --memory-pool path/to/pool.json

    # Dry run (no τ³ required — tests SMTR imports):
    python3 run_smtr_tau3.py --dry-run

Environment:
    OPENAI_API_KEY must be set for LLM API access.
    Or pass --api-key explicitly.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# τ³-bench paths
# ---------------------------------------------------------------------------
TAU2_DATA_DIR = Path("/home/ecs-user/tau2-bench/data/tau2")


def check_tau3_available() -> bool:
    """Check if τ³-bench is importable."""
    try:
        import tau2  # noqa: F401
        return True
    except ImportError:
        return False


def load_retail_tasks(split: str | None = None) -> list[dict[str, Any]]:
    """Load retail domain tasks from τ³-bench data directory.

    Args:
        split: Optional split name ("train", "test", "base").
            If None, loads all tasks.

    Returns:
        List of task dicts.
    """
    tasks_file = TAU2_DATA_DIR / "domains" / "retail" / "tasks.json"
    with open(tasks_file) as f:
        all_tasks = json.load(f)

    if split is None:
        return all_tasks

    split_file = TAU2_DATA_DIR / "domains" / "retail" / "split_tasks.json"
    with open(split_file) as f:
        splits = json.load(f)

    task_ids_in_split = set(splits.get(split, []))
    return [t for t in all_tasks if t["id"] in task_ids_in_split]


def load_memory_pool(pool_path: str) -> Any:
    """Load a SharedMemoryPool from a JSON file."""
    from smtr.memory.pool import SharedMemoryPool
    from smtr.memory.schemas import MemoryRoutingCard, ProcedurePayload

    data = json.loads(Path(pool_path).read_text())
    cards = [MemoryRoutingCard.model_validate(c) for c in data.get("routing_cards", [])]
    payloads = [ProcedurePayload.model_validate(p) for p in data.get("payloads", [])]
    return SharedMemoryPool(routing_cards=cards, payloads=payloads)


def create_smtr_agent(
    tools: list[Any],
    domain_policy: str,
    *,
    llm: str,
    llm_args: dict | None = None,
    memory_pool: Any | None = None,
    critic_path: str | None = None,
) -> Any:
    """Create an SMTRTauAgent with optional memory pool and critic."""
    from smtr.runtime.tau3_agent import SMTRTauAgent

    return SMTRTauAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=llm,
        llm_args=llm_args,
        memory_pool=memory_pool,
        critic_path=critic_path,
    )


# ---------------------------------------------------------------------------
# Baseline evaluation (no SMTR memory)
# ---------------------------------------------------------------------------


def run_baseline_eval(
    *,
    domain: str = "retail",
    agent_llm: str = "openai/qwen3.5-plus",
    user_llm: str = "openai/qwen3.5-plus",
    llm_args: dict | None = None,
    num_tasks: int = 3,
    split: str | None = None,
    output_dir: str = "outputs/tau3_baseline",
    max_steps: int = 100,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Run baseline τ³ evaluation with a plain LLM agent (no SMTR memory)."""
    from tau2.data_model.tasks import Task
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.runner.build import build_agent, build_environment, build_user
    from tau2.runner.simulation import run_simulation

    from smtr.counterfactual.tau_eval import extract_outcome, summarize_outcomes

    env = build_environment(domain)

    tasks_raw = load_retail_tasks(split=split)
    tasks_raw = tasks_raw[:num_tasks]
    tasks = [Task.model_validate(t) for t in tasks_raw]

    logger.info(f"Running baseline eval on {len(tasks)} {domain} tasks")

    results = []
    for task in tasks:
        logger.info(f"Task {task.id}: {str(task.user_scenario)[:100]}...")

        agent = build_agent(
            "llm_agent", env,
            llm=agent_llm,
            llm_args=llm_args,
            task=task,
        )
        user = build_user(
            "user_simulator", env, task,
            llm=user_llm,
            llm_args=llm_args,
        )

        orchestrator = Orchestrator(
            domain=domain,
            agent=agent,
            user=user,
            environment=env,
            task=task,
            max_steps=max_steps,
            seed=seed,
        )

        simulation_run = run_simulation(orchestrator)
        outcome = extract_outcome(simulation_run, domain=domain)
        result = outcome.model_dump()
        result["num_messages"] = len(simulation_run.messages) if simulation_run.messages else 0
        result["termination_reason"] = simulation_run.termination_reason
        results.append(result)
        logger.info(f"  Reward: {outcome.reward}, Success: {outcome.success}")

    # Summary
    from smtr.counterfactual.tau_eval import TauOutcome

    outcomes = [TauOutcome.model_validate(r) for r in results]
    summary = summarize_outcomes(outcomes)
    logger.info(f"Summary: {json.dumps(summary, indent=2)}")

    # Save results
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    results_path = out_path / "tau3_baseline_results.json"
    results_path.write_text(json.dumps(results, indent=2, default=str))
    summary_path = out_path / "tau3_baseline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info(f"Results saved to {results_path}")

    return results


# ---------------------------------------------------------------------------
# SMTR memory-augmented evaluation
# ---------------------------------------------------------------------------


def run_smtr_eval(
    *,
    domain: str = "retail",
    agent_llm: str = "openai/qwen3.5-plus",
    user_llm: str = "openai/qwen3.5-plus",
    llm_args: dict | None = None,
    num_tasks: int = 3,
    split: str | None = None,
    memory_pool_path: str | None = None,
    critic_path: str | None = None,
    output_dir: str = "outputs/tau3_smtr",
    max_steps: int = 100,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Run SMTR-augmented τ³ evaluation with memory injection."""
    from tau2.data_model.tasks import Task
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.runner.build import build_environment, build_user
    from tau2.runner.simulation import run_simulation

    from smtr.counterfactual.tau_eval import extract_outcome, summarize_outcomes

    env = build_environment(domain)

    tasks_raw = load_retail_tasks(split=split)
    tasks_raw = tasks_raw[:num_tasks]
    tasks = [Task.model_validate(t) for t in tasks_raw]

    # Load memory pool
    memory_pool = None
    if memory_pool_path:
        memory_pool = load_memory_pool(memory_pool_path)
        logger.info(f"Loaded memory pool from {memory_pool_path}")

    logger.info(f"Running SMTR eval on {len(tasks)} {domain} tasks")

    results = []
    for task in tasks:
        logger.info(f"Task {task.id}: {str(task.user_scenario)[:100]}...")

        agent = create_smtr_agent(
            tools=env.get_tools(),
            domain_policy=env.get_policy(),
            llm=agent_llm,
            llm_args=llm_args,
            memory_pool=memory_pool,
            critic_path=critic_path,
        )
        user = build_user(
            "user_simulator", env, task,
            llm=user_llm,
            llm_args=llm_args,
        )

        orchestrator = Orchestrator(
            domain=domain,
            agent=agent,
            user=user,
            environment=env,
            task=task,
            max_steps=max_steps,
            seed=seed,
        )

        simulation_run = run_simulation(orchestrator)
        outcome = extract_outcome(simulation_run, domain=domain)
        result = outcome.model_dump()
        result["num_messages"] = len(simulation_run.messages) if simulation_run.messages else 0
        result["termination_reason"] = simulation_run.termination_reason
        result["selected_memory_ids"] = (
            simulation_run.info.get("smtr_selected_ids", [])
            if simulation_run.info else []
        )
        results.append(result)
        logger.info(f"  Reward: {outcome.reward}, Success: {outcome.success}")

    # Summary
    from smtr.counterfactual.tau_eval import TauOutcome

    outcomes = [TauOutcome.model_validate(r) for r in results]
    summary = summarize_outcomes(outcomes)
    logger.info(f"Summary: {json.dumps(summary, indent=2)}")

    # Save results
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    results_path = out_path / "tau3_smtr_results.json"
    results_path.write_text(json.dumps(results, indent=2, default=str))
    summary_path = out_path / "tau3_smtr_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info(f"Results saved to {results_path}")

    return results


# ---------------------------------------------------------------------------
# Paired rollout (share vs withhold)
# ---------------------------------------------------------------------------


def run_paired_eval(
    *,
    domain: str = "retail",
    agent_llm: str = "openai/qwen3.5-plus",
    user_llm: str = "openai/qwen3.5-plus",
    llm_args: dict | None = None,
    num_tasks: int = 3,
    split: str | None = None,
    memory_pool_path: str | None = None,
    critic_path: str | None = None,
    output_dir: str = "outputs/tau3_paired",
    max_steps: int = 100,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Run paired rollout: share (with memory) vs withhold (without memory)."""
    from tau2.data_model.tasks import Task
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.runner.build import build_environment, build_user
    from tau2.runner.simulation import run_simulation

    from smtr.counterfactual.tau3_paired_rollout import (
        Tau3BranchResult,
        Tau3PairedOutcome,
        summarize_paired_outcomes,
    )
    from smtr.counterfactual.tau_eval import extract_outcome

    env = build_environment(domain)

    tasks_raw = load_retail_tasks(split=split)
    tasks_raw = tasks_raw[:num_tasks]
    tasks = [Task.model_validate(t) for t in tasks_raw]

    # Load memory pool
    memory_pool = None
    if memory_pool_path:
        memory_pool = load_memory_pool(memory_pool_path)
        logger.info(f"Loaded memory pool from {memory_pool_path}")

    logger.info(f"Running paired eval on {len(tasks)} {domain} tasks")

    paired_outcomes = []
    for task in tasks:
        task_id = task.id
        logger.info(f"Task {task_id}: {str(task.user_scenario)[:100]}...")

        # Branch A: share (with memory)
        share_agent = create_smtr_agent(
            tools=env.get_tools(),
            domain_policy=env.get_policy(),
            llm=agent_llm,
            llm_args=llm_args,
            memory_pool=memory_pool,
            critic_path=critic_path,
        )
        share_user = build_user("user_simulator", env, task, llm=user_llm, llm_args=llm_args)
        share_orch = Orchestrator(
            domain=domain, agent=share_agent, user=share_user,
            environment=env, task=task, max_steps=max_steps, seed=seed,
        )
        share_run = run_simulation(share_orch)
        share_outcome = extract_outcome(share_run, domain=domain)
        share_result = Tau3BranchResult(
            outcome=share_outcome,
            termination_reason=share_run.termination_reason or "unknown",
            num_messages=len(share_run.messages) if share_run.messages else 0,
        )

        # Branch B: withhold (no memory)
        withhold_agent = create_smtr_agent(
            tools=env.get_tools(),
            domain_policy=env.get_policy(),
            llm=agent_llm,
            llm_args=llm_args,
            memory_pool=None,  # No memory
        )
        withhold_user = build_user("user_simulator", env, task, llm=user_llm, llm_args=llm_args)
        withhold_orch = Orchestrator(
            domain=domain, agent=withhold_agent, user=withhold_user,
            environment=env, task=task, max_steps=max_steps, seed=seed,
        )
        withhold_run = run_simulation(withhold_orch)
        withhold_outcome = extract_outcome(withhold_run, domain=domain)
        withhold_result = Tau3BranchResult(
            outcome=withhold_outcome,
            termination_reason=withhold_run.termination_reason or "unknown",
            num_messages=len(withhold_run.messages) if withhold_run.messages else 0,
        )

        paired = Tau3PairedOutcome.from_branch_results(
            task_id=task_id,
            domain=domain,
            seed=seed,
            share_result=share_result,
            withhold_result=withhold_result,
        )
        paired_outcomes.append(paired)
        logger.info(
            f"  Share: reward={share_outcome.reward}, "
            f"Withhold: reward={withhold_outcome.reward}, "
            f"Transfer: {paired.transfer_class}"
        )

    # Summary
    summary = summarize_paired_outcomes(paired_outcomes)
    logger.info(f"Paired summary: {json.dumps(summary, indent=2)}")

    # Save results
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    results_path = out_path / "tau3_paired_results.json"
    results_path.write_text(
        json.dumps([o.model_dump() for o in paired_outcomes], indent=2, default=str)
    )
    summary_path = out_path / "tau3_paired_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info(f"Results saved to {results_path}")

    return [o.model_dump() for o in paired_outcomes]


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def dry_run() -> None:
    """Test SMTR imports without requiring τ³-bench."""
    print("=== SMTR τ³ Integration Dry Run ===")
    print()

    # Test data models
    from smtr.runtime.tau3_agent import AgentVisibleTauContext, SMTRTauAgentState

    state = SMTRTauAgentState()
    print(f"SMTRTauAgentState: routing_done={state.routing_done}, turn_count={state.turn_count}")

    ctx = AgentVisibleTauContext(
        user_message="I need to return my order",
        domain_policy="Retail return policy...",
        tools=[{"name": "get_order", "description": "Get order details"}],
    )
    print(f"AgentVisibleTauContext: user_message='{ctx.user_message}'")
    print("  Excluded fields: evaluation_criteria, gold_db_state, reward_basis")
    print()

    # Test evaluation wrapper
    from smtr.counterfactual.tau_eval import extract_outcome, summarize_outcomes

    mock_sim_run = {
        "task_id": "retail_001",
        "reward_info": {"reward": 1.0, "db_check": {"passed": True}},
    }
    outcome = extract_outcome(mock_sim_run, domain="retail")
    print(f"TauOutcome: success={outcome.success}, reward={outcome.reward}")

    summary = summarize_outcomes([outcome])
    print(f"Summary: success_rate={summary['success_rate']}, mean_reward={summary['mean_reward']}")
    print()

    # Test data_source field
    from smtr.counterfactual.schemas import EvaluationGroupMetadata

    meta = EvaluationGroupMetadata()
    print(f"EvaluationGroupMetadata.data_source: '{meta.data_source}' (default)")

    meta_tau = EvaluationGroupMetadata(data_source="tau_bench")
    print(f"EvaluationGroupMetadata.data_source: '{meta_tau.data_source}' (set)")
    print()

    # Test SMTRTauAgent
    try:
        from smtr.runtime.tau3_agent import _TAU3_AVAILABLE, SMTRTauAgent

        if _TAU3_AVAILABLE:
            print("τ³-bench is available — SMTRTauAgent can be instantiated")
            agent = SMTRTauAgent(tools=[], domain_policy="test", llm="test/model")
            print(f"  Agent created: system_prompt length={len(agent.system_prompt)}")
        else:
            print("τ³-bench NOT available — SMTRTauAgent will raise ImportError on init")
    except Exception as e:
        print(f"Unexpected error: {e}")

    # Test task loading
    print()
    try:
        tasks = load_retail_tasks(split="test")
        print(f"Loaded {len(tasks)} retail test tasks")
        if tasks:
            desc = str(tasks[0].get("description", ""))[:80]
            print(f"  First task: id={tasks[0]['id']}, desc={desc}")
    except Exception as e:
        print(f"Task loading error: {e}")

    print()
    print("=== Dry run complete ===")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="SMTR τ³-bench integration runner")
    parser.add_argument("--domain", default="retail", help="τ³ domain (default: retail)")
    parser.add_argument("--agent-llm", default="openai/qwen3.5-plus", help="Agent LLM model")
    parser.add_argument(
        "--user-llm", default="openai/qwen3.5-plus",
        help="User simulator LLM model",
    )
    parser.add_argument(
        "--llm-args", type=str, default=None,
        help='JSON dict of LLM args (e.g. \'{"api_base": "...", "max_tokens": 1024}\')',
    )
    parser.add_argument("--api-key", type=str, default=None, help="API key for LLM provider")
    parser.add_argument("--num-tasks", type=int, default=3, help="Number of tasks to run")
    parser.add_argument(
        "--split", type=str, default=None, choices=["train", "test", "base"],
        help="Task split to sample from (default: all tasks)",
    )
    parser.add_argument("--memory-pool", type=str, default=None, help="Path to memory pool JSON")
    parser.add_argument("--critic-path", type=str, default=None, help="Path to critic checkpoint")
    parser.add_argument("--max-steps", type=int, default=100, help="Max dialogue steps per task")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default="outputs/tau3_smtr", help="Output directory")
    parser.add_argument(
        "--mode", choices=["baseline", "smtr", "paired"], default="baseline",
        help="Run mode: baseline (no memory), smtr (with memory), paired (share vs withhold)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Test imports without τ³")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    if args.dry_run:
        dry_run()
        return

    if not check_tau3_available():
        print("ERROR: τ³-bench is not installed.")
        print("Install: cd /home/ecs-user/tau2-bench && uv sync")
        print("Or run with --dry-run to test SMTR imports without τ³.")
        sys.exit(1)

    # Parse LLM args
    llm_args = {}
    if args.llm_args:
        llm_args = json.loads(args.llm_args)
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
        llm_args.setdefault("api_key", args.api_key)

    common = dict(
        domain=args.domain,
        agent_llm=args.agent_llm,
        user_llm=args.user_llm,
        llm_args=llm_args if llm_args else None,
        num_tasks=args.num_tasks,
        split=args.split,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    if args.mode == "baseline":
        run_baseline_eval(output_dir=args.output_dir, **common)
    elif args.mode == "smtr":
        run_smtr_eval(
            output_dir=args.output_dir,
            memory_pool_path=args.memory_pool,
            critic_path=args.critic_path,
            **common,
        )
    elif args.mode == "paired":
        run_paired_eval(
            output_dir=args.output_dir,
            memory_pool_path=args.memory_pool,
            critic_path=args.critic_path,
            **common,
        )


if __name__ == "__main__":
    main()
