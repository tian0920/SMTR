#!/usr/bin/env python3
"""Collect τ³ retail training trajectories with no-memory baseline.

This script runs the τ³-bench CLI on retail training split tasks
and saves trajectories as JSONL for building the memory corpus.

Usage:
    python3 scripts/collect_tau3_trajectories.py --num-tasks 5 --output data/tau3_retail_train_trajectories.jsonl
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

TAU2_PATH = Path("/home/ecs-user/tau2-bench")
SMTR_PATH = Path("/home/ecs-user/SMTR")


def load_retail_train_task_ids() -> list[str]:
    """Load retail training split task IDs."""
    split_file = TAU2_PATH / "data/tau2/domains/retail/split_tasks.json"
    with open(split_file) as f:
        splits = json.load(f)
    return splits["train"]


def run_single_task(
    task_id: str,
    agent_llm: str,
    user_llm: str,
    output_dir: Path,
    seed: int = 42,
    max_steps: int = 200,
) -> dict | None:
    """Run a single τ³ task and return the result."""
    api_base = "https://llm-jhxtd03gjg0gd2o2.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
    
    cmd = [
        str(TAU2_PATH / ".venv/bin/tau2"),
        "run",
        "--domain", "retail",
        "--task-ids", task_id,
        "--agent-llm", agent_llm,
        "--agent-llm-args", json.dumps({"api_base": api_base, "max_tokens": 1024}),
        "--user-llm", user_llm,
        "--user-llm-args", json.dumps({"api_base": api_base, "max_tokens": 1024}),
        "--num-trials", "1",
        "--seed", str(seed),
        "--max-steps", str(max_steps),
        "--save-to", str(output_dir),
    ]
    
    env = {"OPENAI_API_KEY": "sk-8692d980a9e148d3843a64135ae0b0f2", "PATH": "/usr/bin:/bin"}
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(TAU2_PATH),
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        
        # Load results
        results_file = output_dir / "results.json"
        if results_file.exists():
            with open(results_file) as f:
                data = json.load(f)
            if data.get("simulations"):
                return data["simulations"][0]
        return None
    except subprocess.TimeoutExpired:
        print(f"  Timeout for task {task_id}")
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None


def extract_trajectory(sim: dict) -> dict:
    """Extract trajectory from simulation result."""
    messages = []
    tool_calls = []
    for msg in sim.get("messages", []):
        messages.append({
            "role": msg.get("role"),
            "content": msg.get("content"),
        })
        if msg.get("tool_calls"):
            tool_calls.extend(msg["tool_calls"])
    
    reward_info = sim.get("reward_info", {})
    return {
        "task_id": sim.get("task_id"),
        "split": "train",
        "messages": messages,
        "tool_calls": tool_calls,
        "outcome_summary": {
            "reward": reward_info.get("reward"),
            "termination_reason": sim.get("termination_reason"),
            "num_messages": len(messages),
            "num_tool_calls": len(tool_calls),
        },
        "reward": reward_info.get("reward"),
    }


def main():
    parser = argparse.ArgumentParser(description="Collect τ³ retail training trajectories")
    parser.add_argument("--num-tasks", type=int, default=5, help="Number of tasks to run")
    parser.add_argument("--output", type=str, default="data/tau3_retail_train_trajectories.jsonl")
    parser.add_argument("--agent-llm", type=str, default="openai/qwen3.5-plus")
    parser.add_argument("--user-llm", type=str, default="openai/qwen3.5-plus")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=200)
    args = parser.parse_args()

    # Load tasks
    train_ids = load_retail_train_task_ids()
    selected_ids = train_ids[: args.num_tasks]
    print(f"Selected {len(selected_ids)} tasks from retail train split")

    # Run trajectories
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use temp directory for each run
    import tempfile
    temp_dir = Path(tempfile.mkdtemp())

    results = []
    for i, task_id in enumerate(selected_ids):
        print(f"[{i+1}/{len(selected_ids)}] Running task {task_id}...")
        
        # Run task
        sim_result = run_single_task(
            task_id=task_id,
            agent_llm=args.agent_llm,
            user_llm=args.user_llm,
            output_dir=temp_dir / f"run_{task_id}",
            seed=args.seed,
            max_steps=args.max_steps,
        )
        
        if sim_result:
            trajectory = extract_trajectory(sim_result)
            results.append(trajectory)
            
            # Write incrementally
            with open(output_path, "a") as f:
                f.write(json.dumps(trajectory) + "\n")
            
            print(f"  Reward: {trajectory['reward']}, Messages: {trajectory['outcome_summary']['num_messages']}")
        else:
            print(f"  Failed to get result")

    print(f"\nCollected {len(results)} trajectories → {output_path}")


if __name__ == "__main__":
    main()
