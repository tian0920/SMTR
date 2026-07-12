from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from statistics import mean
from typing import Any

from smtr.runtime.graph import run_episode
from smtr.runtime.tool_environment import ToolEnvironment

DEFAULT_TASK = "Obtain a target artifact using the valid action sequence."


@dataclass(frozen=True)
class TaskEvaluationConfig:
    seeds: tuple[int, ...] = (7, 42, 123, 256, 999)
    environment: str = "toy"
    task: str = DEFAULT_TASK
    top_k: int = 4


def parse_seed_list(raw: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(raw, str):
        seeds = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    else:
        seeds = tuple(int(seed) for seed in raw)
    if not seeds:
        raise ValueError("at least one evaluation seed is required")
    return seeds


def evaluate_task_execution(
    *,
    llm: Any | None = None,
    config: TaskEvaluationConfig | None = None,
) -> dict[str, Any]:
    """Run complete SMTR episodes and aggregate task-level metrics.

    This mirrors MARBLE's evaluator shape at SMTR scale: per-episode task
    completion is recorded first, then success rate, reward, action execution,
    and plan validity are aggregated into a report-ready JSON object.
    """
    config = config or TaskEvaluationConfig()
    env_factory = _environment_factory(config.environment)
    episodes = []
    for seed in config.seeds:
        initial_observation = env_factory(seed).observe()
        state = run_episode(
            seed=seed,
            top_k=config.top_k,
            task=config.task,
            task_id=f"{config.environment}-task-{seed}",
            episode_id=f"{config.environment}-episode-{seed}",
            environment_observation=initial_observation,
            llm=llm,
            env_factory=env_factory,
        )
        episodes.append(
            _episode_metrics(
                state,
                expected_plan=initial_observation.get("valid_sequence", []),
            )
        )
    return summarize_task_episodes(episodes, config=config)


def summarize_task_episodes(
    episodes: list[dict[str, Any]],
    *,
    config: TaskEvaluationConfig | None = None,
) -> dict[str, Any]:
    episode_count = len(episodes)
    success_count = sum(1 for episode in episodes if episode["team_success"])
    rewards = [float(episode["team_reward"] or 0.0) for episode in episodes]
    plan_matches = [episode["plan_matches_expected"] for episode in episodes]
    action_counts = [episode["action_count"] for episode in episodes]
    successful_action_counts = [episode["successful_action_count"] for episode in episodes]
    total_actions = sum(action_counts)
    total_successful_actions = sum(successful_action_counts)
    failure_errors = Counter(
        error
        for episode in episodes
        for error in episode["action_errors"]
        if error
    )

    summary = {
        "episode_count": episode_count,
        "task_success_count": success_count,
        "task_success_rate": _ratio(success_count, episode_count),
        "mean_reward": mean(rewards) if rewards else 0.0,
        "plan_match_count": sum(1 for matched in plan_matches if matched),
        "plan_match_rate": _ratio(sum(1 for matched in plan_matches if matched), episode_count),
        "total_actions": total_actions,
        "successful_actions": total_successful_actions,
        "action_success_rate": _ratio(total_successful_actions, total_actions),
        "mean_actions_per_episode": mean(action_counts) if action_counts else 0.0,
        "mean_successful_actions_per_episode": (
            mean(successful_action_counts) if successful_action_counts else 0.0
        ),
        "failure_errors": dict(failure_errors),
        "episodes": episodes,
    }
    if config is not None:
        summary["config"] = {
            "seeds": list(config.seeds),
            "environment": config.environment,
            "task": config.task,
            "top_k": config.top_k,
        }
    return summary


def _episode_metrics(state: dict[str, Any], *, expected_plan: list[str]) -> dict[str, Any]:
    planner_output = state.get("agent_outputs", {}).get("planner", {})
    executor_output = state.get("agent_outputs", {}).get("executor", {})
    plan = list(planner_output.get("plan", []))
    action_results = list(executor_output.get("action_results", []))
    action_errors = [
        str(result.get("error"))
        for result in action_results
        if result.get("ok") is not True and result.get("error") is not None
    ]
    return {
        "episode_id": state.get("episode_id"),
        "task_id": state.get("task_id"),
        "seed": state.get("run_seed"),
        "task": state.get("task"),
        "team_success": bool(state.get("team_success")),
        "team_reward": float(state.get("team_reward") or 0.0),
        "team_summary": state.get("team_summary"),
        "plan": plan,
        "expected_plan": list(expected_plan),
        "plan_matches_expected": plan == list(expected_plan),
        "action_count": len(action_results),
        "successful_action_count": sum(1 for result in action_results if result.get("ok")),
        "action_errors": action_errors,
        "selected_memory_ids_by_agent": state.get("selected_memory_ids_by_agent", {}),
    }


def _environment_factory(name: str) -> Callable[[int], Any]:
    if name == "toy":
        from smtr.runtime.environment import ToyEnvironment

        return lambda seed: ToyEnvironment(seed=seed)
    if name == "tool":
        return lambda seed: ToolEnvironment(seed=seed)
    raise ValueError("environment must be one of: toy, tool")


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
