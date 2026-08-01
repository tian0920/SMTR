from typing import Any, TypedDict


class SMTRState(TypedDict):
    episode_id: str
    task_id: str
    task: str
    environment_observation: dict[str, Any]
    current_agent: str | None
    agent_local_context: dict[str, dict[str, Any]]
    agent_outputs: dict[str, dict[str, Any]]
    candidate_memory_ids_by_agent: dict[str, list[str]]
    selected_memory_ids_by_agent: dict[str, list[str]]
    router_trace: list[dict[str, Any]]
    team_success: bool | None
    team_reward: float | None
    team_summary: str | None
    run_seed: int
    top_k: int


def initial_state(
    task: str,
    environment_observation: dict[str, Any],
    run_seed: int,
    episode_id: str | None = None,
    task_id: str | None = None,
    top_k: int = 4,
) -> SMTRState:
    return {
        "episode_id": episode_id or f"episode-{run_seed}",
        "task_id": task_id or f"task-{run_seed}",
        "task": task,
        "environment_observation": environment_observation,
        "current_agent": None,
        "agent_local_context": {
            "planner": {"visible_payloads": []},
            "executor": {"visible_payloads": []},
            "critic": {"visible_payloads": []},
        },
        "agent_outputs": {},
        "candidate_memory_ids_by_agent": {},
        "selected_memory_ids_by_agent": {},
        "router_trace": [],
        "team_success": None,
        "team_reward": None,
        "team_summary": None,
        "run_seed": run_seed,
        "top_k": top_k,
    }
