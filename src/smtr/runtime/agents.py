from typing import Any

from smtr.runtime.environment import ToyEnvironment
from smtr.runtime.fake_llm import DeterministicFakeLLM
from smtr.runtime.state import SMTRState


def run_planner(state: SMTRState, llm: DeterministicFakeLLM | None = None) -> dict[str, Any]:
    llm = llm or DeterministicFakeLLM()
    context = state["agent_local_context"]["planner"]
    output = llm.plan(
        state["task"],
        state["environment_observation"],
        context.get("visible_payloads", []),
    )
    agent_outputs = dict(state["agent_outputs"])
    agent_outputs["planner"] = output
    return {"agent_outputs": agent_outputs}


def run_executor(state: SMTRState, llm: DeterministicFakeLLM | None = None) -> dict[str, Any]:
    llm = llm or DeterministicFakeLLM()
    env = ToyEnvironment(seed=state["run_seed"])
    env.restore(state["environment_observation"])
    plan = state["agent_outputs"]["planner"]["plan"]
    results = [env.apply({"name": action_name}) for action_name in plan]
    final_observation = env.observe()
    output = {
        "actions": [{"name": action_name} for action_name in plan],
        "execution_result": llm.summarize_execution(results),
        "action_results": results,
        "environment_update": final_observation,
        "local_trace": {
            "source": "toy_environment",
            "visible_payload_count": len(
                state["agent_local_context"]["executor"].get("visible_payloads", [])
            ),
        },
    }
    agent_outputs = dict(state["agent_outputs"])
    agent_outputs["executor"] = output
    return {"agent_outputs": agent_outputs, "environment_observation": final_observation}


def run_critic(state: SMTRState) -> dict[str, Any]:
    observation = state["environment_observation"]
    target = observation.get("target_artifact")
    success = target in observation.get("inventory", [])
    reward = 1.0 if success else 0.0
    summary = "Target artifact obtained." if success else "Target artifact was not obtained."
    output = {
        "team_success": success,
        "team_reward": reward,
        "team_summary": summary,
        "local_trace": {
            "source": "deterministic_critic",
            "visible_payload_count": len(
                state["agent_local_context"]["critic"].get("visible_payloads", [])
            ),
        },
    }
    agent_outputs = dict(state["agent_outputs"])
    agent_outputs["critic"] = output
    return {
        "agent_outputs": agent_outputs,
        "team_success": success,
        "team_reward": reward,
        "team_summary": summary,
    }

