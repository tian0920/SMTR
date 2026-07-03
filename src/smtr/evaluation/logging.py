from smtr.runtime.state import SMTRState


def summarize_run(state: SMTRState) -> str:
    lines = [
        f"Task: {state['task']}",
        f"Planner output: {state['agent_outputs'].get('planner', {})}",
        f"Executor output: {state['agent_outputs'].get('executor', {})}",
        f"Critic output: {state['agent_outputs'].get('critic', {})}",
        f"Candidate memories for each agent: {state['candidate_memory_ids_by_agent']}",
        "Router decisions for each agent:",
    ]
    for trace in state["router_trace"]:
        lines.append(f"  {trace['agent']}: {trace['decisions']}")
    lines.extend(
        [
            f"Selected memory IDs for each agent: {state['selected_memory_ids_by_agent']}",
            f"Final team success: {state['team_success']}",
            f"Final team reward: {state['team_reward']}",
            f"Final team summary: {state['team_summary']}",
        ]
    )
    return "\n".join(lines)

