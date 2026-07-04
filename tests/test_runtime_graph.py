from smtr.config import RuntimeConfig
from smtr.memory.seed_memories import build_seed_memory_pool
from smtr.router.sequential_router import ProductionSequentialRouter, SequentialRouterConfig
from smtr.router.transfer_critic import TransferEstimate
from smtr.runtime.environment import ToyEnvironment
from smtr.runtime.graph import build_graph, run_demo
from smtr.runtime.state import initial_state


def test_demo_runtime_is_deterministic() -> None:
    first = run_demo(seed=7)
    second = run_demo(seed=7)

    assert first == second
    assert first["team_success"] is True
    assert first["team_reward"] == 1.0


def test_no_unselected_payloads_enter_global_or_local_state() -> None:
    state = run_demo(seed=7)

    assert set(state["candidate_memory_ids_by_agent"]) == {"planner", "executor", "critic"}
    assert all(ids for ids in state["candidate_memory_ids_by_agent"].values())
    assert state["selected_memory_ids_by_agent"] == {
        "planner": [],
        "executor": [],
        "critic": [],
    }
    assert all(
        context["visible_payloads"] == []
        for context in state["agent_local_context"].values()
    )

    state_text = repr(state)
    assert "Inspect required prerequisites" not in state_text
    assert "Apply each action in order" not in state_text
    assert "Check obtained artifacts" not in state_text


def test_router_trace_has_cards_but_no_payload_steps() -> None:
    state = run_demo(seed=7)

    assert len(state["router_trace"]) == 3
    for trace in state["router_trace"]:
        assert trace["selected_memory_ids"] == []
        assert trace["candidates"]
        assert all(decision["decision"] == "withhold" for decision in trace["decisions"])

    trace_text = repr(state["router_trace"])
    assert "Inspect required prerequisites" not in trace_text
    assert "Apply each action in order" not in trace_text
    assert "Check obtained artifacts" not in trace_text


def test_router_trace_records_revision_versions_and_scores() -> None:
    state = run_demo(seed=7)

    for trace in state["router_trace"]:
        assert trace["memory_store_revision"] == 0
        assert trace["proposer_name"] == "DeterministicHybridCandidateProposer"
        assert trace["proposer_version"] == "1"
        assert trace["router_name"] == "NoMemoryRouter"
        assert trace["router_version"] == "1"
        assert trace["task_stage"] == trace["agent"]
        assert trace["receiver_agent_id"] == trace["agent"]
        assert trace["candidate_scores"] == {
            candidate["memory_id"]: candidate["total_score"]
            for candidate in trace["candidates"]
        }


def test_production_router_runtime_receives_cards_context_and_exposes_only_payloads() -> None:
    class IdBasedCritic:
        critic_version = "runtime_capture_v1"

        def __init__(self):
            self.calls = []

        def predict(self, item):
            self.calls.append(item)
            accept = item.candidate_card.memory_id == "mem_execute_tool_chain"
            return TransferEstimate(
                q00_mean=0.10,
                q01_mean=0.05 if accept else 0.35,
                q10_mean=0.30 if accept else 0.05,
                q11_mean=0.55,
                tau_mean=0.25 if accept else -0.30,
                tau_lcb=0.10 if accept else -0.20,
                tau_ucb=0.40,
                negative_risk_mean=0.05 if accept else 0.35,
                negative_risk_ucb=0.10 if accept else 0.45,
                support_distance=0.0,
                support_threshold=1.0,
                low_support=False,
                ensemble_size=1,
                critic_version=self.critic_version,
            )

    critic = IdBasedCritic()
    router = ProductionSequentialRouter(
        critic=critic,
        config=SequentialRouterConfig(epsilon=0.2),
    )
    env = ToyEnvironment(seed=7)
    app = build_graph(
        memory_pool=build_seed_memory_pool(),
        router=router,
        config=RuntimeConfig(seed=7, top_k=6),
    )
    state = app.invoke(
        initial_state(
            task="Obtain a target artifact using the valid action sequence.",
            environment_observation=env.observe(),
            run_seed=7,
            top_k=6,
        )
    )

    assert critic.calls
    assert any(call.candidate_card.memory_id == "mem_execute_tool_chain" for call in critic.calls)
    assert all(
        call.context.receiver_agent_id in {"planner", "executor", "critic"}
        for call in critic.calls
    )

    executor_trace = next(trace for trace in state["router_trace"] if trace["agent"] == "executor")
    accepted = [d for d in executor_trace["decisions"] if d["action"] == "share"]
    rejected = [d for d in executor_trace["decisions"] if d["action"] == "withhold"]
    assert [d["memory_id"] for d in accepted] == ["mem_execute_tool_chain"]
    assert accepted[0]["decision_reason"] == "accepted"
    assert rejected
    assert any(d["decision_reason"] == "tau_lcb_nonpositive" for d in rejected)
    assert executor_trace["traversal_order"]
    assert set(executor_trace["traversal_order"]) == {
        candidate["memory_id"] for candidate in executor_trace["candidates"]
    }

    executor_payloads = state["agent_local_context"]["executor"]["visible_payloads"]
    assert [payload["memory_id"] for payload in executor_payloads] == ["mem_execute_tool_chain"]
    payload_text = repr(executor_payloads)
    assert "apply gathered key action" in payload_text
    assert "execution_success_alpha" not in payload_text
    assert "required_environment_facts" not in payload_text
    assert "tau_mean" not in payload_text
    assert "negative_risk_ucb" not in payload_text

    trace_text = repr(state["router_trace"])
    assert "apply gathered key action" not in trace_text
    critic_input_text = repr([call.model_dump(mode="json") for call in critic.calls])
    assert "apply gathered key action" not in critic_input_text
    assert "steps" not in critic_input_text
