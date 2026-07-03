from smtr.runtime.graph import run_demo


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
