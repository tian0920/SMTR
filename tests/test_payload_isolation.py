from smtr.memory.seed_memories import build_seed_memories, seed_repository
from smtr.memory.store import SQLiteSharedMemoryRepository
from smtr.runtime.graph import run_demo_with_repository


def test_sqlite_demo_keeps_unselected_payload_steps_isolated(tmp_path) -> None:
    repository = SQLiteSharedMemoryRepository(tmp_path / "memory.sqlite")
    seed_repository(repository)

    state = run_demo_with_repository(repository=repository, seed=7, top_k=4)

    assert all(state["candidate_memory_ids_by_agent"].values())
    assert state["selected_memory_ids_by_agent"] == {
        "planner": [],
        "executor": [],
        "critic": [],
    }
    assert all(
        context["visible_payloads"] == []
        for context in state["agent_local_context"].values()
    )

    payload_steps = [step for _, payload in build_seed_memories() for step in payload.steps]
    state_text = repr(state)
    assert all(step not in state_text for step in payload_steps)
