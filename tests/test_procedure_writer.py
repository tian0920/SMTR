import pytest

from smtr.memory.procedure_writer import DeterministicProcedureWriter


def test_procedure_writer_creates_payload_and_card_from_successful_episode() -> None:
    initial = {
        "location": "workbench",
        "inventory": [],
        "target_artifact": "target_artifact",
        "tags": ["artifact", "tool-chain"],
        "tool_version": "v1",
        "resource_available": True,
        "resource_locked": False,
    }
    final = {**initial, "inventory": ["key", "target_artifact"], "location": "open_chest"}

    payload, card = DeterministicProcedureWriter().write_from_successful_episode(
        episode_id="episode-1",
        writer_agent_id="executor-1",
        task="Obtain a target artifact using the valid action sequence.",
        planner_output={"plan": ["gather_key", "open_chest", "collect_artifact"]},
        executor_output={
            "actions": [
                {"name": "gather_key"},
                {"name": "open_chest"},
                {"name": "collect_artifact"},
            ]
        },
        initial_environment=initial,
        final_environment=final,
    )

    assert payload.memory_id == card.memory_id
    assert payload.steps == ["call gather_key", "call open_chest", "call collect_artifact"]
    assert card.required_environment_facts["tool_version"] == "v1"
    assert card.execution_success_alpha == 1.0
    assert card.execution_success_beta == 1.0
    assert "call gather_key" not in repr(card)


def test_procedure_writer_rejects_failed_episode() -> None:
    initial = {"inventory": [], "target_artifact": "target_artifact"}

    with pytest.raises(ValueError, match="successful"):
        DeterministicProcedureWriter().write_from_successful_episode(
            episode_id="episode-2",
            writer_agent_id="executor-1",
            task="Obtain artifact",
            planner_output={"plan": []},
            executor_output={"actions": []},
            initial_environment=initial,
            final_environment=initial,
        )
