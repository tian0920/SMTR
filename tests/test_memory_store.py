import pytest

from smtr.memory.execution_evidence import build_context_fingerprint
from smtr.memory.schemas import ExecutionEvidence
from smtr.memory.seed_memories import build_seed_memories
from smtr.memory.store import SQLiteSharedMemoryRepository


def _repo(tmp_path):
    return SQLiteSharedMemoryRepository(tmp_path / "memory.sqlite")


def _first_memory():
    return build_seed_memories()[0]


def _context(index: int = 0):
    return build_context_fingerprint(
        task_id=f"task-{index}",
        task_tags=["artifact"],
        receiver_agent_id="planner",
        receiver_role="planner",
        receiver_capabilities=["planning"],
        environment_observation={"resource_available": True, "prompt": "short"},
        task_stage="planner",
        selected_memory_ids=[],
        episode_id=f"episode-{index}",
        decision_index=index,
    )


def test_create_memory_rejects_mismatched_ids(tmp_path) -> None:
    repo = _repo(tmp_path)
    card, payload = _first_memory()
    bad_payload = payload.model_copy(update={"memory_id": "different"})

    with pytest.raises(ValueError, match="memory_id"):
        repo.create_memory(bad_payload, card)


def test_duplicate_memory_version_fails(tmp_path) -> None:
    repo = _repo(tmp_path)
    card, payload = _first_memory()

    repo.create_memory(payload, card)
    with pytest.raises(ValueError, match="already exists"):
        repo.create_memory(payload, card)


def test_routing_cards_do_not_expose_steps(tmp_path) -> None:
    repo = _repo(tmp_path)
    card, payload = _first_memory()
    repo.create_memory(payload, card)

    card_dump = repo.get_routing_cards()[0].model_dump()

    assert "steps" not in card_dump


def test_get_payload_returns_full_procedure(tmp_path) -> None:
    repo = _repo(tmp_path)
    card, payload = _first_memory()
    repo.create_memory(payload, card)

    loaded = repo.get_payload(payload.memory_id, version=1)

    assert loaded.steps == payload.steps


def test_snapshot_is_immutable_after_new_memory(tmp_path) -> None:
    repo = _repo(tmp_path)
    card, payload = build_seed_memories()[0]
    repo.create_memory(payload, card)
    snapshot = repo.create_read_snapshot()
    second_card, second_payload = build_seed_memories()[1]

    repo.create_memory(second_payload, second_card)

    assert [card.memory_id for card in snapshot.get_routing_cards()] == [payload.memory_id]


def test_execution_success_updates_alpha(tmp_path) -> None:
    repo = _repo(tmp_path)
    card, payload = _first_memory()
    repo.create_memory(payload, card)

    repo.record_execution_evidence(
        ExecutionEvidence(
            memory_id=payload.memory_id,
            payload_version=1,
            context=_context(),
            execution_success=True,
            reward=1.0,
            source="synthetic_toy",
        )
    )

    updated = repo.get_routing_card(payload.memory_id)
    assert updated.execution_success_alpha == 2.0
    assert updated.execution_success_beta == 1.0
    assert updated.execution_success_count == 1


def test_execution_failure_updates_beta_and_not_transfer_counts(tmp_path) -> None:
    repo = _repo(tmp_path)
    card, payload = _first_memory()
    repo.create_memory(payload, card)

    repo.record_execution_evidence(
        ExecutionEvidence(
            memory_id=payload.memory_id,
            payload_version=1,
            context=_context(),
            execution_success=False,
            reward=0.0,
            failure_category="postcondition_not_reached",
            source="synthetic_toy",
        )
    )

    updated = repo.get_routing_card(payload.memory_id)
    assert updated.execution_success_alpha == 1.0
    assert updated.execution_success_beta == 2.0
    assert updated.execution_failure_count == 1
    assert updated.paired_positive_transfer_count == 0
    assert updated.paired_negative_transfer_count == 0
    assert updated.paired_neutral_transfer_count == 0


def test_execution_context_buffer_is_bounded_fifo(tmp_path) -> None:
    repo = _repo(tmp_path)
    card, payload = _first_memory()
    repo.create_memory(payload, card)

    for index in range(40):
        repo.record_execution_evidence(
            ExecutionEvidence(
                memory_id=payload.memory_id,
                payload_version=1,
                context=_context(index),
                execution_success=True,
                reward=1.0,
                source="synthetic_toy",
            )
        )

    contexts = repo.get_routing_card(payload.memory_id).execution_success_contexts
    assert len(contexts) == 32
    assert contexts[0].episode_id == "episode-8"
    assert contexts[-1].episode_id == "episode-39"
