import pytest

from smtr.counterfactual.snapshot import (
    ReadOnlyCounterfactualStoreError,
    ReadOnlyPinnedMemoryView,
)
from smtr.memory.schemas import MemoryRoutingCard
from smtr.memory.seed_memories import build_seed_memories
from smtr.memory.store import SQLiteSharedMemoryRepository


def test_read_only_view_rejects_writes(tmp_path) -> None:
    repo = SQLiteSharedMemoryRepository(tmp_path / "memory.sqlite")
    card, payload = build_seed_memories()[0]
    repo.create_memory(payload, card)
    view = ReadOnlyPinnedMemoryView(repo, repo.create_read_snapshot())

    with pytest.raises(ReadOnlyCounterfactualStoreError):
        view.create_memory(payload, card)


def test_pinned_view_keeps_old_active_payload_version(tmp_path) -> None:
    repo = SQLiteSharedMemoryRepository(tmp_path / "memory.sqlite")
    card, payload = build_seed_memories()[0]
    repo.create_memory(payload, card)
    snapshot = repo.create_read_snapshot()
    view = ReadOnlyPinnedMemoryView(repo, snapshot)

    payload_v2 = payload.model_copy(update={"version": 2, "steps": ["new secret step"]})
    card_v2 = MemoryRoutingCard(
        memory_id=card.memory_id,
        active_payload_version=2,
        goal_summary=card.goal_summary,
        task_tags=card.task_tags,
        compatible_receiver_roles=card.compatible_receiver_roles,
    )
    repo.create_memory(payload_v2, card_v2)

    assert repo.get_payload(card.memory_id).version == 2
    assert view.get_payload(card.memory_id).version == 1
