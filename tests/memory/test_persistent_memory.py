"""Unit tests for persistent memory bank and TCI admission gate (Tasks 1-2)."""

import pytest

from smtr.memory.consolidation import MemoryAdmissionController
from smtr.memory.memory_schema import PersistentMemoryEntry
from smtr.memory.persistent_memory import PersistentMemoryBank


def _make_bank() -> PersistentMemoryBank:
    bank = PersistentMemoryBank()
    bank.add_candidate(
        memory_id="m1", content="use index scan", source_episode=0,
        receiver="agent1", created_step=0,
    )
    bank.add_candidate(
        memory_id="m2", content="check pg_stat_statements", source_episode=1,
        receiver="agent2", created_step=1,
    )
    return bank


class TestPersistentMemoryEntry:
    def test_defaults(self) -> None:
        entry = PersistentMemoryEntry(
            memory_id="m", content="c", source_episode=0,
            receiver="agent1", created_step=0,
        )
        assert entry.status == "candidate"
        assert entry.validation_count == 0
        assert entry.tci_effect is None

    def test_frozen(self) -> None:
        entry = PersistentMemoryEntry(
            memory_id="m", content="c", source_episode=0,
            receiver="agent1", created_step=0,
        )
        with pytest.raises(Exception):
            entry.status = "validated"  # type: ignore[misc]


class TestPersistentMemoryBank:
    def test_add_candidate(self) -> None:
        bank = _make_bank()
        entry = bank.get("m1")
        assert entry.status == "candidate"
        assert entry.content == "use index scan"

    def test_duplicate_id_rejected(self) -> None:
        bank = _make_bank()
        with pytest.raises(ValueError, match="duplicate"):
            bank.add_candidate(
                memory_id="m1", content="x", source_episode=2,
                receiver="agent1", created_step=2,
            )

    def test_validate_memory(self) -> None:
        bank = _make_bank()
        entry = bank.validate_memory("m1", tci_effect=0.5)
        assert entry.status == "validated"
        assert entry.tci_effect == 0.5
        assert entry.validation_count == 1

    def test_reject_memory(self) -> None:
        bank = _make_bank()
        entry = bank.reject_memory("m2", tci_effect=-0.3)
        assert entry.status == "rejected"
        assert entry.tci_effect == -0.3
        assert entry.validation_count == 1

    def test_revalidate_increments_count(self) -> None:
        bank = _make_bank()
        bank.reject_memory("m1", tci_effect=-0.1)
        entry = bank.validate_memory("m1", tci_effect=0.4)
        assert entry.status == "validated"
        assert entry.validation_count == 2

    def test_retrieve_validated_filters_and_sorts(self) -> None:
        bank = _make_bank()
        bank.validate_memory("m2", tci_effect=1.0)
        bank.validate_memory("m1", tci_effect=1.0)
        validated = bank.retrieve_validated()
        assert [e.memory_id for e in validated] == ["m1", "m2"]
        assert bank.retrieve_validated(receiver="agent2")
        assert not bank.retrieve_validated(receiver="agent3")

    def test_get_statistics(self) -> None:
        bank = _make_bank()
        bank.validate_memory("m1", tci_effect=1.0)
        stats = bank.get_statistics()
        assert stats["total"] == 2
        assert stats["validated"] == 1
        assert stats["candidate"] == 1
        assert stats["rejected"] == 0
        assert stats["mean_tci_effect_validated"] == 1.0

    def test_save_load_roundtrip(self, tmp_path) -> None:
        bank = _make_bank()
        bank.validate_memory("m1", tci_effect=0.7)
        path = tmp_path / "bank.jsonl"
        bank.save(path)
        loaded = PersistentMemoryBank.load(path)
        assert loaded.get("m1").status == "validated"
        assert loaded.get("m1").tci_effect == 0.7
        assert loaded.get("m2").status == "candidate"
        assert loaded.get_statistics() == bank.get_statistics()

    def test_unknown_memory_raises(self) -> None:
        with pytest.raises(KeyError):
            PersistentMemoryBank().get("missing")


class TestMemoryAdmissionController:
    def test_positive_delta_validates(self) -> None:
        bank = _make_bank()
        controller = MemoryAdmissionController(bank)
        decision = controller.admit("m1", reward_expose=1.0, reward_withhold=0.0)
        assert decision.decision == "validated"
        assert decision.delta == 1.0
        assert bank.get("m1").status == "validated"

    def test_zero_delta_rejects(self) -> None:
        bank = _make_bank()
        controller = MemoryAdmissionController(bank)
        decision = controller.admit("m1", reward_expose=0.0, reward_withhold=0.0)
        assert decision.decision == "rejected"
        assert bank.get("m1").status == "rejected"

    def test_negative_delta_rejects(self) -> None:
        bank = _make_bank()
        controller = MemoryAdmissionController(bank)
        decision = controller.admit("m2", reward_expose=0.0, reward_withhold=1.0)
        assert decision.decision == "rejected"
        assert decision.delta == -1.0

    def test_decision_log_and_summary(self) -> None:
        bank = _make_bank()
        controller = MemoryAdmissionController(bank)
        controller.admit("m1", reward_expose=1.0, reward_withhold=0.0)
        controller.admit("m2", reward_expose=0.0, reward_withhold=1.0)
        assert len(controller.decisions) == 2
        assert controller.summary() == {"total": 2, "validated": 1, "rejected": 1}

    def test_admit_from_pair_record(self) -> None:
        bank = _make_bank()
        controller = MemoryAdmissionController(bank)
        record = {
            "candidate_memory_id": "m1",
            "reward_expose": 1.0,
            "reward_withhold": 0.0,
        }
        decision = controller.admit_from_pair_record(record)
        assert decision.decision == "validated"

    def test_admit_from_pair_record_alt_keys(self) -> None:
        bank = _make_bank()
        controller = MemoryAdmissionController(bank)
        record = {
            "candidate_memory_id": "m2",
            "Y_expose": 0,
            "Y_withhold": 1,
        }
        decision = controller.admit_from_pair_record(record)
        assert decision.decision == "rejected"

    def test_missing_outcome_raises(self) -> None:
        bank = _make_bank()
        controller = MemoryAdmissionController(bank)
        with pytest.raises(KeyError):
            controller.admit_from_pair_record({"candidate_memory_id": "m1"})
