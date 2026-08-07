"""Memory provenance propagation to paired records (清单 Writer-Agnostic §17).

The ``read_memory_source_provenance`` accessor reads provenance from
``payload.provenance`` and propagates it through to the serialized
paired record.
"""

from __future__ import annotations

import pytest

from smtr.marble.real_pairs import (
    MemorySourceProvenance,
    read_memory_source_provenance,
)


def _memory_record(
    *,
    source_agent_id: str = "agent-1",
    source_task_id: str = "train-task-1",
    source_trajectory_id: str = "traj-1",
    source_split: str = "train",
) -> dict:
    return {
        "memory_id": "m1",
        "payload": {
            "procedure": "Step 1. Use pg_stat_statements.",
            "provenance": {
                "source_agent_id": source_agent_id,
                "source_task_id": source_task_id,
                "source_trajectory_id": source_trajectory_id,
                "source_split": source_split,
            },
        },
        "routing_card": {
            "goal_summary": "Guide diagnosis",
            "task_tags": ["database"],
            "required_tools": [],
        },
    }


class TestReadMemorySourceProvenance:
    def test_reads_all_fields(self):
        prov = read_memory_source_provenance(_memory_record())
        assert prov.source_agent_id == "agent-1"
        assert prov.source_task_id == "train-task-1"
        assert prov.source_trajectory_id == "traj-1"
        assert prov.source_split == "train"

    def test_strips_whitespace(self):
        rec = _memory_record(source_agent_id="  agent-1  ")
        prov = read_memory_source_provenance(rec)
        assert prov.source_agent_id == "agent-1"

    def test_missing_payload_raises(self):
        with pytest.raises(ValueError, match="payload.provenance"):
            read_memory_source_provenance({"memory_id": "m1"})

    def test_missing_provenance_raises(self):
        with pytest.raises(ValueError, match="payload.provenance"):
            read_memory_source_provenance(
                {"payload": {"procedure": "x"}}
            )

    def test_missing_source_agent_id_raises(self):
        rec = _memory_record()
        del rec["payload"]["provenance"]["source_agent_id"]
        with pytest.raises(ValueError, match="payload.provenance"):
            read_memory_source_provenance(rec)

    def test_empty_source_split_raises(self):
        rec = _memory_record(source_split="")
        with pytest.raises(ValueError, match="empty required fields"):
            read_memory_source_provenance(rec)


class TestMemorySourceProvenanceFrozen:
    def test_immutable(self):
        prov = MemorySourceProvenance(
            source_agent_id="a",
            source_task_id="t",
            source_trajectory_id="tr",
            source_split="train",
        )
        with pytest.raises(AttributeError):
            prov.source_agent_id = "b"  # type: ignore[misc]
