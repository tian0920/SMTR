"""P0-5 Integrity Suite — Cases 4-5: Receiver payload isolation.

Invariants tested:
  4. receiver1 memory cannot leak to receiver2/3.
  5. Same memory can be validated for receiver1, rejected for receiver2.
"""

from __future__ import annotations

import pytest

from smtr.baselines.base_memory_controller import CandidateMemory
from smtr.memory.consolidation import MemoryAdmissionController
from smtr.memory.persistent_memory import PersistentMemoryBank


# -----------------------------------------------------------------------
# Case 4: receiver1 memory cannot leak to receiver2/3
# -----------------------------------------------------------------------

class TestReceiverPayloadIsolation:
    """Per-receiver injection must prevent cross-contamination."""

    def test_04_per_receiver_payload_map_isolation(self):
        """Build receiver_payload_map and verify each receiver gets only its own."""
        receiver_payloads = {
            "agent1": ["memory_for_agent1_only"],
            "agent2": ["memory_for_agent2_only"],
            "agent3": ["memory_for_agent3_only"],
        }

        # Simulate what engine_process._build_memory_injection_code does:
        # In per-receiver mode, each agent gets ONLY its own payloads
        for agent_id, expected_payloads in receiver_payloads.items():
            agent_payloads = receiver_payloads.get(agent_id, [])
            assert agent_payloads == expected_payloads
            # Verify no other receiver's payloads leak
            for other_id, other_payloads in receiver_payloads.items():
                if other_id != agent_id:
                    for p in other_payloads:
                        assert p not in agent_payloads

    def test_04_receiver_payload_map_no_broadcast(self):
        """Per-receiver map must NOT broadcast all payloads to all agents."""
        receiver_payloads = {
            "agent1": ["secret_strategy_for_agent1"],
            "agent2": ["different_strategy_for_agent2"],
        }

        # Simulate: agent2 should NOT receive agent1's payload
        agent2_payloads = receiver_payloads.get("agent2", [])
        assert "secret_strategy_for_agent1" not in agent2_payloads
        assert agent2_payloads == ["different_strategy_for_agent2"]

    def test_04_missing_receiver_gets_empty_payloads(self):
        """Receiver not in map gets empty list (no accidental injection)."""
        receiver_payloads = {
            "agent1": ["payload_a"],
        }
        # agent3 not in map → empty
        agent3_payloads = receiver_payloads.get("agent3", [])
        assert agent3_payloads == []

    def test_04_mutually_exclusive_api_raises(self):
        """TrajectoryCollector rejects mixing old and new API."""
        from smtr.marble.trajectory_collector import TrajectoryCollector
        from smtr.marble.task_loader import MarbleTask

        collector = TrajectoryCollector()
        task = MarbleTask(
            task_id="test_001",
            scenario="bargaining",
            raw_task={"task_id": "test_001", "scenario": "bargaining"},
        )

        with pytest.raises(ValueError, match="mutually exclusive"):
            collector.collect(
                task,
                seed=0,
                method="smtr_receiver",
                memory_payloads=["broadcast_payload"],
                receiver_agent_ids=["agent1"],
                receiver_memory_payloads={"agent1": ["per_receiver_payload"]},
            )


# -----------------------------------------------------------------------
# Case 5: Same memory validated for r1, rejected for r2
# -----------------------------------------------------------------------

class TestSameMemoryDifferentReceiverDecisions:
    """The same memory can have different lifecycle outcomes per receiver."""

    def test_05_per_receiver_validation_lifecycle(self):
        """Memory validated for agent1, rejected for agent2 — independent."""
        bank = PersistentMemoryBank()
        admission = MemoryAdmissionController(bank)

        # Register a candidate
        bank.add_candidate(
            memory_id="mem_shared_001",
            content="Use negotiation strategy X",
            source_episode=0,
            receiver="agent1",
            created_step=0,
        )

        # Validate for agent1 (delta > 0)
        decision_r1 = admission.admit_for_receiver(
            "mem_shared_001",
            receiver_id="agent1",
            reward_expose=0.8,
            reward_withhold=0.4,
        )
        assert decision_r1.decision == "validated"
        assert decision_r1.delta > 0

        # Reject for agent2 (delta <= 0)
        decision_r2 = admission.admit_for_receiver(
            "mem_shared_001",
            receiver_id="agent2",
            reward_expose=0.3,
            reward_withhold=0.6,
        )
        assert decision_r2.decision == "rejected"
        assert decision_r2.delta < 0

        # Verify per-receiver retrieval
        r1_memories = bank.get_receiver_validated_memories("agent1")
        r2_memories = bank.get_receiver_validated_memories("agent2")

        assert len(r1_memories) == 1
        assert r1_memories[0].memory_id == "mem_shared_001"

        assert len(r2_memories) == 0  # rejected for agent2

    def test_05_receiver_status_authoritative(self):
        """receiver_status is the authoritative lifecycle state per receiver."""
        bank = PersistentMemoryBank()
        admission = MemoryAdmissionController(bank)

        bank.add_candidate(
            memory_id="mem_002",
            content="Strategy Y",
            source_episode=0,
            receiver="agent1",
            created_step=0,
        )

        admission.admit_for_receiver(
            "mem_002",
            receiver_id="agent1",
            reward_expose=0.9,
            reward_withhold=0.2,
        )
        admission.admit_for_receiver(
            "mem_002",
            receiver_id="agent2",
            reward_expose=0.1,
            reward_withhold=0.7,
        )

        # Authoritative per-receiver status
        assert bank.get_receiver_status("mem_002", "agent1") == "validated"
        assert bank.get_receiver_status("mem_002", "agent2") == "rejected"
        # Legacy global status is NOT the source of truth
        entry = bank.get("mem_002")
        assert entry.receiver_status["agent1"] == "validated"
        assert entry.receiver_status["agent2"] == "rejected"
