"""Phase 8 invariant: current-task-generated memory must never affect
current-task execution.

``M_t`` contains only memories from tasks strictly before ``t``; new
memories extracted at task ``t`` are added to ``M_{t+1}`` and must never
enter the candidate pool of task ``t``.
"""

from __future__ import annotations

from smtr.memory.shared_memory_pool import SharedMemory, SharedMemoryPool


def _mem(mid: str, position: int, source: str = "agent2") -> SharedMemory:
    return SharedMemory(
        memory_id=mid,
        source_agent_id=source,
        origin_task_id=f"task{position}",
        origin_task_position=position,
        routing_card={"goal_summary": "negotiate", "task_tags": ["bargain"]},
        procedure_payload=f"payload-{mid}",
        scenario="bargaining",
    )


def test_current_task_memory_cannot_enter_current_candidate_pool():
    pool = SharedMemoryPool()
    pool.add(_mem("past", 0))
    # Task t=3 runs; a memory "born" at task 3 must not be visible at t=3.
    pool.add(_mem("born_at_3", 3))
    cands = pool.retrieve(
        {"text": "negotiate bargain"}, "agent1", 10, current_task_position=3
    )
    ids = {m.memory_id for m in cands}
    assert "past" in ids
    assert "born_at_3" not in ids


def test_memory_born_at_t_visible_from_t_plus_1():
    pool = SharedMemoryPool()
    pool.add(_mem("born_at_3", 3))
    assert pool.retrieve(
        {"text": "negotiate"}, "agent1", 10, current_task_position=4
    )[0].memory_id == "born_at_3"


def test_memories_before_is_strict():
    pool = SharedMemoryPool()
    for pos in (0, 1, 2):
        pool.add(_mem(f"m{pos}", pos))
    before = pool.memories_before(2)
    assert {m.memory_id for m in before} == {"m0", "m1"}
