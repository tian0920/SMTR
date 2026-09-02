"""Tests for SharedMemoryPool subset/unseen retrieval (Commit 4 / §15-16)."""

from __future__ import annotations

import pytest

from smtr.memory.shared_memory_pool import (
    SharedMemory,
    SharedMemoryPool,
    memory_task_relevance_score,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _mem(
    mid: str,
    *,
    source: str = "agent_A",
    origin_task: str = "t0",
    position: int = 0,
    tags: list[str] | None = None,
    goal: str = "",
) -> SharedMemory:
    return SharedMemory(
        memory_id=mid,
        source_agent_id=source,
        origin_task_id=origin_task,
        origin_task_position=position,
        routing_card={"task_tags": tags or [], "goal_summary": goal},
    )


def _task(text: str = "", tags: list[str] | None = None) -> dict:
    return {"text": text, "tags": tags or []}


# ---------------------------------------------------------------------------
# memory_task_relevance_score (public)
# ---------------------------------------------------------------------------

class TestMemoryTaskRelevanceScore:
    def test_public_function_callable(self):
        assert callable(memory_task_relevance_score)

    def test_relevance_zero_when_no_overlap(self):
        m = _mem("m1", tags=["alpha"])
        assert memory_task_relevance_score(m, _task(text="beta")) == 0.0

    def test_relevance_positive_on_overlap(self):
        m = _mem("m1", tags=["alpha", "beta"])
        score = memory_task_relevance_score(m, _task(tags=["alpha"]))
        assert score > 0.0

    def test_backward_compat_alias(self):
        from smtr.memory.shared_memory_pool import _relevance_score
        assert _relevance_score is memory_task_relevance_score


# ---------------------------------------------------------------------------
# rank_subset
# ---------------------------------------------------------------------------

class TestRankSubset:
    def _pool(self) -> SharedMemoryPool:
        pool = SharedMemoryPool()
        pool.add(_mem("m1", position=0, tags=["alpha"]))
        pool.add(_mem("m2", position=1, tags=["beta"]))
        pool.add(_mem("m3", position=2, tags=["alpha", "gamma"]))
        pool.add(_mem("m4", position=3, tags=["alpha"]))
        return pool

    def test_basic_ranking(self):
        pool = self._pool()
        result = pool.rank_subset(
            memory_ids=["m1", "m3"],
            task=_task(tags=["alpha"]),
            receiver_id="agent_B",
            current_task_position=10,
            top_k=10,
        )
        ids = [m.memory_id for m in result]
        assert "m1" in ids
        assert "m3" in ids

    def test_respects_top_k(self):
        pool = self._pool()
        result = pool.rank_subset(
            memory_ids=["m1", "m2", "m3", "m4"],
            task=_task(tags=["alpha"]),
            receiver_id="agent_B",
            current_task_position=10,
            top_k=2,
        )
        assert len(result) == 2

    def test_excludes_future_memories(self):
        pool = self._pool()
        # m3 has position=2, m4 has position=3
        result = pool.rank_subset(
            memory_ids=["m3", "m4"],
            task=_task(tags=["alpha"]),
            receiver_id="agent_B",
            current_task_position=3,  # m4 (pos=3) excluded
            top_k=10,
        )
        ids = [m.memory_id for m in result]
        assert "m3" in ids
        assert "m4" not in ids

    def test_silently_skips_missing_ids(self):
        pool = self._pool()
        result = pool.rank_subset(
            memory_ids=["m1", "nonexistent"],
            task=_task(tags=["alpha"]),
            receiver_id="agent_B",
            current_task_position=10,
            top_k=10,
        )
        assert len(result) == 1
        assert result[0].memory_id == "m1"

    def test_empty_ids_returns_empty(self):
        pool = self._pool()
        result = pool.rank_subset(
            memory_ids=[],
            task=_task(tags=["alpha"]),
            receiver_id="agent_B",
            current_task_position=10,
            top_k=10,
        )
        assert result == []

    def test_deterministic_order(self):
        pool = self._pool()
        r1 = pool.rank_subset(
            memory_ids=["m1", "m3"],
            task=_task(tags=["alpha"]),
            receiver_id="agent_B",
            current_task_position=10,
            top_k=10,
        )
        r2 = pool.rank_subset(
            memory_ids=["m3", "m1"],  # reversed input order
            task=_task(tags=["alpha"]),
            receiver_id="agent_B",
            current_task_position=10,
            top_k=10,
        )
        assert [m.memory_id for m in r1] == [m.memory_id for m in r2]


# ---------------------------------------------------------------------------
# retrieve_unseen
# ---------------------------------------------------------------------------

class TestRetrieveUnseen:
    def _pool(self) -> SharedMemoryPool:
        pool = SharedMemoryPool()
        pool.add(_mem("m1", position=0, tags=["alpha"]))
        pool.add(_mem("m2", position=1, tags=["beta"]))
        pool.add(_mem("m3", position=2, tags=["alpha", "gamma"]))
        pool.add(_mem("m4", position=3, tags=["alpha"]))
        return pool

    def test_excludes_known_ids(self):
        pool = self._pool()
        result = pool.retrieve_unseen(
            _task(tags=["alpha"]),
            "agent_B",
            top_k=10,
            current_task_position=10,
            exclude_memory_ids={"m1", "m3"},
        )
        ids = {m.memory_id for m in result}
        assert ids == {"m2", "m4"}

    def test_historical_only(self):
        pool = self._pool()
        result = pool.retrieve_unseen(
            _task(tags=["alpha"]),
            "agent_B",
            top_k=10,
            current_task_position=2,  # only m1(0), m2(1) eligible
            exclude_memory_ids=set(),
        )
        ids = {m.memory_id for m in result}
        assert ids == {"m1", "m2"}

    def test_current_task_memory_cannot_be_explored(self):
        pool = self._pool()
        result = pool.retrieve_unseen(
            _task(tags=["alpha"]),
            "agent_B",
            top_k=10,
            current_task_position=1,  # only m1(0) eligible
            exclude_memory_ids=set(),
        )
        ids = {m.memory_id for m in result}
        assert ids == {"m1"}

    def test_respects_top_k(self):
        pool = self._pool()
        result = pool.retrieve_unseen(
            _task(tags=["alpha"]),
            "agent_B",
            top_k=1,
            current_task_position=10,
            exclude_memory_ids=set(),
        )
        assert len(result) == 1

    def test_deterministic_order(self):
        pool = self._pool()
        r1 = pool.retrieve_unseen(
            _task(tags=["alpha"]),
            "agent_B",
            top_k=10,
            current_task_position=10,
            exclude_memory_ids=set(),
        )
        r2 = pool.retrieve_unseen(
            _task(tags=["alpha"]),
            "agent_B",
            top_k=10,
            current_task_position=10,
            exclude_memory_ids=set(),
        )
        assert [m.memory_id for m in r1] == [m.memory_id for m in r2]

    def test_all_excluded_returns_empty(self):
        pool = self._pool()
        result = pool.retrieve_unseen(
            _task(tags=["alpha"]),
            "agent_B",
            top_k=10,
            current_task_position=10,
            exclude_memory_ids={"m1", "m2", "m3", "m4"},
        )
        assert result == []

    def test_negative_known_not_in_unseen(self):
        """A negative known memory must NOT reappear as unseen global.

        It should be reconsidered through K recall (rank_subset), not
        through M_global \\ K (retrieve_unseen).
        """
        pool = self._pool()
        # Simulate: m1 is known (explored, negative)
        known_ids = {"m1"}
        unseen = pool.retrieve_unseen(
            _task(tags=["alpha"]),
            "agent_B",
            top_k=10,
            current_task_position=10,
            exclude_memory_ids=known_ids,
        )
        ids = {m.memory_id for m in unseen}
        assert "m1" not in ids

        # But m1 is still available through rank_subset (K recall)
        known = pool.rank_subset(
            memory_ids=known_ids,
            task=_task(tags=["alpha"]),
            receiver_id="agent_B",
            current_task_position=10,
            top_k=10,
        )
        assert len(known) == 1
        assert known[0].memory_id == "m1"


# ---------------------------------------------------------------------------
# retrieve (unchanged behavior)
# ---------------------------------------------------------------------------

class TestRetrieveUnchanged:
    def test_existing_retrieve_still_works(self):
        pool = SharedMemoryPool()
        pool.add(_mem("m1", position=0, tags=["alpha"]))
        pool.add(_mem("m2", position=1, tags=["beta"]))
        result = pool.retrieve(
            _task(tags=["alpha"]),
            "agent_B",
            top_k=10,
            current_task_position=10,
        )
        assert len(result) == 2
        # m1 (alpha) should rank first
        assert result[0].memory_id == "m1"
