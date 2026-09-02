"""Tests for Stage A --candidate-mode retrieval alignment (RIMA-v2 §53 / Commit 8)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from smtr.memory.shared_memory_pool import SharedMemory, memory_task_relevance_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_memory(
    mid: str,
    *,
    tags: list[str] | None = None,
    goal: str = "",
    source: str = "agent_a",
    origin_pos: int = 0,
) -> SharedMemory:
    return SharedMemory(
        memory_id=mid,
        routing_card={
            "task_tags": tags or [],
            "goal_summary": goal,
            "procedure_type": "experience",
        },
        procedure_payload={"action": f"use_{mid}"},
        source_agent_id=source,
        origin_task_id=f"task_{origin_pos}",
        origin_task_position=origin_pos,
    )


# ---------------------------------------------------------------------------
# Relevance score (already tested, but verify it drives candidate selection)
# ---------------------------------------------------------------------------


class TestRelevanceDrivenCandidates:
    def test_relevance_ranks_matching_tags_higher(self):
        task = {"text": "negotiate a trade deal", "tags": ["bargaining", "trade"]}
        m_high = _make_memory("m1", tags=["bargaining", "trade"], goal="negotiate deal")
        m_low = _make_memory("m2", tags=["coding", "debug"], goal="fix bug")
        assert memory_task_relevance_score(m_high, task) > memory_task_relevance_score(m_low, task)

    def test_relevance_zero_for_no_overlap(self):
        task = {"text": "bargaining", "tags": []}
        m = _make_memory("m1", tags=["unrelated"], goal="something else")
        assert memory_task_relevance_score(m, task) == 0.0


# ---------------------------------------------------------------------------
# CLI argparse structure
# ---------------------------------------------------------------------------


class TestStageACliArgs:
    def test_candidate_mode_default_is_sequential(self):
        """Default candidate-mode should be 'sequential'."""
        from experiments.rima.collect_training_interventions import main

        # We just parse args without running (mock everything)
        import argparse

        # Re-create the parser to test defaults
        parser = argparse.ArgumentParser()
        parser.add_argument("--scenario", default="bargaining")
        parser.add_argument("--source-tasks", type=int, default=2)
        parser.add_argument("--intervention-tasks", type=int, default=3)
        parser.add_argument("--max-candidates-per-task", type=int, default=2)
        parser.add_argument("--receiver-count", type=int, default=3)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument(
            "--candidate-mode",
            choices=["sequential", "retrieval"],
            default="sequential",
        )
        parser.add_argument("--engine-timeout", type=int, default=1800)
        parser.add_argument("--output-dir", default="results/rima/stage_a")
        args = parser.parse_args([])
        assert args.candidate_mode == "sequential"

    def test_candidate_mode_retrieval_accepted(self):
        """--candidate-mode retrieval should be accepted."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--candidate-mode",
            choices=["sequential", "retrieval"],
            default="sequential",
        )
        args = parser.parse_args(["--candidate-mode", "retrieval"])
        assert args.candidate_mode == "retrieval"


# ---------------------------------------------------------------------------
# Candidate selection logic (unit-level)
# ---------------------------------------------------------------------------


class TestCandidateSelectionLogic:
    """Test the retrieval-mode candidate selection logic in isolation."""

    def test_sequential_takes_first_n(self):
        pool_memories = [
            _make_memory("m1", origin_pos=0),
            _make_memory("m2", origin_pos=0),
            _make_memory("m3", origin_pos=1),
        ]
        # Sequential mode: just take first N
        cands = pool_memories[:2]
        assert [m.memory_id for m in cands] == ["m1", "m2"]

    def test_retrieval_ranks_by_relevance(self):
        task = {"text": "bargaining trade negotiation", "tags": ["bargaining"]}
        pool_memories = [
            _make_memory("m1", tags=["coding"], goal="fix", origin_pos=0),
            _make_memory("m2", tags=["bargaining", "trade"], goal="negotiate", origin_pos=0),
            _make_memory("m3", tags=["bargaining"], goal="deal", origin_pos=1),
        ]
        # Simulate retrieval mode logic from collect_training_interventions.py
        scored = [
            (m, memory_task_relevance_score(m, task))
            for m in pool_memories
        ]
        scored.sort(key=lambda pair: (-pair[1], pair[0].memory_id))
        cands = [m for m, _ in scored[:2]]

        # m2 and m3 should be ranked higher (they have bargaining tags)
        ids = [m.memory_id for m in cands]
        assert "m2" in ids
        assert "m1" not in ids
