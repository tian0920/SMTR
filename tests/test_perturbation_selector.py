"""Tests for P2 perturbation selector (清单 §12-§13).

Verifies:
  - Tier 1 (observed 01/10) edges prioritised.
  - Budget honoured exactly.
  - Test-split edges excluded.
  - One perturbation per edge.
"""

from __future__ import annotations

import pytest

from smtr.intervention.perturbation_selector import (
    PerturbationSelection,
    edge_has_transfer_event,
    select_balanced_operator,
    select_perturbation_edges,
)
from smtr.intervention.transfer_perturbation import (
    PreconditionPerturbation,
    RequiredToolPerturbation,
    RequiredCapabilityPerturbation,
)


def _make_paired_record(
    *,
    edge_id: str = "e1",
    task_id: str = "task_1",
    receiver_agent_id: str = "agent_a",
    candidate_memory_id: str = "mem_1",
    generation_seed: int = 42,
    y0: bool = False,
    y1: bool = True,
    valid: bool = True,
    split_name: str = "train",
    scenario: str = "database",
    task_instruction: str = "do stuff",
    receiver_role: str = "executor",
    receiver_capabilities: tuple = ("query_optimization",),
    receiver_tool_names: tuple = ("sql_read", "sql_write"),
    environment_signature: tuple = ("read_write_access",),
) -> dict:
    return {
        "edge_id": edge_id,
        "task_id": task_id,
        "receiver_agent_id": receiver_agent_id,
        "candidate_memory_id": candidate_memory_id,
        "generation_seed": generation_seed,
        "valid": valid,
        "split_name": split_name,
        "scenario": scenario,
        "task_instruction": task_instruction,
        "receiver_role": receiver_role,
        "receiver_capabilities": list(receiver_capabilities),
        "receiver_tool_names": list(receiver_tool_names),
        "environment_signature": list(environment_signature),
        "withhold": {"team_success": y0},
        "share": {"team_success": y1},
    }


def _make_memory_pool(
    memory_id: str = "mem_1",
) -> dict[str, dict]:
    return {
        memory_id: {
            "memory_id": memory_id,
            "routing_card": {
                "memory_id": memory_id,
                "goal_summary": "Step 1: inspect\nStep 2: query",
                "task_tags": ["database"],
                "required_tools": ["sql_read"],
                "required_capabilities": ["query_optimization"],
                "execution_role_tags": ["executor"],
                "environment_constraints": ["read_write_access"],
                "precondition_tags": ["schema_inspected"],
            },
        }
    }


class TestEdgeHasTransferEvent:
    def test_positive_transfer(self) -> None:
        recs = [_make_paired_record(y0=False, y1=True)]
        assert edge_has_transfer_event(recs) is True

    def test_negative_transfer(self) -> None:
        recs = [_make_paired_record(y0=True, y1=False)]
        assert edge_has_transfer_event(recs) is True

    def test_no_transfer(self) -> None:
        recs = [_make_paired_record(y0=True, y1=True)]
        assert edge_has_transfer_event(recs) is False

    def test_no_transfer_both_false(self) -> None:
        recs = [_make_paired_record(y0=False, y1=False)]
        assert edge_has_transfer_event(recs) is False


class TestSelectPerturbationEdges:
    def test_tier1_prioritised(self) -> None:
        # edge_transfer has y0≠y1 (Tier 1), edge_other has y0=y1 (Tier 2).
        records = [
            _make_paired_record(
                edge_id="edge_other",
                candidate_memory_id="mem_1",
                y0=True,
                y1=True,
            ),
            _make_paired_record(
                edge_id="edge_transfer",
                candidate_memory_id="mem_1",
                y0=False,
                y1=True,
            ),
        ]
        pool = _make_memory_pool("mem_1")
        sels = select_perturbation_edges(
            paired_records=records,
            memory_pool=pool,
            perturbation_budget=2,
            seed=7,
        )
        assert len(sels) >= 1
        # First selection should be the transfer edge.
        assert sels[0].edge_id == "edge_transfer"

    def test_budget_respected(self) -> None:
        records = [
            _make_paired_record(
                edge_id=f"e{i}",
                candidate_memory_id="mem_1",
                y0=False,
                y1=True,
                generation_seed=i,
            )
            for i in range(10)
        ]
        pool = _make_memory_pool("mem_1")
        sels = select_perturbation_edges(
            paired_records=records,
            memory_pool=pool,
            perturbation_budget=3,
            seed=7,
        )
        assert len(sels) <= 3

    def test_test_split_excluded(self) -> None:
        records = [
            _make_paired_record(
                edge_id="e_test",
                candidate_memory_id="mem_1",
                split_name="test",
                y0=False,
                y1=True,
            ),
        ]
        pool = _make_memory_pool("mem_1")
        sels = select_perturbation_edges(
            paired_records=records,
            memory_pool=pool,
            perturbation_budget=10,
            seed=7,
        )
        assert len(sels) == 0

    def test_one_perturbation_per_edge(self) -> None:
        records = [
            _make_paired_record(
                edge_id="e1",
                candidate_memory_id="mem_1",
                generation_seed=1,
                y0=False,
                y1=True,
            ),
            _make_paired_record(
                edge_id="e1",
                candidate_memory_id="mem_1",
                generation_seed=2,
                y0=True,
                y1=False,
            ),
        ]
        pool = _make_memory_pool("mem_1")
        sels = select_perturbation_edges(
            paired_records=records,
            memory_pool=pool,
            perturbation_budget=10,
            seed=7,
        )
        # Only one perturbation per edge.
        edge_ids = [s.edge_id for s in sels]
        assert len(edge_ids) == len(set(edge_ids))

    def test_deterministic_under_seed(self) -> None:
        records = [
            _make_paired_record(
                edge_id=f"e{i}",
                candidate_memory_id="mem_1",
                generation_seed=i,
                y0=False,
                y1=True,
            )
            for i in range(5)
        ]
        pool = _make_memory_pool("mem_1")
        s1 = select_perturbation_edges(
            paired_records=records, memory_pool=pool,
            perturbation_budget=3, seed=42,
        )
        s2 = select_perturbation_edges(
            paired_records=records, memory_pool=pool,
            perturbation_budget=3, seed=42,
        )
        assert [s.edge_id for s in s1] == [s.edge_id for s in s2]

    def test_invalid_records_skipped(self) -> None:
        records = [
            _make_paired_record(
                edge_id="e1",
                candidate_memory_id="mem_1",
                valid=False,
                y0=False,
                y1=True,
            ),
        ]
        pool = _make_memory_pool("mem_1")
        sels = select_perturbation_edges(
            paired_records=records, memory_pool=pool,
            perturbation_budget=10, seed=7,
        )
        assert len(sels) == 0


class TestSelectBalancedOperator:
    def test_balanced_operator_selection(self) -> None:
        """Select the less-used operator."""
        ops = [PreconditionPerturbation(), RequiredToolPerturbation()]
        usage = {"precondition": 50, "required_tool": 2}
        selected = select_balanced_operator(ops, usage)
        assert selected.name == "required_tool"

    def test_operator_selection_deterministic(self) -> None:
        """Same inputs produce same output across calls."""
        ops = [
            PreconditionPerturbation(),
            RequiredCapabilityPerturbation(),
        ]
        usage = {"precondition": 5, "required_capability": 5}
        r1 = select_balanced_operator(ops, usage)
        r2 = select_balanced_operator(ops, usage)
        assert r1.name == r2.name

    def test_operator_priority_tiebreak(self) -> None:
        """Equal usage counts fall back to priority order."""
        ops = [
            RequiredToolPerturbation(),
            PreconditionPerturbation(),
        ]
        usage = {"precondition": 10, "required_tool": 10}
        selected = select_balanced_operator(ops, usage)
        assert selected.name == "precondition"
