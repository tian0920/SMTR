"""Tests for P2 perturbation runner (清单 §16-§17).

Verifies:
  - Y0 and Y_original reused from original paired record.
  - Fail-fast on task_id / receiver / seed mismatch.
  - Dry-run produces placeholder.
"""

from __future__ import annotations

import random

import pytest

from smtr.core.types import AgentProfile, MemoryRoutingCard, ReceiverState
from smtr.intervention.perturbation_runner import (
    find_original_paired_record,
    run_perturbed_exposure_branch,
)
from smtr.intervention.perturbation_schema import PerturbationSpec
from smtr.intervention.transfer_perturbation import PreconditionPerturbation


def _make_card(**kw: object) -> MemoryRoutingCard:
    base = dict(
        memory_id="mem_001",
        goal_summary="Step 1: inspect\nStep 2: query",
        task_tags=("database",),
        required_tools=("sql_read",),
        required_capabilities=("query_optimization",),
        execution_role_tags=("executor",),
        environment_constraints=("read_write_access",),
        precondition_tags=("schema_inspected",),
    )
    base.update(kw)
    return MemoryRoutingCard.model_validate(base)


def _make_receiver() -> ReceiverState:
    return ReceiverState(
        task_id="task_1",
        scenario="database",
        task_instruction="do stuff",
        receiver=AgentProfile(
            agent_id="agent_a",
            role="executor",
            capabilities=("query_optimization",),
            tool_names=("sql_read", "sql_write"),
        ),
        environment_signature=("read_write_access",),
    )


def _make_paired_record(
    y0: bool = False, y1: bool = True,
) -> dict:
    return {
        "task_id": "task_1",
        "receiver_agent_id": "agent_a",
        "candidate_memory_id": "mem_001",
        "generation_seed": 42,
        "edge_id": "edge_1",
        "valid": True,
        "withhold": {"team_success": y0},
        "share": {"team_success": y1},
    }


def _make_spec(card: MemoryRoutingCard, perturbed: MemoryRoutingCard) -> PerturbationSpec:
    from smtr.intervention.transfer_perturbation import build_perturbation_spec
    from smtr.intervention.transfer_perturbation import PerturbedMemory

    pm = PerturbedMemory(
        card=perturbed,
        changed_field="precondition_tags",
        original_value=card.precondition_tags,
        perturbed_value=perturbed.precondition_tags,
        perturbation_type="precondition",
    )
    return build_perturbation_spec(
        task_id="task_1",
        receiver_agent_id="agent_a",
        candidate_memory_id="mem_001",
        perturbed=pm,
        original_card=card,
        source_record_id="edge_1",
        control_group_key="task_1::agent_a::42",
        generation_seed=42,
    )


class TestRunPerturbedBranch:
    def test_reuses_y0_and_y_original(self) -> None:
        card = _make_card()
        recv = _make_receiver()
        op = PreconditionPerturbation()
        result = op.perturb(card, recv, rng=random.Random(7))
        # Perturbed card must have different memory_id for validation.
        pert_card = result.card.model_copy(
            update={"memory_id": "mem_001_pert"}
        )
        spec = _make_spec(card, pert_card)
        rec = _make_paired_record(y0=True, y1=False)

        outcome = run_perturbed_exposure_branch(
            original_paired_record=rec,
            perturbation_spec=spec,
            perturbed_memory_card=pert_card,
            original_memory_card=card,
            dry_run=True,
        )
        assert outcome.y0 is True
        assert outcome.y_original is False

    def test_dry_run_placeholder(self) -> None:
        card = _make_card()
        recv = _make_receiver()
        op = PreconditionPerturbation()
        result = op.perturb(card, recv, rng=random.Random(7))
        pert_card = result.card.model_copy(
            update={"memory_id": "mem_001_pert"}
        )
        spec = _make_spec(card, pert_card)
        rec = _make_paired_record()

        outcome = run_perturbed_exposure_branch(
            original_paired_record=rec,
            perturbation_spec=spec,
            perturbed_memory_card=pert_card,
            original_memory_card=card,
            dry_run=True,
        )
        assert outcome.y_perturbed is False
        assert outcome.runtime_metadata["dry_run"] is True
        assert outcome.perturbed_branch_id.startswith("dry_")

    def test_task_id_mismatch_raises(self) -> None:
        card = _make_card()
        recv = _make_receiver()
        op = PreconditionPerturbation()
        result = op.perturb(card, recv, rng=random.Random(7))
        pert_card = result.card.model_copy(
            update={"memory_id": "mem_001_pert"}
        )
        spec = _make_spec(card, pert_card)
        rec = _make_paired_record()
        rec["task_id"] = "wrong_task"

        with pytest.raises(ValueError, match="task_id mismatch"):
            run_perturbed_exposure_branch(
                original_paired_record=rec,
                perturbation_spec=spec,
                perturbed_memory_card=pert_card,
                original_memory_card=card,
                dry_run=True,
            )

    def test_seed_mismatch_raises(self) -> None:
        card = _make_card()
        recv = _make_receiver()
        op = PreconditionPerturbation()
        result = op.perturb(card, recv, rng=random.Random(7))
        pert_card = result.card.model_copy(
            update={"memory_id": "mem_001_pert"}
        )
        spec = _make_spec(card, pert_card)
        rec = _make_paired_record()
        rec["generation_seed"] = 999

        with pytest.raises(ValueError, match="generation_seed mismatch"):
            run_perturbed_exposure_branch(
                original_paired_record=rec,
                perturbation_spec=spec,
                perturbed_memory_card=pert_card,
                original_memory_card=card,
                dry_run=True,
            )

    def test_receiver_mismatch_raises(self) -> None:
        card = _make_card()
        recv = _make_receiver()
        op = PreconditionPerturbation()
        result = op.perturb(card, recv, rng=random.Random(7))
        pert_card = result.card.model_copy(
            update={"memory_id": "mem_001_pert"}
        )
        spec = _make_spec(card, pert_card)
        rec = _make_paired_record()
        rec["receiver_agent_id"] = "wrong_agent"

        with pytest.raises(ValueError, match="receiver_agent_id mismatch"):
            run_perturbed_exposure_branch(
                original_paired_record=rec,
                perturbation_spec=spec,
                perturbed_memory_card=pert_card,
                original_memory_card=card,
                dry_run=True,
            )


class TestFindOriginalPairedRecord:
    def test_found(self) -> None:
        records = [
            _make_paired_record(),
            {**_make_paired_record(), "task_id": "task_2"},
        ]
        found = find_original_paired_record(
            records, "task_1", "agent_a", "mem_001", 42,
        )
        assert found is not None
        assert found["task_id"] == "task_1"

    def test_not_found(self) -> None:
        records = [_make_paired_record()]
        found = find_original_paired_record(
            records, "task_99", "agent_a", "mem_001", 42,
        )
        assert found is None
